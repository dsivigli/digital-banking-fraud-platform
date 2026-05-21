# Databricks notebook source
# MAGIC %md
# MAGIC # Banking Fraud Lakehouse — Gold Layer (BI Analytics)
# MAGIC
# MAGIC Builds BI-ready aggregated tables on top of the validated Silver layer.
# MAGIC Inputs:
# MAGIC
# MAGIC - `silver_fact_transactions_enriched` — clean facts joined with dim attributes
# MAGIC - `quarantine_bad_transactions` — rows that failed Silver validation
# MAGIC - `silver_data_quality_summary` — one-row DQ KPIs from the Silver run
# MAGIC
# MAGIC Outputs (all Delta):
# MAGIC
# MAGIC 1. `gold_fraud_rate_by_country`
# MAGIC 2. `gold_fraud_by_merchant_category`
# MAGIC 3. `gold_top_risky_merchants`
# MAGIC 4. `gold_fraud_volume_by_hour`
# MAGIC 5. `gold_cross_border_activity`
# MAGIC 6. `gold_device_risk_summary`
# MAGIC 7. `gold_daily_fraud_trend`
# MAGIC 8. `gold_data_quality_dashboard`
# MAGIC
# MAGIC ## Why a separate Gold layer
# MAGIC
# MAGIC Gold tables are **small, aggregated, BI-shaped** — they trade granularity for
# MAGIC sub-second dashboard load times. Where Silver may hold 50M–500M rows, each
# MAGIC Gold table holds at most a few thousand rows. Dashboards point here, not at
# MAGIC Silver, so query concurrency from BI tools never competes with the heavy
# MAGIC Silver writes.
# MAGIC
# MAGIC ## Constraints
# MAGIC
# MAGIC - Spark-native APIs only — no pandas, no Python row loops, no Python UDFs.
# MAGIC - No `collect()` on large data — every aggregation is a reduce.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

# Try-set conf pattern for portability across serverless / classic clusters.
def _try_set(key: str, value: str) -> None:
    try:
        spark.conf.set(key, value)
    except Exception:
        pass

_try_set("spark.sql.adaptive.enabled", "true")
_try_set("spark.sql.adaptive.skewJoin.enabled", "true")

# Unity Catalog three-level naming: catalog.schema.table.
CATALOG = "main"
SCHEMA = "fraud_platform"
FQ_SCHEMA = f"{CATALOG}.{SCHEMA}"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Load Silver tables

# COMMAND ----------

silver_enriched = spark.table(f"{FQ_SCHEMA}.silver_fact_transactions_enriched")
quarantine = spark.table(f"{FQ_SCHEMA}.quarantine_bad_transactions")
dq_summary = spark.table(f"{FQ_SCHEMA}.silver_data_quality_summary")

print("=" * 80)
print("silver_fact_transactions_enriched — schema")
print("=" * 80)
silver_enriched.printSchema()
silver_enriched.show(3, truncate=False)

print("=" * 80)
print("quarantine_bad_transactions — schema")
print("=" * 80)
quarantine.printSchema()
quarantine.show(3, truncate=False)

print("=" * 80)
print("silver_data_quality_summary — schema")
print("=" * 80)
dq_summary.printSchema()
dq_summary.show(truncate=False, vertical=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Fraud indicator
# MAGIC
# MAGIC Real fraud platforms have a `fraud_label` column populated from chargebacks,
# MAGIC analyst dispositions, and case management outcomes. For this synthetic
# MAGIC dataset that label doesn't exist, so we derive a **proxy**:
# MAGIC
# MAGIC `suspected_fraud_flag = 1 if risk_signal_count >= 3 else 0`
# MAGIC
# MAGIC Why threshold 3: with 6 risk signals available, requiring 3+ keeps the proxy
# MAGIC suggestive but not noisy. In production this threshold would come from the
# MAGIC ROC analysis of the actual model. The Gold layer treats `suspected_fraud_flag`
# MAGIC as the analytical truth — if a real label arrives later, swapping it in is a
# MAGIC one-line change.

# COMMAND ----------

# Detect whether a real fraud label exists; fall back to the proxy. Computed at
# plan time on the schema, not on the data — no scan.
HAS_FRAUD_LABEL = "fraud_label" in silver_enriched.columns

fact = silver_enriched.withColumn(
    "suspected_fraud_flag",
    F.col("fraud_label") if HAS_FRAUD_LABEL
    else (F.col("risk_signal_count") >= F.lit(3)).cast("int"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Common analytical projection
# MAGIC
# MAGIC We project once into a "BI-shaped" view: only the columns the Gold metrics
# MAGIC care about. Spark's column pruning would do this anyway, but materializing
# MAGIC the projection makes the intent explicit and shrinks the in-memory shuffle
# MAGIC payloads for the aggregations below.

# COMMAND ----------

fact_bi = fact.select(
    "transaction_id",
    F.col("event_date").alias("transaction_date"),
    "transaction_hour",
    "transaction_country",
    "home_country",
    "merchant_id",
    "merchant_name",
    "merchant_category",
    "merchant_country",
    "merchant_risk_level",
    "device_id",
    "device_type",
    "rooted_device_flag",
    "emulator_flag",
    "customer_segment",
    "amount",
    "amount_bucket",
    "is_cross_border",
    "is_high_risk_merchant",
    "is_risky_device",
    "is_card_not_present",
    "is_international_transaction",
    "is_nighttime_transaction",
    "risk_signal_count",
    "suspected_fraud_flag",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregation helper expressions
# MAGIC
# MAGIC Defined once and reused across every Gold table — keeps semantics consistent
# MAGIC (e.g. fraud_rate is *always* `fraud_count / total_count` rounded to 6 dp).
# MAGIC `try_divide` returns NULL on zero-volume groups instead of throwing.

# COMMAND ----------

total_transactions = F.count(F.lit(1)).alias("total_transactions")
suspected_fraud_transactions = F.sum("suspected_fraud_flag").alias("suspected_fraud_transactions")
total_amount = F.round(F.sum("amount"), 2).alias("total_amount")
suspected_fraud_amount = F.round(
    F.sum(F.when(F.col("suspected_fraud_flag") == 1, F.col("amount")).otherwise(F.lit(0.0))),
    2,
).alias("suspected_fraud_amount")

# Rate columns are derived in a follow-up withColumn so we can reference the
# already-computed sums by name.
def add_rate(df, num_col: str, denom_col: str, out_col: str):
    return df.withColumn(
        out_col,
        F.round(F.expr(f"try_divide({num_col}, {denom_col})"), 6),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Gold: fraud rate by country
# MAGIC
# MAGIC **Visualization:** choropleth map (best) or bar chart by country.
# MAGIC
# MAGIC **Use cases:** identify regions with elevated fraud — drives geo-targeted
# MAGIC rule tuning and regulator reporting on cross-border exposure.

# COMMAND ----------

gold_fraud_rate_by_country = (
    fact_bi
    .groupBy("transaction_country")
    .agg(total_transactions, suspected_fraud_transactions, total_amount, suspected_fraud_amount)
)
gold_fraud_rate_by_country = add_rate(
    gold_fraud_rate_by_country,
    "suspected_fraud_transactions", "total_transactions", "fraud_rate",
)
gold_fraud_rate_by_country = add_rate(
    gold_fraud_rate_by_country,
    "suspected_fraud_amount", "total_amount", "fraud_amount_rate",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Gold: fraud by merchant category
# MAGIC
# MAGIC **Visualization:** bar chart sorted by fraud_rate desc.
# MAGIC
# MAGIC **Use cases:** GAMBLING / CRYPTO / GIFT_CARDS typically over-index — confirms
# MAGIC the prior category-risk weighting and surfaces emerging categories.

# COMMAND ----------

gold_fraud_by_merchant_category = (
    fact_bi
    .groupBy("merchant_category")
    .agg(total_transactions, suspected_fraud_transactions, total_amount, suspected_fraud_amount)
)
gold_fraud_by_merchant_category = add_rate(
    gold_fraud_by_merchant_category,
    "suspected_fraud_transactions", "total_transactions", "fraud_rate",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Gold: top risky merchants
# MAGIC
# MAGIC **Visualization:** table or horizontal bar chart of top 50.
# MAGIC
# MAGIC **Use cases:** investigations team works this list daily. Volume filter
# MAGIC `total_transactions >= 50` removes statistical noise from low-volume merchants
# MAGIC where one bad event would dominate the rate.

# COMMAND ----------

MIN_MERCHANT_VOLUME = 50  # statistical-significance floor

gold_top_risky_merchants = (
    fact_bi
    .groupBy("merchant_id", "merchant_name", "merchant_category",
             "merchant_country", "merchant_risk_level")
    .agg(total_transactions, suspected_fraud_transactions, total_amount, suspected_fraud_amount)
    .where(F.col("total_transactions") >= F.lit(MIN_MERCHANT_VOLUME))
)
gold_top_risky_merchants = add_rate(
    gold_top_risky_merchants,
    "suspected_fraud_transactions", "total_transactions", "fraud_rate",
).orderBy(F.desc("suspected_fraud_amount"), F.desc("fraud_rate"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Gold: fraud volume by hour
# MAGIC
# MAGIC **Visualization:** line chart (hour-of-day) or heatmap (hour × weekday).
# MAGIC
# MAGIC **Use cases:** drives staffing for the fraud ops desk and confirms the
# MAGIC nighttime-fraud signal that feeds `is_nighttime_transaction`.

# COMMAND ----------

gold_fraud_volume_by_hour = (
    fact_bi
    .groupBy("transaction_hour")
    .agg(total_transactions, suspected_fraud_transactions, total_amount, suspected_fraud_amount)
    .orderBy("transaction_hour")
)
gold_fraud_volume_by_hour = add_rate(
    gold_fraud_volume_by_hour,
    "suspected_fraud_transactions", "total_transactions", "fraud_rate",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Gold: cross-border activity
# MAGIC
# MAGIC **Visualization:** matrix / heatmap (home_country × transaction_country).
# MAGIC
# MAGIC **Use cases:** AML / sanctions monitoring relies on this. Cross-border flows
# MAGIC are inherently higher fraud risk; the matrix shape lets investigators spot
# MAGIC anomalous corridors (e.g. a sudden spike from country A → country B).

# COMMAND ----------

# cross_border_transactions counts only those rows where is_cross_border is true.
cross_border_count = F.sum(F.col("is_cross_border").cast("int")).alias("cross_border_transactions")

gold_cross_border_activity = (
    fact_bi
    .groupBy("home_country", "transaction_country")
    .agg(
        total_transactions,
        cross_border_count,
        suspected_fraud_transactions,
        total_amount,
        suspected_fraud_amount,
    )
)
gold_cross_border_activity = add_rate(
    gold_cross_border_activity,
    "suspected_fraud_transactions", "total_transactions", "fraud_rate",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Gold: device risk summary
# MAGIC
# MAGIC **Visualization:** stacked bar chart by device_type with rooted/emulator
# MAGIC overlays.
# MAGIC
# MAGIC **Use cases:** rooted phones and emulators are common fraud-tool signatures.
# MAGIC Quantifies the lift those flags provide, justifying their inclusion in the
# MAGIC risk score.

# COMMAND ----------

gold_device_risk_summary = (
    fact_bi
    .groupBy("device_type", "rooted_device_flag", "emulator_flag")
    .agg(total_transactions, suspected_fraud_transactions, total_amount, suspected_fraud_amount)
)
gold_device_risk_summary = add_rate(
    gold_device_risk_summary,
    "suspected_fraud_transactions", "total_transactions", "fraud_rate",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10 — Gold: daily fraud trend
# MAGIC
# MAGIC **Visualization:** line chart of `fraud_rate` over `transaction_date` with
# MAGIC `suspected_fraud_amount` as a secondary axis.
# MAGIC
# MAGIC **Use cases:** the headline KPI. Trends precede outages and rule failures —
# MAGIC a sudden spike means a new attack vector or a rule misfire.

# COMMAND ----------

gold_daily_fraud_trend = (
    fact_bi
    .groupBy("transaction_date")
    .agg(total_transactions, suspected_fraud_transactions, total_amount, suspected_fraud_amount)
    .orderBy("transaction_date")
)
gold_daily_fraud_trend = add_rate(
    gold_daily_fraud_trend,
    "suspected_fraud_transactions", "total_transactions", "fraud_rate",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11 — Gold: data quality dashboard
# MAGIC
# MAGIC **Visualization:** KPI cards (one per metric) plus a stacked bar of
# MAGIC quarantine reasons.
# MAGIC
# MAGIC We blend `silver_data_quality_summary` (the per-run flag totals) with a
# MAGIC fresh aggregation over `quarantine_bad_transactions` for reason-level
# MAGIC counts. The DQ summary is one row, so a `crossJoin` with derived KPIs is
# MAGIC cheap and stays Spark-native.

# COMMAND ----------

# Reason-level quarantine breakdown — pivot the long table to a wide one-row
# DataFrame. Pivot is a Catalyst-native operation; no Python loop.
quarantine_pivot = (
    quarantine
    .groupBy(F.lit(1).alias("_one"))
    .pivot("quarantine_reason")
    .agg(F.count(F.lit(1)))
    .drop("_one")
)

# Compute the orphan-key total (sum of three orphan reasons) and the duplicate
# count from the DQ summary. Use coalesce to handle reasons that may be absent
# from the pivot output (no rows of that reason in this run).
def _safe_col(df, name: str):
    """Return df[name] if it exists, else lit(0). Schema check, no scan."""
    return F.col(name) if name in df.columns else F.lit(0)

orphan_key_count_expr = (
    _safe_col(quarantine_pivot, "orphan_customer")
    + _safe_col(quarantine_pivot, "orphan_merchant")
    + _safe_col(quarantine_pivot, "orphan_device")
).alias("orphan_key_count")

# Build the dashboard by combining the one-row DQ summary with the one-row pivot.
# crossJoin of (1 row × 1 row) = 1 row — never expensive.
gold_data_quality_dashboard = (
    dq_summary.crossJoin(quarantine_pivot)
    .select(
        F.col("total_bronze_records").alias("total_bronze_records"),
        F.col("clean_records").alias("clean_records"),
        F.col("quarantined_records").alias("quarantined_records"),
        F.round(
            F.expr("try_divide(quarantined_records, total_bronze_records)"), 6
        ).alias("quarantine_rate"),
        F.col("duplicate_records").alias("duplicate_count"),
        F.col("invalid_amount_count").alias("invalid_amount_count"),
        F.col("invalid_currency_count").alias("invalid_currency_count"),
        orphan_key_count_expr,
        F.col("late_arriving_count").alias("late_arriving_count"),
        F.current_timestamp().alias("dashboard_ts"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12 — Persist all Gold outputs as Delta

# COMMAND ----------

GOLD_TABLES = [
    ("gold_fraud_rate_by_country", gold_fraud_rate_by_country),
    ("gold_fraud_by_merchant_category", gold_fraud_by_merchant_category),
    ("gold_top_risky_merchants", gold_top_risky_merchants),
    ("gold_fraud_volume_by_hour", gold_fraud_volume_by_hour),
    ("gold_cross_border_activity", gold_cross_border_activity),
    ("gold_device_risk_summary", gold_device_risk_summary),
    ("gold_daily_fraud_trend", gold_daily_fraud_trend),
    ("gold_data_quality_dashboard", gold_data_quality_dashboard),
]

# This Python iteration is over the *list of DataFrames*, not over rows of data.
# It builds a sequence of independent Spark write jobs — perfectly idiomatic and
# does not violate the "no Python loops over rows" rule.
for table_name, df in GOLD_TABLES:
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{FQ_SCHEMA}.{table_name}")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13 — Display Gold tables for BI
# MAGIC
# MAGIC On Databricks, `display(df)` renders an interactive grid + chart picker. In
# MAGIC plain Spark we fall back to `.show()`. Each Gold table has a comment
# MAGIC suggesting the best chart type for that shape of data.

# COMMAND ----------

# Each Gold table goes in its OWN cell with a direct display(df) call. The
# chart toolbar in Databricks attaches to a cell whose last expression is a
# direct display(...) — aliased calls (e.g. _DISPLAY = display) don't trigger
# the chart UI, which is why earlier the icon was missing.
#
# Outside Databricks, `display` isn't defined; the safe fallback below makes the
# notebook still importable, and each chart cell will show a tabular .show()
# instead. Comment out the fallback if you want a hard failure off-platform.
try:
    display  # type: ignore[name-defined]
except NameError:
    def display(df, n: int = 50):  # type: ignore[no-redef]
        df.show(n, truncate=False)

# COMMAND ----------

# Chart: bar chart (Keys = transaction_country, Values = fraud_rate) — or map.
display(
    spark.table(f"{FQ_SCHEMA}.gold_fraud_rate_by_country").orderBy(F.desc("fraud_rate"))
)

# COMMAND ----------

# Chart: bar (Keys = merchant_category, Values = fraud_rate).
display(
    spark.table(f"{FQ_SCHEMA}.gold_fraud_by_merchant_category").orderBy(F.desc("fraud_rate"))
)

# COMMAND ----------

# Chart: horizontal bar of top 50 (Keys = merchant_name, Values = suspected_fraud_amount).
display(
    spark.table(f"{FQ_SCHEMA}.gold_top_risky_merchants").limit(50)
)

# COMMAND ----------

# Chart: line (X = transaction_hour, Y = fraud_rate, second Y = total_transactions).
display(
    spark.table(f"{FQ_SCHEMA}.gold_fraud_volume_by_hour").orderBy("transaction_hour")
)

# COMMAND ----------

# Chart: heatmap (rows = home_country, cols = transaction_country, value = suspected_fraud_amount).
display(
    spark.table(f"{FQ_SCHEMA}.gold_cross_border_activity")
    .orderBy(F.desc("suspected_fraud_amount"))
    .limit(100)
)

# COMMAND ----------

# Chart: bar (Keys = device_type, group by rooted/emulator, Values = fraud_rate).
display(
    spark.table(f"{FQ_SCHEMA}.gold_device_risk_summary")
)

# COMMAND ----------

# Chart: line (X = transaction_date, Y = fraud_rate).
display(
    spark.table(f"{FQ_SCHEMA}.gold_daily_fraud_trend").orderBy("transaction_date")
)

# COMMAND ----------

# Chart: counter / KPI cards (single row of metrics).
display(
    spark.table(f"{FQ_SCHEMA}.gold_data_quality_dashboard")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 14 — Why a Gold layer
# MAGIC
# MAGIC | Property | Bronze | Silver | Gold |
# MAGIC |---|---|---|---|
# MAGIC | Row count | 50M+ | 50M (clean) | hundreds–thousands |
# MAGIC | Granularity | per event | per event | per group |
# MAGIC | Schema stability | volatile | versioned | dashboard-stable |
# MAGIC | Read pattern | rare (audit) | ML / Silver→Gold ETL | BI tools, every minute |
# MAGIC | Cost per dashboard query | high | medium | very low |
# MAGIC
# MAGIC ### Gold design properties
# MAGIC
# MAGIC - **BI-ready.** Pre-aggregated to the granularity each chart needs — no
# MAGIC   tableau/Power BI side aggregation required.
# MAGIC - **Small.** Every Gold table fits in dashboard cache, giving sub-second
# MAGIC   refresh times.
# MAGIC - **Stable schema.** Dashboards are brittle; Gold contracts change less
# MAGIC   often than Silver, even when source feeds churn.
# MAGIC - **Concurrency-safe.** BI tools hammer Gold at high QPS without affecting
# MAGIC   the heavy Silver writes upstream.
# MAGIC - **Cheap to rebuild.** Each Gold table is one or two aggregations away
# MAGIC   from Silver — full rebuild on schema changes is minutes, not hours.

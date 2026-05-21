# Databricks notebook source
# MAGIC %md
# MAGIC # Banking Fraud Lakehouse — ML Feature Table
# MAGIC
# MAGIC Builds a transaction-level feature table for training, evaluating, and
# MAGIC scoring fraud detection models. Inputs:
# MAGIC
# MAGIC - `silver_fact_transactions_enriched` — clean facts joined with dim attributes
# MAGIC - `fraud_labels` — chargebacks / analyst dispositions per transaction
# MAGIC
# MAGIC Output (Delta):
# MAGIC
# MAGIC - `ml_transaction_fraud_features`
# MAGIC
# MAGIC ## Why feature engineering is critical in fraud detection
# MAGIC
# MAGIC - Raw transactions tell you *what* happened. Features tell you *what's
# MAGIC   unusual* — and unusualness is the signal fraud models exploit.
# MAGIC - Behavioral aggregates ("how does this customer normally spend?") consistently
# MAGIC   outperform raw amounts — fraud is defined relative to a customer's baseline.
# MAGIC - Velocity features (count of recent transactions) capture card-testing and
# MAGIC   account takeover patterns that are invisible at the per-row level.
# MAGIC
# MAGIC ## Why point-in-time correctness matters (LEAKAGE)
# MAGIC
# MAGIC A feature that uses a value from *after* the transaction timestamp leaks the
# MAGIC future into training. The model appears spectacular offline (because it
# MAGIC effectively saw the answer key) but collapses in production where the
# MAGIC future doesn't exist yet.
# MAGIC
# MAGIC In this notebook every windowed feature uses
# MAGIC `Window.rangeBetween(-N_seconds, -1)` — strictly **past** events, current
# MAGIC row excluded. We never reference rows downstream of the one being scored.
# MAGIC
# MAGIC ## Constraints
# MAGIC
# MAGIC - Spark-native only — no pandas, no Python row loops, no Python UDFs.
# MAGIC - No `collect()` on large data — every aggregation is a reduce or a window.
# MAGIC - LEFT join `fraud_labels` so unlabeled transactions still produce features
# MAGIC   (the label is filled with 0 = not fraud, the safest assumption for training).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

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

# Time window sizes in seconds. Using seconds (not days/hours) makes the
# rangeBetween math precise and matches the unix_timestamp ordering column.
SECONDS_1H = 60 * 60
SECONDS_24H = 24 * SECONDS_1H
SECONDS_7D = 7 * SECONDS_24H
SECONDS_30D = 30 * SECONDS_24H

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Load and join
# MAGIC
# MAGIC LEFT join `fraud_labels` so every clean transaction yields a feature row
# MAGIC even if the label hasn't been resolved yet (chargebacks have a 60–120 day
# MAGIC settlement window in practice — many recent transactions are unlabeled).
# MAGIC We default missing labels to 0; for *training* you'd typically filter to
# MAGIC labeled rows, but the feature table itself remains complete.

# COMMAND ----------

silver = spark.table(f"{FQ_SCHEMA}.silver_fact_transactions_enriched")

# fraud_labels may not exist yet on a fresh deployment — synthesize a stub so
# the notebook is self-contained for the exercise. The stub is empty (no labels)
# which exercises the "missing label → 0" code path without injecting bias.
def _load_or_stub_fraud_labels():
    if spark.catalog.tableExists(f"{FQ_SCHEMA}.fraud_labels"):
        return spark.table(f"{FQ_SCHEMA}.fraud_labels").select(
            F.col("transaction_id").cast("long").alias("transaction_id"),
            F.col("fraud_label").cast("int").alias("fraud_label"),
        )
    # Stub: empty DataFrame with the expected schema. LEFT join + coalesce(0)
    # then assigns the default label to every row.
    print("fraud_labels not found — using empty stub (every fraud_label defaults to 0).")
    return spark.createDataFrame(
        [], "transaction_id LONG, fraud_label INT"
    )

fraud_labels = _load_or_stub_fraud_labels()

# Join on transaction_id. We do this BEFORE the window features so the label is
# available on every feature row, but we never reference fraud_label inside any
# window — labels must never feed into features (that's the most direct leakage).
labeled = (
    silver
    .join(fraud_labels, on="transaction_id", how="left")
    .withColumn("fraud_label", F.coalesce(F.col("fraud_label"), F.lit(0)))
)

# Cast transaction_ts to long seconds for use as the window ordering column.
# rangeBetween needs a numeric ordering — using seconds keeps the math exact.
# This also normalizes any timezone-arithmetic away into a single epoch dimension.
labeled = labeled.withColumn("ts_seconds", F.unix_timestamp("transaction_ts"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Repartition once for window co-location
# MAGIC
# MAGIC All features below use `Window.partitionBy("customer_id")`. Repartitioning
# MAGIC by `customer_id` once up front means every subsequent window evaluates on
# MAGIC already-co-located data — Spark performs *one* shuffle, not one per window.
# MAGIC
# MAGIC At 500M+ transactions and 1M customers, this is the difference between a
# MAGIC ~10-shuffle plan and a 1-shuffle plan.

# COMMAND ----------

# We don't .cache() because Databricks Serverless blocks PERSIST. AQE + Photon's
# disk cache reuse the read efficiently anyway. On classic clusters you could
# add a guarded cache (see notebook 02 for the pattern) for a small speedup.
labeled = labeled.repartition("customer_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Window specs (past-only, current row EXCLUDED)
# MAGIC
# MAGIC ### Why current row is excluded
# MAGIC
# MAGIC `rangeBetween(-N, -1)` means: include rows whose `ts_seconds` is in the
# MAGIC closed interval `[current - N, current - 1]`. This:
# MAGIC
# MAGIC - **Excludes the current transaction itself** — preventing the trivial
# MAGIC   leakage of "this row's amount inflates its own average".
# MAGIC - **Excludes rows from the future** — which is the main leakage concern.
# MAGIC - **Excludes ties at the exact same second** — pragmatic compromise for
# MAGIC   transactions submitted in the same second; in production you'd add a
# MAGIC   secondary tiebreaker like `transaction_id`.

# COMMAND ----------

# All velocity / behavior windows are partitioned by customer (the natural
# fraud-feature scope) and ordered by event time.
w_customer = Window.partitionBy("customer_id").orderBy("ts_seconds")

# Time-bounded windows. The .rangeBetween(-N, -1) form excludes the current row.
w_1h  = w_customer.rangeBetween(-SECONDS_1H,  -1)
w_24h = w_customer.rangeBetween(-SECONDS_24H, -1)
w_7d  = w_customer.rangeBetween(-SECONDS_7D,  -1)
w_30d = w_customer.rangeBetween(-SECONDS_30D, -1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Engineer features
# MAGIC
# MAGIC We build everything in a single chained projection. Catalyst fuses the
# MAGIC `withColumn` calls into one project node per stage and the underlying
# MAGIC physical plan computes all windows on the same shuffled partitions.
# MAGIC
# MAGIC ### Feature catalog
# MAGIC
# MAGIC | Group | Feature | Why it helps |
# MAGIC |---|---|---|
# MAGIC | Basic | `amount`, `transaction_hour` | Direct context — used as features and as inputs to ratios. |
# MAGIC | Basic | `is_nighttime_transaction` | Fraud disproportionately occurs 00:00–05:59. |
# MAGIC | Basic | `is_cross_border`, `is_international_transaction` | Higher base-rate fraud, especially card-not-present. |
# MAGIC | Basic | `card_present_flag`, `is_card_not_present` | CNP transactions are far easier to commit fraud on. |
# MAGIC | Basic | `merchant_risk_level` | Merchant category prior — gambling/crypto skew the rate. |
# MAGIC | Basic | `rooted_device_flag`, `emulator_flag` | Common fraud-tool signatures. |
# MAGIC | Basic | `risk_signal_count` | Cheap composite from Silver — strong baseline feature. |
# MAGIC | Velocity | `txn_count_1h`, `txn_count_24h`, `txn_count_7d` | Card-testing and ATO show up as bursts of recent activity. |
# MAGIC | Spend | `avg_amount_7d`, `avg_amount_30d`, `max_amount_30d` | Customer baseline — fraud is usually anomalous against this. |
# MAGIC | Spend | `amount_vs_avg_ratio` | Normalizes amount by personal baseline. |
# MAGIC | Geo | `distinct_countries_7d` | Travel patterns; sudden new countries = fraud signal. |
# MAGIC | Geo | `cross_border_txn_ratio_30d` | Customer's natural cross-border behavior. |
# MAGIC | Merchant | `high_risk_merchant_ratio_30d` | Behavioral preference for risky verticals. |
# MAGIC | Merchant | `distinct_merchants_30d` | Many unique merchants in a short window = card testing. |
# MAGIC | Device | `distinct_devices_30d` | Device hopping is a takeover signal. |
# MAGIC | Device | `risky_device_ratio_30d` | Customer's tendency to use rooted/emulator devices. |
# MAGIC | Time | `nighttime_txn_ratio_30d` | Personal "nighttime is normal" baseline. |
# MAGIC | Decline | `declined_txn_ratio_7d` | Recent declines often precede fraud (issuer trying to limit damage). |

# COMMAND ----------

# Helper: ratio with try_divide so zero-volume customers don't throw.
def _ratio(num, denom):
    return F.expr(f"try_divide({num}, {denom})")

# Boolean → int helper (booleans aren't summable in some engines; in Spark they
# are coercible but explicit cast keeps the plan readable).
def _b2i(col):
    return F.col(col).cast("int")

# COMMAND ----------

features = (
    labeled
    # ---------- Velocity (counts of past transactions) ----------
    # We count transaction_id over each time window. count() over a frame is
    # cheap (no aggregation state per row beyond a counter); these are the
    # workhorses of fraud detection.
    .withColumn("txn_count_1h",  F.count("transaction_id").over(w_1h))
    .withColumn("txn_count_24h", F.count("transaction_id").over(w_24h))
    .withColumn("txn_count_7d",  F.count("transaction_id").over(w_7d))

    # ---------- Spend baselines ----------
    # avg / max over time-bounded windows. NULL when no past transactions exist
    # (new customer); leave NULL — downstream model handles missing values.
    .withColumn("avg_amount_7d",  F.avg("amount").over(w_7d))
    .withColumn("avg_amount_30d", F.avg("amount").over(w_30d))
    .withColumn("max_amount_30d", F.max("amount").over(w_30d))

    # amount_vs_avg_ratio: how anomalous is this amount vs. the customer's
    # own 30-day baseline? > 5 means "this is 5x their typical spend" — strong
    # fraud signal. try_divide returns NULL if avg is NULL (new customer) or 0.
    .withColumn(
        "amount_vs_avg_ratio",
        _ratio("amount", "avg_amount_30d"),
    )

    # ---------- Geographic ----------
    # distinct_countries_7d: collect_set over the window, then size. Spark-native
    # — no UDF. Cost is bounded by the customer's recent transaction count.
    .withColumn(
        "distinct_countries_7d",
        F.size(F.collect_set("transaction_country").over(w_7d)),
    )

    # Cross-border ratio over 30 days: sum(is_cross_border)/count.
    .withColumn("_cross_border_sum_30d", F.sum(_b2i("is_cross_border")).over(w_30d))
    .withColumn("_cross_border_cnt_30d", F.count("transaction_id").over(w_30d))
    .withColumn(
        "cross_border_txn_ratio_30d",
        _ratio("_cross_border_sum_30d", "_cross_border_cnt_30d"),
    )

    # ---------- Merchant behavior ----------
    .withColumn(
        "_high_risk_merch_sum_30d",
        F.sum(_b2i("is_high_risk_merchant")).over(w_30d),
    )
    .withColumn(
        "high_risk_merchant_ratio_30d",
        _ratio("_high_risk_merch_sum_30d", "_cross_border_cnt_30d"),
    )
    .withColumn(
        "distinct_merchants_30d",
        F.size(F.collect_set("merchant_id").over(w_30d)),
    )

    # ---------- Device behavior ----------
    .withColumn(
        "distinct_devices_30d",
        F.size(F.collect_set("device_id").over(w_30d)),
    )
    .withColumn(
        "_risky_device_sum_30d",
        F.sum(_b2i("is_risky_device")).over(w_30d),
    )
    .withColumn(
        "risky_device_ratio_30d",
        _ratio("_risky_device_sum_30d", "_cross_border_cnt_30d"),
    )

    # ---------- Time-of-day behavior ----------
    .withColumn(
        "_nighttime_sum_30d",
        F.sum(_b2i("is_nighttime_transaction")).over(w_30d),
    )
    .withColumn(
        "nighttime_txn_ratio_30d",
        _ratio("_nighttime_sum_30d", "_cross_border_cnt_30d"),
    )

    # ---------- Decline behavior ----------
    # Recent declines often precede fraud — issuers try to limit the damage by
    # declining suspect transactions. A spike here is a strong leading indicator.
    .withColumn(
        "_declined_sum_7d",
        F.sum(F.when(F.col("transaction_status") == "DECLINED", 1).otherwise(0)).over(w_7d),
    )
    .withColumn("_declined_cnt_7d", F.count("transaction_id").over(w_7d))
    .withColumn(
        "declined_txn_ratio_7d",
        _ratio("_declined_sum_7d", "_declined_cnt_7d"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Project the final feature schema
# MAGIC
# MAGIC We keep the keys (transaction_id, customer_id, transaction_ts), the basic
# MAGIC per-event features, the engineered features, and the label. Helper columns
# MAGIC (those starting with `_`) are dropped.

# COMMAND ----------

ml_transaction_fraud_features = features.select(
    # Keys / time
    "transaction_id",
    "customer_id",
    "transaction_ts",
    "event_date",

    # Basic per-event features
    "amount",
    "transaction_hour",
    "is_nighttime_transaction",
    "is_cross_border",
    "is_international_transaction",
    "card_present_flag",
    "is_card_not_present",
    "merchant_risk_level",
    "rooted_device_flag",
    "emulator_flag",
    "risk_signal_count",

    # Velocity
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",

    # Spend
    F.round("avg_amount_7d", 2).alias("avg_amount_7d"),
    F.round("avg_amount_30d", 2).alias("avg_amount_30d"),
    F.round("max_amount_30d", 2).alias("max_amount_30d"),
    F.round("amount_vs_avg_ratio", 4).alias("amount_vs_avg_ratio"),

    # Geographic
    "distinct_countries_7d",
    F.round("cross_border_txn_ratio_30d", 4).alias("cross_border_txn_ratio_30d"),

    # Merchant
    F.round("high_risk_merchant_ratio_30d", 4).alias("high_risk_merchant_ratio_30d"),
    "distinct_merchants_30d",

    # Device
    "distinct_devices_30d",
    F.round("risky_device_ratio_30d", 4).alias("risky_device_ratio_30d"),

    # Time-of-day
    F.round("nighttime_txn_ratio_30d", 4).alias("nighttime_txn_ratio_30d"),

    # Decline
    F.round("declined_txn_ratio_7d", 4).alias("declined_txn_ratio_7d"),

    # Label
    "fraud_label",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Persist as Delta
# MAGIC
# MAGIC Partition by `event_date` so training jobs can read a date range with
# MAGIC partition pruning. For 500M+ rows you'd switch to Liquid Clustering on
# MAGIC `(event_date, customer_id)` — the same pattern used for the fact table.

# COMMAND ----------

(
    ml_transaction_fraud_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("event_date")
    .saveAsTable(f"{FQ_SCHEMA}.ml_transaction_fraud_features")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Schema, samples, summaries

# COMMAND ----------

ft = spark.table(f"{FQ_SCHEMA}.ml_transaction_fraud_features")

print("=" * 80)
print("ml_transaction_fraud_features — schema")
print("=" * 80)
ft.printSchema()

print("=" * 80)
print("Sample rows")
print("=" * 80)
ft.orderBy(F.desc("transaction_ts")).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Quality checks
# MAGIC
# MAGIC All checks are aggregations (reduces) — single-row results, never collects
# MAGIC of the full feature table.

# COMMAND ----------

# Null counts per feature. summary() on selected columns is a single pass.
print("Null and basic stats per feature (single distributed pass):")
ft.select(
    "amount",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "avg_amount_7d", "avg_amount_30d", "max_amount_30d", "amount_vs_avg_ratio",
    "distinct_countries_7d", "cross_border_txn_ratio_30d",
    "high_risk_merchant_ratio_30d", "distinct_merchants_30d",
    "distinct_devices_30d", "risky_device_ratio_30d",
    "nighttime_txn_ratio_30d", "declined_txn_ratio_7d",
).summary("count", "min", "max", "mean", "stddev").show(truncate=False)

# COMMAND ----------

# Fraud class imbalance. Fraud rates are typically 0.1%–2% — anything above 5%
# means the label join went wrong or the proxy is too aggressive.
print("Fraud label distribution (class imbalance):")
(
    ft.groupBy("fraud_label")
    .agg(F.count("*").alias("count"))
    .withColumn(
        "share",
        F.round(F.col("count") / F.sum("count").over(Window.partitionBy()), 4),
    )
    .orderBy("fraud_label")
    .show(truncate=False)
)

# COMMAND ----------

# Velocity-feature sanity: brand-new customers will have NULL/0 windowed values
# for their first transaction (no past events). This is correct, but a high
# fraction of all-zero velocity rows would suggest a join or window bug.
print("Velocity feature null/zero coverage:")
(
    ft.agg(
        F.sum(F.when(F.col("txn_count_1h").isNull(), 1).otherwise(0)).alias("null_1h"),
        F.sum(F.when(F.col("txn_count_24h").isNull(), 1).otherwise(0)).alias("null_24h"),
        F.sum(F.when(F.col("txn_count_7d").isNull(), 1).otherwise(0)).alias("null_7d"),
        F.sum(F.when(F.col("txn_count_1h") == 0, 1).otherwise(0)).alias("zero_1h"),
    ).show(truncate=False)
)

# COMMAND ----------

# Range checks: ratio columns should be in [0, 1]; amount_vs_avg_ratio is [0, ∞).
print("Ratio column ranges (should be in [0, 1] except amount_vs_avg_ratio):")
(
    ft.agg(
        F.min("cross_border_txn_ratio_30d").alias("min_xb"),
        F.max("cross_border_txn_ratio_30d").alias("max_xb"),
        F.min("high_risk_merchant_ratio_30d").alias("min_hrm"),
        F.max("high_risk_merchant_ratio_30d").alias("max_hrm"),
        F.min("nighttime_txn_ratio_30d").alias("min_night"),
        F.max("nighttime_txn_ratio_30d").alias("max_night"),
        F.min("declined_txn_ratio_7d").alias("min_dec"),
        F.max("declined_txn_ratio_7d").alias("max_dec"),
        F.max("amount_vs_avg_ratio").alias("max_amt_ratio"),
    ).show(truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Why this design
# MAGIC
# MAGIC ### Feature engineering > raw inputs
# MAGIC
# MAGIC Raw `amount` tells the model nothing about whether $500 is unusual *for
# MAGIC this customer*. `amount_vs_avg_ratio` does. Across every fraud benchmark
# MAGIC published in the last decade, behavioral aggregates dominate raw fields.
# MAGIC
# MAGIC ### Point-in-time correctness is non-negotiable
# MAGIC
# MAGIC Leakage is the #1 cause of fraud-model production failure. A model that
# MAGIC sees future data during training learns to predict things it cannot
# MAGIC actually predict. The `rangeBetween(-N, -1)` pattern guarantees:
# MAGIC
# MAGIC 1. **No future data.** Frame upper bound is `-1` second.
# MAGIC 2. **No self-leakage.** Current row excluded from its own averages.
# MAGIC 3. **Reproducibility.** Re-running on the same input produces identical
# MAGIC    feature values — backfills and live inference agree.
# MAGIC
# MAGIC ### How this table powers the rest of the platform
# MAGIC
# MAGIC | Use case | How |
# MAGIC |---|---|
# MAGIC | **Batch training** | Read with a date filter (`event_date BETWEEN ...`). Partition pruning = fast. |
# MAGIC | **Streaming inference** | The same windowed expressions can be ported to Structured Streaming with `withWatermark` — feature definitions stay identical, training/serving skew avoided. |
# MAGIC | **Real-time fraud scoring** | An online feature store (Feature Engineering on Databricks, Feast, etc.) materializes the latest per-customer aggregates from this table on a schedule; the scoring service reads them with sub-ms latency. |
# MAGIC | **Model monitoring** | Compute the same feature distributions on production traffic and compare to training-time stats. Drift on `nighttime_txn_ratio_30d` or `declined_txn_ratio_7d` often precedes fraud waves. |
# MAGIC
# MAGIC ### Scalability notes
# MAGIC
# MAGIC - **One shuffle, many windows.** All windows share `partitionBy(customer_id)`,
# MAGIC   so the planner shuffles once and computes every window on co-located data.
# MAGIC - **No `collect_list` over time-unbounded windows.** Every window is bounded
# MAGIC   in seconds — state per row is bounded by recent activity, not all history.
# MAGIC - **No Python UDFs.** Every expression is Catalyst-native; whole-stage
# MAGIC   codegen fuses them into one Java class per stage.
# MAGIC - **Partition by `event_date`** for training reads; switch to Liquid
# MAGIC   Clustering on `(event_date, customer_id)` past 200M rows.
# MAGIC - **Skew handling:** customer_id is naturally near-uniform, but a few
# MAGIC   power-users may dominate. AQE skew-join handling (enabled at top of
# MAGIC   notebook) splits oversized partitions transparently.

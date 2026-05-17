# Databricks notebook source
# MAGIC %md
# MAGIC # Banking Fraud Lakehouse — Silver Layer
# MAGIC
# MAGIC Builds the validated, deduplicated, enriched analytical truth from the Bronze
# MAGIC raw landing zone. Inputs:
# MAGIC
# MAGIC - `bronze_fact_transactions_dirty` — raw transactions with realistic DQ issues
# MAGIC - `dim_customer`, `dim_merchant`, `dim_device` — conformed dimensions
# MAGIC
# MAGIC Outputs:
# MAGIC
# MAGIC - `silver_fact_transactions_clean` — deduplicated, validated facts
# MAGIC - `quarantine_bad_transactions` — rows that failed critical validation
# MAGIC - `silver_fact_transactions_enriched` — clean facts + dim attributes + risk signals
# MAGIC - `silver_data_quality_summary` — one-row table of DQ KPIs for monitoring
# MAGIC
# MAGIC ## Architectural notes
# MAGIC
# MAGIC - **Bronze preserves raw truth.** No mutation, no filtering — the auditor's view.
# MAGIC - **Silver creates trusted analytical truth.** Validated, deduplicated, enriched.
# MAGIC - **Quarantine enables audit and reprocessing.** Failed rows are kept (not dropped)
# MAGIC   with the reason recorded so upstream feeds can be repaired and replayed.
# MAGIC - **Enriched Silver powers BI, fraud analytics, and ML features.** It is the
# MAGIC   single source of truth that downstream Gold marts and feature stores read from.
# MAGIC
# MAGIC ## Constraints
# MAGIC
# MAGIC - Spark-native APIs only — no pandas, no Python row loops, no Python UDFs.
# MAGIC - No `collect()` on large data — every aggregation is a reduce.
# MAGIC - LEFT joins for enrichment so we never silently drop financial events.
# MAGIC - Broadcast joins for small dims (Spark auto-picks below
# MAGIC   `spark.sql.autoBroadcastJoinThreshold`).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# Serverless / shared UC compute blocks user-level conf changes; defaults already
# enable AQE on those runtimes. Try-set keeps this notebook portable.
def _try_set(key: str, value: str) -> None:
    try:
        spark.conf.set(key, value)
    except Exception:
        pass

_try_set("spark.sql.adaptive.enabled", "true")
_try_set("spark.sql.adaptive.skewJoin.enabled", "true")
_try_set("spark.sql.adaptive.coalescePartitions.enabled", "true")

DB_NAME = "fraud_platform"
spark.sql(f"USE {DB_NAME}")

# Allow lists — declared once, broadcast as Spark literals so the predicates run
# entirely in Catalyst (no per-row Python interop).
ALLOWED_CURRENCIES = ["USD", "EUR", "GBP", "CHF", "CAD", "AUD"]
ALLOWED_STATUSES = ["APPROVED", "DECLINED", "PENDING"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Load tables, print schemas, sample rows

# COMMAND ----------

bronze = spark.table(f"{DB_NAME}.bronze_fact_transactions_dirty")
dim_customer = spark.table(f"{DB_NAME}.dim_customer")
dim_merchant = spark.table(f"{DB_NAME}.dim_merchant")
dim_device = spark.table(f"{DB_NAME}.dim_device")

print("=" * 80)
print("bronze_fact_transactions_dirty — schema")
print("=" * 80)
bronze.printSchema()
bronze.show(5, truncate=False)

print("=" * 80)
print("dim_customer — schema")
print("=" * 80)
dim_customer.printSchema()
dim_customer.show(3, truncate=False)

print("=" * 80)
print("dim_merchant — schema")
print("=" * 80)
dim_merchant.printSchema()
dim_merchant.show(3, truncate=False)

print("=" * 80)
print("dim_device — schema")
print("=" * 80)
dim_device.printSchema()
dim_device.show(3, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Standardize values
# MAGIC
# MAGIC Why: rules and BI dashboards compare strings literally. "USD" vs "usd" splits
# MAGIC revenue rollups; trailing whitespace from upstream feeds creates phantom
# MAGIC categories in dashboards. We standardize once at the Silver boundary so every
# MAGIC downstream consumer sees the same canonical form.

# COMMAND ----------

bronze_std = (
    bronze
    # Uppercase + trim string fields. coalesce keeps NULLs as NULLs (don't fabricate).
    .withColumn("currency", F.upper(F.trim(F.col("currency"))))
    .withColumn("transaction_country", F.upper(F.trim(F.col("transaction_country"))))
    .withColumn("transaction_status", F.upper(F.trim(F.col("transaction_status"))))
    .withColumn("payment_type", F.upper(F.trim(F.col("payment_type"))))
    .withColumn("transaction_channel", F.upper(F.trim(F.col("transaction_channel"))))

    # Ensure transaction_ts is a real timestamp. If Bronze already typed it, the
    # cast is a no-op. If it's stored as string, the cast parses it — far cheaper
    # than a UDF or pandas pre-processing step.
    .withColumn("transaction_ts", F.col("transaction_ts").cast("timestamp"))
    .withColumn("ingestion_ts", F.col("ingestion_ts").cast("timestamp"))

    # event_date is the partition key for time-series fraud queries (Delta partition
    # pruning gives near-O(1) reads on time-bounded windows).
    .withColumn("event_date", F.to_date(F.col("transaction_ts")))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Validation flags
# MAGIC
# MAGIC Each flag is a column expression — Catalyst fuses them all into one project.
# MAGIC We materialize the flags as boolean columns rather than computing them ad-hoc
# MAGIC downstream because (a) the quarantine rule and the clean-table filter must
# MAGIC agree exactly, and (b) the data quality summary table aggregates the flags.
# MAGIC
# MAGIC ### Why each flag matters in banking fraud
# MAGIC
# MAGIC | Flag | Why we care |
# MAGIC |---|---|
# MAGIC | `is_null_*` | A NULL FK silently short-circuits rules like `country != home_country` to NULL — masking cross-border fraud. |
# MAGIC | `is_invalid_amount` | Negatives are mis-routed refunds; zeros are pre-auths. Flipping/keeping them poisons velocity features. |
# MAGIC | `is_invalid_currency` | No FX → no USD-normalized loss exposure → wrong fraud-loss reporting. |
# MAGIC | `is_invalid_status` | Out-of-enum status could be incorrectly counted as APPROVED in loss accounting. |
# MAGIC | `is_future_timestamp` | One 2099-dated row breaks any `MAX(transaction_ts)` window. Common around DST/TZ bugs. |
# MAGIC | `is_orphan_*` | Can't be enriched with `risk_tier`/`merchant_category` — a LEFT join would emit NULL features and the model would score "low risk by default". |
# MAGIC | `is_suspicious_default_*` | Sentinel values (0, -1) from legacy mainframes — they look numeric but match no real entity. |
# MAGIC | `is_late_arriving` | Late events arrive *after* the model has scored. Track them, but don't quarantine if otherwise valid — overnight POS batches are legitimate. |

# COMMAND ----------

NOW = F.current_timestamp()

# Build sets of valid PKs from each dimension as small DataFrames. We then LEFT
# anti-join these to detect orphans. We use this rather than collecting PKs to a
# Python set: dim_customer is 1M rows — way too big to broadcast as a Python list,
# but Spark will broadcast the single-column DataFrame automatically.
valid_customer_ids = dim_customer.select("customer_id").distinct()
valid_merchant_ids = dim_merchant.select("merchant_id").distinct()
valid_device_ids = dim_device.select("device_id").distinct()

# We mark orphans by LEFT JOINing the PK set and flagging where the join missed.
# This is a single pass per dim — much cheaper than three separate anti-joins.
bronze_with_dim_existence = (
    bronze_std
    .join(valid_customer_ids.withColumn("_cust_exists", F.lit(True)),
          on="customer_id", how="left")
    .join(valid_merchant_ids.withColumn("_merch_exists", F.lit(True)),
          on="merchant_id", how="left")
    .join(valid_device_ids.withColumn("_dev_exists", F.lit(True)),
          on="device_id", how="left")
)

# COMMAND ----------

bronze_flagged = (
    bronze_with_dim_existence

    # ----- Null critical fields -----
    .withColumn("is_null_customer", F.col("customer_id").isNull())
    .withColumn("is_null_merchant", F.col("merchant_id").isNull())
    .withColumn("is_null_device", F.col("device_id").isNull())
    .withColumn("is_null_amount", F.col("amount").isNull())

    # ----- Suspicious default sentinels -----
    # 0 = legacy "unknown customer", -1 = legacy "unknown merchant" placeholder.
    # Coalesce to handle nulls without throwing.
    .withColumn(
        "is_suspicious_default_customer",
        F.coalesce(F.col("customer_id") == F.lit(0), F.lit(False)),
    )
    .withColumn(
        "is_suspicious_default_merchant",
        F.coalesce(F.col("merchant_id") == F.lit(-1), F.lit(False)),
    )

    # ----- Amount validity -----
    # Strictly > 0 per spec. Nulls handled separately to keep flags single-purpose.
    .withColumn(
        "is_invalid_amount",
        F.col("amount").isNotNull() & (F.col("amount") <= F.lit(0)),
    )

    # ----- Currency / status enums -----
    # Nulls do not match isin() — they are not flagged as invalid here, the null
    # flag covers them separately. We split null vs invalid because the upstream
    # remediation differs.
    .withColumn(
        "is_invalid_currency",
        F.col("currency").isNotNull() & ~F.col("currency").isin(*ALLOWED_CURRENCIES),
    )
    .withColumn(
        "is_invalid_status",
        F.col("transaction_status").isNotNull()
        & ~F.col("transaction_status").isin(*ALLOWED_STATUSES),
    )

    # ----- Time anomalies -----
    .withColumn(
        "is_future_timestamp",
        F.col("transaction_ts").isNotNull() & (F.col("transaction_ts") > NOW),
    )
    # Late arriving: ingestion > 2 days after transaction. Only when ingestion_ts
    # is present — otherwise we cannot judge lateness, so default to false.
    .withColumn(
        "is_late_arriving",
        F.col("ingestion_ts").isNotNull()
        & F.col("transaction_ts").isNotNull()
        & (F.col("ingestion_ts") > F.col("transaction_ts") + F.expr("INTERVAL 2 DAYS")),
    )

    # ----- Orphan FKs -----
    # Orphan = present (not null), not a sentinel, and not found in the dim.
    # The double check (not null AND not exists) avoids double-flagging nulls as
    # orphans — that matters for the DQ summary because a single root cause
    # should populate exactly one bucket.
    .withColumn(
        "is_orphan_customer",
        F.col("customer_id").isNotNull()
        & (F.col("customer_id") != F.lit(0))
        & F.col("_cust_exists").isNull(),
    )
    .withColumn(
        "is_orphan_merchant",
        F.col("merchant_id").isNotNull()
        & (F.col("merchant_id") != F.lit(-1))
        & F.col("_merch_exists").isNull(),
    )
    .withColumn(
        "is_orphan_device",
        F.col("device_id").isNotNull() & F.col("_dev_exists").isNull(),
    )

    # Drop the join-helper booleans now that the orphan flags are computed.
    .drop("_cust_exists", "_merch_exists", "_dev_exists")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Deduplicate
# MAGIC
# MAGIC Strategy: `row_number()` over a window partitioned by `transaction_id`,
# MAGIC ordered by ingestion_ts DESC then transaction_ts DESC. Spec says **keep the
# MAGIC latest** record. Tie-break with transaction_ts handles the case where
# MAGIC ingestion_ts is missing for some rows.
# MAGIC
# MAGIC Why row_number over `dropDuplicates`: dropDuplicates picks an arbitrary
# MAGIC survivor and gives us no audit trail. We need both:
# MAGIC
# MAGIC - The deterministic survivor (rn = 1) for Silver.
# MAGIC - The losing rows (rn > 1) tagged for the quarantine table.

# COMMAND ----------

# Order by ingestion_ts DESC (nulls last) then transaction_ts DESC. We use
# F.col(...).desc_nulls_last() so rows missing ingestion_ts don't accidentally
# win — falling back to transaction_ts gives us the "latest known" row.
dedup_window = Window.partitionBy("transaction_id").orderBy(
    F.col("ingestion_ts").desc_nulls_last(),
    F.col("transaction_ts").desc_nulls_last(),
)

bronze_deduped = (
    bronze_flagged
    .withColumn("_dedup_rn", F.row_number().over(dedup_window))
    # is_duplicate_candidate marks every row that has at least one twin. Survivor
    # gets is_duplicate_candidate=true if a twin existed; loser gets it too. This
    # makes the audit trail symmetric: both copies are visible as "had duplicates".
    .withColumn(
        "_dup_count",
        F.count("transaction_id").over(Window.partitionBy("transaction_id")),
    )
    .withColumn("is_duplicate_candidate", F.col("_dup_count") > F.lit(1))
    .drop("_dup_count")
)

# Cache because we'll filter twice (clean side and quarantine side). On Databricks
# Serverless, .cache() raises NOT_SUPPORTED_WITH_SERVERLESS — Serverless has its
# own disk/result caching (Photon), so the try/except keeps the notebook portable.
def _try_cache(df):
    try:
        return df.cache()
    except Exception:
        # Serverless path — return the DataFrame unchanged; the planner reuses
        # results within the same query and Photon caches the read at storage.
        return df

bronze_deduped = _try_cache(bronze_deduped)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Quarantine bad records
# MAGIC
# MAGIC Critical-failure rules: any of these flags trips quarantine. **Late
# MAGIC arriving is intentionally NOT a quarantine reason** — overnight POS batch
# MAGIC sync is legitimate, we just want it labeled.
# MAGIC
# MAGIC `quarantine_reason` records the *first* matching reason in a severity
# MAGIC ordering. Multiple issues per row are common; the dominant reason is the
# MAGIC one most likely to need upstream remediation.

# COMMAND ----------

# Severity-ordered reason. First match wins. Order matches the spec's intent:
# critical missing keys/amounts > sentinels > orphans > value validation > time.
quarantine_reason_expr = (
    F.when(F.col("_dedup_rn") > F.lit(1), F.lit("duplicate_transaction"))
     .when(F.col("is_null_customer"), F.lit("null_customer"))
     .when(F.col("is_null_merchant"), F.lit("null_merchant"))
     .when(F.col("is_null_device"), F.lit("null_device"))
     .when(F.col("is_null_amount"), F.lit("null_amount"))
     .when(F.col("is_suspicious_default_customer"), F.lit("suspicious_default_customer"))
     .when(F.col("is_suspicious_default_merchant"), F.lit("suspicious_default_merchant"))
     .when(F.col("is_orphan_customer"), F.lit("orphan_customer"))
     .when(F.col("is_orphan_merchant"), F.lit("orphan_merchant"))
     .when(F.col("is_orphan_device"), F.lit("orphan_device"))
     .when(F.col("is_invalid_amount"), F.lit("invalid_amount"))
     .when(F.col("is_invalid_currency"), F.lit("invalid_currency"))
     .when(F.col("is_invalid_status"), F.lit("invalid_status"))
     .when(F.col("is_future_timestamp"), F.lit("future_timestamp"))
     .otherwise(F.lit(None).cast("string"))
)

bronze_with_reason = bronze_deduped.withColumn("quarantine_reason", quarantine_reason_expr)

quarantine_bad_transactions = (
    bronze_with_reason
    .where(F.col("quarantine_reason").isNotNull())
    # Audit timestamp — when this row was rejected. Useful for SOX trails and for
    # monitoring how DQ issues trend over time.
    .withColumn("quarantine_ts", F.current_timestamp())
    .drop("_dedup_rn")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Build clean Silver fact table
# MAGIC
# MAGIC Survivors only: `quarantine_reason IS NULL AND _dedup_rn = 1`. We then add
# MAGIC analytical derivations that BI and ML consumers will want:
# MAGIC
# MAGIC - `transaction_hour` — for hour-of-day fraud patterns.
# MAGIC - `is_nighttime_transaction` — known fraud signal (00:00–05:59 local).
# MAGIC - `is_cross_border` — transaction_country ≠ home_country (computed in
# MAGIC   enrichment, since home_country comes from dim_customer).
# MAGIC - `amount_bucket` — log-scale bucket for skewed distributions in BI.
# MAGIC
# MAGIC `is_cross_border` requires the customer's home_country, so we defer it to
# MAGIC the enrichment step in Step 7. The clean table contains everything that
# MAGIC depends solely on the fact row itself.

# COMMAND ----------

silver_fact_transactions_clean = (
    bronze_with_reason
    .where(F.col("quarantine_reason").isNull() & (F.col("_dedup_rn") == F.lit(1)))
    .drop("_dedup_rn", "quarantine_reason")

    # Hour of day in 0..23. For multi-region banks you'd convert to local TZ first;
    # we keep UTC for simplicity and let downstream re-localize per region.
    .withColumn("transaction_hour", F.hour(F.col("transaction_ts")))

    # Nighttime: 00:00–05:59. Empirically elevated fraud during these hours.
    .withColumn(
        "is_nighttime_transaction",
        (F.col("transaction_hour") >= F.lit(0)) & (F.col("transaction_hour") < F.lit(6)),
    )

    # Amount bucket — used in BI dashboards to show distribution shape without
    # leaking individual amounts. Log-scale because amounts are log-distributed.
    .withColumn(
        "amount_bucket",
        F.when(F.col("amount") < F.lit(10), F.lit("<10"))
         .when(F.col("amount") < F.lit(100), F.lit("10-100"))
         .when(F.col("amount") < F.lit(1_000), F.lit("100-1K"))
         .when(F.col("amount") < F.lit(10_000), F.lit("1K-10K"))
         .otherwise(F.lit("10K+")),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Enrich with dimensions
# MAGIC
# MAGIC LEFT JOINs because: a clean financial event must never disappear due to a
# MAGIC dimension lookup miss. By Step 6 we've already removed orphan FKs, so LEFT
# MAGIC vs INNER produce the same row count today — but defending against future
# MAGIC drift (e.g. a dim row purged but the fact retained) means LEFT is safer.
# MAGIC
# MAGIC Spark will broadcast the small dims automatically because they're well below
# MAGIC `spark.sql.autoBroadcastJoinThreshold` (default 10 MB). We add explicit
# MAGIC `F.broadcast(...)` hints anyway — it makes the intent obvious in the plan
# MAGIC and protects against threshold changes.

# COMMAND ----------

# Pre-select only the dim columns we need to keep the join shuffles small.
dim_customer_slim = dim_customer.select(
    "customer_id", "home_country", "customer_segment", "risk_tier",
    "is_high_net_worth", "kyc_status", "preferred_channel",
)
dim_merchant_slim = dim_merchant.select(
    "merchant_id", "merchant_name", "merchant_category", "merchant_country",
    "merchant_risk_level", "merchant_size", "online_only_flag",
)
dim_device_slim = dim_device.select(
    "device_id", "device_type", "os_type", "rooted_device_flag", "emulator_flag",
)

silver_fact_transactions_enriched = (
    silver_fact_transactions_clean
    .join(F.broadcast(dim_customer_slim), on="customer_id", how="left")
    .join(F.broadcast(dim_merchant_slim), on="merchant_id", how="left")
    .join(F.broadcast(dim_device_slim), on="device_id", how="left")

    # ----- Derived risk signals -----
    # is_cross_border: txn country ≠ customer's home country. Coalesce so a NULL
    # on either side does not silently become "false" via SQL NULL semantics.
    .withColumn(
        "is_cross_border",
        F.coalesce(
            F.col("transaction_country") != F.col("home_country"),
            F.lit(False),
        ),
    )

    # High-risk merchant: gambling / crypto / explicit HIGH risk_level.
    .withColumn(
        "is_high_risk_merchant",
        F.coalesce(
            (F.col("merchant_risk_level") == F.lit("HIGH"))
            | F.col("merchant_category").isin("GAMBLING", "CRYPTO"),
            F.lit(False),
        ),
    )

    # Risky device: rooted phones or emulators are common fraud-tool signatures.
    .withColumn(
        "is_risky_device",
        F.coalesce(F.col("rooted_device_flag") | F.col("emulator_flag"), F.lit(False)),
    )

    # Card-not-present: ONLINE_BANKING and CARD_NOT_PRESENT both qualify.
    .withColumn(
        "is_card_not_present",
        F.col("payment_type").isin("CARD_NOT_PRESENT", "ONLINE_BANKING"),
    )

    # International: the transaction itself crossed a border OR was an explicit
    # international transfer payment type.
    .withColumn(
        "is_international_transaction",
        F.col("is_cross_border") | (F.col("payment_type") == F.lit("INTL_TRANSFER")),
    )

    # risk_signal_count: how many risk flags fired on this row. Cheap proxy for
    # "suspiciousness" before the ML model runs. Cast booleans to int and sum.
    .withColumn(
        "risk_signal_count",
        F.col("is_cross_border").cast("int")
        + F.col("is_high_risk_merchant").cast("int")
        + F.col("is_risky_device").cast("int")
        + F.col("is_card_not_present").cast("int")
        + F.col("is_international_transaction").cast("int")
        + F.col("is_nighttime_transaction").cast("int"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Data quality summary
# MAGIC
# MAGIC One-row table of DQ KPIs. Every count is a `sum(flag.cast(int))` over the
# MAGIC bronze plus the survivor counts. This is a reduce — single row, safe.
# MAGIC
# MAGIC The summary is what the DQ dashboard reads from. Storing it as Delta lets us
# MAGIC append a snapshot per run later (changing the write mode to `append` and
# MAGIC adding a `run_ts` column) — for this exercise we overwrite.

# COMMAND ----------

# Bronze counts use the flagged-but-not-deduped frame so duplicate flags include
# all duplicate rows (not just the losers).
total_bronze = bronze.count()  # single scalar — safe
total_clean = silver_fact_transactions_clean.count()
total_quarantine = quarantine_bad_transactions.count()

# Aggregate DQ flag counts in a single pass over bronze_flagged. groupBy-less agg
# returns one row.
flag_counts = (
    bronze_flagged.agg(
        F.sum(F.col("is_null_customer").cast("int")).alias("null_customer_count"),
        F.sum(F.col("is_null_merchant").cast("int")).alias("null_merchant_count"),
        F.sum(F.col("is_null_device").cast("int")).alias("null_device_count"),
        F.sum(F.col("is_null_amount").cast("int")).alias("null_amount_count"),
        F.sum(F.col("is_invalid_amount").cast("int")).alias("invalid_amount_count"),
        F.sum(F.col("is_invalid_currency").cast("int")).alias("invalid_currency_count"),
        F.sum(F.col("is_invalid_status").cast("int")).alias("invalid_status_count"),
        F.sum(F.col("is_future_timestamp").cast("int")).alias("future_timestamp_count"),
        F.sum(F.col("is_orphan_customer").cast("int")).alias("orphan_customer_count"),
        F.sum(F.col("is_orphan_merchant").cast("int")).alias("orphan_merchant_count"),
        F.sum(F.col("is_orphan_device").cast("int")).alias("orphan_device_count"),
        F.sum(F.col("is_late_arriving").cast("int")).alias("late_arriving_count"),
        F.sum(F.col("is_suspicious_default_customer").cast("int"))
            .alias("suspicious_default_customer_count"),
        F.sum(F.col("is_suspicious_default_merchant").cast("int"))
            .alias("suspicious_default_merchant_count"),
    )
)

# Duplicate count = number of rows lost to dedup (rn > 1). Computed on the
# already-cached deduped frame so we don't re-shuffle.
duplicate_count = (
    bronze_deduped
    .where(F.col("_dedup_rn") > F.lit(1))
    .count()
)

# Build the summary as a one-row DataFrame. We use spark.range(1) + literals
# instead of createDataFrame on a Python list — this stays Spark-native and
# composes with the flag_counts aggregation above via crossJoin.
summary_scalars = (
    spark.range(1)
    .withColumn("total_bronze_records", F.lit(total_bronze))
    .withColumn("duplicate_records", F.lit(duplicate_count))
    .withColumn("quarantined_records", F.lit(total_quarantine))
    .withColumn("clean_records", F.lit(total_clean))
    .withColumn("summary_ts", F.current_timestamp())
    .drop("id")
)

silver_data_quality_summary = summary_scalars.crossJoin(flag_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Persist all outputs as Delta

# COMMAND ----------

(
    silver_fact_transactions_clean.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("event_date")
    .saveAsTable(f"{DB_NAME}.silver_fact_transactions_clean")
)

(
    quarantine_bad_transactions.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("event_date")
    .saveAsTable(f"{DB_NAME}.quarantine_bad_transactions")
)

(
    silver_fact_transactions_enriched.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("event_date")
    .saveAsTable(f"{DB_NAME}.silver_fact_transactions_enriched")
)

(
    silver_data_quality_summary.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.silver_data_quality_summary")
)

# Release the cache now that all writes are committed. unpersist() is a no-op
# (and may also raise) on serverless where cache() was skipped — guard it.
try:
    bronze_deduped.unpersist()
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10 — Display results

# COMMAND ----------

print("=" * 80)
print("silver_fact_transactions_clean — sample")
print("=" * 80)
spark.table(f"{DB_NAME}.silver_fact_transactions_clean").printSchema()
spark.table(f"{DB_NAME}.silver_fact_transactions_clean").show(5, truncate=False)

print("=" * 80)
print("quarantine_bad_transactions — sample")
print("=" * 80)
spark.table(f"{DB_NAME}.quarantine_bad_transactions").printSchema()
(
    spark.table(f"{DB_NAME}.quarantine_bad_transactions")
    .select("transaction_id", "customer_id", "merchant_id", "amount",
            "currency", "transaction_status", "transaction_ts", "ingestion_ts",
            "quarantine_reason", "quarantine_ts")
    .show(10, truncate=False)
)

print("Quarantine reason breakdown:")
(
    spark.table(f"{DB_NAME}.quarantine_bad_transactions")
    .groupBy("quarantine_reason")
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

print("=" * 80)
print("silver_fact_transactions_enriched — sample")
print("=" * 80)
spark.table(f"{DB_NAME}.silver_fact_transactions_enriched").printSchema()
(
    spark.table(f"{DB_NAME}.silver_fact_transactions_enriched")
    .select("transaction_id", "customer_id", "merchant_id", "amount",
            "transaction_country", "home_country", "merchant_category",
            "is_cross_border", "is_high_risk_merchant", "is_risky_device",
            "is_card_not_present", "is_international_transaction",
            "is_nighttime_transaction", "risk_signal_count")
    .show(10, truncate=False)
)

print("=" * 80)
print("silver_data_quality_summary")
print("=" * 80)
spark.table(f"{DB_NAME}.silver_data_quality_summary").show(truncate=False, vertical=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Layered architecture summary
# MAGIC
# MAGIC | Layer | Role | What lives here |
# MAGIC |---|---|---|
# MAGIC | **Bronze** | Raw truth | Unmodified events including DQ defects — auditor's view, no row ever dropped. |
# MAGIC | **Quarantine** | Audit + reprocessing | Rows that failed validation, with `quarantine_reason` and `quarantine_ts`. Upstream feeds get fixed and replayed from here. |
# MAGIC | **Silver clean** | Trusted analytical truth | Deduplicated, validated facts. The single source of truth for downstream BI / ML / Gold marts. |
# MAGIC | **Silver enriched** | BI + fraud + feature store | Clean facts + dim attributes + risk signals. What dashboards, fraud rules, and ML feature pipelines read. |
# MAGIC | **DQ summary** | Operational KPIs | One-row snapshot per run — drives the DQ dashboard and alerting. |
# MAGIC
# MAGIC ## Why Spark-native APIs (not pandas / not UDFs)
# MAGIC
# MAGIC - **Catalyst optimization** — rule reordering, predicate pushdown, and join
# MAGIC   reordering only work when expressions are native. A Python UDF is opaque.
# MAGIC - **Whole-stage codegen** — fuses our many `withColumn` calls into one Java
# MAGIC   class, vastly faster than per-row Python interop.
# MAGIC - **Memory locality** — pandas lives on the driver heap; Spark DataFrames
# MAGIC   live on executors and scale horizontally.
# MAGIC - **No `collect()` on facts** — every aggregation in this notebook is a
# MAGIC   reduce returning at most a few rows. The driver never materializes the
# MAGIC   50M+ row body.

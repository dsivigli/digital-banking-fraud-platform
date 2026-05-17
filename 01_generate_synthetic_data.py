# Databricks notebook source
# MAGIC %md
# MAGIC # Global Banking Fraud Detection Platform — Synthetic Data Generation
# MAGIC
# MAGIC This notebook generates a production-style synthetic dataset for a multinational
# MAGIC bank's fraud detection platform. The data covers:
# MAGIC
# MAGIC - **Credit card transactions** (card-present and card-not-present)
# MAGIC - **ATM withdrawals**
# MAGIC - **Online banking payments**
# MAGIC - **Mobile app payments**
# MAGIC - **International transfers**
# MAGIC
# MAGIC ## Design principles
# MAGIC
# MAGIC 1. **Spark-native only** — no Python loops, no pandas, no Python UDFs, no `collect()`
# MAGIC    on large datasets. Every transformation runs in the JVM via Catalyst/Tungsten.
# MAGIC 2. **Deterministic FK generation** — `pmod(hash(natural_key), N) + 1` guarantees
# MAGIC    every foreign key falls within `[1, N]` and is reproducible across reruns.
# MAGIC 3. **Seeded randomness** — `rand(seed=...)` makes the dataset reproducible while
# MAGIC    still producing well-distributed values across partitions.
# MAGIC 4. **Linear scalability** — `spark.range(N)` creates a partitioned DataFrame that
# MAGIC    scales from 1M to 500M+ rows by adjusting `N` and partition count.
# MAGIC 5. **Intentional skew** — a small number of mega merchants and high-velocity
# MAGIC    customers receive disproportionate transaction volume so we can demonstrate
# MAGIC    skew handling (salting, AQE, broadcast joins).
# MAGIC
# MAGIC Output: four Delta tables in the `fraud_platform` database.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports and configuration

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    DoubleType,
    BooleanType,
    DateType,
    TimestampType,
)

# Spark session — in Databricks this is already provided; the assignment below is a
# safety net for local execution. We do NOT call .getOrCreate() unconditionally inside
# Databricks because it would shadow the runtime-tuned session.
spark = SparkSession.builder.getOrCreate()

# Adaptive Query Execution handles skew automatically by splitting heavy partitions
# during shuffles. On Databricks Serverless / Unity Catalog shared compute these
# confs are platform-managed and AQE is already on by default, so attempting to set
# them raises CONFIG_NOT_AVAILABLE (SQLSTATE 42K0I). We try-set and ignore failures
# so the notebook runs unchanged on classic clusters, serverless, and DBR ≥ 7.3.
def _try_set(key: str, value: str) -> None:
    try:
        spark.conf.set(key, value)
    except Exception:
        # Serverless blocks user-level conf changes — that's fine, defaults are right.
        pass

_try_set("spark.sql.adaptive.enabled", "true")
_try_set("spark.sql.adaptive.skewJoin.enabled", "true")
_try_set("spark.sql.adaptive.coalescePartitions.enabled", "true")
_try_set("spark.sql.shuffle.partitions", "400")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Sizing parameters
# MAGIC
# MAGIC Every dataset size is a single constant. To scale from 1M to 500M transactions
# MAGIC we change `N_TRANSACTIONS` only — the rest of the code is dimension-agnostic.

# COMMAND ----------

# Dimension cardinalities. Tuned for a "small prod" footprint that still exposes
# realistic distributed-systems behavior (skew, shuffle pressure, broadcast cutoffs).
N_CUSTOMERS = 1_000_000        # 1M customers
N_MERCHANTS = 100_000          # 100k merchants
N_DEVICES = 5_000_000          # 5M devices (more devices than customers — multi-device users)
N_TRANSACTIONS = 50_000_000    # 50M transactions; bump to 500_000_000 for stress tests

# Database / catalog. In Databricks Unity Catalog this would be `catalog.schema`.
DB_NAME = "fraud_platform"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
spark.sql(f"USE {DB_NAME}")

# Partition counts. Rule of thumb: keep each partition ~128–256 MB after generation.
# spark.range() defaults to spark.default.parallelism which is often too low for
# huge ranges, so we set explicit numPartitions on the largest table.
TX_PARTITIONS = 400            # 50M rows / 400 = 125k rows per partition before joins
DIM_PARTITIONS = 32            # dimensions are small enough to live in few partitions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `dim_customer` — 1M customers
# MAGIC
# MAGIC Customers are generated entirely with Spark SQL expressions:
# MAGIC
# MAGIC - **`spark.range(1, N+1)`** produces a sequential `id` column distributed across
# MAGIC   partitions. We use `1..N` (inclusive) so `pmod(hash(...), N) + 1` produces FKs
# MAGIC   that match the PK domain exactly.
# MAGIC - **Categorical attributes** are derived with `pmod(hash(id), bucket_count)` to
# MAGIC   pick from a fixed list via `element_at(array(...), bucket+1)`. This keeps the
# MAGIC   logic data-parallel and avoids any Python-side lookup table.
# MAGIC - **`rand(seed=...)`** with distinct seeds per column produces statistically
# MAGIC   independent draws while remaining reproducible.
# MAGIC - **Risk tier** is intentionally skewed: ~5% high-risk, ~20% medium, ~75% low.

# COMMAND ----------

# Country list — a representative slice of a multinational customer base. Weights are
# baked in by repeating high-traffic countries (US, GB, IN, DE) inside the array; the
# uniform `pmod(hash, len(array))` pick then yields a non-uniform real-world mix.
country_array = F.array(*[F.lit(c) for c in [
    "US", "US", "US", "US",          # ~28% — largest market
    "GB", "GB",                       # ~14%
    "DE", "DE",                       # ~14%
    "IN", "IN",                       # ~14%
    "FR", "JP", "BR", "MX", "CA", "AU", "SG", "AE", "ZA", "NG"
]])

dim_customer = (
    spark.range(1, N_CUSTOMERS + 1, numPartitions=DIM_PARTITIONS)
    .withColumnRenamed("id", "customer_id")

    # Age band — uniform pmod over 6 buckets, mapped to labels.
    .withColumn(
        "age_band",
        F.element_at(
            F.array(F.lit("18-24"), F.lit("25-34"), F.lit("35-44"),
                    F.lit("45-54"), F.lit("55-64"), F.lit("65+")),
            (F.pmod(F.hash(F.col("customer_id"), F.lit("age")), F.lit(6)) + F.lit(1)).cast("int"),
        ),
    )

    # Home country — picked deterministically per customer; weight via repetition.
    .withColumn(
        "home_country",
        F.element_at(
            country_array,
            (F.pmod(F.hash(F.col("customer_id"), F.lit("country")), F.lit(20)) + F.lit(1)).cast("int"),
        ),
    )

    # Segment — RETAIL is the bulk; PRIVATE/SME smaller; CORPORATE smallest.
    .withColumn(
        "customer_segment",
        F.element_at(
            F.array(F.lit("RETAIL"), F.lit("RETAIL"), F.lit("RETAIL"), F.lit("RETAIL"),
                    F.lit("RETAIL"), F.lit("RETAIL"), F.lit("RETAIL"),
                    F.lit("PRIVATE"), F.lit("SME"), F.lit("CORPORATE")),
            (F.pmod(F.hash(F.col("customer_id"), F.lit("seg")), F.lit(10)) + F.lit(1)).cast("int"),
        ),
    )

    # Account open date — uniform between 2010-01-01 and 2025-12-31. We compute by
    # adding a random day-offset (0..5843) to a base date entirely in Spark SQL.
    .withColumn(
        "account_open_date",
        F.expr("date_add(to_date('2010-01-01'), cast(rand(101) * 5843 as int))"),
    )

    # KYC status — vast majority VERIFIED; a few PENDING/EXPIRED to drive risk signals.
    .withColumn(
        "kyc_status",
        F.element_at(
            F.array(F.lit("VERIFIED"), F.lit("VERIFIED"), F.lit("VERIFIED"),
                    F.lit("VERIFIED"), F.lit("VERIFIED"), F.lit("VERIFIED"),
                    F.lit("VERIFIED"), F.lit("VERIFIED"), F.lit("VERIFIED"),
                    F.lit("VERIFIED"), F.lit("VERIFIED"), F.lit("VERIFIED"),
                    F.lit("VERIFIED"), F.lit("VERIFIED"), F.lit("VERIFIED"),
                    F.lit("VERIFIED"), F.lit("VERIFIED"), F.lit("VERIFIED"),
                    F.lit("PENDING"), F.lit("EXPIRED")),  # ~10% non-verified
            (F.pmod(F.hash(F.col("customer_id"), F.lit("kyc")), F.lit(20)) + F.lit(1)).cast("int"),
        ),
    )

    # Risk tier — distribution is intentionally skewed: 75% LOW, 20% MEDIUM, 5% HIGH.
    # Skew here is desired: it mirrors the real-world long-tail of fraud risk, and
    # provides a label-correlated feature for downstream ML.
    .withColumn(
        "risk_tier",
        F.element_at(
            F.array(*([F.lit("LOW")] * 15 + [F.lit("MEDIUM")] * 4 + [F.lit("HIGH")] * 1)),
            (F.pmod(F.hash(F.col("customer_id"), F.lit("risk")), F.lit(20)) + F.lit(1)).cast("int"),
        ),
    )

    # Annual income band — 5 buckets, roughly normal-ish via repeated mid-bands.
    .withColumn(
        "annual_income_band",
        F.element_at(
            F.array(F.lit("<25K"), F.lit("25-50K"), F.lit("25-50K"),
                    F.lit("50-100K"), F.lit("50-100K"), F.lit("50-100K"),
                    F.lit("100-250K"), F.lit("100-250K"), F.lit("250K+")),
            (F.pmod(F.hash(F.col("customer_id"), F.lit("inc")), F.lit(9)) + F.lit(1)).cast("int"),
        ),
    )

    # Preferred channel — drives transaction-channel weighting later.
    .withColumn(
        "preferred_channel",
        F.element_at(
            F.array(F.lit("MOBILE"), F.lit("MOBILE"), F.lit("MOBILE"),
                    F.lit("ONLINE"), F.lit("ONLINE"),
                    F.lit("CARD"), F.lit("CARD"),
                    F.lit("ATM")),
            (F.pmod(F.hash(F.col("customer_id"), F.lit("chan")), F.lit(8)) + F.lit(1)).cast("int"),
        ),
    )

    # HNW flag — derived from income band so the two are consistent.
    .withColumn(
        "is_high_net_worth",
        F.col("annual_income_band").isin("250K+", "100-250K"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. `dim_merchant` — 100k merchants with intentional skew
# MAGIC
# MAGIC Merchants include risky verticals (gambling, crypto, gift cards, electronics)
# MAGIC alongside grocery/restaurant/utilities. To simulate the real-world Pareto in
# MAGIC payment volume, we mark the first ~50 merchants as `MEGA` size — downstream the
# MAGIC FK distribution biases transactions toward these IDs to reproduce the classic
# MAGIC skew problem (one merchant_id receiving 10% of all transactions).

# COMMAND ----------

merchant_categories = F.array(*[F.lit(c) for c in [
    "GROCERY", "GROCERY", "GROCERY",
    "RESTAURANT", "RESTAURANT",
    "RETAIL", "RETAIL",
    "FUEL",
    "UTILITIES",
    "TRAVEL",
    "ELECTRONICS",     # risky
    "GIFT_CARDS",      # risky
    "GAMBLING",        # risky
    "CRYPTO",          # risky
    "PHARMACY",
    "ENTERTAINMENT",
]])

dim_merchant = (
    spark.range(1, N_MERCHANTS + 1, numPartitions=DIM_PARTITIONS)
    .withColumnRenamed("id", "merchant_id")

    # Synthetic merchant name — concat the id into a stable readable string. Produced
    # via concat_ws inside Spark, never via a Python f-string applied per row.
    .withColumn("merchant_name", F.concat(F.lit("MERCHANT_"), F.col("merchant_id")))

    .withColumn(
        "merchant_category",
        F.element_at(
            merchant_categories,
            (F.pmod(F.hash(F.col("merchant_id"), F.lit("cat")), F.lit(16)) + F.lit(1)).cast("int"),
        ),
    )

    .withColumn(
        "merchant_country",
        F.element_at(
            country_array,
            (F.pmod(F.hash(F.col("merchant_id"), F.lit("mctry")), F.lit(20)) + F.lit(1)).cast("int"),
        ),
    )

    # Risk level is correlated with category: gambling/crypto/gift_cards default HIGH.
    # This is expressed as a CASE WHEN — pure Catalyst, runs in parallel.
    .withColumn(
        "merchant_risk_level",
        F.when(F.col("merchant_category").isin("GAMBLING", "CRYPTO"), F.lit("HIGH"))
         .when(F.col("merchant_category").isin("GIFT_CARDS", "ELECTRONICS"), F.lit("MEDIUM"))
         .otherwise(F.lit("LOW")),
    )

    # Mega merchants: the first 50 merchant_ids are tagged MEGA. Combined with the FK
    # generation strategy in fact_transactions, this produces deliberate skew.
    .withColumn(
        "merchant_size",
        F.when(F.col("merchant_id") <= F.lit(50), F.lit("MEGA"))
         .when(F.col("merchant_id") <= F.lit(5_000), F.lit("LARGE"))
         .when(F.col("merchant_id") <= F.lit(30_000), F.lit("MEDIUM"))
         .otherwise(F.lit("SMALL")),
    )

    # Online-only flag is correlated with category for realism.
    .withColumn(
        "online_only_flag",
        F.col("merchant_category").isin("CRYPTO", "GIFT_CARDS", "GAMBLING", "ELECTRONICS"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `dim_device` — 5M devices
# MAGIC
# MAGIC Devices are not 1:1 with customers — a customer can have multiple devices, and
# MAGIC fraudsters often share/spoof devices. We seed `rooted_device_flag` and
# MAGIC `emulator_flag` with low base rates (~2% and ~1%) so they remain useful signals
# MAGIC when correlated with fraud labels.

# COMMAND ----------

dim_device = (
    spark.range(1, N_DEVICES + 1, numPartitions=DIM_PARTITIONS * 4)
    .withColumnRenamed("id", "device_id")

    .withColumn(
        "device_type",
        F.element_at(
            F.array(F.lit("MOBILE"), F.lit("MOBILE"), F.lit("MOBILE"),
                    F.lit("MOBILE"), F.lit("MOBILE"),
                    F.lit("DESKTOP"), F.lit("DESKTOP"),
                    F.lit("TABLET")),
            (F.pmod(F.hash(F.col("device_id"), F.lit("dt")), F.lit(8)) + F.lit(1)).cast("int"),
        ),
    )

    .withColumn(
        "os_type",
        F.element_at(
            F.array(F.lit("iOS"), F.lit("iOS"), F.lit("iOS"),
                    F.lit("Android"), F.lit("Android"), F.lit("Android"), F.lit("Android"),
                    F.lit("Windows"), F.lit("MacOS"), F.lit("Linux")),
            (F.pmod(F.hash(F.col("device_id"), F.lit("os")), F.lit(10)) + F.lit(1)).cast("int"),
        ),
    )

    # App version — mostly modern with a long tail of stale clients.
    .withColumn(
        "app_version",
        F.element_at(
            F.array(F.lit("5.4.0"), F.lit("5.4.0"), F.lit("5.3.2"), F.lit("5.3.2"),
                    F.lit("5.2.1"), F.lit("5.0.0"), F.lit("4.9.7"), F.lit("4.5.0")),
            (F.pmod(F.hash(F.col("device_id"), F.lit("ver")), F.lit(8)) + F.lit(1)).cast("int"),
        ),
    )

    # ~2% rooted devices, seeded for reproducibility.
    .withColumn("rooted_device_flag", F.rand(seed=2001) < F.lit(0.02))

    # ~1% emulators.
    .withColumn("emulator_flag", F.rand(seed=2002) < F.lit(0.01))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist dimensions as Delta
# MAGIC
# MAGIC Dimensions are written first so subsequent FK generation can reference real PK
# MAGIC ranges. We use `mode("overwrite")` so the notebook is idempotent. Z-ORDERing on
# MAGIC the PK accelerates point lookups during fraud investigations and joins.

# COMMAND ----------

(
    dim_customer.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.dim_customer")
)

(
    dim_merchant.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.dim_merchant")
)

(
    dim_device.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.dim_device")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. `fact_transactions` — 50M transactions
# MAGIC
# MAGIC Generation strategy:
# MAGIC
# MAGIC 1. Start from `spark.range(1, N+1, numPartitions=TX_PARTITIONS)` — this is the
# MAGIC    only source of cardinality. Every other column is derived deterministically.
# MAGIC 2. Foreign keys use **`pmod(hash(transaction_id, salt), N_DIM) + 1`** which
# MAGIC    guarantees `[1, N_DIM]` range — no orphan FKs, ever.
# MAGIC 3. To create skew, ~10% of transactions are forced onto the first 50 mega
# MAGIC    merchants. This produces a Pareto-style distribution that AQE skew join
# MAGIC    handling can demonstrate.
# MAGIC 4. Timestamps span 2024-01-01 to 2025-12-31 via `from_unixtime` arithmetic on
# MAGIC    a seeded `rand`. `event_date` is derived for partition pruning.
# MAGIC 5. Amounts use `pow(10, rand() * 4)` to produce a log-uniform distribution
# MAGIC    (most transactions $1–$1000, a few up to ~$10k).

# COMMAND ----------

# Time window in epoch seconds — computed once on the driver as constants and
# broadcast to executors via .lit(). This avoids per-row date parsing.
TS_START = "2024-01-01 00:00:00"
TS_END = "2025-12-31 23:59:59"
ts_start_epoch = spark.sql(f"SELECT unix_timestamp('{TS_START}') AS s").first()["s"]
ts_end_epoch = spark.sql(f"SELECT unix_timestamp('{TS_END}') AS s").first()["s"]
ts_window_seconds = ts_end_epoch - ts_start_epoch  # ~63M seconds across 2 years

# Number of mega merchants that will absorb the skewed traffic. Keep small so the
# skew is dramatic — a few percent of merchants getting double-digit % of volume.
N_MEGA_MERCHANTS = 50
MEGA_MERCHANT_SHARE = 0.10  # 10% of transactions go to mega merchants

fact_transactions = (
    spark.range(1, N_TRANSACTIONS + 1, numPartitions=TX_PARTITIONS)
    .withColumnRenamed("id", "transaction_id")

    # ---------------- Foreign keys ----------------
    # Customer FK. The salt literal "cust" diversifies the hash so customer_id and
    # merchant_id distributions are independent. pmod guarantees non-negative.
    .withColumn(
        "customer_id",
        F.pmod(F.hash(F.col("transaction_id"), F.lit("cust")), F.lit(N_CUSTOMERS)) + F.lit(1),
    )

    # Merchant FK with deliberate skew. We compute two candidate FKs:
    #   * a uniform FK across all 100k merchants
    #   * a skewed FK that maps to the first N_MEGA_MERCHANTS
    # Then a seeded coin flip selects which to use. Implementing this purely in
    # Catalyst keeps the generation single-pass and parallel.
    .withColumn(
        "_merchant_uniform",
        F.pmod(F.hash(F.col("transaction_id"), F.lit("merch")), F.lit(N_MERCHANTS)) + F.lit(1),
    )
    .withColumn(
        "_merchant_skewed",
        F.pmod(F.hash(F.col("transaction_id"), F.lit("mega")), F.lit(N_MEGA_MERCHANTS)) + F.lit(1),
    )
    .withColumn("_skew_pick", F.rand(seed=3001))
    .withColumn(
        "merchant_id",
        F.when(F.col("_skew_pick") < F.lit(MEGA_MERCHANT_SHARE), F.col("_merchant_skewed"))
         .otherwise(F.col("_merchant_uniform")),
    )

    # Device FK — independent salt; multi-device users are a natural side-effect of
    # the hash collisions across customer_ids.
    .withColumn(
        "device_id",
        F.pmod(F.hash(F.col("transaction_id"), F.lit("dev")), F.lit(N_DEVICES)) + F.lit(1),
    )

    # ---------------- Timestamp / date ----------------
    # Uniform timestamp across the 2-year window. We add a seeded random offset to
    # the start epoch and cast back to a timestamp.
    .withColumn(
        "transaction_ts",
        F.to_timestamp(
            F.from_unixtime(
                F.lit(ts_start_epoch) + (F.rand(seed=4001) * F.lit(ts_window_seconds)).cast("long")
            )
        ),
    )
    # event_date is the partition column for the fact table — Delta partition pruning
    # on this column gives near-O(1) reads for time-bounded fraud queries.
    .withColumn("event_date", F.to_date(F.col("transaction_ts")))

    # ---------------- Channel & payment type ----------------
    # Channel mix mirrors a real bank's traffic — heavy mobile/online with ATM tail.
    .withColumn(
        "transaction_channel",
        F.element_at(
            F.array(F.lit("MOBILE"), F.lit("MOBILE"), F.lit("MOBILE"), F.lit("MOBILE"),
                    F.lit("ONLINE"), F.lit("ONLINE"), F.lit("ONLINE"),
                    F.lit("POS"), F.lit("POS"),
                    F.lit("ATM")),
            (F.pmod(F.hash(F.col("transaction_id"), F.lit("ch")), F.lit(10)) + F.lit(1)).cast("int"),
        ),
    )

    # Payment type — derived partly from channel for internal consistency.
    .withColumn(
        "payment_type",
        F.when(F.col("transaction_channel") == "ATM", F.lit("ATM_WITHDRAWAL"))
         .when(F.col("transaction_channel") == "POS", F.lit("CARD_PRESENT"))
         .when(F.col("transaction_channel") == "MOBILE",
               F.element_at(F.array(F.lit("MOBILE_PAYMENT"), F.lit("MOBILE_PAYMENT"),
                                    F.lit("INTL_TRANSFER")),
                            (F.pmod(F.hash(F.col("transaction_id"), F.lit("pt")), F.lit(3)) + F.lit(1)).cast("int")))
         .otherwise(  # ONLINE
               F.element_at(F.array(F.lit("ONLINE_BANKING"), F.lit("ONLINE_BANKING"),
                                    F.lit("CARD_NOT_PRESENT"), F.lit("INTL_TRANSFER")),
                            (F.pmod(F.hash(F.col("transaction_id"), F.lit("pt2")), F.lit(4)) + F.lit(1)).cast("int"))),
    )

    # card_present_flag aligns with payment_type but stays a separate column because
    # downstream ML treats it as a feature in its own right.
    .withColumn("card_present_flag", F.col("payment_type") == "CARD_PRESENT")

    # ---------------- Amount ----------------
    # Log-uniform: rand uniform in [0,1) → exp range $1 to $10,000. ATM withdrawals
    # are capped lower because real ATMs cap daily limits.
    .withColumn(
        "amount",
        F.when(F.col("transaction_channel") == "ATM",
               F.round(F.lit(20.0) + F.rand(seed=5001) * F.lit(480.0), 2))
         .otherwise(F.round(F.pow(F.lit(10.0), F.rand(seed=5002) * F.lit(4.0)), 2)),
    )

    # ---------------- Currency / country ----------------
    # Currency — biased toward USD/EUR/GBP for realism.
    .withColumn(
        "currency",
        F.element_at(
            F.array(F.lit("USD"), F.lit("USD"), F.lit("USD"), F.lit("USD"),
                    F.lit("EUR"), F.lit("EUR"), F.lit("EUR"),
                    F.lit("GBP"), F.lit("GBP"),
                    F.lit("INR"), F.lit("JPY"), F.lit("BRL"), F.lit("AUD"),
                    F.lit("SGD"), F.lit("AED"), F.lit("CAD")),
            (F.pmod(F.hash(F.col("transaction_id"), F.lit("ccy")), F.lit(16)) + F.lit(1)).cast("int"),
        ),
    )

    # Transaction country — independently distributed; cross-border with home_country
    # is a key fraud feature joined in downstream.
    .withColumn(
        "transaction_country",
        F.element_at(
            country_array,
            (F.pmod(F.hash(F.col("transaction_id"), F.lit("tcty")), F.lit(20)) + F.lit(1)).cast("int"),
        ),
    )

    # ---------------- IP address ----------------
    # Synthetic IPv4 — four octets from independent hashes. Stays inside Catalyst.
    .withColumn(
        "ip_address",
        F.concat_ws(
            ".",
            (F.pmod(F.hash(F.col("transaction_id"), F.lit("ip1")), F.lit(255))).cast("string"),
            (F.pmod(F.hash(F.col("transaction_id"), F.lit("ip2")), F.lit(255))).cast("string"),
            (F.pmod(F.hash(F.col("transaction_id"), F.lit("ip3")), F.lit(255))).cast("string"),
            (F.pmod(F.hash(F.col("transaction_id"), F.lit("ip4")), F.lit(255))).cast("string"),
        ),
    )

    # ---------------- Status ----------------
    # ~96% APPROVED, 3% DECLINED, 1% REVERSED. Seeded so reruns are stable.
    .withColumn("_status_pick", F.rand(seed=6001))
    .withColumn(
        "transaction_status",
        F.when(F.col("_status_pick") < F.lit(0.96), F.lit("APPROVED"))
         .when(F.col("_status_pick") < F.lit(0.99), F.lit("DECLINED"))
         .otherwise(F.lit("REVERSED")),
    )

    # ---------------- Denormalized merchant_category ----------------
    # We denormalize merchant_category onto the fact at write-time only if we want to
    # avoid the join in BI dashboards. We compute it from the merchant_id using the
    # SAME deterministic hashing used in dim_merchant — perfect agreement, no join.
    .withColumn(
        "merchant_category",
        F.element_at(
            merchant_categories,
            (F.pmod(F.hash(F.col("merchant_id"), F.lit("cat")), F.lit(16)) + F.lit(1)).cast("int"),
        ),
    )

    # Drop helper columns — they were only needed to build the skew distribution.
    .drop("_merchant_uniform", "_merchant_skewed", "_skew_pick", "_status_pick")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Persist `fact_transactions` as a partitioned Delta table
# MAGIC
# MAGIC We partition by `event_date`. Two-year horizon × ~50M rows → ~70k rows/day, a
# MAGIC sweet spot for Delta partition sizing (~few MB/partition file). For 500M rows
# MAGIC we'd partition by `event_date` and `month` or switch to liquid clustering on
# MAGIC `(event_date, customer_id)` to avoid small files.

# COMMAND ----------

(
    fact_transactions.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("event_date")
    .saveAsTable(f"{DB_NAME}.fact_transactions")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Sample rows and schemas
# MAGIC
# MAGIC `display()` is the Databricks-native way to render small previews. Each `.show()`
# MAGIC fetches a tiny `LIMIT n` slice — never a full collect.

# COMMAND ----------

print("=" * 80)
print("dim_customer — schema & sample")
print("=" * 80)
spark.table(f"{DB_NAME}.dim_customer").printSchema()
spark.table(f"{DB_NAME}.dim_customer").show(5, truncate=False)

print("=" * 80)
print("dim_merchant — schema & sample")
print("=" * 80)
spark.table(f"{DB_NAME}.dim_merchant").printSchema()
spark.table(f"{DB_NAME}.dim_merchant").show(5, truncate=False)

print("=" * 80)
print("dim_device — schema & sample")
print("=" * 80)
spark.table(f"{DB_NAME}.dim_device").printSchema()
spark.table(f"{DB_NAME}.dim_device").show(5, truncate=False)

print("=" * 80)
print("fact_transactions — schema & sample")
print("=" * 80)
spark.table(f"{DB_NAME}.fact_transactions").printSchema()
spark.table(f"{DB_NAME}.fact_transactions").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Sanity checks (no `collect()` on large data)
# MAGIC
# MAGIC We verify that:
# MAGIC
# MAGIC - All FKs land in the dimension PK domain — done by an `agg(min, max)` which
# MAGIC   returns a single-row DataFrame (safe to `.first()`).
# MAGIC - Per-merchant transaction counts show the expected skew — top-50 merchants
# MAGIC   should cumulatively own ~10% of all transactions.

# COMMAND ----------

# FK domain check — aggregation reduces 50M rows to one. .first() is safe here.
fk_bounds = (
    spark.table(f"{DB_NAME}.fact_transactions")
    .agg(
        F.min("customer_id").alias("min_customer_id"),
        F.max("customer_id").alias("max_customer_id"),
        F.min("merchant_id").alias("min_merchant_id"),
        F.max("merchant_id").alias("max_merchant_id"),
        F.min("device_id").alias("min_device_id"),
        F.max("device_id").alias("max_device_id"),
    )
)
fk_bounds.show(truncate=False)

# Skew check — show top-10 merchants by volume. Aggregation + LIMIT, never a full collect.
(
    spark.table(f"{DB_NAME}.fact_transactions")
    .groupBy("merchant_id")
    .count()
    .orderBy(F.desc("count"))
    .limit(10)
    .show(truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Design notes
# MAGIC
# MAGIC ### Why this scales linearly from 1M to 500M+ rows
# MAGIC
# MAGIC - **`spark.range(N, numPartitions=P)`** creates a fully partitioned DataFrame
# MAGIC   without materializing any driver-side list. Doubling `N` and `P` produces an
# MAGIC   (almost) linear runtime increase — the executors never see a global view.
# MAGIC - **Every column is a column expression.** Catalyst translates the entire DAG
# MAGIC   to whole-stage codegen Java; no row-at-a-time Python interop, no PySpark-to-JVM
# MAGIC   serialization tax (which is what kills Python UDFs).
# MAGIC - **Deterministic hashing replaces lookups.** A naïve approach would broadcast
# MAGIC   a Python dict of "category → weight" — that breaks at scale. `pmod(hash, k)`
# MAGIC   is a pure function of the input columns and runs entirely on executors.
# MAGIC - **Seeded `rand()`** is reproducible *per partition* — the same partition with
# MAGIC   the same seed produces the same values. This makes regression testing of
# MAGIC   downstream models possible.
# MAGIC
# MAGIC ### Where skew shows up (and how to handle it)
# MAGIC
# MAGIC 1. **Mega merchants** — by design, the first 50 merchant_ids absorb ~10% of
# MAGIC    all transactions. Joins to `dim_merchant` on `merchant_id` will hit hot
# MAGIC    partitions. Mitigations:
# MAGIC    - Enable `spark.sql.adaptive.skewJoin.enabled` (already set above) so AQE
# MAGIC      splits oversized partitions at runtime.
# MAGIC    - Broadcast `dim_merchant` (only ~100k rows ≈ a few MB) — Spark will pick
# MAGIC      this automatically when below `spark.sql.autoBroadcastJoinThreshold`.
# MAGIC    - For *aggregations* keyed on `merchant_id`, salt the key:
# MAGIC      `groupBy(merchant_id, pmod(rand_id, 16))` then re-aggregate. The salt
# MAGIC      column is a Spark expression, not a Python loop.
# MAGIC 2. **Risk-tier skew on `dim_customer`** — 75% of customers are LOW risk. Any
# MAGIC    `groupBy("risk_tier")` will see one heavy partition. Use `repartition(N,
# MAGIC    "risk_tier")` only when downstream needs co-location, otherwise let AQE
# MAGIC    coalesce.
# MAGIC 3. **Date skew** — peak shopping days (Black Friday, year-end) exist if you
# MAGIC    add seasonality. Partitioning by `event_date` gives O(1) pruning but you
# MAGIC    can still hit a hot date during writes. Use `optimizeWrite + autoCompact`
# MAGIC    on the Delta table.
# MAGIC
# MAGIC ### Why Spark-native APIs (no pandas / no Python UDFs)
# MAGIC
# MAGIC - **Catalyst optimizer** can reorder, push down, and fuse only Spark-native
# MAGIC   expressions. A Python UDF is opaque — the planner treats it as a black box
# MAGIC   and disables predicate pushdown around it.
# MAGIC - **Whole-stage codegen** compiles native expressions into a single Java class
# MAGIC   per stage — orders of magnitude faster than Python UDFs which require row
# MAGIC   serialization JVM ↔ Python ↔ JVM.
# MAGIC - **Memory locality** — pandas DataFrames live on the driver's heap. Once data
# MAGIC   exceeds driver RAM, pandas is unusable. Spark DataFrames live on executors.
# MAGIC - **Reproducibility** — `pmod(hash(...), N) + 1` is the same on every executor,
# MAGIC   every cluster, every Spark version. Random Python state across workers is not.
# MAGIC - **Safety from `collect()`** — every aggregation we run is a reduce, returning
# MAGIC   a tiny result. The driver never sees the 50M-row body.
# MAGIC
# MAGIC ### Production hardening checklist (out of scope for this notebook)
# MAGIC
# MAGIC - Liquid clustering on `(event_date, customer_id)` for very-large fact tables
# MAGIC - Z-ORDER on `customer_id` for dim_customer to speed point-lookups
# MAGIC - Bloom filter indexes on `merchant_id` and `device_id`
# MAGIC - Photon enablement for ~3–5× speedup on these aggregation patterns
# MAGIC - Unity Catalog row-level filters for PII (`ip_address`, `customer_id`)

# COMMAND ----------

# MAGIC %md
# MAGIC # Bronze: Inject realistic raw-data quality issues
# MAGIC
# MAGIC Real ingestion landing zones contain duplicates, nulls, sentinels, future
# MAGIC timestamps, late events, and orphan FKs. Models trained or rules tested on
# MAGIC the clean `fact_transactions` will fail in production. We derive
# MAGIC `bronze_fact_transactions_dirty` from the clean fact and inject reproducible
# MAGIC issues.
# MAGIC
# MAGIC ## Strategy
# MAGIC
# MAGIC Each transaction is assigned a deterministic **issue bucket** in `[0, 9999]`
# MAGIC via `pmod(hash(transaction_id, "issue"), 10000)`. Disjoint bucket ranges
# MAGIC map to disjoint issue types, so:
# MAGIC
# MAGIC - **No double-tagging.** Each row gets exactly one issue type (or `clean`).
# MAGIC - **Reproducibility.** Same `transaction_id` → same issue across runs.
# MAGIC - **Distribution control.** Bucket-range width = % of rows for that issue.
# MAGIC
# MAGIC `data_quality_issue_type` is set in lock-step with the mutated columns. The
# MAGIC Silver step **does not trust** this label — it re-derives the verdict from
# MAGIC the data itself; the label only verifies cleaning recall.

# COMMAND ----------

# Read the clean fact table generated above.
fact_clean = spark.table(f"{DB_NAME}.fact_transactions")

# Add ingestion_ts: when the transaction *arrived* in the warehouse. For most
# rows this is transaction_ts + a small random delay (0..600 seconds). The
# `late_event` bucket below overrides this. rand(seed=...) keeps it reproducible.
bronze_base = (
    fact_clean
    .withColumn(
        "_issue_bucket",
        F.pmod(F.hash(F.col("transaction_id"), F.lit("issue")), F.lit(10000)),
    )
    .withColumn(
        "ingestion_ts",
        F.col("transaction_ts")
        + F.expr("make_interval(0, 0, 0, 0, 0, 0, cast(rand(7001) * 600 as bigint))"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bucket → issue mapping
# MAGIC
# MAGIC | Buckets | Width | Issue type | Production analogue |
# MAGIC |---|---|---|---|
# MAGIC | 0–19 | 0.20% | `null_customer` | Token decode failure on legacy ATM stream |
# MAGIC | 20–39 | 0.20% | `null_merchant` | Acquirer feed missing MID for new sign-ups |
# MAGIC | 40–59 | 0.20% | `null_device` | Web channel — no device fingerprint captured |
# MAGIC | 60–89 | 0.30% | `null_amount` | Partial settlement message before posting |
# MAGIC | 90–119 | 0.30% | `negative_amount` | Refund/reversal mis-routed as txn |
# MAGIC | 120–139 | 0.20% | `invalid_currency` | Cross-network message with corrupted ISO code |
# MAGIC | 140–159 | 0.20% | `future_timestamp` | Mis-set acquirer clock / TZ bug |
# MAGIC | 160–179 | 0.20% | `orphan_customer` | Race: txn before customer onboarding lands |
# MAGIC | 180–199 | 0.20% | `orphan_merchant` | New merchant onboarded mid-batch |
# MAGIC | 200–249 | 0.50% | `late_event` | Offline POS terminal syncing on reconnect |
# MAGIC | 250–259 | 0.10% | `zero_amount` | $0 pre-auth leaking into settlement feed |
# MAGIC | 260–279 | 0.20% | `invalid_status` | New upstream status code outside our enum |
# MAGIC | 280–299 | 0.20% | `suspicious_default` | Sentinels: 0 / -1 / "UNKNOWN" |
# MAGIC | 300–319 | 0.20% | `null_country` | IP-to-country lookup timed out |
# MAGIC | 320+ | ~96.8% | `clean` | Healthy rows |

# COMMAND ----------

# Build the issue label as a single Catalyst CASE WHEN expression. Python is
# only used to compose the column tree; evaluation is per-row on executors.
issue_label = (
    F.when((F.col("_issue_bucket") >= 0)   & (F.col("_issue_bucket") < 20),  F.lit("null_customer"))
     .when((F.col("_issue_bucket") >= 20)  & (F.col("_issue_bucket") < 40),  F.lit("null_merchant"))
     .when((F.col("_issue_bucket") >= 40)  & (F.col("_issue_bucket") < 60),  F.lit("null_device"))
     .when((F.col("_issue_bucket") >= 60)  & (F.col("_issue_bucket") < 90),  F.lit("null_amount"))
     .when((F.col("_issue_bucket") >= 90)  & (F.col("_issue_bucket") < 120), F.lit("negative_amount"))
     .when((F.col("_issue_bucket") >= 120) & (F.col("_issue_bucket") < 140), F.lit("invalid_currency"))
     .when((F.col("_issue_bucket") >= 140) & (F.col("_issue_bucket") < 160), F.lit("future_timestamp"))
     .when((F.col("_issue_bucket") >= 160) & (F.col("_issue_bucket") < 180), F.lit("orphan_customer"))
     .when((F.col("_issue_bucket") >= 180) & (F.col("_issue_bucket") < 200), F.lit("orphan_merchant"))
     .when((F.col("_issue_bucket") >= 200) & (F.col("_issue_bucket") < 250), F.lit("late_event"))
     .when((F.col("_issue_bucket") >= 250) & (F.col("_issue_bucket") < 260), F.lit("zero_amount"))
     .when((F.col("_issue_bucket") >= 260) & (F.col("_issue_bucket") < 280), F.lit("invalid_status"))
     .when((F.col("_issue_bucket") >= 280) & (F.col("_issue_bucket") < 300), F.lit("suspicious_default"))
     .when((F.col("_issue_bucket") >= 300) & (F.col("_issue_bucket") < 320), F.lit("null_country"))
     .otherwise(F.lit("clean"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply mutations
# MAGIC
# MAGIC Tag the issue type first, then mutate columns in lock-step. Each `withColumn`
# MAGIC is a single Catalyst project — the planner fuses them into one stage.

# COMMAND ----------

bronze_mutated = (
    bronze_base
    .withColumn("data_quality_issue_type", issue_label)

    # ---------- Nulls / sentinels in critical FKs ----------
    # A null customer_id silently short-circuits fraud rules like
    # `txn_country != home_country` to NULL — hiding cross-border fraud.
    .withColumn(
        "customer_id",
        F.when(F.col("data_quality_issue_type") == "null_customer", F.lit(None).cast("long"))
         # Sentinel 0 from legacy systems = "unknown customer". Same effect as NULL
         # for risk scoring, but tracking the distinction helps prioritize fixes.
         .when(F.col("data_quality_issue_type") == "suspicious_default", F.lit(0).cast("long"))
         .otherwise(F.col("customer_id")),
    )
    .withColumn(
        "merchant_id",
        F.when(F.col("data_quality_issue_type") == "null_merchant", F.lit(None).cast("long"))
         # -1 sentinel: classic "unknown merchant" placeholder from older mainframes.
         .when(F.col("data_quality_issue_type") == "suspicious_default", F.lit(-1).cast("long"))
         .otherwise(F.col("merchant_id")),
    )
    .withColumn(
        "device_id",
        F.when(F.col("data_quality_issue_type") == "null_device", F.lit(None).cast("long"))
         .otherwise(F.col("device_id")),
    )
    .withColumn(
        "transaction_country",
        F.when(F.col("data_quality_issue_type") == "null_country", F.lit(None).cast("string"))
         # "UNKNOWN" placeholder — must be standardized to NULL in Silver.
         .when(F.col("data_quality_issue_type") == "suspicious_default", F.lit("UNKNOWN"))
         .otherwise(F.col("transaction_country")),
    )

    # ---------- Amount issues ----------
    # Amount drives every velocity rule. Negatives invert velocity scores; zeros
    # pollute customer-baseline averages used as ML features.
    .withColumn(
        "amount",
        F.when(F.col("data_quality_issue_type") == "null_amount", F.lit(None).cast("double"))
         # Negative: flip the sign so the magnitude stays realistic.
         .when(F.col("data_quality_issue_type") == "negative_amount", -F.col("amount"))
         # Zero: simulates a $0 pre-auth leaking into the settlement feed.
         .when(F.col("data_quality_issue_type") == "zero_amount", F.lit(0.0))
         .otherwise(F.col("amount")),
    )

    # ---------- Currency / status standardization issues ----------
    # Invalid ISO codes break FX normalization; out-of-enum statuses can be
    # incorrectly counted as APPROVED in fraud-loss reporting.
    .withColumn(
        "currency",
        F.when(F.col("data_quality_issue_type") == "invalid_currency", F.lit("XXX"))
         .otherwise(F.col("currency")),
    )
    .withColumn(
        "transaction_status",
        F.when(F.col("data_quality_issue_type") == "invalid_status", F.lit("UNKNOWN_BAD"))
         .otherwise(F.col("transaction_status")),
    )

    # ---------- Timestamp anomalies ----------
    # Future timestamps poison time-windowed aggregations: one 2099-dated row
    # makes 24h velocity counts span 70+ years. Late events arriving after the
    # model has already scored cause feature/score drift.
    .withColumn(
        "ingestion_ts",
        # Late events: ingestion 3..10 days *after* transaction. Day offset is
        # deterministic via hash so reruns produce the same lateness.
        F.when(
            F.col("data_quality_issue_type") == "late_event",
            F.col("transaction_ts")
            + F.expr("make_interval(0, 0, 0, "
                     "cast(3 + pmod(hash(transaction_id, 'late'), 8) as int), "
                     "0, 0, 0)"),
        ).otherwise(F.col("ingestion_ts")),
    )
    .withColumn(
        "transaction_ts",
        # Future timestamps push *transaction_ts* (not ingestion_ts) forward —
        # that's the realistic shape: source clock bug, not warehouse bug.
        F.when(
            F.col("data_quality_issue_type") == "future_timestamp",
            F.col("transaction_ts")
            + F.expr("make_interval(0, 0, 0, "
                     "cast(30 + pmod(hash(transaction_id, 'future'), 90) as int), "
                     "0, 0, 0)"),
        ).otherwise(F.col("transaction_ts")),
    )

    # ---------- Orphan FKs ----------
    # A customer_id outside dim_customer can't be enriched with risk_tier. An
    # inner-join silently drops the row (masking attacks); a left-join produces
    # NULL features the model treats as "low risk by default". Push outside the
    # dim range deliberately so the bug is impossible to miss.
    .withColumn(
        "customer_id",
        F.when(
            F.col("data_quality_issue_type") == "orphan_customer",
            F.lit(N_CUSTOMERS) + F.lit(1)
            + F.pmod(F.hash(F.col("transaction_id"), F.lit("orph_c")), F.lit(1_000_000)),
        ).otherwise(F.col("customer_id")),
    )
    .withColumn(
        "merchant_id",
        F.when(
            F.col("data_quality_issue_type") == "orphan_merchant",
            F.lit(N_MERCHANTS) + F.lit(1)
            + F.pmod(F.hash(F.col("transaction_id"), F.lit("orph_m")), F.lit(100_000)),
        ).otherwise(F.col("merchant_id")),
    )
    .drop("_issue_bucket")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inject duplicates (extra rows, not mutated values)
# MAGIC
# MAGIC Duplicates need real extra rows — payment networks legitimately re-emit
# MAGIC messages on retries. We take ~2% of rows, stamp them as `duplicate`, bump
# MAGIC `ingestion_ts` forward by 60–3600 seconds, and union them on. Both rows
# MAGIC share the same `transaction_id` — the Silver dedup must keep the *first*
# MAGIC arrival (which is the auth message; later messages are clearing/settlement,
# MAGIC and fraud decisioning runs at auth time).

# COMMAND ----------

DUP_SHARE_PCT = 2  # 2% — within the requested 1–3% band

duplicates = (
    bronze_mutated
    .where(F.pmod(F.hash(F.col("transaction_id"), F.lit("dup")), F.lit(100)) < F.lit(DUP_SHARE_PCT))
    .withColumn("data_quality_issue_type", F.lit("duplicate"))
    .withColumn(
        "ingestion_ts",
        F.col("ingestion_ts")
        + F.expr("make_interval(0, 0, 0, 0, 0, 0, "
                 "cast(60 + pmod(hash(transaction_id, 'dupoff'), 3540) as bigint))"),
    )
)

bronze_dirty = bronze_mutated.unionByName(duplicates)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist Bronze

# COMMAND ----------

(
    bronze_dirty.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("event_date")
    .saveAsTable(f"{DB_NAME}.bronze_fact_transactions_dirty")
)

print("Bronze issue-type distribution (groupBy reduce — never a full collect):")
(
    spark.table(f"{DB_NAME}.bronze_fact_transactions_dirty")
    .groupBy("data_quality_issue_type")
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample dirty rows + schema

# COMMAND ----------

print("=" * 80)
print("bronze_fact_transactions_dirty — schema & sample")
print("=" * 80)
spark.table(f"{DB_NAME}.bronze_fact_transactions_dirty").printSchema()

# Show one example of each issue type. groupBy + first() is a reduce — safe.
print("One sample row per issue type:")
(
    spark.table(f"{DB_NAME}.bronze_fact_transactions_dirty")
    .select("transaction_id", "customer_id", "merchant_id", "device_id",
            "amount", "currency", "transaction_country", "transaction_status",
            "transaction_ts", "ingestion_ts", "data_quality_issue_type")
    .where(F.col("data_quality_issue_type") != "clean")
    .orderBy("data_quality_issue_type", "transaction_id")
    .dropDuplicates(["data_quality_issue_type"])
    .show(20, truncate=False)
)

# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search setup for the Fraud Investigation RAG
# MAGIC
# MAGIC One-shot bootstrap. Creates everything notebook 06 expects:
# MAGIC
# MAGIC 1. Source Delta table `main.fraud_platform.fraud_policy_chunks` with
# MAGIC    synthetic policy / SOP / investigation-note chunks.
# MAGIC 2. A Vector Search endpoint `fraud_vs_endpoint` (if missing).
# MAGIC 3. A Delta-Sync index `main.fraud_platform.fraud_policy_index` using
# MAGIC    Databricks-managed embeddings (`databricks-gte-large-en`).
# MAGIC
# MAGIC Idempotent — safe to re-run. Tables get `MERGE`-style upserts; endpoint
# MAGIC and index creation skip if they already exist.
# MAGIC
# MAGIC The sample policy chunks cover the scenarios our prompts will hit:
# MAGIC
# MAGIC - High-risk merchant categories (gambling, crypto, gift cards)
# MAGIC - Cross-border AML thresholds and SAR obligations
# MAGIC - Card-not-present (CNP) and 3-D Secure rules
# MAGIC - Device fraud (rooted, emulator) playbooks
# MAGIC - Nighttime / velocity rules
# MAGIC - Analyst escalation matrix

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install + restart

# COMMAND ----------

# MAGIC %pip install --quiet databricks-vectorsearch databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC
# MAGIC These names must match what notebook 06 reads.

# COMMAND ----------

CATALOG = "main"
SCHEMA = "fraud_platform"
FQ_SCHEMA = f"{CATALOG}.{SCHEMA}"

CHUNKS_TABLE = f"{FQ_SCHEMA}.fraud_policy_chunks"
VS_ENDPOINT = "fraud_vs_endpoint"
VS_INDEX = f"{FQ_SCHEMA}.fraud_policy_index"

# Databricks-hosted embedding model. No API key needed when calling from
# inside the workspace; the index pulls embeddings server-side.
EMBEDDING_MODEL = "databricks-gte-large-en"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Source Delta table
# MAGIC
# MAGIC Vector Search Delta-Sync indexes need:
# MAGIC
# MAGIC - A primary key (`chunk_id` here)
# MAGIC - The text column that gets embedded (`chunk_text`)
# MAGIC - Change Data Feed enabled (`delta.enableChangeDataFeed=true`) so the
# MAGIC   index can incrementally sync inserts/updates.
# MAGIC
# MAGIC Extra columns (`document_name`, `section`, `access_level`) ride along as
# MAGIC retrieval metadata — they're returned alongside the chunk text.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
  chunk_id        STRING NOT NULL,
  document_name   STRING NOT NULL,
  section         STRING NOT NULL,
  access_level    STRING NOT NULL,
  chunk_text      STRING NOT NULL,
  CONSTRAINT fraud_policy_chunks_pk PRIMARY KEY (chunk_id)
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print(f"OK — {CHUNKS_TABLE} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Seed policy chunks
# MAGIC
# MAGIC Synthetic but plausible. Each chunk is short on purpose — embedding
# MAGIC quality drops on very long passages, and citations are easier to map
# MAGIC back to a specific section.

# COMMAND ----------

POLICY_CHUNKS = [
    # --- Merchant-category risk ---------------------------------------------
    {
        "chunk_id": "MCR-001",
        "document_name": "Merchant Category Risk Policy",
        "section": "1.1 High-risk categories",
        "access_level": "internal",
        "chunk_text": (
            "The following merchant categories are designated HIGH RISK and "
            "subject to enhanced monitoring: GAMBLING, CRYPTO, GIFT_CARDS, "
            "ADULT_ENTERTAINMENT, MONEY_TRANSFER. Transactions in these "
            "categories above 1,000 USD-equivalent require automated 3-D "
            "Secure verification when the channel supports it."
        ),
    },
    {
        "chunk_id": "MCR-002",
        "document_name": "Merchant Category Risk Policy",
        "section": "1.2 Volume rule",
        "access_level": "internal",
        "chunk_text": (
            "Merchants in high-risk categories whose 30-day fraud rate exceeds "
            "5% are placed on the watch list. Watch-listed merchants trigger "
            "an automatic HOLD_FOR_REVIEW on any transaction above 500 "
            "USD-equivalent, regardless of customer risk tier."
        ),
    },
    # --- Cross-border / AML --------------------------------------------------
    {
        "chunk_id": "AML-001",
        "document_name": "Cross-Border AML Procedures",
        "section": "2.1 Reporting thresholds",
        "access_level": "restricted",
        "chunk_text": (
            "Cross-border transactions above 10,000 USD-equivalent must be "
            "flagged for Currency Transaction Report (CTR) review within 24 "
            "hours. Aggregated daily flow above 10,000 USD-equivalent across "
            "multiple smaller transactions ('structuring') triggers SAR review."
        ),
    },
    {
        "chunk_id": "AML-002",
        "document_name": "Cross-Border AML Procedures",
        "section": "2.2 Sanctions and OFAC",
        "access_level": "restricted",
        "chunk_text": (
            "Any transaction whose origin or destination country appears on "
            "the OFAC sanctions list must be blocked at authorisation and "
            "escalated to the AML desk. Customer accounts with two or more "
            "sanctions hits in 90 days are frozen pending compliance review."
        ),
    },
    {
        "chunk_id": "AML-003",
        "document_name": "Cross-Border AML Procedures",
        "section": "2.3 High-risk corridors",
        "access_level": "restricted",
        "chunk_text": (
            "Transactions between FATF-grey-listed countries are scored at "
            "double the normal cross-border weight. Recurring transfers "
            "(>3 in 30 days) on these corridors must be reviewed by Tier-2 "
            "analysts even when individually below threshold."
        ),
    },
    # --- Card-not-present ---------------------------------------------------
    {
        "chunk_id": "CNP-001",
        "document_name": "CNP and 3-D Secure Policy",
        "section": "3.1 Authentication requirements",
        "access_level": "internal",
        "chunk_text": (
            "All ECOMMERCE transactions where card_present_flag = false must "
            "carry a successful 3-D Secure authentication unless the merchant "
            "has been granted a documented frictionless-flow exemption. "
            "Authentication failure is a hard decline; do not retry."
        ),
    },
    {
        "chunk_id": "CNP-002",
        "document_name": "CNP and 3-D Secure Policy",
        "section": "3.2 Velocity",
        "access_level": "internal",
        "chunk_text": (
            "More than five CNP transactions on a single card in a one-hour "
            "window are treated as suspected card-testing. Block subsequent "
            "attempts, queue the customer for outbound verification."
        ),
    },
    # --- Device fraud --------------------------------------------------------
    {
        "chunk_id": "DEV-001",
        "document_name": "Device Risk Playbook",
        "section": "4.1 Rooted / emulator devices",
        "access_level": "internal",
        "chunk_text": (
            "Transactions originating from devices flagged as rooted or "
            "emulated are treated as high-risk regardless of customer "
            "history. First occurrence: HOLD_FOR_REVIEW. Repeat occurrence "
            "from the same device id within 7 days: ESCALATE_TO_TIER2."
        ),
    },
    {
        "chunk_id": "DEV-002",
        "document_name": "Device Risk Playbook",
        "section": "4.2 Device sharing",
        "access_level": "internal",
        "chunk_text": (
            "A single device id used by more than three distinct customer ids "
            "in 30 days is a strong account-takeover signal. Place the device "
            "on the device blocklist; require step-up authentication on all "
            "future transactions from that device."
        ),
    },
    # --- Nighttime / velocity ------------------------------------------------
    {
        "chunk_id": "VEL-001",
        "document_name": "Velocity and Time-of-Day Rules",
        "section": "5.1 Nighttime",
        "access_level": "internal",
        "chunk_text": (
            "Transactions between 00:00 and 05:00 local merchant time are "
            "weighted at 1.5x the daytime fraud risk for retail categories. "
            "A nighttime transaction from a customer who has never previously "
            "transacted at night is a notable behavioural anomaly."
        ),
    },
    {
        "chunk_id": "VEL-002",
        "document_name": "Velocity and Time-of-Day Rules",
        "section": "5.2 Burst patterns",
        "access_level": "internal",
        "chunk_text": (
            "Three or more declines followed by a successful authorisation "
            "within 10 minutes is a card-testing pattern. Reverse-and-hold "
            "the successful transaction, queue for analyst review."
        ),
    },
    # --- Analyst escalation matrix ------------------------------------------
    {
        "chunk_id": "ESC-001",
        "document_name": "Analyst Escalation Matrix",
        "section": "6.1 Recommended actions",
        "access_level": "internal",
        "chunk_text": (
            "APPROVE: zero risk signals, no policy hits, customer in good "
            "standing. HOLD_FOR_REVIEW: 1-2 risk signals or one watch-list "
            "match. ESCALATE_TO_TIER2: 3+ risk signals, repeat device/IP "
            "anomaly, or merchant on watch list. FILE_SAR: any AML / "
            "sanctions hit, structuring evidence, or confirmed fraud > "
            "10,000 USD-equivalent."
        ),
    },
    {
        "chunk_id": "ESC-002",
        "document_name": "Analyst Escalation Matrix",
        "section": "6.2 Customer communication",
        "access_level": "restricted",
        "chunk_text": (
            "Customers must not be told a transaction was held due to "
            "AML/SAR review (tipping-off prohibition under BSA §5318(g)). "
            "Use neutral language: 'additional verification required.' "
            "Document the actual reason in the case management system only."
        ),
    },
    # --- Prior investigation notes ------------------------------------------
    {
        "chunk_id": "INV-001",
        "document_name": "Investigation Notes — Q1 Trends",
        "section": "Pattern A",
        "access_level": "internal",
        "chunk_text": (
            "Pattern A (observed Jan-Mar): low-value gambling top-ups from "
            "newly opened accounts (<30 days), followed within 24 hours by "
            "a single large CRYPTO purchase. Treat the gambling top-ups as "
            "account-grooming and block the crypto leg."
        ),
    },
    {
        "chunk_id": "INV-002",
        "document_name": "Investigation Notes — Q1 Trends",
        "section": "Pattern B",
        "access_level": "internal",
        "chunk_text": (
            "Pattern B: device emulator + cross-border + nighttime + "
            "card-not-present is a near-certain fraud cluster. Of 412 cases "
            "matching all four signals last quarter, 387 were confirmed "
            "fraud. Treat as ESCALATE_TO_TIER2 by default."
        ),
    },
]

print(f"Prepared {len(POLICY_CHUNKS)} chunks.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upsert chunks
# MAGIC
# MAGIC `MERGE` so re-runs don't duplicate. Non-PK columns get overwritten if a
# MAGIC chunk's text or metadata is edited.

# COMMAND ----------

chunks_df = spark.createDataFrame(POLICY_CHUNKS)

chunks_df.createOrReplaceTempView("_staged_chunks")

spark.sql(f"""
MERGE INTO {CHUNKS_TABLE} t
USING _staged_chunks s
ON t.chunk_id = s.chunk_id
WHEN MATCHED THEN UPDATE SET
  document_name = s.document_name,
  section       = s.section,
  access_level  = s.access_level,
  chunk_text    = s.chunk_text
WHEN NOT MATCHED THEN INSERT (chunk_id, document_name, section, access_level, chunk_text)
VALUES (s.chunk_id, s.document_name, s.section, s.access_level, s.chunk_text)
""")

print(f"Row count in {CHUNKS_TABLE}: {spark.table(CHUNKS_TABLE).count()}")
spark.table(CHUNKS_TABLE).show(3, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vector Search endpoint
# MAGIC
# MAGIC The endpoint is the compute layer that hosts indexes. Endpoints scale
# MAGIC independently of the indexes attached to them. For a demo a single
# MAGIC `STANDARD` endpoint is plenty.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing_endpoints = {ep["name"] for ep in vsc.list_endpoints().get("endpoints", [])}

if VS_ENDPOINT in existing_endpoints:
    print(f"Endpoint '{VS_ENDPOINT}' already exists.")
else:
    print(f"Creating endpoint '{VS_ENDPOINT}' (this can take a few minutes)...")
    vsc.create_endpoint_and_wait(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print("Endpoint ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Vector Search index
# MAGIC
# MAGIC `delta_sync` index = stays in sync with the source Delta table via CDC.
# MAGIC `pipeline_type="TRIGGERED"` runs a sync on demand (we trigger it below);
# MAGIC for prod, switch to `"CONTINUOUS"` for near-real-time updates.

# COMMAND ----------

existing_indexes = {ix["name"] for ix in vsc.list_indexes(name=VS_ENDPOINT).get("vector_indexes", [])}

if VS_INDEX in existing_indexes:
    print(f"Index '{VS_INDEX}' already exists.")
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
else:
    print(f"Creating index '{VS_INDEX}'...")
    index = vsc.create_delta_sync_index_and_wait(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        primary_key="chunk_id",
        source_table_name=CHUNKS_TABLE,
        pipeline_type="TRIGGERED",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=EMBEDDING_MODEL,
    )
    print("Index ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Trigger an initial sync
# MAGIC
# MAGIC On a freshly-created TRIGGERED index, the first sync starts automatically
# MAGIC at creation. `sync()` is here as an explicit re-run after edits to the
# MAGIC source table.

# COMMAND ----------

try:
    index.sync()
    print("Sync triggered.")
except Exception as e:
    # First sync is already running on a brand-new index — that's fine.
    print(f"sync() reported: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Smoke test
# MAGIC
# MAGIC Top-3 chunks for a typical retrieval query. If this returns content the
# MAGIC index is live and notebook 06 will work.

# COMMAND ----------

results = index.similarity_search(
    query_text="cross-border high-amount gambling merchant",
    columns=["chunk_text", "document_name", "section", "access_level"],
    num_results=3,
)

for row in results.get("result", {}).get("data_array", []):
    chunk_text, doc, section, access, score = row[0], row[1], row[2], row[3], row[-1]
    print(f"[{score:.3f}] {doc} §{section} ({access})")
    print(f"  {chunk_text[:140]}...\n")

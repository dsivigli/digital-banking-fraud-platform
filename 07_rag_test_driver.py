# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Fraud-Investigation — Test Driver
# MAGIC
# MAGIC Exercises every layer of `06_rag_fraud_investigation.py`:
# MAGIC
# MAGIC 1. Pick real `transaction_id`s from Gold (fraud + clean + cross-border).
# MAGIC 2. Test each retrieval function in isolation — fail fast if a Gold table
# MAGIC    is missing a column or the Vector Search index isn't reachable.
# MAGIC 3. Run the full `investigate_transaction` flow on each scenario.
# MAGIC 4. Test the refusal path with a non-existent id.
# MAGIC 5. Inspect the audit table.
# MAGIC 6. (Optional) Call the deployed Databricks App over HTTP.
# MAGIC
# MAGIC Run cells top-to-bottom. Each scenario is independent — if one fails,
# MAGIC skip ahead to isolate the issue.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Import the library
# MAGIC
# MAGIC `06_rag_fraud_investigation` defines the functions; this notebook only
# MAGIC calls them. `%run` brings the module's symbols into this namespace.

# COMMAND ----------

# MAGIC %run ./06_rag_fraud_investigation

# COMMAND ----------

# Sanity check — these come from the imported notebook.
print("Catalog/schema:", FQ_SCHEMA)
print("Vector Search index:", VS_INDEX)
print("LLM endpoint:", LLM_ENDPOINT)
print("Audit table:", AUDIT_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pick representative transaction IDs from Gold
# MAGIC
# MAGIC Pulling real ids beats hard-coding strings — the test stays valid as the
# MAGIC data evolves. Three buckets so we exercise different prompt shapes:
# MAGIC
# MAGIC - **fraud**: `fraud_label = 1` — the model should recommend ESCALATE / SAR.
# MAGIC - **clean**: `fraud_label = 0` and unremarkable — should recommend APPROVE.
# MAGIC - **cross-border**: `is_cross_border = true` — exercises the AML policy chunks.

# COMMAND ----------

from pyspark.sql import functions as F

gold_tx = spark.table(GOLD_TRANSACTIONS)

def _pick_one(df, predicate, label):
    rows = df.where(predicate).select("transaction_id").limit(1).collect()
    if not rows:
        print(f"[warn] no {label} transaction available")
        return None
    return rows[0]["transaction_id"]

ID_FRAUD = _pick_one(gold_tx, F.col("fraud_label") == 1, "fraud")
ID_CLEAN = _pick_one(
    gold_tx,
    (F.col("fraud_label") == 0) & (F.col("is_cross_border") == False),
    "clean",
)
ID_CROSS_BORDER = _pick_one(gold_tx, F.col("is_cross_border") == True, "cross-border")
ID_MISSING = "txn_does_not_exist_xxxxxx"

print({"fraud": ID_FRAUD, "clean": ID_CLEAN, "cross_border": ID_CROSS_BORDER})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test each retrieval function in isolation
# MAGIC
# MAGIC Cheaper than running the LLM and pinpoints which layer is broken when
# MAGIC something fails end-to-end.

# COMMAND ----------

# 2a. Transaction lookup — happy path.
tx = get_transaction_context(ID_FRAUD)
assert tx.found, "expected to find the fraud transaction"
print(f"transaction_id={tx.transaction_id} customer={tx.row.get('customer_id')} merchant={tx.row.get('merchant_id')}")
tx.row

# COMMAND ----------

# 2b. Transaction lookup — missing id.
missing = get_transaction_context(ID_MISSING)
assert not missing.found, "expected missing flag"
print("OK — refusal path engaged for missing id")

# COMMAND ----------

# 2c. Entity lookup — populated from the fraud transaction.
ent = get_entity_context(
    customer_id=tx.row.get("customer_id"),
    merchant_id=tx.row.get("merchant_id"),
    device_id=tx.row.get("device_id"),
    ip_address=tx.row.get("ip_address"),
)
print("customer:", ent.customer)
print("merchant:", ent.merchant)
print("device:  ", ent.device)
print("ip:      ", ent.ip)

# COMMAND ----------

# 2d. Vector Search — small k so we can eyeball the chunks.
query = _build_retrieval_query(tx, ent)
print("retrieval query:", query)
chunks = retrieve_policy_context(query, k=3)
for i, c in enumerate(chunks, 1):
    print(f"--- chunk {i} (score={c.score:.3f}) {c.document_name} §{c.section} ---")
    print(c.chunk_text[:300], "...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. End-to-end on each scenario

# COMMAND ----------

def _show(label, result):
    print("=" * 80)
    print(f"{label}: transaction_id={result.transaction_id}  latency={result.latency_ms} ms  chunks={len(result.policy_chunks)}")
    print("=" * 80)
    print(result.answer)
    print()

if ID_FRAUD:
    _show("FRAUD", investigate_transaction(ID_FRAUD))

if ID_CLEAN:
    _show("CLEAN", investigate_transaction(ID_CLEAN))

if ID_CROSS_BORDER:
    _show("CROSS-BORDER", investigate_transaction(ID_CROSS_BORDER))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Refusal path
# MAGIC
# MAGIC The contract: missing transaction → exact string "Insufficient evidence to
# MAGIC investigate." (no LLM call). Asserts the cheap path stays cheap.

# COMMAND ----------

import time
t0 = time.monotonic()
refused = investigate_transaction(ID_MISSING)
elapsed = (time.monotonic() - t0) * 1000

print(refused.answer)
print(f"latency: {elapsed:.0f} ms")
assert refused.answer == "Insufficient evidence to investigate."
assert elapsed < 1500, f"refusal path took too long ({elapsed:.0f} ms) — should skip the LLM"
print("OK — refusal short-circuits the LLM")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspect the audit table
# MAGIC
# MAGIC Every call should leave a row. Newest first.

# COMMAND ----------

audit = (
    spark.table(AUDIT_TABLE)
    .orderBy(F.col("generated_at").desc())
    .select("transaction_id", "generated_at", "latency_ms",
            "model_endpoint", F.length("answer").alias("answer_chars"))
    .limit(10)
)
audit.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. (Optional) Call the deployed Databricks App over HTTP
# MAGIC
# MAGIC If you've deployed `rag_investigation_app/` as a Databricks App, set
# MAGIC `APP_URL` to its base URL and run this cell. The App authenticates the
# MAGIC caller via OBO; from a notebook the simplest path is a PAT token.

# COMMAND ----------

# Set these two and uncomment to exercise the App.
# APP_URL = "https://fraud-investigation-rag-<workspace-id>.aws.databricksapps.com"
# DATABRICKS_TOKEN = dbutils.secrets.get(scope="my-scope", key="pat")  # or paste a PAT
#
# import requests
# r = requests.post(
#     f"{APP_URL}/investigate",
#     headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
#     json={"transaction_id": ID_FRAUD, "k": 5},
#     timeout=60,
# )
# r.raise_for_status()
# data = r.json()
# print(data["answer"])
# print(f"latency: {data['latency_ms']} ms, chunks: {len(data['policy_chunks'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Tear-down (optional)
# MAGIC
# MAGIC Wipe the audit table between demo runs so the table stays small. Comment
# MAGIC out for real use — the audit log is append-only on purpose.

# COMMAND ----------

# spark.sql(f"TRUNCATE TABLE {AUDIT_TABLE}")

# Databricks notebook source
# MAGIC %md
# MAGIC # Banking Fraud Lakehouse — RAG Fraud-Investigation Assistant
# MAGIC
# MAGIC Given a `transaction_id`, retrieve grounded context from Gold Delta tables and
# MAGIC a Vector Search index of fraud policies / investigation notes, then ask a
# MAGIC Databricks-served LLM to produce a structured investigation summary.
# MAGIC
# MAGIC ## Why Gold (and not Bronze) for retrieval
# MAGIC
# MAGIC - **Bronze** is raw, unvalidated, possibly duplicated. Hallucination risk goes
# MAGIC   up when the model sees rows that contradict each other.
# MAGIC - **Silver** is clean but row-level — too verbose for a prompt context window
# MAGIC   and missing the engineered features fraud analysts actually use.
# MAGIC - **Gold** holds the engineered, aggregated, business-meaningful signals
# MAGIC   (customer fraud rate, merchant risk score, device reuse stats). These are
# MAGIC   the same numbers that drive dashboards and rules — the model and the
# MAGIC   analyst are reading from the same source of truth.
# MAGIC
# MAGIC ## Anti-hallucination posture
# MAGIC
# MAGIC The retrieval functions are deliberately strict: missing transaction → empty
# MAGIC context, not a fabricated row. The prompt instructs the model to refuse if
# MAGIC the evidence is empty. Every citation in the answer must trace back to a
# MAGIC retrieved chunk or column.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

logger = logging.getLogger("rag_fraud_investigation")
logger.setLevel(logging.INFO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC
# MAGIC Single source of truth for table, index, and endpoint names. Keep these as
# MAGIC module-level constants so unit tests can monkey-patch them, and so a
# MAGIC promote-to-prod step can swap them without code edits.
# MAGIC
# MAGIC In a real deployment these would come from a Databricks job parameter,
# MAGIC `dbutils.widgets`, or a config file in a Unity Catalog volume — never from
# MAGIC user input (SQL injection risk on `spark.table(...)`).

# COMMAND ----------

CATALOG = "main"
SCHEMA = "fraud_platform"
FQ_SCHEMA = f"{CATALOG}.{SCHEMA}"

# Gold tables — engineered, BI-shaped, the trustworthy retrieval surface.
GOLD_TRANSACTIONS = f"{FQ_SCHEMA}.gold_fact_transactions"
GOLD_CUSTOMER_FEATURES = f"{FQ_SCHEMA}.gold_customer_fraud_features"
GOLD_MERCHANT_RISK = f"{FQ_SCHEMA}.gold_top_risky_merchants"
GOLD_DEVICE_RISK = f"{FQ_SCHEMA}.gold_device_risk_summary"
GOLD_IP_RISK = f"{FQ_SCHEMA}.gold_ip_risk_summary"

# Vector Search — fraud policies, SOPs, and prior investigation notes.
VS_ENDPOINT = "fraud_vs_endpoint"
VS_INDEX = f"{FQ_SCHEMA}.fraud_policy_index"

# LLM. Keep model name in one place so a swap is a one-line change.
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800

# Audit log — every prompt + answer is appended here for compliance review.
AUDIT_TABLE = f"{FQ_SCHEMA}.rag_investigation_audit"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data classes
# MAGIC
# MAGIC Typed return values make the orchestration function readable and let the
# MAGIC audit log have a stable schema. Dataclasses serialize cleanly to JSON via
# MAGIC `asdict()` — useful for both the prompt and the audit row.

# COMMAND ----------

@dataclass
class TransactionContext:
    transaction_id: str
    found: bool
    row: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityContext:
    customer: dict[str, Any] = field(default_factory=dict)
    merchant: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    ip: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyChunk:
    chunk_text: str
    document_name: str
    section: str
    access_level: str
    score: float


@dataclass
class InvestigationResult:
    transaction_id: str
    answer: str
    transaction: TransactionContext
    entities: EntityContext
    policy_chunks: list[PolicyChunk]
    latency_ms: int
    generated_at: str


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `get_transaction_context`
# MAGIC
# MAGIC Filtered read on the Gold transaction table.
# MAGIC
# MAGIC - **Why projected columns and not `SELECT *`**: Gold tables can have 80+
# MAGIC   columns. Trimming to the 12 the model actually needs (a) shrinks prompt
# MAGIC   tokens, (b) keeps PII out of the LLM context unless explicitly required,
# MAGIC   (c) lets Photon prune the read at the file level.
# MAGIC - **Why `.limit(1).collect()`**: `transaction_id` is the natural key; we
# MAGIC   expect 0 or 1 row. `collect()` is safe here because the result is bounded.

# COMMAND ----------

# Columns the analyst summary genuinely needs. Adjust to your schema.
_TRANSACTION_COLS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "ip_address",
    "transaction_ts",
    "amount",
    "currency",
    "merchant_category",
    "country_code",
    "is_cross_border",
    "fraud_label",
]


def get_transaction_context(transaction_id: str) -> TransactionContext:
    """Return a single transaction row from the Gold table, or empty if missing."""
    if not transaction_id or not isinstance(transaction_id, str):
        return TransactionContext(transaction_id=str(transaction_id), found=False)

    df = (
        spark.table(GOLD_TRANSACTIONS)
        .where(F.col("transaction_id") == transaction_id)
        .select(*_TRANSACTION_COLS)
        .limit(1)
    )
    rows = df.collect()
    if not rows:
        logger.warning("transaction_not_found id=%s", transaction_id)
        return TransactionContext(transaction_id=transaction_id, found=False)

    return TransactionContext(
        transaction_id=transaction_id,
        found=True,
        row=rows[0].asDict(recursive=True),
    )


# SQL equivalent (for analysts and for a SQL-native variant of the function):
#
#   SELECT transaction_id, customer_id, merchant_id, device_id, ip_address,
#          transaction_ts, amount, currency, merchant_category, country_code,
#          is_cross_border, fraud_label
#   FROM   main.fraud_platform.gold_fact_transactions
#   WHERE  transaction_id = :transaction_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. `get_entity_context`
# MAGIC
# MAGIC Pulls customer / merchant / device / IP risk signals. Each lookup is its own
# MAGIC narrow query against a Gold table — no wide join — because:
# MAGIC
# MAGIC - Gold risk tables are small (thousands to low millions of rows). Photon
# MAGIC   handles a point lookup in milliseconds.
# MAGIC - Decoupled lookups make it trivial to add/remove a signal source without
# MAGIC   touching a join graph.
# MAGIC - If a signal source is missing for an entity, the function still returns
# MAGIC   partial context — the prompt explicitly tells the model how to handle that.

# COMMAND ----------

_CUSTOMER_FEATURE_COLS = [
    "customer_id",
    "txn_count_30d",
    "fraud_count_30d",
    "fraud_rate_30d",
    "avg_amount_30d",
    "distinct_merchants_30d",
    "distinct_countries_30d",
    "account_age_days",
    "kyc_risk_tier",
]

_MERCHANT_RISK_COLS = [
    "merchant_id",
    "merchant_name",
    "merchant_category",
    "fraud_rate",
    "txn_count",
    "risk_tier",
]

_DEVICE_RISK_COLS = [
    "device_id",
    "distinct_customers",
    "fraud_rate",
    "is_emulator",
    "is_rooted",
    "first_seen",
]

_IP_RISK_COLS = [
    "ip_address",
    "country_code",
    "is_tor_exit_node",
    "is_known_vpn",
    "fraud_rate",
]


def _lookup_one(table: str, key_col: str, key_val: Any, cols: list[str]) -> dict[str, Any]:
    """Single-row Gold lookup. Returns {} if key is None or row absent."""
    if key_val is None:
        return {}
    df = (
        spark.table(table)
        .where(F.col(key_col) == key_val)
        .select(*cols)
        .limit(1)
    )
    rows = df.collect()
    return rows[0].asDict(recursive=True) if rows else {}


def get_entity_context(
    customer_id: str | None,
    merchant_id: str | None,
    device_id: str | None,
    ip_address: str | None = None,
) -> EntityContext:
    """Return enrichment from Gold risk tables for the entities on the transaction."""
    return EntityContext(
        customer=_lookup_one(
            GOLD_CUSTOMER_FEATURES, "customer_id", customer_id, _CUSTOMER_FEATURE_COLS
        ),
        merchant=_lookup_one(
            GOLD_MERCHANT_RISK, "merchant_id", merchant_id, _MERCHANT_RISK_COLS
        ),
        device=_lookup_one(
            GOLD_DEVICE_RISK, "device_id", device_id, _DEVICE_RISK_COLS
        ),
        ip=_lookup_one(GOLD_IP_RISK, "ip_address", ip_address, _IP_RISK_COLS),
    )


# SQL equivalents:
#
#   SELECT * EXCEPT(_metadata)
#   FROM main.fraud_platform.gold_customer_fraud_features
#   WHERE customer_id = :customer_id;
#
#   SELECT merchant_id, merchant_name, merchant_category, fraud_rate, txn_count, risk_tier
#   FROM   main.fraud_platform.gold_top_risky_merchants
#   WHERE  merchant_id = :merchant_id;
#
#   ... and so on for device / ip.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `retrieve_policy_context`
# MAGIC
# MAGIC Calls the Databricks Vector Search index that holds chunked fraud policy
# MAGIC documents and prior investigation notes.
# MAGIC
# MAGIC The query string is built from the transaction context (merchant category,
# MAGIC country, cross-border flag, amount band) — *not* from raw user input. This
# MAGIC keeps the retrieval grounded in actual transaction attributes and avoids
# MAGIC prompt-injection from any free-text the analyst types.

# COMMAND ----------

def retrieve_policy_context(query: str, k: int = 5) -> list[PolicyChunk]:
    """Top-k chunks from the fraud policy / investigation-notes index."""
    if not query:
        return []

    # Lazy import — keeps the notebook importable on clusters without the SDK.
    from databricks.vector_search.client import VectorSearchClient

    client = VectorSearchClient(disable_notice=True)
    index = client.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)

    results = index.similarity_search(
        query_text=query,
        columns=["chunk_text", "document_name", "section", "access_level"],
        num_results=k,
    )

    # Vector Search returns {"result": {"data_array": [[col1, col2, ...], ...]}}
    data_array = results.get("result", {}).get("data_array", []) or []
    out: list[PolicyChunk] = []
    for row in data_array:
        # Last element is the similarity score.
        chunk_text, document_name, section, access_level, score = (
            row[0], row[1], row[2], row[3], float(row[-1])
        )
        out.append(
            PolicyChunk(
                chunk_text=chunk_text,
                document_name=document_name,
                section=section,
                access_level=access_level,
                score=score,
            )
        )
    return out


def _build_retrieval_query(tx: TransactionContext, ent: EntityContext) -> str:
    """Compose a deterministic, structured query string for the index."""
    if not tx.found:
        return ""
    r = tx.row
    parts = [
        f"merchant_category={r.get('merchant_category')}",
        f"country={r.get('country_code')}",
        f"cross_border={r.get('is_cross_border')}",
        f"amount_band={'high' if (r.get('amount') or 0) >= 10000 else 'normal'}",
    ]
    if ent.merchant.get("risk_tier"):
        parts.append(f"merchant_risk_tier={ent.merchant['risk_tier']}")
    if ent.device.get("is_emulator"):
        parts.append("device_emulator=true")
    if ent.ip.get("is_tor_exit_node"):
        parts.append("ip_tor=true")
    return " ".join(parts)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. `build_prompt`
# MAGIC
# MAGIC The prompt is the only contract between retrieval and generation. Three
# MAGIC rules baked in:
# MAGIC
# MAGIC 1. **Refuse on empty evidence.** If retrieval returned nothing, the model
# MAGIC    must say so and stop — not invent a plausible-sounding investigation.
# MAGIC 2. **Cite or omit.** Every claim must reference either a transaction column
# MAGIC    or a `[doc:…]` policy chunk. No outside knowledge.
# MAGIC 3. **Structured output.** Fixed sections so a downstream consumer (case
# MAGIC    management UI, Slack bot) can parse without an LLM second pass.

# COMMAND ----------

_SYSTEM_INSTRUCTIONS = """\
You are a fraud-investigation assistant for a regulated bank. You answer ONLY
from the EVIDENCE block below. Do not use outside knowledge. Do not infer values
that are not present. If the EVIDENCE block is empty or the transaction was not
found, reply exactly: "Insufficient evidence to investigate."

Every factual claim must be supported by either a transaction/entity field or a
policy chunk cited as [doc:<document_name> §<section>].

Return your answer in this exact structure, using these section headers:

## Summary
One paragraph, plain English, no jargon.

## Evidence
Bulleted list of the specific fields and policy chunks you used.

## Policy references
Bulleted list of [doc:<document_name> §<section>] citations.

## Risk explanation
Why this transaction does or does not match a fraud pattern, grounded in the evidence.

## Recommended analyst action
One of: APPROVE / HOLD_FOR_REVIEW / ESCALATE_TO_TIER2 / FILE_SAR. Justify in one sentence.
"""


def build_prompt(
    tx: TransactionContext,
    ent: EntityContext,
    policy_chunks: list[PolicyChunk],
) -> str:
    """Assemble the system + evidence prompt."""
    if not tx.found:
        evidence = "EVIDENCE: (transaction not found)"
    else:
        evidence_obj = {
            "transaction": tx.row,
            "customer_features": ent.customer,
            "merchant_risk": ent.merchant,
            "device_risk": ent.device,
            "ip_risk": ent.ip,
            "policy_chunks": [
                {
                    "document_name": c.document_name,
                    "section": c.section,
                    "access_level": c.access_level,
                    "text": c.chunk_text,
                }
                for c in policy_chunks
            ],
        }
        # default=str handles datetimes, Decimals, etc.
        evidence = "EVIDENCE:\n" + json.dumps(evidence_obj, indent=2, default=str)

    return f"{_SYSTEM_INSTRUCTIONS}\n\n{evidence}\n"


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. `generate_answer`
# MAGIC
# MAGIC Single thin wrapper around the LLM call. Two implementations are provided:
# MAGIC the Python SDK path (preferred — typed, retries handled by the SDK) and a
# MAGIC SQL `ai_query` path (useful for batch backfills over a column of prompts).
# MAGIC
# MAGIC Swap to a different LLM by changing `LLM_ENDPOINT` only.

# COMMAND ----------

def generate_answer(prompt: str) -> str:
    """Call the Databricks model-serving endpoint and return the text response."""
    # Lazy import — same reason as the Vector Search client.
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name=LLM_ENDPOINT,
        messages=[
            {"role": "system", "content": "You are a fraud-investigation assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    # SDK returns a typed object; pull the first choice's text.
    return response.choices[0].message.content


# Batch / SQL alternative — useful when you want to score thousands of historical
# transactions in one job. `ai_query` runs server-side, no Python round-trip.
#
#   SELECT
#     transaction_id,
#     ai_query(
#       'databricks-meta-llama-3-3-70b-instruct',
#       prompt_text,
#       modelParameters => named_struct('temperature', 0.0, 'max_tokens', 800)
#     ) AS investigation_summary
#   FROM main.fraud_platform.rag_prompts_to_score;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. `investigate_transaction` — orchestration
# MAGIC
# MAGIC End-to-end flow. Times each stage so the audit row carries latency
# MAGIC breakdown — useful for the "how to reduce latency" follow-up.

# COMMAND ----------

def investigate_transaction(transaction_id: str, k: int = 5) -> InvestigationResult:
    """Full RAG flow: retrieve → prompt → generate → audit → return."""
    started = time.monotonic()

    tx = get_transaction_context(transaction_id)

    if not tx.found:
        result = InvestigationResult(
            transaction_id=transaction_id,
            answer="Insufficient evidence to investigate.",
            transaction=tx,
            entities=EntityContext(),
            policy_chunks=[],
            latency_ms=int((time.monotonic() - started) * 1000),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        _audit(result, prompt="(skipped: tx not found)")
        return result

    r = tx.row
    ent = get_entity_context(
        customer_id=r.get("customer_id"),
        merchant_id=r.get("merchant_id"),
        device_id=r.get("device_id"),
        ip_address=r.get("ip_address"),
    )

    query = _build_retrieval_query(tx, ent)
    chunks = retrieve_policy_context(query, k=k)

    prompt = build_prompt(tx, ent, chunks)
    answer = generate_answer(prompt)

    result = InvestigationResult(
        transaction_id=transaction_id,
        answer=answer,
        transaction=tx,
        entities=ent,
        policy_chunks=chunks,
        latency_ms=int((time.monotonic() - started) * 1000),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    _audit(result, prompt=prompt)
    return result


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Audit logging
# MAGIC
# MAGIC Compliance + RAG-evaluation both need a persisted record of `(transaction,
# MAGIC retrieved evidence, prompt, answer, model, version, timestamp)`. Append-only
# MAGIC Delta table — never updated, never deleted (governed by retention policy).

# COMMAND ----------

def _audit(result: InvestigationResult, prompt: str) -> None:
    """Append a single audit row. Failures must not break the user flow."""
    try:
        row = {
            "transaction_id": result.transaction_id,
            "generated_at": result.generated_at,
            "latency_ms": result.latency_ms,
            "model_endpoint": LLM_ENDPOINT,
            "vs_index": VS_INDEX,
            "prompt": prompt,
            "answer": result.answer,
            "evidence_json": json.dumps(
                {
                    "transaction": asdict(result.transaction),
                    "entities": asdict(result.entities),
                    "policy_chunks": [asdict(c) for c in result.policy_chunks],
                },
                default=str,
            ),
        }
        spark.createDataFrame([row]).write.mode("append").saveAsTable(AUDIT_TABLE)
    except Exception as e:
        # Log but don't raise — auditing is important but not user-blocking.
        logger.exception("audit_write_failed transaction_id=%s err=%s",
                         result.transaction_id, e)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Example invocation

# COMMAND ----------

# Example — replace with a real id from your Gold table.
# result = investigate_transaction("txn_000123")
# print(result.answer)
# print(f"latency: {result.latency_ms} ms, chunks: {len(result.policy_chunks)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Design notes (for the interview follow-up)
# MAGIC
# MAGIC ### Unity Catalog protects sensitive data
# MAGIC
# MAGIC - Grant `SELECT` on the Gold tables to the investigation service principal
# MAGIC   only — no analyst access to raw `ip_address` or `customer_id`.
# MAGIC - Column masks: a UC column-mask function on `ip_address` /
# MAGIC   `customer_email` returns the unmasked value only for callers in the
# MAGIC   `fraud_investigators` group; everyone else (and the LLM service principal
# MAGIC   when running for self-service queries) sees a hash.
# MAGIC - Row filters: restrict the policy index to chunks whose `access_level`
# MAGIC   matches the caller's clearance — same Vector Search index, different
# MAGIC   visible chunks per persona.
# MAGIC - Lineage: every Gold→prompt→answer flow shows up in UC lineage, so an
# MAGIC   auditor can prove which data fed which model run.
# MAGIC
# MAGIC ### Preventing hallucinations
# MAGIC
# MAGIC 1. Strict retrieval — never pad with "related" context the model didn't ask
# MAGIC    for.
# MAGIC 2. Refusal contract — empty evidence ⇒ "Insufficient evidence" reply.
# MAGIC 3. Forced citations — `[doc:…]` references checked by a post-filter (not
# MAGIC    shown here): if a citation doesn't match any retrieved chunk, mark the
# MAGIC    answer as suspect and queue for human review.
# MAGIC 4. Temperature 0, deterministic prompt order.
# MAGIC 5. Output schema — the structured headers make missing-section regressions
# MAGIC    easy to detect.
# MAGIC
# MAGIC ### Auditability
# MAGIC
# MAGIC - Every call writes to `rag_investigation_audit` with prompt + answer +
# MAGIC   retrieved chunks + model id + latency.
# MAGIC - Audit table is append-only; UC governs delete/update permissions.
# MAGIC - Periodic Delta Live Table can roll up answers to dashboards: % refusals,
# MAGIC   median latency, distribution of recommended actions.
# MAGIC
# MAGIC ### Scaling
# MAGIC
# MAGIC - **Concurrent calls**: model-serving endpoint with autoscaling; provisioned
# MAGIC   throughput for steady traffic, scale-to-zero for bursty.
# MAGIC - **Batch backfill**: use the `ai_query` SQL form across a Spark DataFrame
# MAGIC   of prompts — no Python driver bottleneck, runs on Photon.
# MAGIC - **Vector Search**: a single endpoint serves many indices; size the index
# MAGIC   to the retrieval QPS, not the document count.
# MAGIC - **Hot transactions**: cache `get_transaction_context` and
# MAGIC   `get_entity_context` per `(transaction_id)` for a short TTL — these
# MAGIC   change at most once when a label is updated.
# MAGIC
# MAGIC ### PII masking
# MAGIC
# MAGIC - UC column masks on Gold (preferred — masking happens before the row
# MAGIC   leaves the table).
# MAGIC - Tokenize `customer_id` / `ip_address` in the prompt: the model sees
# MAGIC   `customer_token=ABC123`, not the real id. Detokenize only in the audit
# MAGIC   row, which is access-controlled.
# MAGIC - Strip the `EVIDENCE` JSON of any field tagged `pii=true` in the UC tag
# MAGIC   catalog before it enters the prompt.
# MAGIC
# MAGIC ### Evaluating RAG quality
# MAGIC
# MAGIC - **Retrieval**: hold-out set of `(query, expected chunk ids)` pairs;
# MAGIC   measure recall@k and MRR. Re-run on every index rebuild.
# MAGIC - **Generation**: MLflow Evaluation with LLM-as-judge on faithfulness,
# MAGIC   answer relevance, and citation precision. Track in MLflow, gate
# MAGIC   promotions on regression thresholds.
# MAGIC - **End-to-end**: shadow the production endpoint against a champion model
# MAGIC   for a week; compare `recommended_action` distributions and fraud-analyst
# MAGIC   override rate.
# MAGIC - **Production drift**: the audit table powers a dashboard tracking
# MAGIC   refusal rate, citation-mismatch rate, and average chunks per answer over
# MAGIC   time.
# MAGIC
# MAGIC ### Reducing latency
# MAGIC
# MAGIC - Most time is in the LLM call. Use a smaller model for first-pass
# MAGIC   triage; escalate to a 70B model only when the small model returns
# MAGIC   `ESCALATE` or low confidence.
# MAGIC - Run retrieval and entity lookups in parallel — `concurrent.futures` over
# MAGIC   `_lookup_one` and `retrieve_policy_context`. Each is independent.
# MAGIC - Photon + Z-ORDER on `transaction_id` makes the Gold lookup a sub-100ms
# MAGIC   point read.
# MAGIC - Provisioned throughput on the LLM endpoint to remove cold-start.
# MAGIC - Cache the rendered `_SYSTEM_INSTRUCTIONS` system message — most providers
# MAGIC   support prompt caching for the static prefix, which here is the bulk of
# MAGIC   the input tokens.

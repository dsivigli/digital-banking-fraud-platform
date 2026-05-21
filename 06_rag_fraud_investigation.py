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

# Per-transaction context lives in the Silver enriched fact — already joined
# with customer / merchant / device dim attributes and engineered risk flags.
# This is the single best row-level source for the prompt.
SILVER_ENRICHED = f"{FQ_SCHEMA}.silver_fact_transactions_enriched"

# Customer master data — used to enrich beyond what's already on the silver row.
DIM_CUSTOMER = f"{FQ_SCHEMA}.dim_customer"

# Gold risk surfaces — small, aggregated, fast point lookups.
GOLD_MERCHANT_RISK = f"{FQ_SCHEMA}.gold_top_risky_merchants"
GOLD_DEVICE_RISK = f"{FQ_SCHEMA}.gold_device_risk_summary"

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

# Columns the analyst summary genuinely needs from the silver enriched fact.
# Trimmed from 50+ columns down to the ones that drive the prompt.
_TRANSACTION_COLS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "ip_address",
    "transaction_ts",
    "amount",
    "currency",
    "transaction_country",
    "merchant_category",
    "merchant_country",
    "merchant_risk_level",
    "device_type",
    "rooted_device_flag",
    "emulator_flag",
    "is_cross_border",
    "is_high_risk_merchant",
    "is_risky_device",
    "is_card_not_present",
    "is_nighttime_transaction",
    "risk_signal_count",
    "transaction_status",
]


def get_transaction_context(transaction_id) -> TransactionContext:
    """Return a single transaction row, or empty if missing.

    Accepts int or string — silver stores transaction_id as bigint.
    """
    if transaction_id is None or transaction_id == "":
        return TransactionContext(transaction_id=str(transaction_id), found=False)

    # Coerce to int when the caller passed a numeric string, so the predicate
    # doesn't accidentally do a string compare against a bigint column.
    try:
        key = int(transaction_id)
    except (TypeError, ValueError):
        return TransactionContext(transaction_id=str(transaction_id), found=False)

    df = (
        spark.table(SILVER_ENRICHED)
        .where(F.col("transaction_id") == F.lit(key))
        .select(*_TRANSACTION_COLS)
        .limit(1)
    )
    rows = df.collect()
    if not rows:
        logger.warning("transaction_not_found id=%s", transaction_id)
        return TransactionContext(transaction_id=str(transaction_id), found=False)

    return TransactionContext(
        transaction_id=str(transaction_id),
        found=True,
        row=rows[0].asDict(recursive=True),
    )


# SQL equivalent:
#
#   SELECT transaction_id, customer_id, merchant_id, device_id, ip_address,
#          transaction_ts, amount, currency, transaction_country,
#          merchant_category, merchant_country, merchant_risk_level,
#          device_type, rooted_device_flag, emulator_flag,
#          is_cross_border, is_high_risk_merchant, is_risky_device,
#          is_card_not_present, is_nighttime_transaction,
#          risk_signal_count, transaction_status
#   FROM   main.fraud_platform.silver_fact_transactions_enriched
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

# Customer master attributes from dim_customer. Note the column list will need
# to match your dim_customer schema — adjust if your dim has different columns.
_CUSTOMER_COLS = [
    "customer_id",
    "home_country",
    "customer_segment",
    "risk_tier",
    "is_high_net_worth",
    "kyc_status",
    "preferred_channel",
]

# gold_top_risky_merchants schema (verified against your workspace):
# merchant_id, merchant_name, merchant_category, merchant_country,
# merchant_risk_level, total_transactions, suspected_fraud_transactions,
# total_amount, suspected_fraud_amount, fraud_rate
_MERCHANT_RISK_COLS = [
    "merchant_id",
    "merchant_name",
    "merchant_category",
    "merchant_country",
    "merchant_risk_level",
    "total_transactions",
    "suspected_fraud_transactions",
    "fraud_rate",
]

# gold_device_risk_summary is keyed by (device_type, rooted_device_flag,
# emulator_flag) — not device_id. Pass the matching attributes from the
# transaction row to look up the population-level risk for that device profile.
_DEVICE_RISK_COLS = [
    "device_type",
    "rooted_device_flag",
    "emulator_flag",
    "total_transactions",
    "suspected_fraud_transactions",
    "fraud_rate",
]


def _lookup_customer(customer_id) -> dict[str, Any]:
    if customer_id is None:
        return {}
    df = (
        spark.table(DIM_CUSTOMER)
        .where(F.col("customer_id") == F.lit(customer_id))
        .select(*[c for c in _CUSTOMER_COLS if c])
        .limit(1)
    )
    rows = df.collect()
    return rows[0].asDict(recursive=True) if rows else {}


def _lookup_merchant_risk(merchant_id) -> dict[str, Any]:
    if merchant_id is None:
        return {}
    df = (
        spark.table(GOLD_MERCHANT_RISK)
        .where(F.col("merchant_id") == F.lit(merchant_id))
        .select(*_MERCHANT_RISK_COLS)
        .limit(1)
    )
    rows = df.collect()
    return rows[0].asDict(recursive=True) if rows else {}


def _lookup_device_risk(
    device_type: str | None,
    rooted: bool | None,
    emulator: bool | None,
) -> dict[str, Any]:
    """Device risk is summarized by *profile*, not per-device id."""
    if device_type is None:
        return {}
    df = (
        spark.table(GOLD_DEVICE_RISK)
        .where(F.col("device_type") == F.lit(device_type))
        .where(F.col("rooted_device_flag") == F.lit(bool(rooted)))
        .where(F.col("emulator_flag") == F.lit(bool(emulator)))
        .select(*_DEVICE_RISK_COLS)
        .limit(1)
    )
    rows = df.collect()
    return rows[0].asDict(recursive=True) if rows else {}


def get_entity_context(tx_row: dict[str, Any]) -> EntityContext:
    """Pull customer / merchant / device enrichment for one transaction row."""
    return EntityContext(
        customer=_lookup_customer(tx_row.get("customer_id")),
        merchant=_lookup_merchant_risk(tx_row.get("merchant_id")),
        device=_lookup_device_risk(
            device_type=tx_row.get("device_type"),
            rooted=tx_row.get("rooted_device_flag"),
            emulator=tx_row.get("emulator_flag"),
        ),
        ip={},  # No IP risk table in this workspace; left empty by design.
    )


# SQL equivalents:
#
#   SELECT customer_id, home_country, customer_segment, risk_tier,
#          is_high_net_worth, kyc_status, preferred_channel
#   FROM   main.fraud_platform.dim_customer
#   WHERE  customer_id = :customer_id;
#
#   SELECT merchant_id, merchant_name, merchant_category, merchant_country,
#          merchant_risk_level, total_transactions, suspected_fraud_transactions, fraud_rate
#   FROM   main.fraud_platform.gold_top_risky_merchants
#   WHERE  merchant_id = :merchant_id;
#
#   SELECT device_type, rooted_device_flag, emulator_flag, total_transactions,
#          suspected_fraud_transactions, fraud_rate
#   FROM   main.fraud_platform.gold_device_risk_summary
#   WHERE  device_type = :dt AND rooted_device_flag = :rooted AND emulator_flag = :emu;

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
        f"country={r.get('transaction_country')}",
        f"cross_border={r.get('is_cross_border')}",
        f"amount_band={'high' if (r.get('amount') or 0) >= 10000 else 'normal'}",
    ]
    if r.get("is_high_risk_merchant"):
        parts.append("high_risk_merchant=true")
    if r.get("is_risky_device") or r.get("emulator_flag") or r.get("rooted_device_flag"):
        parts.append("risky_device=true")
    if r.get("is_card_not_present"):
        parts.append("card_not_present=true")
    if r.get("is_nighttime_transaction"):
        parts.append("nighttime=true")
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

    ent = get_entity_context(tx.row)
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

# Databricks notebook source
# MAGIC %md
# MAGIC # Banking Fraud Lakehouse — ML Training & Evaluation
# MAGIC
# MAGIC Trains and evaluates two binary classifiers on `ml_transaction_fraud_features`
# MAGIC to predict `fraud_label`. Spark MLlib only — no pandas, no scikit-learn.
# MAGIC
# MAGIC Models trained:
# MAGIC
# MAGIC 1. **Logistic Regression** — fast, interpretable baseline. Coefficients map
# MAGIC    directly to feature contributions, which matters for banking compliance.
# MAGIC 2. **Random Forest** — captures non-linear interactions (e.g.
# MAGIC    "high amount AND nighttime AND new device") that linear models miss.
# MAGIC
# MAGIC Outputs:
# MAGIC
# MAGIC - `ml_fraud_model_logistic` / `ml_fraud_model_rf` — saved Spark MLlib pipelines.
# MAGIC - `ml_fraud_model_metrics` — comparison table (Delta).
# MAGIC - `ml_fraud_scored_transactions` — test-set scores (Delta).
# MAGIC
# MAGIC ## Why Spark MLlib (not scikit-learn) for banking fraud
# MAGIC
# MAGIC - **Distributed training** — fraud datasets are 100M+ rows. sklearn fits in
# MAGIC   a single process and tops out at what driver memory can hold.
# MAGIC - **Same engine for ETL and ML** — features and training share Catalyst
# MAGIC   plans, no `toPandas()` boundary that breaks at scale.
# MAGIC - **Identical pipeline for training and scoring** — the trained `PipelineModel`
# MAGIC   is the same artifact used for batch and streaming inference, eliminating
# MAGIC   training/serving skew (the #1 production ML failure mode).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.getOrCreate()

def _try_set(key: str, value: str) -> None:
    try:
        spark.conf.set(key, value)
    except Exception:
        pass

_try_set("spark.sql.adaptive.enabled", "true")
_try_set("spark.sql.adaptive.skewJoin.enabled", "true")

DB_NAME = "fraud_platform"
spark.sql(f"USE {DB_NAME}")

# Reproducibility seed. Single source of truth — every random op uses this so
# train/test splits, model init, and undersampling are bit-identical across runs.
# Why this matters: a regulator may ask "show me the exact data and parameters
# you used to make this prediction six months ago." Non-deterministic training
# makes that answer impossible.
SEED = 42

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Load feature table

# COMMAND ----------

ft = spark.table(f"{DB_NAME}.ml_transaction_fraud_features")

print("=" * 80)
print("ml_transaction_fraud_features — schema")
print("=" * 80)
ft.printSchema()

print("Sample rows:")
ft.show(5, truncate=False)

print("Row count:")
print(ft.count())  # single scalar — safe

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Filter labels and critical features
# MAGIC
# MAGIC ### Why feature preparation matters
# MAGIC
# MAGIC Spark MLlib refuses to train on rows with `null` in any feature vector
# MAGIC component — the model would silently propagate NaN. We make the missing-
# MAGIC value handling explicit:
# MAGIC
# MAGIC 1. Drop rows where the **label** is null (we have no ground truth there).
# MAGIC 2. Drop rows where critical **keys** are null (transaction_id, customer_id).
# MAGIC 3. **Impute** numeric features (median) — losing a row because
# MAGIC    `txn_count_1h` is null on a brand-new customer is wasteful; "0 prior
# MAGIC    transactions" is the correct interpretation.
# MAGIC 4. **Default categorical** features to a sentinel `"UNKNOWN"` so the
# MAGIC    StringIndexer doesn't drop the row.

# COMMAND ----------

# Critical-row filter. We use isNotNull() on the bare minimum that must exist.
filtered = ft.where(
    F.col("fraud_label").isNotNull()
    & F.col("transaction_id").isNotNull()
    & F.col("customer_id").isNotNull()
    & F.col("amount").isNotNull()
)

print(f"Rows after filtering: {filtered.count()}")
print("Fraud label distribution after filtering:")
filtered.groupBy("fraud_label").count().orderBy("fraud_label").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Categorical vs numeric features
# MAGIC
# MAGIC We split by **what the column means**, not by physical type. `transaction_hour`
# MAGIC is a numeric int but cyclical (0 and 23 are adjacent), so a tree can use it
# MAGIC raw; we keep it numeric. Boolean flags also stay numeric (cast to int).

# COMMAND ----------

# Categorical: low-cardinality strings that the model should treat as discrete.
# transaction_country has high cardinality (20+ countries) — we still index it
# but skip OneHot to avoid exploding the feature space. The model will treat
# the index as ordinal, which is fine for tree models; LR may suffer slightly
# but the alternative (one-hot of 20 countries) is worse for memory.
# Some columns may not exist in the feature table — we filter to those present.
candidate_categorical = [
    "merchant_risk_level",
    "amount_bucket",
    # transaction_country / customer_segment / transaction_channel were defined
    # in earlier layers but may not all be carried into the feature table; we
    # add them only if present.
    "transaction_country",
    "customer_segment",
    "transaction_channel",
]
categorical_cols = [c for c in candidate_categorical if c in filtered.columns]

# Numeric features. Boolean flags are cast to int below.
candidate_numeric = [
    "amount",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "avg_amount_7d", "avg_amount_30d", "amount_vs_avg_ratio",
    "distinct_countries_7d",
    "high_risk_merchant_ratio_30d", "distinct_merchants_30d",
    "distinct_devices_30d", "risky_device_ratio_30d",
    "nighttime_txn_ratio_30d", "declined_txn_ratio_7d",
    "risk_signal_count",
]
numeric_cols = [c for c in candidate_numeric if c in filtered.columns]

# Boolean flags as 0/1 ints. These are technically numeric but we list them
# separately so the cast is explicit.
candidate_boolean = [
    "is_nighttime_transaction", "is_cross_border", "is_international_transaction",
    "card_present_flag", "is_card_not_present",
    "rooted_device_flag", "emulator_flag",
]
boolean_cols = [c for c in candidate_boolean if c in filtered.columns]

print(f"Categorical: {categorical_cols}")
print(f"Numeric: {numeric_cols}")
print(f"Boolean (cast to int): {boolean_cols}")

# COMMAND ----------

# Cast booleans to int and fill categorical NULLs with "UNKNOWN" sentinel.
# Done in Spark SQL — no Python iteration over rows.
prepared = filtered
for c in boolean_cols:
    prepared = prepared.withColumn(c, F.col(c).cast("int"))
for c in categorical_cols:
    prepared = prepared.withColumn(c, F.coalesce(F.col(c), F.lit("UNKNOWN")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Train / test split
# MAGIC
# MAGIC ### Why reproducibility matters
# MAGIC
# MAGIC Banking models are audited. Six months from now an investigator may ask
# MAGIC *"what data did this model learn from?"* — `seed=42` means the answer is
# MAGIC always the same, given the same input table.
# MAGIC
# MAGIC ### Why class imbalance handling matters
# MAGIC
# MAGIC Real fraud rates are 0.1%–2%. A trivial model that always predicts "not
# MAGIC fraud" achieves >98% accuracy and zero business value. **Accuracy is
# MAGIC misleading.** What we care about:
# MAGIC
# MAGIC | Metric | Banking meaning |
# MAGIC |---|---|
# MAGIC | **Recall** (sensitivity) | Of all real frauds, how many did we catch? Misses cost the bank money. |
# MAGIC | **Precision** | Of flagged transactions, how many were really fraud? False positives cause customer friction. |
# MAGIC | **AUC** | Threshold-free ranking quality — does the model rank fraud above non-fraud? |
# MAGIC | **F1** | Harmonic mean of precision and recall — single number for imbalanced problems. |
# MAGIC
# MAGIC We split BEFORE imbalance handling so the test set retains the natural
# MAGIC class ratio — otherwise test metrics are no longer representative of
# MAGIC production traffic.

# COMMAND ----------

train_raw, test = prepared.randomSplit([0.8, 0.2], seed=SEED)
print(f"Train rows (before rebalancing): {train_raw.count()}")
print(f"Test rows: {test.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Class imbalance: weighted training
# MAGIC
# MAGIC We use **class weights** rather than undersampling because:
# MAGIC
# MAGIC - Undersampling discards data — at high imbalance ratios you throw away
# MAGIC   the majority of the training set.
# MAGIC - Weights let the optimizer "pay more attention" to fraud rows without
# MAGIC   losing the negative signal.
# MAGIC
# MAGIC Weight = `(total_rows / class_count) / 2`. The fraud class gets a high
# MAGIC weight (~50× if fraud rate is ~1%); non-fraud gets ~1.

# COMMAND ----------

# Compute per-row weights with a single window aggregation. No collect.
# weight = (N / class_count) / 2 — this normalizes both classes around 1 and
# inflates the rare class.
class_counts = (
    train_raw.groupBy("fraud_label").agg(F.count("*").alias("class_count"))
)
total = train_raw.count()
class_weights = class_counts.withColumn(
    "class_weight",
    F.lit(total) / (F.col("class_count") * F.lit(2.0)),
)
class_weights.show()

train = train_raw.join(class_weights.select("fraud_label", "class_weight"),
                       on="fraud_label", how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Build the ML pipeline
# MAGIC
# MAGIC One `Pipeline` per model so we can fit and persist atomically. Stages:
# MAGIC
# MAGIC 1. **StringIndexer** for each categorical → `<col>_idx`.
# MAGIC 2. **OneHotEncoder** on low-cardinality indices → `<col>_oh`.
# MAGIC    (We skip OneHot for `transaction_country` to avoid blowing up the
# MAGIC    feature dimension on high-cardinality columns; tree models handle the
# MAGIC    raw index fine, and LR loses a small amount of expressiveness.)
# MAGIC 3. **Imputer** (median) for numeric NULLs — windowed features are NULL
# MAGIC    on the customer's first transaction.
# MAGIC 4. **VectorAssembler** → single `features` vector for the classifier.
# MAGIC 5. **Classifier** with `weightCol="class_weight"`.

# COMMAND ----------

# StringIndexer: handleInvalid="keep" prevents the indexer from throwing when a
# test-set category was unseen at training time (it gets bucket = numLabels).
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]

# OneHotEncode the low-cardinality indices. Skip the high-cardinality ones.
HIGH_CARD = {"transaction_country"}  # kept as ordinal index
ohe_input = [f"{c}_idx" for c in categorical_cols if c not in HIGH_CARD]
ohe_output = [f"{c}_oh" for c in categorical_cols if c not in HIGH_CARD]
encoder = OneHotEncoder(inputCols=ohe_input, outputCols=ohe_output, handleInvalid="keep")

# Imputer: median is robust to outliers (long-tailed amount distribution).
# Output columns overwrite the originals — pipeline stays compact.
imputed_numeric = [f"{c}_imp" for c in numeric_cols]
imputer = Imputer(inputCols=numeric_cols, outputCols=imputed_numeric, strategy="median")

# Final assembly: imputed numeric + boolean flags + OneHot + high-card index.
assembler_inputs = (
    imputed_numeric
    + boolean_cols
    + ohe_output
    + [f"{c}_idx" for c in categorical_cols if c in HIGH_CARD]
)
assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features",
    handleInvalid="keep",  # propagates rare NaN as 0; keeps row in training
)

# Common preprocessing — the same stages feed both classifiers.
preprocessing_stages = indexers + [encoder, imputer, assembler]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression
# MAGIC
# MAGIC **Why interpretable:** the trained model exposes one coefficient per
# MAGIC feature. Coefficient × feature value = log-odds contribution to the fraud
# MAGIC score. A compliance officer can read off "this transaction was flagged
# MAGIC because `amount_vs_avg_ratio` contributed +2.3 log-odds." That direct
# MAGIC traceability is required by EU AI Act / OCC SR 11-7 explainability rules.

# COMMAND ----------

lr = LogisticRegression(
    labelCol="fraud_label",
    featuresCol="features",
    weightCol="class_weight",
    maxIter=50,
    regParam=0.01,    # mild L2 regularization to prevent overfit on rare-class weights
    elasticNetParam=0.0,
)
lr_pipeline = Pipeline(stages=preprocessing_stages + [lr])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest
# MAGIC
# MAGIC **Why nonlinear matters:** fraud signals interact. Nighttime alone is
# MAGIC weak; a new device alone is weak; high amount alone is weak. **All three
# MAGIC together** is overwhelmingly fraud. Linear models can't capture that
# MAGIC AND-interaction without manual feature crossing. Trees split on it
# MAGIC naturally — that's why ensembles dominate fraud benchmarks.

# COMMAND ----------

rf = RandomForestClassifier(
    labelCol="fraud_label",
    featuresCol="features",
    weightCol="class_weight",
    numTrees=100,
    maxDepth=8,
    seed=SEED,
    subsamplingRate=0.8,
    featureSubsetStrategy="sqrt",
)
rf_pipeline = Pipeline(stages=preprocessing_stages + [rf])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Train both models

# COMMAND ----------

print("Training Logistic Regression...")
lr_model = lr_pipeline.fit(train)

print("Training Random Forest...")
rf_model = rf_pipeline.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Score the test set

# COMMAND ----------

lr_pred = lr_model.transform(test)
rf_pred = rf_model.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Evaluate
# MAGIC
# MAGIC ### False positives vs false negatives in fraud
# MAGIC
# MAGIC | | Cost |
# MAGIC |---|---|
# MAGIC | **False positive** (flagged but legit) | Customer's card is declined at checkout — friction, complaints, churn risk. Recoverable. |
# MAGIC | **False negative** (fraud missed) | Direct loss to the bank or customer. Often unrecoverable. Regulatory risk. |
# MAGIC
# MAGIC Most banks tolerate a higher false positive rate to drive false negatives
# MAGIC down — recall is prioritized. The threshold (default 0.5 for LR) would
# MAGIC be tuned in production to hit a target recall, accepting whatever
# MAGIC precision falls out.

# COMMAND ----------

bin_eval_auc = BinaryClassificationEvaluator(
    labelCol="fraud_label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
bin_eval_pr = BinaryClassificationEvaluator(
    labelCol="fraud_label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
)
multi_eval_f1 = MulticlassClassificationEvaluator(
    labelCol="fraud_label", predictionCol="prediction", metricName="f1"
)
multi_eval_prec = MulticlassClassificationEvaluator(
    labelCol="fraud_label", predictionCol="prediction",
    metricName="weightedPrecision",
)
multi_eval_rec = MulticlassClassificationEvaluator(
    labelCol="fraud_label", predictionCol="prediction",
    metricName="weightedRecall",
)

def _evaluate(name: str, predictions):
    """Return a one-row Spark Row of metrics. Each .evaluate() is a reduce → safe scalar."""
    return {
        "model_name": name,
        "AUC": float(bin_eval_auc.evaluate(predictions)),
        "AUPRC": float(bin_eval_pr.evaluate(predictions)),
        "precision": float(multi_eval_prec.evaluate(predictions)),
        "recall": float(multi_eval_rec.evaluate(predictions)),
        "F1": float(multi_eval_f1.evaluate(predictions)),
    }

lr_metrics = _evaluate("logistic_regression", lr_pred)
rf_metrics = _evaluate("random_forest", rf_pred)

print("Logistic Regression:", lr_metrics)
print("Random Forest:      ", rf_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion matrix
# MAGIC
# MAGIC Cross-tab of `fraud_label` vs `prediction`. Driven by an aggregation, not
# MAGIC a collect — the result is a tiny 2×2 table.

# COMMAND ----------

print("Confusion matrix — Logistic Regression:")
lr_pred.groupBy("fraud_label").pivot("prediction", [0, 1]).count().show()

print("Confusion matrix — Random Forest:")
rf_pred.groupBy("fraud_label").pivot("prediction", [0, 1]).count().show()

# COMMAND ----------

print("Sample predictions — Random Forest:")
(
    rf_pred.select(
        "transaction_id", "customer_id", "amount",
        "fraud_label", "prediction",
        F.round(F.col("probability").getItem(1), 4).alias("fraud_score"),
    )
    .orderBy(F.desc("fraud_score"))
    .show(10, truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10 — Feature importance (Random Forest)
# MAGIC
# MAGIC Tree-based feature importance = mean decrease in impurity attributed to
# MAGIC each feature. The top features tell us *what the model actually learned*.
# MAGIC Typically the strongest fraud signals are:
# MAGIC
# MAGIC - `amount_vs_avg_ratio` — anomaly relative to personal baseline.
# MAGIC - `txn_count_1h` — velocity (card-testing pattern).
# MAGIC - `risky_device_ratio_30d` / `rooted_device_flag` — device hygiene.
# MAGIC - `declined_txn_ratio_7d` — issuer pre-warning signal.
# MAGIC - `is_card_not_present` — base-rate fraud differential.

# COMMAND ----------

# Pull the trained RF stage and align importances with the assembled features.
rf_stage = rf_model.stages[-1]
feature_names = rf_model.stages[-2].getInputCols()  # VectorAssembler input names
importances = rf_stage.featureImportances

# Build a small DataFrame of (feature, importance). spark.createDataFrame on
# this small list is safe — it's len(features) rows, not data rows.
fi_rows = [(name, float(importances[i])) for i, name in enumerate(feature_names)]
fi_df = (
    spark.createDataFrame(fi_rows, "feature STRING, importance DOUBLE")
    .orderBy(F.desc("importance"))
)
print("Top 20 features by importance:")
fi_df.show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11 — Model comparison
# MAGIC
# MAGIC | Aspect | Logistic Regression | Random Forest |
# MAGIC |---|---|---|
# MAGIC | **Interpretability** | High — per-feature coefficients | Medium — feature importance + SHAP |
# MAGIC | **Latency** | Very low (vector dot product) | Higher (tree traversal × N trees) |
# MAGIC | **Captures interactions** | No (without manual crosses) | Yes (natural splits) |
# MAGIC | **Tuning surface** | Small (regParam, elasticNet) | Large (numTrees, depth, subsampling) |
# MAGIC | **Production fit** | Real-time scoring, regulator-friendly | Batch scoring, ensemble bake-offs |
# MAGIC
# MAGIC Many fraud teams **deploy both**: LR for latency-critical real-time
# MAGIC scoring at the auth path, RF (or XGBoost) for offline batch re-scoring
# MAGIC and case prioritization.

# COMMAND ----------

metrics_table = spark.createDataFrame(
    [lr_metrics, rf_metrics],
    "model_name STRING, AUC DOUBLE, AUPRC DOUBLE, precision DOUBLE, recall DOUBLE, F1 DOUBLE",
)
metrics_table.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12 — Persist artifacts
# MAGIC
# MAGIC We save:
# MAGIC
# MAGIC - The trained `PipelineModel` (preprocessing + classifier in one bundle).
# MAGIC - The metrics comparison as Delta.
# MAGIC - The scored test set as Delta — useful for downstream calibration,
# MAGIC   threshold tuning, and post-hoc analysis.

# COMMAND ----------

import os

# Models are saved to DBFS (or workspace files outside Databricks). The path
# pattern below works on both. Pipelines round-trip exactly via load().
LR_MODEL_PATH = "/tmp/fraud_platform/ml_fraud_model_logistic"
RF_MODEL_PATH = "/tmp/fraud_platform/ml_fraud_model_rf"

lr_model.write().overwrite().save(LR_MODEL_PATH)
rf_model.write().overwrite().save(RF_MODEL_PATH)
print(f"Saved LR model → {LR_MODEL_PATH}")
print(f"Saved RF model → {RF_MODEL_PATH}")

# COMMAND ----------

# Metrics table as Delta — small, but Delta gives us version history.
(
    metrics_table.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.ml_fraud_model_metrics")
)

# COMMAND ----------

# Scored test set: keep the keys, the label, and the predicted fraud probability
# from each model. Useful for threshold tuning and audit.
scored = (
    rf_pred.select(
        "transaction_id", "customer_id", "transaction_ts", "event_date",
        "amount", "fraud_label",
        F.col("prediction").alias("rf_prediction"),
        F.col("probability").getItem(1).alias("rf_fraud_score"),
    )
    .join(
        lr_pred.select(
            "transaction_id",
            F.col("prediction").alias("lr_prediction"),
            F.col("probability").getItem(1).alias("lr_fraud_score"),
        ),
        on="transaction_id",
        how="inner",
    )
)

(
    scored.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("event_date")
    .saveAsTable(f"{DB_NAME}.ml_fraud_scored_transactions")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13 — Production considerations
# MAGIC
# MAGIC ### Real-time fraud scoring
# MAGIC
# MAGIC The same `PipelineModel` deserialized in a streaming job scores incoming
# MAGIC transactions sub-100ms per record. Critical: the **feature definitions
# MAGIC must match exactly** between training and serving — that's the value of
# MAGIC putting all preprocessing inside the Pipeline. Hand-rolled feature code
# MAGIC at the serving layer is the #1 source of training/serving skew.
# MAGIC
# MAGIC ### Streaming feature integration
# MAGIC
# MAGIC The windowed features in `ml_transaction_fraud_features` (txn_count_1h,
# MAGIC etc.) port to Structured Streaming with `withWatermark` + windowed
# MAGIC aggregations. An online feature store (Databricks Feature Engineering,
# MAGIC Feast) materializes the latest per-customer aggregates with sub-ms read
# MAGIC latency for the auth path.
# MAGIC
# MAGIC ### MLflow experiment tracking
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC with mlflow.start_run():
# MAGIC     mlflow.log_params({"numTrees": 100, "maxDepth": 8, "seed": SEED})
# MAGIC     mlflow.log_metrics(rf_metrics)
# MAGIC     mlflow.spark.log_model(rf_model, "model")
# MAGIC ```
# MAGIC
# MAGIC MLflow gives every training run a unique ID, captures the full parameter
# MAGIC and metric history, and lets you reload any past model. Required for
# MAGIC banking model risk management (SR 11-7 lineage requirements).
# MAGIC
# MAGIC ### Model drift monitoring
# MAGIC
# MAGIC Compute the same feature distributions (and the prediction distribution)
# MAGIC on production traffic and compare to training-time stats. Distribution
# MAGIC drift on `nighttime_txn_ratio_30d` or `declined_txn_ratio_7d` often
# MAGIC precedes fraud-attack waves. Concept drift on `fraud_label` rates triggers
# MAGIC retraining.
# MAGIC
# MAGIC ### Explainability and compliance
# MAGIC
# MAGIC Banks must explain adverse decisions to customers (FCRA in the US,
# MAGIC GDPR Article 22 in the EU). For LR, coefficient × feature is a direct
# MAGIC explanation. For tree ensembles, SHAP values give per-prediction
# MAGIC attributions. The pipeline must surface either as part of the scoring
# MAGIC response payload — not as an afterthought.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 14 — Summary
# MAGIC
# MAGIC ### Why behavioral features are powerful
# MAGIC
# MAGIC Fraud is defined relative to a customer's normal pattern. A $500 dinner
# MAGIC is normal for one customer and screamingly anomalous for another. Raw
# MAGIC `amount` cannot capture that. `amount_vs_avg_ratio`, `txn_count_1h`,
# MAGIC `distinct_devices_30d` — these *behavioral* aggregates are what drive
# MAGIC every modern fraud system.
# MAGIC
# MAGIC ### Why point-in-time correctness matters
# MAGIC
# MAGIC Every windowed feature in this notebook uses past-only frames. Without
# MAGIC that guarantee, the model trains on data it cannot see at scoring time
# MAGIC — offline metrics look spectacular and production performance collapses.
# MAGIC Leakage is the most expensive bug in ML and the easiest to miss.
# MAGIC
# MAGIC ### Why fraud is an imbalanced problem
# MAGIC
# MAGIC Real fraud rates are 0.1%–2%. Naive accuracy is meaningless. Models must
# MAGIC be evaluated on AUC / AUPRC / recall / precision and trained with class
# MAGIC weights or rebalancing so the rare class actually contributes to the loss.
# MAGIC
# MAGIC ### Why scalable Spark ML in banking
# MAGIC
# MAGIC Production fraud datasets are 100M–billions of rows across years of
# MAGIC history. Single-node ML libraries top out at driver memory; Spark MLlib
# MAGIC trains in parallel across the cluster. The same `PipelineModel` then
# MAGIC serves batch scoring (Spark) and real-time scoring (mleap / model
# MAGIC export) without re-implementing features — the strongest defense
# MAGIC against training/serving skew.

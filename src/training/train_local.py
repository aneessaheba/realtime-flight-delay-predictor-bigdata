"""
train_local.py
--------------
MLlib model training (LR + GBT) with hyperparameter tuning and validation.
Runs in Spark local mode — no HDFS required.

Reads BTS flight data from a local CSV or Parquet file, trains Logistic
Regression (baseline) and Gradient Boosted Trees (primary) models inside
full Spark ML Pipelines with CrossValidator, evaluates on a held-out test
set, and saves both serialized pipelines to the local models/ directory.

Usage:
    python src/train_local.py
    python src/train_local.py --input data/sample_flights.csv --cv-folds 3
    python src/train_local.py --input data/sample_flights.csv --skip-lr

The saved pipelines are compatible with the team's streaming consumer
(src/streaming/streaming_consumer.py) — load with PipelineModel.load().
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import Imputer, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_local")

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]
NUMERIC_COLS = [
    "DAY_OF_WEEK",
    "CRS_DEP_TIME",
    "DEP_DELAY",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "MONTH",
]
LABEL_COL = "label"
FEATURE_COL = "features"
WEIGHT_COL = "class_weight"
DELAY_THRESHOLD = 15.0

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ── Spark session ─────────────────────────────────────────────────────────────


def build_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder.appName("FlightDelay_LocalTraining")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ── Data loading ──────────────────────────────────────────────────────────────


BTS_COLUMN_MAP = {
    "Year": "YEAR", "Month": "MONTH", "DayofMonth": "DAY_OF_MONTH",
    "DayOfWeek": "DAY_OF_WEEK", "Reporting_Airline": "OP_UNIQUE_CARRIER",
    "Origin": "ORIGIN", "Dest": "DEST", "CRSDepTime": "CRS_DEP_TIME",
    "DepDelay": "DEP_DELAY", "CRSArrTime": "CRS_ARR_TIME",
    "ArrDelay": "ARR_DELAY", "CRSElapsedTime": "CRS_ELAPSED_TIME",
    "Distance": "DISTANCE", "CarrierDelay": "CARRIER_DELAY",
    "WeatherDelay": "WEATHER_DELAY", "NASDelay": "NAS_DELAY",
    "SecurityDelay": "SECURITY_DELAY", "LateAircraftDelay": "LATE_AIRCRAFT_DELAY",
}


def load_data(spark: SparkSession, input_path: str):
    logger.info("Loading data from: %s", input_path)

    if input_path.endswith(".parquet") or os.path.isdir(input_path):
        df = spark.read.parquet(input_path)
    else:
        df = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .option("nullValue", "")
            .option("mode", "PERMISSIVE")
            .csv(input_path)
        )
        for raw, std in BTS_COLUMN_MAP.items():
            if raw in df.columns:
                df = df.withColumnRenamed(raw, std)

    # Drop cancelled / diverted flights before creating the label
    df = df.filter(F.col("ARR_DELAY").isNotNull())

    # Create binary label: 1 if arrival delay > 15 min
    df = df.withColumn(
        LABEL_COL,
        F.when(F.col("ARR_DELAY") > DELAY_THRESHOLD, 1.0).otherwise(0.0),
    )

    # Keep only the columns the pipeline needs
    required = CATEGORICAL_COLS + NUMERIC_COLS + [LABEL_COL]
    available = set(df.columns)
    missing = [c for c in required if c not in available]
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")

    df = df.select(required)

    # Cast numerics
    for col in NUMERIC_COLS:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))
    df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast(DoubleType()))

    count = df.count()
    logger.info("Loaded %d records.", count)
    return df


def compute_class_weights(df) -> Dict[int, float]:
    counts = df.groupBy(LABEL_COL).count().collect()
    total = sum(row["count"] for row in counts)
    n_classes = len(counts)
    weights = {
        int(row[LABEL_COL]): total / (n_classes * row["count"])
        for row in counts
    }
    logger.info("Class weights: %s", weights)
    return weights


def add_class_weights(df, weights: Dict[int, float]):
    weight_expr = F.when(F.col(LABEL_COL) == 0.0, weights[0]).otherwise(weights[1])
    return df.withColumn(WEIGHT_COL, weight_expr.cast(DoubleType()))


def split_data(df):
    train_df, val_df, test_df = df.randomSplit(
        [TRAIN_RATIO, VAL_RATIO, TEST_RATIO], seed=RANDOM_SEED
    )
    logger.info(
        "Split — train: %d | val: %d | test: %d",
        train_df.count(),
        val_df.count(),
        test_df.count(),
    )
    return train_df, val_df, test_df


# ── Pipeline builders ─────────────────────────────────────────────────────────


def build_preprocessing_stages() -> List:
    """
    Preprocessing stages shared by both models:
      1. StringIndexer per categorical column
      2. Imputer (median) for numeric columns
      3. VectorAssembler → raw_features
      4. StandardScaler → features  (LR benefits; GBT ignores scale)
    """
    stages = []

    # StringIndexer for each categorical column
    indexed_cat_cols = []
    for col in CATEGORICAL_COLS:
        out = f"{col}_idx"
        stages.append(
            StringIndexer(inputCol=col, outputCol=out, handleInvalid="keep")
        )
        indexed_cat_cols.append(out)

    # Imputer for numeric columns
    imputed_num_cols = [f"{c}_imp" for c in NUMERIC_COLS]
    stages.append(
        Imputer(inputCols=NUMERIC_COLS, outputCols=imputed_num_cols, strategy="median")
    )

    # VectorAssembler
    stages.append(
        VectorAssembler(
            inputCols=indexed_cat_cols + imputed_num_cols,
            outputCol="raw_features",
            handleInvalid="keep",
        )
    )

    # StandardScaler
    stages.append(
        StandardScaler(
            inputCol="raw_features",
            outputCol=FEATURE_COL,
            withMean=True,
            withStd=True,
        )
    )

    return stages


def build_lr_pipeline(preprocessing_stages: List) -> Tuple[Pipeline, list]:
    lr = LogisticRegression(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
        family="binomial",
        standardization=False,
    )
    pipeline = Pipeline(stages=preprocessing_stages + [lr])
    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.001, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )
    return pipeline, param_grid


def build_gbt_pipeline(preprocessing_stages: List) -> Tuple[Pipeline, list]:
    gbt = GBTClassifier(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxIter=50,
        maxDepth=5,
        stepSize=0.1,
        subsamplingRate=0.8,
        seed=RANDOM_SEED,
    )
    pipeline = Pipeline(stages=preprocessing_stages + [gbt])
    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [4, 5])
        .addGrid(gbt.maxIter, [30, 50])
        .addGrid(gbt.stepSize, [0.05, 0.1])
        .build()
    )
    return pipeline, param_grid


# ── Training & evaluation ─────────────────────────────────────────────────────


def train_with_cv(pipeline: Pipeline, param_grid: list, train_df, num_folds: int):
    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=num_folds,
        seed=RANDOM_SEED,
        parallelism=2,
    )
    logger.info("Starting %d-fold cross-validation (%d param combos)...",
                num_folds, len(param_grid))
    t0 = time.time()
    cv_model = cv.fit(train_df)
    elapsed = time.time() - t0

    best_auc = max(cv_model.avgMetrics)
    logger.info("CV complete in %.1f s — best AUC-ROC: %.4f", elapsed, best_auc)
    return cv_model.bestModel, cv_model.avgMetrics


def evaluate_model(model, test_df, model_name: str) -> Dict:
    preds = model.transform(test_df)

    binary_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction"
    )
    mc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction"
    )

    metrics = {
        "model": model_name,
        "auc_roc":   round(binary_eval.evaluate(preds, {binary_eval.metricName: "areaUnderROC"}), 4),
        "auc_pr":    round(binary_eval.evaluate(preds, {binary_eval.metricName: "areaUnderPR"}), 4),
        "f1":        round(mc_eval.evaluate(preds, {mc_eval.metricName: "f1"}), 4),
        "precision": round(mc_eval.evaluate(preds, {mc_eval.metricName: "weightedPrecision"}), 4),
        "recall":    round(mc_eval.evaluate(preds, {mc_eval.metricName: "weightedRecall"}), 4),
        "accuracy":  round(mc_eval.evaluate(preds, {mc_eval.metricName: "accuracy"}), 4),
    }

    logger.info("=" * 60)
    logger.info("Results — %s", model_name)
    for k, v in metrics.items():
        if k != "model":
            logger.info("  %-12s : %s", k.upper(), v)
    logger.info("=" * 60)
    return metrics


def print_feature_importance(model, model_name: str) -> None:
    try:
        classifier = model.stages[-1]
        importances = classifier.featureImportances
        cat_indexed = [f"{c}_idx" for c in CATEGORICAL_COLS]
        imputed_num = [f"{c}_imp" for c in NUMERIC_COLS]
        names = cat_indexed + imputed_num
        ranked = sorted(zip(names, importances.toArray()), key=lambda x: -x[1])
        logger.info("Top feature importances (%s):", model_name)
        for name, imp in ranked[:10]:
            logger.info("  %-30s %.4f", name, imp)
    except AttributeError:
        logger.info("Feature importances not available for %s.", model_name)


def save_model(model, path: str, model_name: str) -> None:
    logger.info("Saving %s → %s", model_name, path)
    model.write().overwrite().save(path)
    logger.info("Saved.")


def save_metrics_json(all_metrics: List[Dict], out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Metrics written to %s", out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flight delay models locally.")
    parser.add_argument(
        "--input", default="data/sample_flights.csv",
        help="Local CSV or Parquet path (default: data/sample_flights.csv)"
    )
    parser.add_argument(
        "--model-dir", default="models/",
        help="Directory to save trained pipelines (default: models/)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=3,
        help="Cross-validation folds (default: 3 for local speed; use 5 for HDFS run)"
    )
    parser.add_argument(
        "--skip-lr", action="store_true",
        help="Skip Logistic Regression (faster iteration)"
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DATA-228 Flight Delay — Local ML Training")
    logger.info("  Input      : %s", args.input)
    logger.info("  Model dir  : %s", args.model_dir)
    logger.info("  CV folds   : %d", args.cv_folds)
    logger.info("=" * 60)

    spark = build_spark_session()

    try:
        # ── 1. Load & prepare data ───────────────────────────────────
        df = load_data(spark, args.input)
        weights = compute_class_weights(df)
        df = add_class_weights(df, weights)
        df.cache()

        train_df, val_df, test_df = split_data(df)
        train_df.cache()
        test_df.cache()

        preprocessing_stages = build_preprocessing_stages()
        all_metrics = []

        # ── 2. GBT (primary model) ───────────────────────────────────
        logger.info("\n>>> Training GBTClassifier (primary model)...")
        gbt_pipeline, gbt_grid = build_gbt_pipeline(preprocessing_stages)
        gbt_model, gbt_cv_scores = train_with_cv(
            gbt_pipeline, gbt_grid, train_df, args.cv_folds
        )
        gbt_metrics = evaluate_model(gbt_model, test_df, "GBTClassifier")
        gbt_metrics["cv_scores"] = [round(s, 4) for s in gbt_cv_scores]
        all_metrics.append(gbt_metrics)
        print_feature_importance(gbt_model, "GBTClassifier")
        save_model(gbt_model, os.path.join(args.model_dir, "gbt_pipeline"), "gbt_pipeline")

        # ── 3. Logistic Regression (baseline) ───────────────────────
        if not args.skip_lr:
            logger.info("\n>>> Training LogisticRegression (baseline)...")
            lr_pipeline, lr_grid = build_lr_pipeline(preprocessing_stages)
            lr_model, lr_cv_scores = train_with_cv(
                lr_pipeline, lr_grid, train_df, args.cv_folds
            )
            lr_metrics = evaluate_model(lr_model, test_df, "LogisticRegression")
            lr_metrics["cv_scores"] = [round(s, 4) for s in lr_cv_scores]
            all_metrics.append(lr_metrics)
            save_model(lr_model, os.path.join(args.model_dir, "lr_pipeline"), "lr_pipeline")

        # ── 4. Summary ───────────────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("FINAL TRAINING SUMMARY")
        logger.info("%-22s %8s %8s %8s %8s %8s",
                    "Model", "AUC-ROC", "AUC-PR", "F1", "Prec", "Recall")
        logger.info("-" * 66)
        for m in all_metrics:
            logger.info(
                "%-22s %8.4f %8.4f %8.4f %8.4f %8.4f",
                m["model"], m["auc_roc"], m["auc_pr"],
                m["f1"], m["precision"], m["recall"],
            )
        logger.info("=" * 60)

        # Check mid-term target from proposal (GBT F1 >= 0.70)
        gbt_f1 = gbt_metrics["f1"]
        status = "PASS" if gbt_f1 >= 0.70 else "below target (target: >= 0.70)"
        logger.info("Mid-term checkpoint — GBT F1: %.4f [%s]", gbt_f1, status)

        # Save metrics JSON for benchmarking report
        metrics_path = os.path.join(args.model_dir, "metrics.json")
        save_metrics_json(all_metrics, metrics_path)

        logger.info("\nTraining complete. Models saved to: %s", args.model_dir)

    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""
train_model.py
--------------
Batch ML training for the flight delay binary classification task.

Reads cleaned Parquet data from HDFS (years 2018-2023), trains a
LogisticRegression baseline and a GBTClassifier primary model using
Spark MLlib Pipelines with cross-validation and class-weight handling.
Serialises both full Pipelines to HDFS.

Usage:
    spark-submit src/training/train_model.py \
        --hdfs-path hdfs://hdfs-namenode:9000/data/flights \
        --model-path hdfs://hdfs-namenode:9000/models \
        --train-years 2018 2019 2020 2021 2022 2023
"""

import argparse
import logging
import sys
import time
from typing import Dict, List, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler,
    StandardScaler,
    Imputer,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_model")

# ─── Constants ────────────────────────────────────────────────────────────────

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

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ─── Spark session ────────────────────────────────────────────────────────────


def build_spark_session(app_name: str = "FlightDelay_Training") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "6g")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_data(spark: SparkSession, hdfs_path: str, years: List[int]):
    """Load Parquet data from HDFS and filter to the requested years."""
    logger.info("Loading data from %s for years %s", hdfs_path, years)
    df = spark.read.parquet(hdfs_path)
    df = df.filter(F.col("YEAR").isin(years))

    required_cols = CATEGORICAL_COLS + NUMERIC_COLS + [LABEL_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Cast numeric columns to double for MLlib compatibility
    for col_name in NUMERIC_COLS:
        df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
    df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast(DoubleType()))

    count = df.count()
    logger.info("Loaded %d records.", count)
    return df


def compute_class_weights(df) -> Dict[int, float]:
    """
    Compute balanced class weights:
        weight_i = total_samples / (n_classes * count_i)
    Returns a dict mapping label value (int) to weight (float).
    """
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
    """Add a per-row weight column to the DataFrame."""
    weight_expr = F.when(F.col(LABEL_COL) == 0, weights[0]).otherwise(weights[1])
    return df.withColumn(WEIGHT_COL, weight_expr.cast(DoubleType()))


def split_data(df, seed: int = RANDOM_SEED):
    """Stratified-ish split: 70 / 15 / 15 using randomSplit with a fixed seed."""
    train_df, val_df, test_df = df.randomSplit(
        [TRAIN_RATIO, VAL_RATIO, TEST_RATIO], seed=seed
    )
    logger.info(
        "Split sizes – train: %d, val: %d, test: %d",
        train_df.count(),
        val_df.count(),
        test_df.count(),
    )
    return train_df, val_df, test_df


# ─── Pipeline builders ────────────────────────────────────────────────────────


def build_preprocessing_stages() -> List:
    """
    Return a list of pipeline stages that transform categorical + numeric
    features into a single 'features' vector:

    1. StringIndexer for each categorical column
    2. Imputer for numeric columns (fills NaN/null with median)
    3. VectorAssembler → raw feature vector
    4. StandardScaler → scaled features (used by LR; GBT ignores scale)
    """
    stages = []

    # StringIndexer for categoricals
    indexed_cat_cols = []
    for col_name in CATEGORICAL_COLS:
        output_col = f"{col_name}_idx"
        indexer = StringIndexer(
            inputCol=col_name,
            outputCol=output_col,
            handleInvalid="keep",  # unknown categories → last index
        )
        stages.append(indexer)
        indexed_cat_cols.append(output_col)

    # Imputer for numeric columns
    imputed_num_cols = [f"{c}_imp" for c in NUMERIC_COLS]
    imputer = Imputer(
        inputCols=NUMERIC_COLS,
        outputCols=imputed_num_cols,
        strategy="median",
    )
    stages.append(imputer)

    # VectorAssembler
    all_feature_cols = indexed_cat_cols + imputed_num_cols
    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="raw_features",
        handleInvalid="keep",
    )
    stages.append(assembler)

    # StandardScaler
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol=FEATURE_COL,
        withMean=True,
        withStd=True,
    )
    stages.append(scaler)

    return stages


def build_lr_pipeline(preprocessing_stages: List) -> Tuple[Pipeline, object]:
    """Build a LogisticRegression pipeline and its parameter grid."""
    lr = LogisticRegression(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
        family="binomial",
        standardization=False,  # already scaled
    )
    pipeline = Pipeline(stages=preprocessing_stages + [lr])

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.001, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )
    return pipeline, param_grid


def build_gbt_pipeline(preprocessing_stages: List) -> Tuple[Pipeline, object]:
    """Build a GBTClassifier pipeline and its parameter grid."""
    gbt = GBTClassifier(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxIter=100,
        maxDepth=6,
        stepSize=0.1,
        subsamplingRate=0.8,
        seed=RANDOM_SEED,
    )
    pipeline = Pipeline(stages=preprocessing_stages + [gbt])

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [4, 6])
        .addGrid(gbt.maxIter, [50, 100])
        .addGrid(gbt.stepSize, [0.05, 0.1])
        .build()
    )
    return pipeline, param_grid


# ─── Training & evaluation ────────────────────────────────────────────────────


def train_with_cv(
    pipeline: Pipeline,
    param_grid,
    train_df,
    num_folds: int = 5,
) -> object:
    """Wrap the pipeline in a CrossValidator and fit on train_df."""
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
    logger.info("Starting cross-validated training (%d folds) …", num_folds)
    start = time.time()
    cv_model = cv.fit(train_df)
    elapsed = time.time() - start
    logger.info("Training completed in %.1f s.", elapsed)
    logger.info(
        "Best CV AUC-ROC: %.4f",
        max(cv_model.avgMetrics),
    )
    return cv_model.bestModel


def evaluate_model(model, test_df, model_name: str) -> Dict[str, float]:
    """Compute F1, AUC-ROC, Precision, and Recall on the test set."""
    predictions = model.transform(test_df)

    binary_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
    )
    auc_roc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})
    auc_pr = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderPR"})

    mc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
    )
    f1 = mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"})
    precision = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"})
    recall = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"})
    accuracy = mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"})

    metrics = {
        "model": model_name,
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
    }

    logger.info("=" * 60)
    logger.info("Evaluation results for: %s", model_name)
    for k, v in metrics.items():
        if k != "model":
            logger.info("  %-15s : %s", k.upper(), v)
    logger.info("=" * 60)
    return metrics


def save_pipeline(model, path: str, model_name: str) -> None:
    """Persist the best pipeline model to HDFS."""
    logger.info("Saving %s to %s …", model_name, path)
    model.write().overwrite().save(path)
    logger.info("Saved successfully.")


def print_feature_importance(model, model_name: str, preprocessing_stages: List) -> None:
    """Print feature importances if the model supports it (GBT/RF)."""
    try:
        # Navigate to the classifier stage in the best pipeline
        classifier = model.stages[-1]
        importances = classifier.featureImportances
        # Build feature names list matching VectorAssembler order
        cat_indexed = [f"{c}_idx" for c in CATEGORICAL_COLS]
        imputed_num = [f"{c}_imp" for c in NUMERIC_COLS]
        feature_names = cat_indexed + imputed_num
        logger.info("Top feature importances (%s):", model_name)
        pairs = sorted(
            zip(feature_names, importances.toArray()),
            key=lambda x: -x[1],
        )
        for name, imp in pairs[:10]:
            logger.info("  %-30s %.4f", name, imp)
    except AttributeError:
        logger.info("Feature importances not available for %s.", model_name)


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flight delay prediction models.")
    parser.add_argument(
        "--hdfs-path",
        default="hdfs://hdfs-namenode:9000/data/flights",
        help="HDFS path to cleaned Parquet data.",
    )
    parser.add_argument(
        "--model-path",
        default="hdfs://hdfs-namenode:9000/models",
        help="HDFS base path to save trained models.",
    )
    parser.add_argument(
        "--train-years",
        nargs="+",
        type=int,
        default=list(range(2018, 2024)),
        help="Years to use for training (default: 2018-2023).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--skip-lr",
        action="store_true",
        help="Skip Logistic Regression training (faster iteration).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting training job.")
    logger.info("  HDFS data path : %s", args.hdfs_path)
    logger.info("  Model path     : %s", args.model_path)
    logger.info("  Training years : %s", args.train_years)

    spark = build_spark_session()

    try:
        # Load and split data
        df = load_data(spark, args.hdfs_path, args.train_years)
        class_weights = compute_class_weights(df)
        df = add_class_weights(df, class_weights)
        df.cache()

        train_df, val_df, test_df = split_data(df)
        train_df.cache()
        test_df.cache()

        preprocessing_stages = build_preprocessing_stages()
        all_metrics = []

        # ── GBT (primary model) ──────────────────────────────────────
        logger.info("Training GBTClassifier …")
        gbt_pipeline, gbt_grid = build_gbt_pipeline(preprocessing_stages)
        gbt_model = train_with_cv(gbt_pipeline, gbt_grid, train_df, args.cv_folds)
        gbt_metrics = evaluate_model(gbt_model, test_df, "GBTClassifier")
        all_metrics.append(gbt_metrics)
        print_feature_importance(gbt_model, "GBTClassifier", preprocessing_stages)
        save_pipeline(gbt_model, f"{args.model_path}/gbt_pipeline", "gbt_pipeline")

        # ── Logistic Regression (baseline) ───────────────────────────
        if not args.skip_lr:
            logger.info("Training LogisticRegression baseline …")
            lr_pipeline, lr_grid = build_lr_pipeline(preprocessing_stages)
            lr_model = train_with_cv(lr_pipeline, lr_grid, train_df, args.cv_folds)
            lr_metrics = evaluate_model(lr_model, test_df, "LogisticRegression")
            all_metrics.append(lr_metrics)
            save_pipeline(lr_model, f"{args.model_path}/lr_pipeline", "lr_pipeline")

        # ── Summary ──────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("%-20s %8s %8s %8s %8s", "Model", "AUC-ROC", "F1", "Prec", "Recall")
        logger.info("-" * 60)
        for m in all_metrics:
            logger.info(
                "%-20s %8.4f %8.4f %8.4f %8.4f",
                m["model"],
                m["auc_roc"],
                m["f1"],
                m["precision"],
                m["recall"],
            )
        logger.info("=" * 60)
        logger.info("Training job completed successfully.")

    except Exception as exc:
        logger.exception("Training job failed: %s", exc)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

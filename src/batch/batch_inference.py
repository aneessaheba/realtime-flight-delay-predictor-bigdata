"""
batch_inference.py
------------------
Batch inference script: loads test data from HDFS, applies the saved GBT Pipeline,
evaluates classification metrics, and writes predictions to HDFS.

Usage:
    spark-submit src/batch/batch_inference.py \
        --data-path hdfs://namenode:9000/data/flights \
        --model-path hdfs://namenode:9000/models/gbt_pipeline \
        --output-path hdfs://namenode:9000/output/batch_predictions \
        --test-years 2024
"""

import argparse
import json
import logging
import sys
import time
from typing import Dict, List

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("batch_inference")

LABEL_COL = "label"


# ─── Spark session ────────────────────────────────────────────────────────────


def build_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder.appName("FlightDelay_BatchInference")
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


def load_test_data(spark: SparkSession, data_path: str, test_years: List[int]):
    """Load Parquet test data from HDFS and filter to the requested years."""
    logger.info("Loading test data from %s for years %s …", data_path, test_years)
    df = spark.read.parquet(data_path)

    if test_years:
        df = df.filter(F.col("YEAR").isin(test_years))

    # Ensure numeric features are cast to double for the pipeline
    numeric_cols = [
        "DAY_OF_WEEK", "CRS_DEP_TIME", "DEP_DELAY", "CRS_ELAPSED_TIME",
        "DISTANCE", "MONTH",
    ]
    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    if LABEL_COL in df.columns:
        df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast(DoubleType()))

    count = df.count()
    logger.info("Test data loaded: %d records.", count)
    return df


# ─── Model loading ────────────────────────────────────────────────────────────


def load_model(model_path: str) -> PipelineModel:
    """Load the saved GBT Pipeline from HDFS."""
    logger.info("Loading pipeline from %s …", model_path)
    model = PipelineModel.load(model_path)
    logger.info("Pipeline loaded successfully.")
    return model


# ─── Inference ────────────────────────────────────────────────────────────────


def run_inference(model: PipelineModel, test_df) -> tuple:
    """
    Apply the pipeline to the test DataFrame.
    Returns (predictions_df, inference_time_seconds).
    """
    logger.info("Running batch inference …")
    start = time.time()
    predictions_df = model.transform(test_df)
    # Force evaluation (Spark is lazy)
    predictions_df.cache()
    count = predictions_df.count()
    elapsed = time.time() - start
    logger.info(
        "Inference completed: %d records in %.2f s (%.1f records/s).",
        count,
        elapsed,
        count / elapsed if elapsed > 0 else 0,
    )
    return predictions_df, elapsed


# ─── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_predictions(predictions_df, model_name: str = "GBTClassifier") -> Dict:
    """Compute classification metrics on predictions that include true labels."""
    if LABEL_COL not in predictions_df.columns:
        logger.warning("Label column '%s' not found – skipping evaluation.", LABEL_COL)
        return {}

    binary_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
    )
    auc_roc = binary_eval.evaluate(
        predictions_df, {binary_eval.metricName: "areaUnderROC"}
    )
    auc_pr = binary_eval.evaluate(
        predictions_df, {binary_eval.metricName: "areaUnderPR"}
    )

    mc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
    )
    f1 = mc_eval.evaluate(predictions_df, {mc_eval.metricName: "f1"})
    precision = mc_eval.evaluate(
        predictions_df, {mc_eval.metricName: "weightedPrecision"}
    )
    recall = mc_eval.evaluate(
        predictions_df, {mc_eval.metricName: "weightedRecall"}
    )
    accuracy = mc_eval.evaluate(predictions_df, {mc_eval.metricName: "accuracy"})

    # Confusion matrix counts
    tp = predictions_df.filter(
        (F.col("prediction") == 1.0) & (F.col(LABEL_COL) == 1.0)
    ).count()
    tn = predictions_df.filter(
        (F.col("prediction") == 0.0) & (F.col(LABEL_COL) == 0.0)
    ).count()
    fp = predictions_df.filter(
        (F.col("prediction") == 1.0) & (F.col(LABEL_COL) == 0.0)
    ).count()
    fn = predictions_df.filter(
        (F.col("prediction") == 0.0) & (F.col(LABEL_COL) == 1.0)
    ).count()

    metrics = {
        "model": model_name,
        "mode": "batch",
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    logger.info("=" * 60)
    logger.info("Batch Inference Metrics – %s", model_name)
    logger.info("  AUC-ROC   : %.4f", auc_roc)
    logger.info("  AUC-PR    : %.4f", auc_pr)
    logger.info("  F1        : %.4f", f1)
    logger.info("  Precision : %.4f", precision)
    logger.info("  Recall    : %.4f", recall)
    logger.info("  Accuracy  : %.4f", accuracy)
    logger.info("  TP=%d  TN=%d  FP=%d  FN=%d", tp, tn, fp, fn)
    logger.info("=" * 60)

    return metrics


# ─── Output ───────────────────────────────────────────────────────────────────


def write_predictions(predictions_df, output_path: str) -> None:
    """Write prediction output to HDFS as Parquet."""
    logger.info("Writing batch predictions to %s …", output_path)

    output_cols = [
        "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
        "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
        "CRS_DEP_TIME", "DEP_DELAY", "ARR_DELAY",
        "DISTANCE", "prediction", "probability", "rawPrediction",
    ]
    # Only include columns that exist in the DataFrame
    available_cols = [c for c in output_cols if c in predictions_df.columns]
    if LABEL_COL in predictions_df.columns:
        available_cols.append(LABEL_COL)

    start = time.time()
    (
        predictions_df.select(available_cols)
        .write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(output_path)
    )
    elapsed = time.time() - start
    logger.info("Predictions written in %.2f s.", elapsed)


def save_metrics_json(metrics: Dict, local_path: str = "/output/batch_metrics.json") -> None:
    """Save evaluation metrics as a JSON file locally."""
    try:
        import os
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Metrics saved to %s.", local_path)
    except Exception as exc:
        logger.warning("Could not save metrics JSON: %s", exc)


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference using the trained GBT Pipeline."
    )
    parser.add_argument(
        "--data-path",
        default="hdfs://namenode:9000/data/flights",
        help="HDFS path to cleaned Parquet data.",
    )
    parser.add_argument(
        "--model-path",
        default="hdfs://namenode:9000/models/gbt_pipeline",
        help="HDFS path to the saved GBT PipelineModel.",
    )
    parser.add_argument(
        "--output-path",
        default="hdfs://namenode:9000/output/batch_predictions",
        help="HDFS output path for batch prediction Parquet files.",
    )
    parser.add_argument(
        "--test-years",
        nargs="+",
        type=int,
        default=[2024],
        help="Years to use as test data (default: 2024).",
    )
    parser.add_argument(
        "--metrics-json",
        default="/output/batch_metrics.json",
        help="Local path to save metrics JSON (default: /output/batch_metrics.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting batch inference job.")
    logger.info("  Data path   : %s", args.data_path)
    logger.info("  Model path  : %s", args.model_path)
    logger.info("  Output path : %s", args.output_path)
    logger.info("  Test years  : %s", args.test_years)

    spark = build_spark_session()
    job_start = time.time()

    try:
        test_df = load_test_data(spark, args.data_path, args.test_years)
        model = load_model(args.model_path)
        predictions_df, inference_time = run_inference(model, test_df)
        metrics = evaluate_predictions(predictions_df)
        metrics["inference_time_s"] = round(inference_time, 3)
        metrics["total_records"] = predictions_df.count()
        metrics["throughput_records_per_s"] = round(
            metrics["total_records"] / inference_time if inference_time > 0 else 0, 1
        )
        write_predictions(predictions_df, args.output_path)
        save_metrics_json(metrics, args.metrics_json)

        total_elapsed = time.time() - job_start
        logger.info("Batch inference job completed in %.2f s.", total_elapsed)

    except Exception as exc:
        logger.exception("Batch inference failed: %s", exc)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

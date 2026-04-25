"""
benchmark.py
------------
Compare batch vs. streaming prediction results, compute latency statistics
from streaming output timestamps, and produce a consolidated benchmark report.

Usage:
    python src/evaluation/benchmark.py \
        --batch-path hdfs://hdfs-namenode:9000/output/batch_predictions \
        --streaming-path hdfs://hdfs-namenode:9000/output/streaming_predictions \
        --report-path /output/benchmark_report.json \
        --spark-master spark://spark-master:7077
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark")

LABEL_COL = "label"


# ─── Spark session ────────────────────────────────────────────────────────────


def build_spark_session(master: Optional[str] = None) -> SparkSession:
    builder = SparkSession.builder.appName("FlightDelay_Benchmark")
    if master:
        builder = builder.master(master)
    spark = (
        builder
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ─── Data loading ────────────────────────────────────────────────────────────


def load_predictions(spark: SparkSession, path: str, mode_name: str):
    """Load prediction Parquet output from HDFS; return None if unavailable."""
    try:
        df = spark.read.parquet(path)
        count = df.count()
        logger.info("[%s] Loaded %d prediction records from %s.", mode_name, count, path)
        return df
    except Exception as exc:
        logger.warning("[%s] Could not load predictions from %s: %s", mode_name, path, exc)
        return None


# ─── Metric computation ───────────────────────────────────────────────────────


def compute_classification_metrics(df, mode_name: str) -> Dict[str, Any]:
    """Compute AUC-ROC, AUC-PR, F1, Precision, Recall, Accuracy from prediction df."""
    metrics: Dict[str, Any] = {"mode": mode_name}

    if df is None:
        logger.warning("[%s] No data – metrics unavailable.", mode_name)
        metrics["status"] = "no_data"
        return metrics

    has_label = LABEL_COL in df.columns
    has_raw_pred = "rawPrediction" in df.columns
    has_pred = "prediction" in df.columns

    if not has_pred:
        logger.warning("[%s] 'prediction' column not found.", mode_name)
        metrics["status"] = "missing_prediction_column"
        return metrics

    # Cast label to double if present
    if has_label:
        df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast(DoubleType()))

    record_count = df.count()
    metrics["record_count"] = record_count

    if has_label and has_raw_pred:
        binary_eval = BinaryClassificationEvaluator(
            labelCol=LABEL_COL,
            rawPredictionCol="rawPrediction",
        )
        try:
            metrics["auc_roc"] = round(
                binary_eval.evaluate(df, {binary_eval.metricName: "areaUnderROC"}), 4
            )
            metrics["auc_pr"] = round(
                binary_eval.evaluate(df, {binary_eval.metricName: "areaUnderPR"}), 4
            )
        except Exception as exc:
            logger.warning("[%s] Binary evaluation failed: %s", mode_name, exc)

    if has_label and has_pred:
        mc_eval = MulticlassClassificationEvaluator(
            labelCol=LABEL_COL,
            predictionCol="prediction",
        )
        try:
            metrics["f1"] = round(
                mc_eval.evaluate(df, {mc_eval.metricName: "f1"}), 4
            )
            metrics["precision"] = round(
                mc_eval.evaluate(df, {mc_eval.metricName: "weightedPrecision"}), 4
            )
            metrics["recall"] = round(
                mc_eval.evaluate(df, {mc_eval.metricName: "weightedRecall"}), 4
            )
            metrics["accuracy"] = round(
                mc_eval.evaluate(df, {mc_eval.metricName: "accuracy"}), 4
            )
        except Exception as exc:
            logger.warning("[%s] Multi-class evaluation failed: %s", mode_name, exc)

    # Prediction distribution
    if has_pred:
        dist = (
            df.groupBy("prediction")
            .count()
            .collect()
        )
        metrics["prediction_distribution"] = {
            str(int(row["prediction"])): row["count"] for row in dist
        }
        delayed = metrics["prediction_distribution"].get("1", 0)
        metrics["pct_predicted_delayed"] = round(
            100.0 * delayed / record_count if record_count > 0 else 0.0, 2
        )

    metrics["status"] = "ok"
    return metrics


def compute_latency_stats(streaming_df) -> Dict[str, Any]:
    """
    Compute end-to-end latency statistics from streaming predictions.
    Requires 'producer_ts' and 'consumer_ts' columns (epoch seconds).
    """
    stats: Dict[str, Any] = {}

    if streaming_df is None:
        return stats

    if "producer_ts" not in streaming_df.columns or "consumer_ts" not in streaming_df.columns:
        logger.info("Latency columns not present in streaming output.")
        return stats

    latency_df = streaming_df.withColumn(
        "latency_ms",
        (F.col("consumer_ts") - F.col("producer_ts")) * 1000.0,
    ).filter(F.col("latency_ms").isNotNull() & (F.col("latency_ms") >= 0))

    if latency_df.rdd.isEmpty():
        logger.info("No valid latency rows found.")
        return stats

    result = latency_df.select(
        F.mean("latency_ms").alias("mean_ms"),
        F.stddev("latency_ms").alias("stddev_ms"),
        F.min("latency_ms").alias("min_ms"),
        F.max("latency_ms").alias("max_ms"),
        F.percentile_approx("latency_ms", 0.50).alias("p50_ms"),
        F.percentile_approx("latency_ms", 0.90).alias("p90_ms"),
        F.percentile_approx("latency_ms", 0.95).alias("p95_ms"),
        F.percentile_approx("latency_ms", 0.99).alias("p99_ms"),
        F.count("latency_ms").alias("sample_count"),
    ).collect()[0]

    stats = {
        "mean_ms": round(result["mean_ms"] or 0, 2),
        "stddev_ms": round(result["stddev_ms"] or 0, 2),
        "min_ms": round(result["min_ms"] or 0, 2),
        "max_ms": round(result["max_ms"] or 0, 2),
        "p50_ms": round(result["p50_ms"] or 0, 2),
        "p90_ms": round(result["p90_ms"] or 0, 2),
        "p95_ms": round(result["p95_ms"] or 0, 2),
        "p99_ms": round(result["p99_ms"] or 0, 2),
        "sample_count": result["sample_count"],
    }
    return stats


def compute_throughput_stats(streaming_df) -> Dict[str, Any]:
    """
    Estimate per-batch throughput from streaming output.
    Requires 'batch_id' and 'consumer_ts' columns.
    """
    stats: Dict[str, Any] = {}

    if streaming_df is None:
        return stats

    if "batch_id" not in streaming_df.columns or "consumer_ts" not in streaming_df.columns:
        return stats

    batch_stats = (
        streaming_df.groupBy("batch_id")
        .agg(
            F.count("*").alias("events"),
            F.min("consumer_ts").alias("batch_start_ts"),
            F.max("consumer_ts").alias("batch_end_ts"),
        )
        .withColumn(
            "batch_duration_s",
            F.col("batch_end_ts") - F.col("batch_start_ts"),
        )
        .filter(F.col("batch_duration_s") > 0)
        .withColumn(
            "events_per_sec",
            F.col("events") / F.col("batch_duration_s"),
        )
    )

    if batch_stats.rdd.isEmpty():
        return stats

    agg = batch_stats.select(
        F.mean("events_per_sec").alias("mean_throughput"),
        F.max("events_per_sec").alias("max_throughput"),
        F.min("events_per_sec").alias("min_throughput"),
        F.sum("events").alias("total_events"),
        F.count("batch_id").alias("total_batches"),
    ).collect()[0]

    stats = {
        "mean_throughput_events_per_sec": round(agg["mean_throughput"] or 0, 1),
        "max_throughput_events_per_sec": round(agg["max_throughput"] or 0, 1),
        "min_throughput_events_per_sec": round(agg["min_throughput"] or 0, 1),
        "total_events": int(agg["total_events"] or 0),
        "total_batches": int(agg["total_batches"] or 0),
    }
    return stats


# ─── Comparison ──────────────────────────────────────────────────────────────


def compare_metrics(
    batch_metrics: Dict,
    streaming_metrics: Dict,
) -> Dict[str, Any]:
    """
    Compute deltas between batch and streaming classification metrics.
    Positive delta means streaming > batch.
    """
    comparison: Dict[str, Any] = {}
    metric_keys = ["auc_roc", "auc_pr", "f1", "precision", "recall", "accuracy"]
    for key in metric_keys:
        b = batch_metrics.get(key)
        s = streaming_metrics.get(key)
        if b is not None and s is not None:
            comparison[f"{key}_batch"] = b
            comparison[f"{key}_streaming"] = s
            comparison[f"{key}_delta"] = round(s - b, 4)
    return comparison


# ─── Report generation ────────────────────────────────────────────────────────


def build_report(
    batch_metrics: Dict,
    streaming_metrics: Dict,
    latency_stats: Dict,
    throughput_stats: Dict,
    comparison: Dict,
) -> Dict[str, Any]:
    report = {
        "report_generated_at": datetime.utcnow().isoformat() + "Z",
        "batch_metrics": batch_metrics,
        "streaming_metrics": streaming_metrics,
        "metric_comparison": comparison,
        "streaming_latency": latency_stats,
        "streaming_throughput": throughput_stats,
    }
    return report


def print_report(report: Dict) -> None:
    """Pretty-print the benchmark report to stdout."""
    sep = "=" * 70
    print(sep)
    print("FLIGHT DELAY PREDICTOR – BENCHMARK REPORT")
    print(f"Generated at: {report['report_generated_at']}")
    print(sep)

    def _fmt(label: str, val: Any, indent: int = 2) -> str:
        return " " * indent + f"{label:<35} {val}"

    print("\n[BATCH INFERENCE METRICS]")
    bm = report["batch_metrics"]
    for k in ["record_count", "auc_roc", "auc_pr", "f1", "precision", "recall", "accuracy"]:
        if k in bm:
            print(_fmt(k, bm[k]))

    print("\n[STREAMING INFERENCE METRICS]")
    sm = report["streaming_metrics"]
    for k in ["record_count", "auc_roc", "auc_pr", "f1", "precision", "recall", "accuracy"]:
        if k in sm:
            print(_fmt(k, sm[k]))

    print("\n[BATCH vs STREAMING COMPARISON (streaming - batch)]")
    comp = report["metric_comparison"]
    for key in ["auc_roc", "auc_pr", "f1", "precision", "recall", "accuracy"]:
        delta_key = f"{key}_delta"
        if delta_key in comp:
            sign = "+" if comp[delta_key] >= 0 else ""
            print(_fmt(delta_key, f"{sign}{comp[delta_key]:.4f}"))

    print("\n[STREAMING LATENCY (end-to-end: producer → prediction)]")
    lat = report["streaming_latency"]
    for k in ["mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms", "sample_count"]:
        if k in lat:
            print(_fmt(k, lat[k]))

    print("\n[STREAMING THROUGHPUT]")
    thr = report["streaming_throughput"]
    for k in ["mean_throughput_events_per_sec", "max_throughput_events_per_sec",
              "total_events", "total_batches"]:
        if k in thr:
            print(_fmt(k, thr[k]))

    print(sep)


def save_report(report: Dict, report_path: str) -> None:
    """Save the report as JSON to the local filesystem."""
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Benchmark report saved to %s.", report_path)
    except Exception as exc:
        logger.warning("Could not save report to %s: %s", report_path, exc)


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batch vs. streaming flight delay prediction."
    )
    parser.add_argument(
        "--batch-path",
        default="hdfs://hdfs-namenode:9000/output/batch_predictions",
        help="HDFS path to batch prediction Parquet output.",
    )
    parser.add_argument(
        "--streaming-path",
        default="hdfs://hdfs-namenode:9000/output/streaming_predictions",
        help="HDFS path to streaming prediction Parquet output.",
    )
    parser.add_argument(
        "--report-path",
        default="/output/benchmark_report.json",
        help="Local path to save the benchmark report JSON.",
    )
    parser.add_argument(
        "--spark-master",
        default=None,
        help="Spark master URL (optional; uses default if omitted).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting benchmark.")
    logger.info("  Batch path     : %s", args.batch_path)
    logger.info("  Streaming path : %s", args.streaming_path)
    logger.info("  Report path    : %s", args.report_path)

    spark = build_spark_session(args.spark_master)

    try:
        batch_df = load_predictions(spark, args.batch_path, "batch")
        streaming_df = load_predictions(spark, args.streaming_path, "streaming")

        batch_metrics = compute_classification_metrics(batch_df, "batch")
        streaming_metrics = compute_classification_metrics(streaming_df, "streaming")
        latency_stats = compute_latency_stats(streaming_df)
        throughput_stats = compute_throughput_stats(streaming_df)
        comparison = compare_metrics(batch_metrics, streaming_metrics)

        report = build_report(
            batch_metrics,
            streaming_metrics,
            latency_stats,
            throughput_stats,
            comparison,
        )

        print_report(report)
        save_report(report, args.report_path)
        logger.info("Benchmark completed successfully.")

    except Exception as exc:
        logger.exception("Benchmark failed: %s", exc)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""
evaluate_streaming.py
---------------------
Evaluate a finished streaming replay: predictive accuracy of the streamed
predictions against realised arrival outcomes, batch/streaming parity, and the
end-to-end latency distribution with a tail breakdown.

Inputs are the prediction Parquet written by streaming_consumer.py and its
--batch-log JSONL (per-batch HDFS write-completion times).

  * Accuracy: label = ARR_DELAY > 15 (records without ARR_DELAY are excluded
    and counted). AUC-ROC / AUC-PR from the streamed probability; positive-class
    and weighted metrics at 0.5 and at --threshold (the validation-selected
    threshold from train_chronological.py).
  * Parity: the same records are re-scored in batch mode with the same
    PipelineModel and the probabilities compared.
  * Latency: end-to-end = write_done_ts(batch) - producer_ts(record);
    queueing = latency_ms column (producer publish -> batch start).

Usage (inside the spark-master container):
    spark-submit src/evaluation/evaluate_streaming.py \
        --predictions hdfs://hdfs-namenode:9000/output/streaming_2024 \
        --batch-log /tmp/streaming_batches.jsonl \
        --model-path hdfs://hdfs-namenode:9000/models_chrono/full/gbt_pipeline \
        --threshold 0.79 --output /tmp/streaming_eval.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from train_chronological import confusion_at, metrics_from_confusion, PROB_COL  # noqa: E402

LABEL_COL = "label"
PERCENTILES = [50, 90, 95, 99, 99.9]
# Columns the producer sends that the model consumes (raw, pre-feature-derivation).
RAW_INPUT_COLS = [
    "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DEP_DELAY", "TAXI_OUT", "DISTANCE",
]


def accuracy_block(scored, threshold: float) -> dict:
    def auc(metric):
        return BinaryClassificationEvaluator(
            labelCol=LABEL_COL, rawPredictionCol=PROB_COL, metricName=metric
        ).evaluate(scored)

    return {
        "auc_roc": round(auc("areaUnderROC"), 4),
        "auc_pr": round(auc("areaUnderPR"), 4),
        "at_0.5": metrics_from_confusion(confusion_at(scored, 0.5)),
        "threshold": threshold,
        "at_threshold": metrics_from_confusion(confusion_at(scored, threshold)),
    }


def percentiles(values: np.ndarray) -> dict:
    out = {f"p{p:g}": round(float(np.percentile(values, p)), 1) for p in PERCENTILES}
    out.update(mean=round(float(values.mean()), 1), max=round(float(values.max()), 1), n=int(values.size))
    return out


def latency_block(pdf: pd.DataFrame, batches: pd.DataFrame) -> dict:
    df = pdf.merge(batches[["batch_id", "write_done_ts", "batch_start_ts"]], on="batch_id", how="left")
    df["e2e_ms"] = (df["write_done_ts"] - df["producer_ts"]) * 1000.0
    df = df.dropna(subset=["e2e_ms"])
    e2e = df["e2e_ms"].to_numpy()
    p99 = np.percentile(e2e, 99)

    order = sorted(df["batch_id"].unique())
    rank = {b: i for i, b in enumerate(order)}
    df["batch_rank"] = df["batch_id"].map(rank)
    tail = df[df["e2e_ms"] > p99]

    per_batch = df.groupby("batch_rank").agg(events=("e2e_ms", "size"), max_ms=("e2e_ms", "max"))
    batch_proc_ms = (batches["write_done_ts"] - batches["batch_start_ts"]) * 1000.0
    return {
        "definition": "write_done_ts(batch) - producer_ts(record): Kafka publish -> prediction written to HDFS",
        "end_to_end_ms": percentiles(e2e),
        "queueing_ms_publish_to_batch_start": percentiles(df["latency_ms"].dropna().to_numpy()),
        "batch_processing_ms_start_to_write_done": percentiles(batch_proc_ms.to_numpy()),
        "tail_above_p99": {
            "threshold_ms": round(float(p99), 1),
            "events": int(len(tail)),
            "share_in_first_3_batches": round(float((tail["batch_rank"] < 3).mean()), 4) if len(tail) else None,
            "batches_containing_tail": int(tail["batch_rank"].nunique()),
            "first_batch_rank": int(tail["batch_rank"].min()) if len(tail) else None,
        },
        "first_3_batches": per_batch.head(3).round(1).reset_index().to_dict(orient="records"),
        "batches": int(len(order)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a streaming replay against realised outcomes.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--batch-log", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("FlightDelay_StreamingEval").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    preds = spark.read.parquet(args.predictions)
    total = preds.count()
    labelled = (
        preds.filter(F.col("ARR_DELAY").isNotNull())
        .withColumn(LABEL_COL, (F.col("ARR_DELAY") > 15.0).cast("double"))
        .withColumn(PROB_COL, vector_to_array(F.col("probability"))[1])
    ).persist()
    n_labelled = labelled.count()

    # Parity: re-score the identical rows in batch mode with the same artifact.
    model = PipelineModel.load(args.model_path)
    rescored = (
        model.transform(labelled.select("batch_id", "kafka_partition", "kafka_offset", *RAW_INPUT_COLS))
        .withColumn("p_batch", vector_to_array(F.col("probability"))[1])
        .select("kafka_partition", "kafka_offset", "p_batch", F.col("prediction").alias("pred_batch"))
    )
    joined = labelled.join(rescored, on=["kafka_partition", "kafka_offset"], how="inner")
    parity = joined.select(
        F.count("*").alias("compared"),
        F.max(F.abs(F.col(PROB_COL) - F.col("p_batch"))).alias("max_abs_prob_diff"),
        F.sum((F.col("prediction") != F.col("pred_batch")).cast("int")).alias("prediction_mismatches"),
    ).collect()[0].asDict()

    batches = pd.read_json(args.batch_log, lines=True)
    pdf = labelled.select("batch_id", "producer_ts", "latency_ms").toPandas()

    result = {
        "records_scored": total,
        "records_with_outcome": n_labelled,
        "delayed_rate_actual": round(labelled.agg(F.avg(LABEL_COL)).collect()[0][0], 4),
        "accuracy": accuracy_block(labelled, args.threshold),
        "batch_streaming_parity": parity,
        "latency": latency_block(pdf, batches),
        "bloom_final": batches["bloom"].iloc[-1] if "bloom" in batches else None,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    spark.stop()


if __name__ == "__main__":
    main()

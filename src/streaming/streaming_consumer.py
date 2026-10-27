"""
streaming_consumer.py
---------------------
CANONICAL streaming consumer — this is the only Spark Structured Streaming
consumer invoked by scripts/run_pipeline.sh (see the start_streaming_consumer
step) and the only one that should be used or modified. Default trigger
interval is 10 seconds (see --trigger-interval below), matching the value
recorded in models/metrics.json and README.md. An older, unused duplicate
previously lived at src/spark_streaming_consumer.py (5s trigger default);
it has been moved to deprecated/spark_streaming_consumer.py and is not wired
into any entry point.

PySpark Structured Streaming consumer that reads flight events from Kafka,
applies the trained GBT Pipeline, and writes predictions to HDFS while
printing throughput and latency statistics.

Usage:
    spark-submit \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
        src/streaming/streaming_consumer.py \
        --kafka-bootstrap kafka:9092 \
        --topic flight-events \
        --model-path hdfs://hdfs-namenode:9000/models/gbt_pipeline \
        --output-path hdfs://hdfs-namenode:9000/output/streaming_predictions \
        --checkpoint-path hdfs://hdfs-namenode:9000/checkpoints/streaming \
        --batch-log /tmp/streaming_batches.jsonl

The model artifact is expected to contain its own feature derivation as its
first stage (see FEATURE_SQL in src/training/train_chronological.py), so this
consumer passes raw event columns straight to model.transform().

Latency: latency_ms in the prediction Parquet is producer publish -> batch
start (queueing only). --batch-log records, per batch, the wall-clock time the
HDFS write finished; end-to-end latency per record is
write_done_ts - producer_ts, joined on batch_id.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict

from pybloom_live import ScalableBloomFilter

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("streaming_consumer")

# ─── Schema for incoming Kafka JSON messages ──────────────────────────────────

# Every numeric field is DoubleType: kafka_producer.py sends all numerics as
# JSON floats (e.g. 1430.0), and from_json yields NULL for a float value in an
# IntegerType field.
FLIGHT_EVENT_SCHEMA = StructType(
    [
        StructField("YEAR", DoubleType(), True),
        StructField("MONTH", DoubleType(), True),
        StructField("DAY_OF_MONTH", DoubleType(), True),
        StructField("DAY_OF_WEEK", DoubleType(), True),
        StructField("OP_UNIQUE_CARRIER", StringType(), True),
        StructField("ORIGIN", StringType(), True),
        StructField("DEST", StringType(), True),
        StructField("CRS_DEP_TIME", DoubleType(), True),
        StructField("DEP_DELAY", DoubleType(), True),
        StructField("TAXI_OUT", DoubleType(), True),
        StructField("CRS_ARR_TIME", DoubleType(), True),
        StructField("ARR_DELAY", DoubleType(), True),
        StructField("CRS_ELAPSED_TIME", DoubleType(), True),
        StructField("DISTANCE", DoubleType(), True),
        StructField("CARRIER_DELAY", DoubleType(), True),
        StructField("WEATHER_DELAY", DoubleType(), True),
        StructField("NAS_DELAY", DoubleType(), True),
        StructField("SECURITY_DELAY", DoubleType(), True),
        StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True),
        StructField("producer_ts", DoubleType(), True),  # epoch seconds from producer
    ]
)


# ─── Bloom filter deduplication (driver-side) ────────────────────────────────

# Flight identity: date + carrier + origin + destination + scheduled departure.
FLIGHT_KEY_COLS = ["YEAR", "MONTH", "DAY_OF_MONTH", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME"]
BLOOM_ERROR_RATE = 0.01

_bloom = ScalableBloomFilter(
    initial_capacity=100_000,
    error_rate=BLOOM_ERROR_RATE,
    mode=ScalableBloomFilter.SMALL_SET_GROWTH,
)
_bloom_stats = {"seen": 0, "dropped_within_batch": 0, "dropped_seen_before": 0}


def bloom_deduplicate(batch_df: DataFrame):
    """
    Drop events whose flight key was already scored, within this micro-batch
    (exact) or in an earlier one (Bloom filter, ~1% false-positive rate).

    Every key dropped as "seen before" is either a genuine replay or a Bloom
    false positive; the counters are logged per batch so the realised rate
    can be measured against a replay known to contain no duplicates.

    Returns (deduplicated frame, number of rows kept). Uses a single collect
    of the keys, so it adds one Spark job per batch.
    """
    keyed = batch_df.withColumn("_flight_key", F.concat_ws("|", *[F.col(c) for c in FLIGHT_KEY_COLS]))
    all_keys = [row["_flight_key"] for row in keyed.select("_flight_key").collect()]
    keys = list(dict.fromkeys(all_keys))

    seen_before = [k for k in keys if k in _bloom]
    for k in keys:
        _bloom.add(k)

    _bloom_stats["seen"] += len(all_keys)
    _bloom_stats["dropped_within_batch"] += len(all_keys) - len(keys)
    _bloom_stats["dropped_seen_before"] += len(seen_before)
    if len(keys) < len(all_keys):
        keyed = keyed.dropDuplicates(["_flight_key"])
    if seen_before:
        keyed = keyed.filter(~F.col("_flight_key").isin(seen_before))
    return keyed.drop("_flight_key"), len(keys) - len(seen_before)


# ─── Metrics accumulator (driver-side) ───────────────────────────────────────

_metrics: Dict = {
    "total_events": 0,
    "total_batches": 0,
    "total_processing_time_ms": 0,
    "batch_start_wall": None,
    "stream_start_wall": None,
}


# ─── Spark session ────────────────────────────────────────────────────────────


def build_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder.appName("FlightDelay_StreamingConsumer")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.streaming.schemaInference", "true")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        # Reduce per-batch overhead for low-latency micro-batches
        .config("spark.sql.streaming.minBatchesToRetain", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ─── Model loading ────────────────────────────────────────────────────────────


def load_pipeline(model_path: str) -> PipelineModel:
    """Load the serialised GBT PipelineModel from HDFS."""
    logger.info("Loading pipeline model from %s …", model_path)
    model = PipelineModel.load(model_path)
    logger.info("Pipeline model loaded successfully.")
    return model


def warm_up(spark: SparkSession, model: PipelineModel, output_path: str, rounds: int = 3) -> None:
    """
    Score and write one synthetic event a few times before the stream starts,
    so JVM/codegen warm-up and the first HDFS write are not paid by the first
    real micro-batch. The warm-up output goes to a separate path.
    """
    event = {
        "YEAR": 2024.0, "MONTH": 1.0, "DAY_OF_MONTH": 1.0, "DAY_OF_WEEK": 1.0,
        "OP_UNIQUE_CARRIER": "AA", "ORIGIN": "JFK", "DEST": "LAX",
        "CRS_DEP_TIME": 900.0, "DEP_DELAY": 0.0, "TAXI_OUT": 15.0, "CRS_ARR_TIME": 1200.0,
        "ARR_DELAY": 0.0, "CRS_ELAPSED_TIME": 360.0, "DISTANCE": 2475.0,
        "producer_ts": time.time(),
    }
    row = {f.name: event.get(f.name) for f in FLIGHT_EVENT_SCHEMA.fields}
    df = spark.createDataFrame([row], schema=FLIGHT_EVENT_SCHEMA)
    for i in range(rounds):
        start = time.time()
        model.transform(df).select("prediction").write.mode("overwrite").parquet(f"{output_path}_warmup")
        logger.info("Warm-up round %d: %.0f ms", i + 1, (time.time() - start) * 1000.0)


# ─── Stream reading ───────────────────────────────────────────────────────────


def read_kafka_stream(spark: SparkSession, bootstrap: str, topic: str) -> DataFrame:
    """Create a streaming DataFrame from a Kafka topic."""
    logger.info("Subscribing to Kafka topic '%s' at %s …", topic, bootstrap)
    raw_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", 10_000)   # back-pressure: max 10k msgs/batch
        .option("kafka.session.timeout.ms", "30000")
        .option("kafka.request.timeout.ms", "40000")
        .load()
    )
    return raw_df


def parse_kafka_messages(raw_df: DataFrame) -> DataFrame:
    """Extract and parse the JSON payload from Kafka messages."""
    parsed_df = (
        raw_df.select(
            F.col("timestamp").alias("kafka_timestamp"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.from_json(F.col("value").cast("string"), FLIGHT_EVENT_SCHEMA).alias("data"),
        )
        .select(
            "kafka_timestamp",
            "kafka_partition",
            "kafka_offset",
            "data.*",
        )
    )
    return parsed_df


# ─── foreachBatch handler ─────────────────────────────────────────────────────


def make_batch_handler(model: PipelineModel, output_path: str, batch_log_path: str = None):
    """
    Return a foreachBatch function that:
      1. Applies the ML pipeline to the micro-batch.
      2. Computes latency from producer_ts to now.
      3. Appends predictions to HDFS Parquet.
      4. Logs throughput and latency statistics.
    """

    def process_batch(batch_df: DataFrame, batch_id: int) -> None:
        batch_start = time.time()

        # ── 1. Deduplicate, then apply the ML pipeline ────────────────
        raw_batch_df = batch_df.persist()
        deduped_df, batch_count = bloom_deduplicate(raw_batch_df)
        if batch_count == 0:
            logger.info("Batch %d has no new events – skipping.", batch_id)
            raw_batch_df.unpersist()
            return
        predictions_df = model.transform(deduped_df)

        # Compute consumer-side timestamp and end-to-end latency
        consumer_ts = time.time()
        predictions_df = predictions_df.withColumn(
            "consumer_ts", F.lit(consumer_ts).cast(DoubleType())
        ).withColumn(
            "latency_ms",
            F.when(
                F.col("producer_ts").isNotNull(),
                (F.lit(consumer_ts) - F.col("producer_ts")) * 1000.0,
            ).otherwise(F.lit(None).cast(DoubleType())),
        ).withColumn(
            "batch_id", F.lit(batch_id).cast(LongType())
        ).withColumn(
            "processing_time", F.lit(batch_start).cast(DoubleType())
        )

        # Select output columns
        output_df = predictions_df.select(
            "batch_id",
            "kafka_timestamp",
            "kafka_partition",
            "kafka_offset",
            "YEAR",
            "MONTH",
            "DAY_OF_MONTH",
            "DAY_OF_WEEK",
            "OP_UNIQUE_CARRIER",
            "ORIGIN",
            "DEST",
            "CRS_DEP_TIME",
            "CRS_ARR_TIME",
            "CRS_ELAPSED_TIME",
            "DEP_DELAY",
            "TAXI_OUT",
            "ARR_DELAY",
            "DISTANCE",
            "prediction",
            "probability",
            "rawPrediction",
            "producer_ts",
            "consumer_ts",
            "latency_ms",
        )

        # ── 2. Write to HDFS (the only job that runs the model) ──────
        (
            output_df.write.mode("append")
            .option("compression", "snappy")
            .parquet(output_path)
        )
        write_done_ts = time.time()

        # Per-record latency and prediction statistics are computed offline
        # from the written Parquet (src/evaluation/evaluate_streaming.py);
        # computing them here would add Spark jobs to every micro-batch.
        logger.info(
            "Batch %d | events: %d | start -> HDFS write done: %.1f ms",
            batch_id, batch_count, (write_done_ts - batch_start) * 1000.0,
        )

        if batch_log_path:
            with open(batch_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "batch_id": batch_id,
                    "events": batch_count,
                    "batch_start_ts": batch_start,
                    "write_done_ts": write_done_ts,
                    "bloom": dict(_bloom_stats),
                }) + "\n")
        raw_batch_df.unpersist()

    return process_batch


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spark Structured Streaming consumer for flight delay prediction."
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default="kafka:9092",
        help="Kafka bootstrap server (default: kafka:9092).",
    )
    parser.add_argument(
        "--topic",
        default="flight-events",
        help="Kafka topic to subscribe to (default: flight-events).",
    )
    parser.add_argument(
        "--model-path",
        default="hdfs://hdfs-namenode:9000/models/gbt_pipeline",
        help="HDFS path to the saved GBT PipelineModel.",
    )
    parser.add_argument(
        "--output-path",
        default="hdfs://hdfs-namenode:9000/output/streaming_predictions",
        help="HDFS output path for prediction Parquet files.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="hdfs://hdfs-namenode:9000/checkpoints/streaming",
        help="HDFS checkpoint directory for Structured Streaming.",
    )
    parser.add_argument(
        "--trigger-interval",
        default="10 seconds",
        help="Spark Structured Streaming trigger interval (default: '10 seconds').",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=3,
        help="Synthetic score+write rounds before the stream starts (0 disables).",
    )
    parser.add_argument(
        "--batch-log",
        default=None,
        help="Local JSONL path for per-batch write-completion times (end-to-end latency).",
    )
    parser.add_argument(
        "--await-termination",
        type=int,
        default=0,
        help="Seconds to await termination (0 = run indefinitely).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting Spark Structured Streaming consumer.")
    logger.info("  Kafka bootstrap  : %s", args.kafka_bootstrap)
    logger.info("  Topic            : %s", args.topic)
    logger.info("  Model path       : %s", args.model_path)
    logger.info("  Output path      : %s", args.output_path)
    logger.info("  Checkpoint path  : %s", args.checkpoint_path)
    logger.info("  Trigger interval : %s", args.trigger_interval)

    spark = build_spark_session()

    try:
        # Load the trained model once on the driver
        model = load_pipeline(args.model_path)
        if args.warmup_rounds > 0:
            warm_up(spark, model, args.output_path, args.warmup_rounds)

        # Build the streaming pipeline
        raw_stream = read_kafka_stream(spark, args.kafka_bootstrap, args.topic)
        parsed_stream = parse_kafka_messages(raw_stream)

        batch_handler = make_batch_handler(model, args.output_path, args.batch_log)

        query = (
            parsed_stream.writeStream.foreachBatch(batch_handler)
            .option("checkpointLocation", args.checkpoint_path)
            .trigger(processingTime=args.trigger_interval)
            .start()
        )

        logger.info("Streaming query started. Query ID: %s", query.id)

        if args.await_termination > 0:
            query.awaitTermination(args.await_termination)
        else:
            query.awaitTermination()

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping query …")
    except Exception as exc:
        logger.exception("Streaming consumer failed: %s", exc)
        sys.exit(1)
    finally:
        spark.stop()
        logger.info("SparkSession stopped.")


if __name__ == "__main__":
    main()

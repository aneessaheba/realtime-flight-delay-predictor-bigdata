"""
streaming_consumer.py
---------------------
PySpark Structured Streaming consumer that reads flight events from Kafka,
applies the trained GBT Pipeline, and writes predictions to HDFS while
printing throughput and latency statistics.

Usage:
    spark-submit \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,io.delta:delta-spark_2.12:3.2.0 \
        src/streaming/streaming_consumer.py \
        --kafka-bootstrap kafka:9092 \
        --topic flight-events \
        --model-path hdfs://namenode:9000/models/gbt_pipeline \
        --output-path hdfs://namenode:9000/output/streaming_predictions \
        --checkpoint-path hdfs://namenode:9000/checkpoints/streaming
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import Dict

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
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

FLIGHT_EVENT_SCHEMA = StructType(
    [
        StructField("YEAR", IntegerType(), True),
        StructField("MONTH", IntegerType(), True),
        StructField("DAY_OF_MONTH", IntegerType(), True),
        StructField("DAY_OF_WEEK", DoubleType(), True),
        StructField("OP_UNIQUE_CARRIER", StringType(), True),
        StructField("ORIGIN", StringType(), True),
        StructField("DEST", StringType(), True),
        StructField("CRS_DEP_TIME", DoubleType(), True),
        StructField("DEP_DELAY", DoubleType(), True),
        StructField("CRS_ARR_TIME", IntegerType(), True),
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
        .config("spark.sql.streaming.minBatchesToRetain", "2")
        # Delta Lake
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
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


def make_batch_handler(model: PipelineModel, output_path: str):
    """
    Return a foreachBatch function that:
      1. Applies the ML pipeline to the micro-batch.
      2. Computes latency from producer_ts to now.
      3. Appends predictions to HDFS Parquet.
      4. Logs throughput and latency statistics.
    """

    def process_batch(batch_df: DataFrame, batch_id: int) -> None:
        batch_start = time.time()

        if batch_df.rdd.isEmpty():
            logger.info("Batch %d is empty – skipping.", batch_id)
            return

        # ── 1. Apply ML pipeline ──────────────────────────────────────
        predictions_df = model.transform(batch_df)

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
            "DEP_DELAY",
            "ARR_DELAY",
            "DISTANCE",
            "prediction",
            "probability",
            "rawPrediction",
            "producer_ts",
            "consumer_ts",
            "latency_ms",
        )

        # ── 2. Write to Delta Lake ───────────────────────────────────
        (
            output_df.write.format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .partitionBy("YEAR", "MONTH")
            .save(output_path)
        )

        # ── 3. Console preview ───────────────────────────────────────
        logger.info("Batch %d – sample predictions:", batch_id)
        output_df.select(
            "ORIGIN", "DEST", "OP_UNIQUE_CARRIER",
            "DEP_DELAY", "ARR_DELAY", "prediction",
            "latency_ms",
        ).show(5, truncate=False)

        # ── 4. Metrics ───────────────────────────────────────────────
        batch_count = output_df.count()
        batch_elapsed_ms = (time.time() - batch_start) * 1000.0

        latency_stats = output_df.select(
            F.mean("latency_ms").alias("mean_latency_ms"),
            F.percentile_approx("latency_ms", 0.50).alias("p50_latency_ms"),
            F.percentile_approx("latency_ms", 0.95).alias("p95_latency_ms"),
            F.percentile_approx("latency_ms", 0.99).alias("p99_latency_ms"),
        ).collect()

        if latency_stats:
            stats = latency_stats[0]
            logger.info(
                "Batch %d | events: %d | processing: %.1f ms | "
                "latency mean/p50/p95/p99: %.1f/%.1f/%.1f/%.1f ms",
                batch_id,
                batch_count,
                batch_elapsed_ms,
                stats["mean_latency_ms"] or 0,
                stats["p50_latency_ms"] or 0,
                stats["p95_latency_ms"] or 0,
                stats["p99_latency_ms"] or 0,
            )

        throughput = batch_count / (batch_elapsed_ms / 1000.0) if batch_elapsed_ms > 0 else 0
        logger.info(
            "Batch %d | throughput: %.1f events/sec | "
            "delayed predictions: %d / %d",
            batch_id,
            throughput,
            output_df.filter(F.col("prediction") == 1.0).count(),
            batch_count,
        )

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
        default="hdfs://namenode:9000/models/gbt_pipeline",
        help="HDFS path to the saved GBT PipelineModel.",
    )
    parser.add_argument(
        "--output-path",
        default="hdfs://namenode:9000/output/streaming_predictions",
        help="HDFS output path for prediction Parquet files.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="hdfs://namenode:9000/checkpoints/streaming",
        help="HDFS checkpoint directory for Structured Streaming.",
    )
    parser.add_argument(
        "--trigger-interval",
        default="10 seconds",
        help="Spark Structured Streaming trigger interval (default: '10 seconds').",
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

        # Build the streaming pipeline
        raw_stream = read_kafka_stream(spark, args.kafka_bootstrap, args.topic)
        parsed_stream = parse_kafka_messages(raw_stream)

        batch_handler = make_batch_handler(model, args.output_path)

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

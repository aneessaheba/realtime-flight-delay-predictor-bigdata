"""
spark_streaming_consumer.py  –  Kartheek Alluri (Weeks 7-8)
Spark Structured Streaming job that consumes flight events from Kafka,
runs the saved ML Pipeline on each micro-batch, and writes predictions
to HDFS via Delta Lake.

Streaming Algorithms added:
  - Bloom Filter (pybloom-live):
      A ScalableBloomFilter tracks flight IDs (FL_DATE + carrier + origin +
      dep_time) seen across micro-batches. Rows that are already in the
      filter are skipped before model inference, avoiding redundant
      predictions when the Kafka producer loops or events are replicated.
      The filter uses a target false-positive rate of 1% and scales
      automatically as new unique IDs are added.

  - Delta Lake output:
      Predictions are written in Delta format instead of plain Parquet.
      This is a new tool not used in any course homework. Delta provides
      ACID transactions, time-travel queries, and schema enforcement on
      top of HDFS — enabling the benchmark script to query any prior
      run with a simple VERSION AS OF clause.

Usage:
spark-submit \
  --master spark://spark-master:7077 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,\
io.delta:delta-core_2.12:2.4.0 \
  --conf spark.hadoop.fs.defaultFS=hdfs://hdfs-namenode:9000 \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  src/spark_streaming_consumer.py \
  --model   hdfs:///models/gbt_pipeline \
  --kafka   kafka:9092 \
  --topic   flight-events \
  --output  hdfs:///predictions/streaming \
  --trigger 5
"""

import argparse
import time

from pybloom_live import ScalableBloomFilter

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType
)
from pyspark.ml import PipelineModel

FLIGHT_SCHEMA = StructType([
    StructField("FL_DATE",              StringType(),  True),
    StructField("OP_UNIQUE_CARRIER",    StringType(),  True),
    StructField("ORIGIN",               StringType(),  True),
    StructField("DEST",                 StringType(),  True),
    StructField("CRS_DEP_TIME",         DoubleType(),  True),
    StructField("DEP_TIME",             DoubleType(),  True),
    StructField("DEP_DELAY",            DoubleType(),  True),
    StructField("CRS_ARR_TIME",         DoubleType(),  True),
    StructField("CRS_ELAPSED_TIME",     DoubleType(),  True),
    StructField("ACTUAL_ELAPSED_TIME",  DoubleType(),  True),
    StructField("AIR_TIME",             DoubleType(),  True),
    StructField("DISTANCE",             DoubleType(),  True),
    StructField("_event_ts",            DoubleType(),  True),
    StructField("_label",               IntegerType(), True),
])

bloom_filter = ScalableBloomFilter(
    initial_capacity=100_000,
    error_rate=0.01,
    mode=ScalableBloomFilter.SMALL_SET_GROWTH,
)
bloom_total_seen = 0
bloom_total_skipped = 0


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("FlightDelayStreamingConsumer")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .getOrCreate()
    )


def bloom_deduplicate(batch_df, spark):
    """
    Bloom Filter deduplication (streaming algorithm).

    Constructs a unique flight key from FL_DATE + carrier + origin +
    CRS_DEP_TIME for each row. Rows whose key is already in the filter
    are dropped as duplicates. New keys are added to the filter.

    The ScalableBloomFilter grows automatically so memory stays bounded
    relative to the number of unique flights seen, not total events.

    False positive rate: ~1% — a small fraction of genuinely new flights
    may be incorrectly treated as duplicates. This is acceptable for a
    streaming inference pipeline where occasional missed predictions are
    preferable to redundant scoring at scale.
    """
    global bloom_total_seen, bloom_total_skipped

    rows = batch_df.select(
        "FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "CRS_DEP_TIME"
    ).collect()

    new_keys = set()
    duplicate_keys = set()

    for row in rows:
        key = f"{row.FL_DATE}|{row.OP_UNIQUE_CARRIER}|{row.ORIGIN}|{row.CRS_DEP_TIME}"
        bloom_total_seen += 1
        if key in bloom_filter:
            duplicate_keys.add(key)
            bloom_total_skipped += 1
        else:
            bloom_filter.add(key)
            new_keys.add(key)

    if duplicate_keys:
        print(
            f"[bloom] Deduplicated {len(duplicate_keys)} rows this batch "
            f"| total seen={bloom_total_seen:,} skipped={bloom_total_skipped:,} "
            f"(fp_rate≈1%)",
            flush=True,
        )

    def make_key(fl_date, carrier, origin, dep_time):
        return f"{fl_date}|{carrier}|{origin}|{dep_time}"

    key_udf = F.udf(make_key, StringType())
    batch_with_key = batch_df.withColumn(
        "_flight_key",
        key_udf(
            F.col("FL_DATE"),
            F.col("OP_UNIQUE_CARRIER"),
            F.col("ORIGIN"),
            F.col("CRS_DEP_TIME").cast(StringType()),
        )
    )

    new_keys_broadcast = spark.sparkContext.broadcast(new_keys)
    is_new_udf = F.udf(lambda k: k in new_keys_broadcast.value, "boolean")

    deduped = (
        batch_with_key
        .filter(is_new_udf(F.col("_flight_key")))
        .drop("_flight_key")
    )
    return deduped


def write_batch(batch_df, batch_id, model, output_path, spark):
    global bloom_total_seen, bloom_total_skipped

    count = batch_df.count()
    if count == 0:
        print(f"[consumer] Batch {batch_id}: empty, skipping.")
        return

    deduped_df = bloom_deduplicate(batch_df, spark)
    deduped_count = deduped_df.count()

    if deduped_count == 0:
        print(f"[consumer] Batch {batch_id}: all {count} rows were duplicates, skipping inference.")
        return

    t0 = time.time()
    predictions = model.transform(deduped_df)
    predictions = predictions.withColumn(
        "end_to_end_latency_s",
        (F.unix_timestamp(F.current_timestamp()) - F.col("_event_ts"))
    )

    labelled = predictions.filter(F.col("_label").isNotNull())
    total_labelled = labelled.count()
    if total_labelled > 0:
        correct = labelled.filter(
            F.col("prediction") == F.col("_label").cast("double")
        ).count()
        accuracy = correct / total_labelled
        avg_latency = labelled.agg(F.avg("end_to_end_latency_s")).collect()[0][0] or 0.0
        throughput = deduped_count / max(time.time() - t0, 0.001)
        print(
            f"[consumer] Batch {batch_id:4d} | rows={deduped_count:6,} "
            f"(deduped from {count:,}) | "
            f"accuracy={accuracy:.4f} | avg_latency={avg_latency:.3f}s | "
            f"throughput={throughput:,.0f} rows/s",
            flush=True,
        )

    out_cols = [
        "kafka_ts", "kafka_partition", "kafka_offset",
        "FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
        "CRS_DEP_TIME", "DEP_DELAY", "DISTANCE",
        "prediction", "probability", "rawPrediction",
        "_label", "end_to_end_latency_s",
    ]
    available = [c for c in out_cols if c in predictions.columns]

    (
        predictions.select(available)
        .write
        .format("delta")
        .mode("append")
        .save(f"{output_path}")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="hdfs:///models/gbt_pipeline")
    parser.add_argument("--kafka",   default="kafka:9092")
    parser.add_argument("--topic",   default="flight-events")
    parser.add_argument("--output",  default="hdfs:///predictions/streaming")
    parser.add_argument("--trigger", type=int, default=5)
    parser.add_argument("--offset",  default="latest", choices=["latest", "earliest"])
    args = parser.parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[consumer] Loading ML Pipeline from {args.model} ...")
    model = PipelineModel.load(args.model)
    print("[consumer] Model loaded.")
    print("[consumer] Bloom Filter initialized (capacity=100k, fp_rate=1%).")
    print("[consumer] Delta Lake output enabled.")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", args.kafka)
        .option("subscribe", args.topic)
        .option("startingOffsets", args.offset)
        .option("maxOffsetsPerTrigger", 20000)
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed_stream = (
        raw_stream
        .select(
            F.col("timestamp").alias("kafka_ts"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.from_json(F.col("value").cast("string"), FLIGHT_SCHEMA).alias("data")
        )
        .select("kafka_ts", "kafka_partition", "kafka_offset", "data.*")
    )

    query = (
        parsed_stream.writeStream
        .foreachBatch(lambda df, bid: write_batch(df, bid, model, args.output, spark))
        .trigger(processingTime=f"{args.trigger} seconds")
        .option("checkpointLocation", f"{args.output}/_checkpoints")
        .start()
    )

    print(f"[consumer] Streaming query started. Trigger every {args.trigger}s.")
    query.awaitTermination()


if __name__ == "__main__":
    main()
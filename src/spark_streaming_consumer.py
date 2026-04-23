"""
spark_streaming_consumer.py  –  Kartheek Alluri (Weeks 7-8)
Spark Structured Streaming job that consumes flight events from Kafka,
runs the saved ML Pipeline on each micro-batch, and writes predictions to HDFS.

Usage:
spark-submit \
  --master spark://spark-master:7077 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  --conf spark.hadoop.fs.defaultFS=hdfs://hdfs-namenode:9000 \
  src/spark_streaming_consumer.py \
  --model   hdfs:///models/gbt_pipeline \
  --kafka   kafka:9092 \
  --topic   flight-events \
  --output  hdfs:///predictions/streaming \
  --trigger 5
"""

import argparse
import time

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


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("FlightDelayStreamingConsumer")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )


def write_batch(batch_df, batch_id, model, output_path):
    count = batch_df.count()
    if count == 0:
        print(f"[consumer] Batch {batch_id}: empty, skipping.")
        return

    t0 = time.time()
    predictions = model.transform(batch_df)
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
        throughput = count / max(time.time() - t0, 0.001)
        print(
            f"[consumer] Batch {batch_id:4d} | rows={count:6,} | "
            f"accuracy={accuracy:.4f} | avg_latency={avg_latency:.3f}s | "
            f"throughput={throughput:,.0f} rows/s", flush=True
        )

    out_cols = [
        "kafka_ts", "kafka_partition", "kafka_offset",
        "FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
        "CRS_DEP_TIME", "DEP_DELAY", "DISTANCE",
        "prediction", "probability",
        "_label", "end_to_end_latency_s",
    ]
    available = [c for c in out_cols if c in predictions.columns]
    (
        predictions.select(available)
        .write.mode("append")
        .parquet(f"{output_path}/batch_{batch_id:06d}")
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
        .foreachBatch(lambda df, bid: write_batch(df, bid, model, args.output))
        .trigger(processingTime=f"{args.trigger} seconds")
        .option("checkpointLocation", f"{args.output}/_checkpoints")
        .start()
    )

    print(f"[consumer] Streaming query started. Trigger every {args.trigger}s.")
    query.awaitTermination()


if __name__ == "__main__":
    main()

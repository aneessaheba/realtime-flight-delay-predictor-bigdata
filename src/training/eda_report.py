import argparse
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

DEFAULT_INPUT = "hdfs:///data/bts/processed/cleaned"
DEFAULT_OUTPUT = "hdfs:///reports/eda"

def get_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )

def write_single_csv(df: DataFrame, path: str):
    (
        df.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(path)
    )

def main():
    parser = argparse.ArgumentParser(description="Generate EDA summaries for BTS data")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input cleaned parquet path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="EDA output directory")
    args = parser.parse_args()

    spark = get_spark("BTS_EDA_Report")
    df = spark.read.parquet(args.input)

    total_rows = df.count()
    delayed_rows = df.filter(F.col("label") == 1).count() if "label" in df.columns else 0
    delayed_pct = (delayed_rows / total_rows * 100.0) if total_rows else 0.0

    summary_rows = [
        ("total_rows", str(total_rows)),
        ("delayed_rows", str(delayed_rows)),
        ("delayed_pct", f"{delayed_pct:.2f}"),
    ]
    summary_df = spark.createDataFrame(summary_rows, ["metric", "value"])
    write_single_csv(summary_df, f"{args.output}/summary")

    missing_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    missing_df = df.select(*missing_exprs)
    missing_long = missing_df.select(
        F.explode(
            F.array(*[
                F.struct(F.lit(c).alias("column"), F.col(c).cast("long").alias("missing_count"))
                for c in missing_df.columns
            ])
        ).alias("x")
    ).select("x.column", "x.missing_count")

    write_single_csv(missing_long.orderBy(F.desc("missing_count")), f"{args.output}/missing_values")

    if "label" in df.columns:
        label_dist = (
            df.groupBy("label")
            .count()
            .orderBy("label")
        )
        write_single_csv(label_dist, f"{args.output}/label_distribution")

    if "OP_UNIQUE_CARRIER" in df.columns and "label" in df.columns:
        carrier_delay = (
            df.groupBy("OP_UNIQUE_CARRIER")
            .agg(
                F.count("*").alias("total_flights"),
                F.sum("label").alias("delayed_flights"),
                (F.avg("label") * 100.0).alias("delay_rate_pct")
            )
            .orderBy(F.desc("delay_rate_pct"))
        )
        write_single_csv(carrier_delay, f"{args.output}/carrier_delay_rates")

    if "ORIGIN" in df.columns and "label" in df.columns:
        origin_delay = (
            df.groupBy("ORIGIN")
            .agg(
                F.count("*").alias("total_flights"),
                F.sum("label").alias("delayed_flights"),
                (F.avg("label") * 100.0).alias("delay_rate_pct")
            )
            .filter(F.col("total_flights") >= 100)
            .orderBy(F.desc("delay_rate_pct"))
        )
        write_single_csv(origin_delay, f"{args.output}/origin_delay_rates")

    # Destination airport delays
    if "DEST" in df.columns and "label" in df.columns:
        dest_delay = (
            df.groupBy("DEST")
            .agg(
                F.count("*").alias("total_flights"),
                F.sum("label").alias("delayed_flights"),
                (F.avg("label") * 100.0).alias("delay_rate_pct")
            )
            .filter(F.col("total_flights") >= 100)
            .orderBy(F.desc("delay_rate_pct"))
        )
        write_single_csv(dest_delay, f"{args.output}/dest_delay_rates")

    if "MONTH" in df.columns and "label" in df.columns:
        month_trend = (
            df.groupBy("MONTH")
            .agg(
                F.count("*").alias("total_flights"),
                (F.avg("label") * 100.0).alias("delay_rate_pct")
            )
            .orderBy("MONTH")
        )
        write_single_csv(month_trend, f"{args.output}/month_trends")

    if "DAY_OF_WEEK" in df.columns and "label" in df.columns:
        dow_trend = (
            df.groupBy("DAY_OF_WEEK")
            .agg(
                F.count("*").alias("total_flights"),
                (F.avg("label") * 100.0).alias("delay_rate_pct")
            )
            .orderBy("DAY_OF_WEEK")
        )
        write_single_csv(dow_trend, f"{args.output}/day_of_week_trends")

    if "dep_hour" in df.columns and "label" in df.columns:
        hour_trend = (
            df.groupBy("dep_hour")
            .agg(
                F.count("*").alias("total_flights"),
                (F.avg("label") * 100.0).alias("delay_rate_pct")
            )
            .orderBy("dep_hour")
        )
        write_single_csv(hour_trend, f"{args.output}/departure_hour_trends")

    print(f"EDA outputs written to: {args.output}")
    spark.stop()

if __name__ == "__main__":
    main()
"""
ingest_bts_to_hdfs.py
---------------------
Read BTS Airline On-Time Performance CSV files from a local path,
clean and transform the data, and write Parquet files to HDFS
partitioned by YEAR and MONTH.

Usage:
    spark-submit src/ingestion/ingest_bts_to_hdfs.py \
        --input-path /data/raw/bts \
        --hdfs-path hdfs://namenode:9000/data/flights \
        --years 2018 2019 2020 2021 2022 2023
"""

import argparse
import logging
import os
import sys
import time
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingest_bts_to_hdfs")

# ─── Column definitions ───────────────────────────────────────────────────────

# All columns we care about from the raw BTS CSV
SELECTED_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_DELAY",
    "CRS_ARR_TIME",
    "ARR_DELAY",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]

# Target cast types for each column
COLUMN_TYPES = {
    "YEAR": IntegerType(),
    "MONTH": IntegerType(),
    "DAY_OF_MONTH": IntegerType(),
    "DAY_OF_WEEK": IntegerType(),
    "OP_UNIQUE_CARRIER": StringType(),
    "ORIGIN": StringType(),
    "DEST": StringType(),
    "CRS_DEP_TIME": IntegerType(),
    "DEP_DELAY": DoubleType(),
    "CRS_ARR_TIME": IntegerType(),
    "ARR_DELAY": DoubleType(),
    "CRS_ELAPSED_TIME": DoubleType(),
    "DISTANCE": DoubleType(),
    "CARRIER_DELAY": DoubleType(),
    "WEATHER_DELAY": DoubleType(),
    "NAS_DELAY": DoubleType(),
    "SECURITY_DELAY": DoubleType(),
    "LATE_AIRCRAFT_DELAY": DoubleType(),
}

# Delay breakdown columns that may be null for on-time flights – fill with 0
NULLABLE_DELAY_COLS = [
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "DEP_DELAY",
]

# Map from BTS raw CSV column names - internal SELECTED_COLUMNS names
BTS_COLUMN_MAP = {
    "Year":               "YEAR",
    "Month":              "MONTH",
    "DayofMonth":         "DAY_OF_MONTH",
    "DayOfWeek":          "DAY_OF_WEEK",
    "Reporting_Airline":  "OP_UNIQUE_CARRIER",
    "Origin":             "ORIGIN",
    "Dest":               "DEST",
    "CRSDepTime":         "CRS_DEP_TIME",
    "DepDelay":           "DEP_DELAY",
    "CRSArrTime":         "CRS_ARR_TIME",
    "ArrDelay":           "ARR_DELAY",
    "CRSElapsedTime":     "CRS_ELAPSED_TIME",
    "Distance":           "DISTANCE",
    "CarrierDelay":       "CARRIER_DELAY",
    "WeatherDelay":       "WEATHER_DELAY",
    "NASDelay":           "NAS_DELAY",
    "SecurityDelay":      "SECURITY_DELAY",
    "LateAircraftDelay":  "LATE_AIRCRAFT_DELAY",
}

ARR_DELAY_THRESHOLD = 15.0  # minutes


# ─── Helper functions ─────────────────────────────────────────────────────────


def build_spark_session(app_name: str = "BTS_Ingestion") -> SparkSession:
    """Create and return a SparkSession configured for HDFS access."""
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def resolve_csv_paths(input_path: str, years: List[int]) -> List[str]:
    """
    Return a list of glob patterns covering all CSV files for the requested
    years. Supports both flat directories (all CSVs in one folder) and
    year-partitioned sub-directories.
    """
    paths = []
    for year in years:
        year_dir = os.path.join(input_path, str(year))
        if os.path.isdir(year_dir):
            paths.append(os.path.join(year_dir, "*.csv"))
            logger.info("Adding year directory: %s", year_dir)
        else:
            # Fall back to flat layout – filter by year after reading
            paths.append(os.path.join(input_path, "*.csv"))
            logger.info(
                "Year directory not found for %d; will use flat path and filter.", year
            )
            break  # Only need to add the flat path once
    return list(set(paths))


def read_raw_csv(spark: SparkSession, csv_paths: List[str]):
    """Read raw BTS CSV files into a DataFrame."""
    logger.info("Reading CSV files from: %s", csv_paths)
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "false")
        .option("nullValue", "")
        .option("mode", "PERMISSIVE")
        .csv(csv_paths)
    )
    logger.info("Raw schema has %d columns.", len(df.columns))

    # Rename BTS raw column names to internal standard names
    for raw_name, std_name in BTS_COLUMN_MAP.items():
        if raw_name in df.columns:
            df = df.withColumnRenamed(raw_name, std_name)
    logger.info("Column rename complete.")

    return df


def clean_and_transform(df, years: List[int]):
    """
    Select relevant columns, cast types, fill nulls, create label column,
    and filter to requested years.
    """
    # Keep only the columns that exist in this dataset
    available = set(df.columns)
    missing = [c for c in SELECTED_COLUMNS if c not in available]
    if missing:
        logger.warning(
            "The following columns are missing from the source data and will be "
            "filled with NULL: %s",
            missing,
        )
        for col_name in missing:
            df = df.withColumn(col_name, F.lit(None).cast(COLUMN_TYPES[col_name]))

    df = df.select(SELECTED_COLUMNS)

    # Cast each column to its target type
    for col_name, col_type in COLUMN_TYPES.items():
        df = df.withColumn(col_name, F.col(col_name).cast(col_type))

    # Drop rows where ARR_DELAY is null (cancelled / diverted flights)
    before_count = df.count()
    df = df.filter(F.col("ARR_DELAY").isNotNull())
    after_count = df.count()
    logger.info(
        "Dropped %d rows with null ARR_DELAY (%.2f%%). Remaining: %d",
        before_count - after_count,
        100.0 * (before_count - after_count) / max(before_count, 1),
        after_count,
    )

    # Fill nullable delay breakdown columns with 0 (on-time flights have no delay)
    fill_map = {c: 0.0 for c in NULLABLE_DELAY_COLS}
    df = df.fillna(fill_map)

    # Drop rows with null in critical feature columns
    critical_cols = ["DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "DISTANCE"]
    df = df.dropna(subset=critical_cols)

    # Filter to requested years
    if years:
        df = df.filter(F.col("YEAR").isin(years))
        logger.info("Filtered to years %s. Row count: %d", years, df.count())

    # Binary label: 1 if arrival delay > 15 minutes, else 0
    df = df.withColumn(
        "label",
        F.when(F.col("ARR_DELAY") > ARR_DELAY_THRESHOLD, 1).otherwise(0).cast(IntegerType()),
    )

    # Repartition to balance files on HDFS: ~500k rows per partition
    total_rows = df.count()
    num_partitions = max(4, int(total_rows / 500_000) + 1)
    df = df.repartition(num_partitions, "YEAR", "MONTH")

    logger.info(
        "Final cleaned dataset: %d rows, %d partitions.",
        total_rows,
        num_partitions,
    )
    return df


def write_to_hdfs(df, hdfs_path: str) -> None:
    """Write the cleaned DataFrame to HDFS in Parquet format, partitioned by YEAR and MONTH."""
    logger.info("Writing Parquet to HDFS path: %s", hdfs_path)
    start = time.time()
    (
        df.write.mode("append")
        .partitionBy("YEAR", "MONTH")
        .option("compression", "snappy")
        .parquet(hdfs_path)
    )
    elapsed = time.time() - start
    logger.info("Write completed in %.2f seconds.", elapsed)


def print_data_summary(df) -> None:
    """Log basic statistics about the cleaned dataset."""
    total = df.count()
    delayed = df.filter(F.col("label") == 1).count()
    pct_delayed = 100.0 * delayed / max(total, 1)
    logger.info("=" * 60)
    logger.info("Data summary:")
    logger.info("  Total rows     : %d", total)
    logger.info("  Delayed (label=1): %d (%.1f%%)", delayed, pct_delayed)
    logger.info("  On-time (label=0): %d (%.1f%%)", total - delayed, 100.0 - pct_delayed)
    logger.info(
        "  Year range     : %s",
        df.select(F.min("YEAR"), F.max("YEAR")).collect()[0],
    )
    logger.info("=" * 60)


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest BTS On-Time Performance CSV data into HDFS as Parquet."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Local directory containing BTS CSV files (may include year sub-dirs).",
    )
    parser.add_argument(
        "--hdfs-path",
        default="hdfs://namenode:9000/data/flights",
        help="HDFS destination path for Parquet output.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2018, 2025)),
        help="Years to ingest (default: 2018-2024).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting BTS ingestion job.")
    logger.info("  Input path : %s", args.input_path)
    logger.info("  HDFS path  : %s", args.hdfs_path)
    logger.info("  Years      : %s", args.years)

    spark = build_spark_session()

    try:
        csv_paths = resolve_csv_paths(args.input_path, args.years)
        raw_df = read_raw_csv(spark, csv_paths)
        clean_df = clean_and_transform(raw_df, args.years)
        print_data_summary(clean_df)
        write_to_hdfs(clean_df, args.hdfs_path)
        logger.info("Ingestion completed successfully.")
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

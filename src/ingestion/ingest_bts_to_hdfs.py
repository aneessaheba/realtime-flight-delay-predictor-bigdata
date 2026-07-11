"""
ingest_bts_to_hdfs.py
---------------------
Read BTS Airline On-Time Performance CSV files from a local path,
clean and transform the data, and write Parquet files to HDFS
partitioned by YEAR and MONTH.

Usage:
    spark-submit src/ingestion/ingest_bts_to_hdfs.py \
        --input-path /data/raw/bts \
        --hdfs-path hdfs://hdfs-namenode:9000/data/flights \
        --years 2021 2022 2023
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

# ─── Column definitions ───

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

ARR_DELAY_THRESHOLD = 15.0  # minutes
LOG_DIR = Path("logs")

# BTS CSVs from different download periods use mixed-case column names.
# This map normalises them to the uppercase standard used throughout the project.
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
    """Read raw BTS CSV files into a DataFrame and normalise column names."""
    logger.info("Reading CSV files from: %s", csv_paths)
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "false")  # All columns as strings initially
        .option("nullValue", "")
        .option("mode", "PERMISSIVE")
        .csv(csv_paths)
    )
    for raw_name, std_name in BTS_COLUMN_MAP.items():
        if raw_name in df.columns:
            df = df.withColumnRenamed(raw_name, std_name)
    logger.info("Raw schema has %d columns.", len(df.columns))
    return df


def clean_and_transform(df, years: List[int]) -> Tuple[object, Dict]:
    """
    Select relevant columns, cast types, fill nulls, create label column,
    and filter to requested years.

    Returns (cleaned_df, stats) where `stats` records row counts at each
    cleaning stage plus the actual (YEAR, MONTH) combinations present in the
    output, so every ingestion run is traceable back to exactly what data
    produced it.
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

    # Repartition to balance files on HDFS
    total_rows = df.count()
    num_partitions = max(4, int(total_rows / 500_000) + 1)
    df = df.repartition(num_partitions, "YEAR", "MONTH")

    logger.info(
        "Final cleaned dataset: %d rows, %d partitions.",
        total_rows,
        num_partitions,
    )

    year_months = sorted(
        {
            (row["YEAR"], row["MONTH"])
            for row in df.select("YEAR", "MONTH").distinct().collect()
        }
    )
    logger.info("Years/months covered in cleaned output: %s", year_months)

    stats = {
        "rows_before_arr_delay_filter": before_count,
        "rows_dropped_null_arr_delay": before_count - after_count,
        "rows_after_arr_delay_filter": after_count,
        "cleaned_row_count": total_rows,
        "num_output_partitions": num_partitions,
        "years_months_covered": [f"{y}-{m:02d}" for y, m in year_months],
    }
    return df, stats


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


def resolve_actual_csv_files(input_path: str, years: List[int]) -> List[str]:
    """Expand the glob patterns from resolve_csv_paths() into concrete file paths, for logging."""
    if input_path.startswith("hdfs://"):
        # Can't glob a remote HDFS path from the driver's local filesystem;
        # record the pattern strings themselves rather than an empty list.
        return resolve_csv_paths(input_path, years)
    files: List[str] = []
    for pattern in resolve_csv_paths(input_path, years):
        local_pattern = pattern[len("file://"):] if pattern.startswith("file://") else pattern
        files.extend(sorted(glob.glob(local_pattern)))
    return sorted(set(files))


def write_ingestion_log(
    input_path: str,
    csv_files: List[str],
    years_requested: List[int],
    raw_row_count: int,
    clean_stats: Dict,
    hdfs_path: str,
    elapsed_seconds: float,
) -> Path:
    """
    Persist per-run ingestion stats (input files, raw row count, cleaned row
    count, years/months actually covered) to logs/ingest_run_<timestamp>.json
    so every future ingestion run is traceable back to concrete numbers
    instead of being estimated after the fact.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_path = LOG_DIR / f"ingest_run_{timestamp}.json"

    payload = {
        "run_timestamp_utc": timestamp,
        "input_path": input_path,
        "input_files": csv_files,
        "input_file_count": len(csv_files),
        "years_requested": years_requested,
        "raw_row_count": raw_row_count,
        "hdfs_output_path": hdfs_path,
        "elapsed_seconds": round(elapsed_seconds, 2),
        **clean_stats,
    }

    with log_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Ingestion run log written to %s", log_path)
    logger.info("Ingestion run summary: %s", json.dumps(payload, indent=2))
    return log_path


# ─── Entry point ───

# Makes script reusable because we can change input, output location, and years without editing code
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
        default="hdfs://hdfs-namenode:9000/data/flights",
        help="HDFS destination path for Parquet output.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2021, 2024)),
        help="Years to ingest (default: 2021-2023).",
    )
    return parser.parse_args()

# Calls each function in order and stops Spark at the end
def main() -> None:
    args = parse_args()
    logger.info("Starting BTS ingestion job.")
    logger.info("  Input path : %s", args.input_path)
    logger.info("  HDFS path  : %s", args.hdfs_path)
    logger.info("  Years      : %s", args.years)

    spark = build_spark_session()
    run_start = time.time()

    try:
        csv_paths = resolve_csv_paths(args.input_path, args.years)
        csv_files = resolve_actual_csv_files(args.input_path, args.years)
        logger.info("Resolved %d input CSV file(s): %s", len(csv_files), csv_files)

        raw_df = read_raw_csv(spark, csv_paths)
        raw_row_count = raw_df.count()
        logger.info("Raw row count (post-parse, pre-clean): %d", raw_row_count)

        clean_df, clean_stats = clean_and_transform(raw_df, args.years)
        print_data_summary(clean_df)
        write_to_hdfs(clean_df, args.hdfs_path)

        write_ingestion_log(
            input_path=args.input_path,
            csv_files=csv_files,
            years_requested=args.years,
            raw_row_count=raw_row_count,
            clean_stats=clean_stats,
            hdfs_path=args.hdfs_path,
            elapsed_seconds=time.time() - run_start,
        )

        logger.info("Ingestion completed successfully.")
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prepare_features")

DEFAULT_INPUT = "data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2021_12.csv"
DEFAULT_CLEANED_OUTPUT = "data/cleaned_local"
DEFAULT_FEATURED_OUTPUT = "data/featured_local"
DEFAULT_PIPELINE_MODEL_OUTPUT = "models/preprocessing_pipeline_local"
LOG_DIR = Path("logs")

POSSIBLE_NUMERIC_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "DISTANCE",
    "TAXI_OUT",
    "TAXI_IN",
    "AIR_TIME",
    "CRS_ELAPSED_TIME",
    "ACTUAL_ELAPSED_TIME",
    "DEP_DELAY",
    "ARR_DELAY",
    "DEP_DEL15",
    "ARR_DEL15",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "CARRIER_DELAY",
]

POSSIBLE_CATEGORICAL_COLUMNS = [
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "ORIGIN_STATE_ABR",
    "DEST_STATE_ABR",
]

# Columns known only after the flight lands — must not be used as features
LEAKAGE_COLUMNS = [
    "ARR_DELAY",
    "ARR_DEL15",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "TAXI_IN",
    "WHEELS_ON",
    "ARR_TIME",
    "ARR_TIME_BLK",
    "CANCELLED",
    "DIVERTED",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "CARRIER_DELAY",
]

OPTIONAL_DROP_COLUMNS = [
    "TAIL_NUM",
    "FLIGHTS",
    "WHEELS_OFF",
    "DEP_TIME",
    "CRS_ARR_TIME",
    "FL_DATE",
]

FULL_FEATURE_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "DISTANCE",
    "CRS_ELAPSED_TIME",
    "DEP_DELAY",
    "DEP_DEL15",
    "dep_hour",
    "arr_sched_hour",
    "is_weekend",
    "is_holiday_season",
]

PRE_DEPARTURE_FEATURE_COLUMNS = [
    "MONTH",
    "DAY_OF_WEEK",
    "DISTANCE",
    "CRS_ELAPSED_TIME",
    "dep_hour",
    "arr_sched_hour",
    "is_weekend",
    "is_holiday_season",
]

POST_DEPARTURE_EXCLUDED_FEATURES = {"DEP_DELAY", "DEP_DEL15", "TAXI_OUT"}

BTS_COLUMN_MAP = {
    "Year": "YEAR",
    "Month": "MONTH",
    "DayofMonth": "DAY_OF_MONTH",
    "DayOfWeek": "DAY_OF_WEEK",
    "Reporting_Airline": "OP_UNIQUE_CARRIER",
    "Origin": "ORIGIN",
    "Dest": "DEST",
    "CRSDepTime": "CRS_DEP_TIME",
    "DepDelay": "DEP_DELAY",
    "CRSArrTime": "CRS_ARR_TIME",
    "ArrDelay": "ARR_DELAY",
    "CRSElapsedTime": "CRS_ELAPSED_TIME",
    "Distance": "DISTANCE",
    "CarrierDelay": "CARRIER_DELAY",
    "WeatherDelay": "WEATHER_DELAY",
    "NASDelay": "NAS_DELAY",
    "SecurityDelay": "SECURITY_DELAY",
    "LateAircraftDelay": "LATE_AIRCRAFT_DELAY",
    "TaxiOut": "TAXI_OUT",
    "DepDel15": "DEP_DEL15",
    "ArrDel15": "ARR_DEL15",
}


def get_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "3g")
        .getOrCreate()
    )


def normalize_column_names(df: DataFrame) -> DataFrame:
    for raw_col, standard_col in BTS_COLUMN_MAP.items():
        if raw_col in df.columns:
            df = df.withColumnRenamed(raw_col, standard_col)
    return df


def existing_columns(df: DataFrame, cols: List[str]) -> List[str]:
    df_cols = set(df.columns)
    return [c for c in cols if c in df_cols]

def safe_cast_numeric(df: DataFrame, cols: List[str]) -> DataFrame:
    for c in cols:
        df = df.withColumn(c, F.col(c).cast("double"))
    return df

def parse_hhmm_to_hour(col_name: str):
    padded = F.lpad(F.col(col_name).cast("string"), 4, "0")
    return F.substring(padded, 1, 2).cast("int")

def add_time_features(df: DataFrame) -> DataFrame:
    if "CRS_DEP_TIME" in df.columns:
        df = df.withColumn("dep_hour", parse_hhmm_to_hour("CRS_DEP_TIME"))
    if "CRS_ARR_TIME" in df.columns:
        df = df.withColumn("arr_sched_hour", parse_hhmm_to_hour("CRS_ARR_TIME"))

    if "DAY_OF_WEEK" in df.columns:
        df = df.withColumn(
            "is_weekend",
            F.when(F.col("DAY_OF_WEEK").isin([6, 7]), F.lit(1)).otherwise(F.lit(0))
        )

    if "MONTH" in df.columns:
        df = df.withColumn(
            "is_holiday_season",
            F.when(F.col("MONTH").isin([11, 12]), F.lit(1)).otherwise(F.lit(0))
        )

    if "ORIGIN" in df.columns and "DEST" in df.columns:
        df = df.withColumn("route", F.concat_ws("_", F.col("ORIGIN"), F.col("DEST")))

    return df

def filter_bad_rows(df: DataFrame) -> DataFrame:
    if "CANCELLED" in df.columns:
        df = df.filter((F.col("CANCELLED").isNull()) | (F.col("CANCELLED") == 0))
    if "DIVERTED" in df.columns:
        df = df.filter((F.col("DIVERTED").isNull()) | (F.col("DIVERTED") == 0))

    if "ARR_DELAY" not in df.columns:
        raise ValueError("ARR_DELAY column is required to create the label.")

    df = df.filter(F.col("ARR_DELAY").isNotNull())
    return df

def create_label(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "label",
        F.when(F.col("ARR_DELAY") > 15, F.lit(1)).otherwise(F.lit(0))
    )

def fill_missing_values(df: DataFrame) -> DataFrame:
    numeric_cols = existing_columns(df, POSSIBLE_NUMERIC_COLUMNS + [
        "dep_hour", "arr_sched_hour", "is_weekend", "is_holiday_season"
    ])
    categorical_cols = existing_columns(df, POSSIBLE_CATEGORICAL_COLUMNS + ["route"])

    if numeric_cols:
        fill_numeric = {c: 0.0 for c in numeric_cols}
        df = df.fillna(fill_numeric)

    if categorical_cols:
        fill_cats = {c: "UNKNOWN" for c in categorical_cols}
        df = df.fillna(fill_cats)

    return df

def clean_dataframe(df: DataFrame) -> DataFrame:
    numeric_cols = existing_columns(df, POSSIBLE_NUMERIC_COLUMNS + ["CRS_DEP_TIME", "CRS_ARR_TIME"])
    df = safe_cast_numeric(df, numeric_cols)

    df = filter_bad_rows(df)
    df = add_time_features(df)
    df = create_label(df)
    df = fill_missing_values(df)

    for required_col in ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]:
        if required_col in df.columns:
            df = df.filter(F.col(required_col).isNotNull())

    return df

def get_feature_columns_for_mode(df: DataFrame, mode: str = "full") -> Tuple[List[str], List[str]]:
    categorical_cols = existing_columns(df, POSSIBLE_CATEGORICAL_COLUMNS + ["route"])

    if mode == "pre_departure":
        numeric_feature_cols = existing_columns(df, PRE_DEPARTURE_FEATURE_COLUMNS)
    else:
        numeric_feature_cols = existing_columns(df, FULL_FEATURE_COLUMNS)

    numeric_feature_cols = [
        c for c in numeric_feature_cols
        if c not in LEAKAGE_COLUMNS and c not in POST_DEPARTURE_EXCLUDED_FEATURES
    ]
    return categorical_cols, numeric_feature_cols


def build_preprocessing_pipeline(df: DataFrame, mode: str = "full") -> Tuple[Pipeline, List[str], List[str]]:
    categorical_cols, numeric_feature_cols = get_feature_columns_for_mode(df, mode=mode)

    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=f"{c}_idx",
            handleInvalid="keep"
        )
        for c in categorical_cols
    ]

    encoders = [
        OneHotEncoder(
            inputCol=f"{c}_idx",
            outputCol=f"{c}_ohe",
            handleInvalid="keep"
        )
        for c in categorical_cols
    ]

    assembled_inputs = numeric_feature_cols + [f"{c}_ohe" for c in categorical_cols]

    assembler = VectorAssembler(
        inputCols=assembled_inputs,
        outputCol="features",
        handleInvalid="keep"
    )

    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    return pipeline, categorical_cols, numeric_feature_cols

def drop_non_feature_columns(df: DataFrame) -> DataFrame:
    cols_to_drop = existing_columns(df, OPTIONAL_DROP_COLUMNS)
    for c in cols_to_drop:
        df = df.drop(c)
    return df

def resolve_input_files(input_path: str) -> List[str]:
    """Best-effort expansion of the --input argument into concrete file paths, for logging."""
    if input_path.startswith("hdfs://"):
        # Can't glob a remote HDFS path from the driver without an extra Hadoop
        # client call; record the path itself rather than guessing its contents.
        return [input_path]
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(str(f) for f in p.rglob("*") if f.is_file() and f.suffix in (".csv", ".parquet"))
        return files if files else [str(p)]
    matched = sorted(glob.glob(input_path))
    return matched if matched else [input_path]

def years_months_covered(df: DataFrame) -> List[str]:
    """Return the sorted 'YYYY-MM' combinations actually present in df, if YEAR/MONTH exist."""
    if "YEAR" not in df.columns or "MONTH" not in df.columns:
        return []
    rows = df.select("YEAR", "MONTH").distinct().collect()
    pairs = sorted({(int(r["YEAR"]), int(r["MONTH"])) for r in rows if r["YEAR"] is not None and r["MONTH"] is not None})
    return [f"{y}-{m:02d}" for y, m in pairs]

def write_prepare_features_log(
    input_path: str,
    input_files: List[str],
    raw_row_count: int,
    cleaned_row_count: int,
    featured_row_count: int,
    covered: List[str],
    sample_fraction: float,
    mode: str,
    cleaned_output: str,
    featured_output: str,
    elapsed_seconds: float,
) -> Path:
    """
    Persist per-run feature-prep stats to logs/prepare_features_run_<timestamp>.json
    so cleaned/featured row counts are always traceable to a concrete run.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_path = LOG_DIR / f"prepare_features_run_{timestamp}.json"

    payload = {
        "run_timestamp_utc": timestamp,
        "input_path": input_path,
        "input_files": input_files,
        "input_file_count": len(input_files),
        "raw_row_count": raw_row_count,
        "cleaned_row_count": cleaned_row_count,
        "featured_row_count": featured_row_count,
        "years_months_covered": covered,
        "sample_fraction": sample_fraction,
        "mode": mode,
        "cleaned_output": cleaned_output,
        "featured_output": featured_output,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    with log_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Feature-prep run log written to %s", log_path)
    logger.info("Feature-prep run summary: %s", json.dumps(payload, indent=2))
    return log_path

def main():
    parser = argparse.ArgumentParser(description="Prepare cleaned and feature-engineered BTS data")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input Parquet path")
    parser.add_argument("--cleaned-output", default=DEFAULT_CLEANED_OUTPUT, help="Cleaned output path")
    parser.add_argument("--featured-output", default=DEFAULT_FEATURED_OUTPUT, help="Featured output path")
    parser.add_argument("--pipeline-model-output", default=DEFAULT_PIPELINE_MODEL_OUTPUT, help="Saved preprocessing pipeline model path")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Optional sampling fraction for faster testing")
    parser.add_argument(
        "--mode",
        choices=["full", "pre_departure"],
        default="full",
        help="Feature set to build: full includes post-departure signals, pre_departure excludes them.",
    )
    args = parser.parse_args()

    spark = get_spark("PrepareBTSFeatures")
    run_start = time.time()

    input_files = resolve_input_files(args.input)
    logger.info("Reading input from: %s", args.input)
    logger.info("Resolved %d input file(s): %s", len(input_files), input_files)

    if args.input.endswith(".csv"):
        df = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .option("nullValue", "")
            .option("mode", "PERMISSIVE")
            .csv(args.input)
        )
        df = normalize_column_names(df)
    else:
        df = spark.read.parquet(args.input)

    if args.sample_fraction < 1.0:
        df = df.sample(withReplacement=False, fraction=args.sample_fraction, seed=42)

    raw_row_count = df.count()
    logger.info("Initial row count: %d", raw_row_count)
    df.printSchema()

    cleaned_df = clean_dataframe(df)
    cleaned_df = drop_non_feature_columns(cleaned_df)

    cleaned_row_count = cleaned_df.count()
    logger.info("Cleaned row count: %d", cleaned_row_count)
    covered = years_months_covered(cleaned_df)
    logger.info("Years/months covered in cleaned output: %s", covered)

    logger.info("Writing cleaned data to: %s", args.cleaned_output)
    (
        cleaned_df.write
        .mode("overwrite")
        .partitionBy(*[c for c in ["YEAR", "MONTH"] if c in cleaned_df.columns])
        .parquet(args.cleaned_output)
    )

    pipeline, cat_cols, num_cols = build_preprocessing_pipeline(cleaned_df, mode=args.mode)
    logger.info("Feature mode: %s", args.mode)
    logger.info("Categorical feature columns: %s", cat_cols)
    logger.info("Numeric feature columns: %s", num_cols)

    pipeline_model: PipelineModel = pipeline.fit(cleaned_df)
    featured_df = pipeline_model.transform(cleaned_df)

    keep_cols = [c for c in ["label", "features", "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK"] if c in featured_df.columns]
    featured_df = featured_df.select(*keep_cols)

    featured_row_count = featured_df.count()
    logger.info("Featured row count: %d", featured_row_count)
    featured_df.printSchema()

    logger.info("Writing featured data to: %s", args.featured_output)
    (
        featured_df.write
        .mode("overwrite")
        .partitionBy(*[c for c in ["YEAR", "MONTH"] if c in featured_df.columns])
        .parquet(args.featured_output)
    )

    logger.info("Saving pipeline model to: %s", args.pipeline_model_output)
    pipeline_model.write().overwrite().save(args.pipeline_model_output)

    write_prepare_features_log(
        input_path=args.input,
        input_files=input_files,
        raw_row_count=raw_row_count,
        cleaned_row_count=cleaned_row_count,
        featured_row_count=featured_row_count,
        covered=covered,
        sample_fraction=args.sample_fraction,
        mode=args.mode,
        cleaned_output=args.cleaned_output,
        featured_output=args.featured_output,
        elapsed_seconds=time.time() - run_start,
    )

    logger.info("Done.")
    spark.stop()

if __name__ == "__main__":
    main()

import argparse
import os
from typing import List, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

DEFAULT_INPUT = "hdfs://hdfs-namenode:9000/data/flights"
DEFAULT_CLEANED_OUTPUT = "hdfs://hdfs-namenode:9000/data/flights/cleaned"
DEFAULT_FEATURED_OUTPUT = "hdfs://hdfs-namenode:9000/data/flights/featured"
DEFAULT_PIPELINE_MODEL_OUTPUT = "hdfs://hdfs-namenode:9000/models/preprocessing_pipeline"

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

def get_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )


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

def build_preprocessing_pipeline(df: DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    categorical_cols = existing_columns(df, POSSIBLE_CATEGORICAL_COLUMNS + ["route"])
    numeric_feature_cols = existing_columns(df, [
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
    ])

    numeric_feature_cols = [c for c in numeric_feature_cols if c not in LEAKAGE_COLUMNS]

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

def main():
    parser = argparse.ArgumentParser(description="Prepare cleaned and feature-engineered BTS data")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input Parquet path")
    parser.add_argument("--cleaned-output", default=DEFAULT_CLEANED_OUTPUT, help="Cleaned output path")
    parser.add_argument("--featured-output", default=DEFAULT_FEATURED_OUTPUT, help="Featured output path")
    parser.add_argument("--pipeline-model-output", default=DEFAULT_PIPELINE_MODEL_OUTPUT, help="Saved preprocessing pipeline model path")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Optional sampling fraction for faster testing")
    args = parser.parse_args()

    spark = get_spark("PrepareBTSFeatures")

    print(f"Reading input from: {args.input}")
    df = spark.read.parquet(args.input)

    if args.sample_fraction < 1.0:
        df = df.sample(withReplacement=False, fraction=args.sample_fraction, seed=42)

    print("Initial row count:", df.count())
    print("Initial schema:")
    df.printSchema()

    cleaned_df = clean_dataframe(df)
    cleaned_df = drop_non_feature_columns(cleaned_df)

    print("Cleaned row count:", cleaned_df.count())

    print(f"Writing cleaned data to: {args.cleaned_output}")
    (
        cleaned_df.write
        .mode("overwrite")
        .partitionBy(*[c for c in ["YEAR", "MONTH"] if c in cleaned_df.columns])
        .parquet(args.cleaned_output)
    )

    pipeline, cat_cols, num_cols = build_preprocessing_pipeline(cleaned_df)
    print("Categorical feature columns:", cat_cols)
    print("Numeric feature columns:", num_cols)

    pipeline_model: PipelineModel = pipeline.fit(cleaned_df)
    featured_df = pipeline_model.transform(cleaned_df)

    keep_cols = [c for c in ["label", "features", "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK"] if c in featured_df.columns]
    featured_df = featured_df.select(*keep_cols)

    print(f"Writing featured data to: {args.featured_output}")
    (
        featured_df.write
        .mode("overwrite")
        .partitionBy(*[c for c in ["YEAR", "MONTH"] if c in featured_df.columns])
        .parquet(args.featured_output)
    )

    print(f"Saving pipeline model to: {args.pipeline_model_output}")
    pipeline_model.write().overwrite().save(args.pipeline_model_output)

    print("Done.")
    spark.stop()

if __name__ == "__main__":
    main()

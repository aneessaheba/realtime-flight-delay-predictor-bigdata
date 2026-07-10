"""
train_pre_departure_comparison.py
---------------------------------
Train and compare the existing full-feature flight delay model against a new
pre-departure-only model using the same train/val/test split, class weighting,
and cross-validation setup as the repository's existing Spark ML workflow.

Outputs are written to outputs/pre_departure_comparison/ and
models/pre_departure_comparison/ so the new results can be reported side-by-side.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pre_departure_comparison")

LABEL_COL = "label"
FEATURE_COL = "features"
WEIGHT_COL = "class_weight"
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SHAP_SAMPLE_SIZE = 2000
DELAY_THRESHOLD = 15.0

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


def build_spark_session(app_name: str = "FlightDelay_PreDepartureComparison") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "100")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "3g")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def parse_hhmm_to_hour(col_name: str):
    padded = F.lpad(F.col(col_name).cast("string"), 4, "0")
    return F.substring(padded, 1, 2).cast("int")


def add_time_features(df):
    if "CRS_DEP_TIME" in df.columns:
        df = df.withColumn("dep_hour", parse_hhmm_to_hour("CRS_DEP_TIME"))
    if "CRS_ARR_TIME" in df.columns:
        df = df.withColumn("arr_sched_hour", parse_hhmm_to_hour("CRS_ARR_TIME"))
    if "DAY_OF_WEEK" in df.columns:
        df = df.withColumn(
            "is_weekend",
            F.when(F.col("DAY_OF_WEEK").isin([6, 7]), F.lit(1)).otherwise(F.lit(0)),
        )
    if "MONTH" in df.columns:
        df = df.withColumn(
            "is_holiday_season",
            F.when(F.col("MONTH").isin([11, 12]), F.lit(1)).otherwise(F.lit(0)),
        )
    if "ORIGIN" in df.columns and "DEST" in df.columns:
        df = df.withColumn("route", F.concat_ws("_", F.col("ORIGIN"), F.col("DEST")))
    return df


def load_data(spark: SparkSession, input_path: str) -> SparkSession:
    logger.info("Loading data from %s", input_path)
    if input_path.endswith(".parquet") or os.path.isdir(input_path):
        df = spark.read.parquet(input_path)
    else:
        df = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .option("nullValue", "")
            .option("mode", "PERMISSIVE")
            .csv(input_path)
        )
        for raw, std in BTS_COLUMN_MAP.items():
            if raw in df.columns:
                df = df.withColumnRenamed(raw, std)

    if "ARR_DELAY" not in df.columns:
        raise ValueError("Input data must include ARR_DELAY to create the delay label.")

    df = df.filter(F.col("ARR_DELAY").isNotNull())
    df = df.withColumn(LABEL_COL, F.when(F.col("ARR_DELAY") > DELAY_THRESHOLD, F.lit(1.0)).otherwise(F.lit(0.0)))

    numeric_cols = [
        "YEAR",
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "DISTANCE",
        "CRS_ELAPSED_TIME",
        "DEP_DELAY",
        "DEP_DEL15",
        "TAXI_OUT",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "ARR_DELAY",
    ]
    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    df = add_time_features(df)

    num_cols = [c for c in [
        "YEAR",
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "DISTANCE",
        "CRS_ELAPSED_TIME",
        "DEP_DELAY",
        "DEP_DEL15",
        "TAXI_OUT",
        "dep_hour",
        "arr_sched_hour",
        "is_weekend",
        "is_holiday_season",
    ] if c in df.columns]

    for col_name in num_cols:
        df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    categorical_cols = [c for c in ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "route"] if c in df.columns]
    for col_name in categorical_cols:
        df = df.withColumn(col_name, F.col(col_name).cast("string"))

    fill_numeric = {c: 0.0 for c in num_cols}
    fill_cats = {c: "UNKNOWN" for c in categorical_cols}
    df = df.fillna(fill_numeric)
    df = df.fillna(fill_cats)

    df = df.filter(F.col("OP_UNIQUE_CARRIER").isNotNull())
    df = df.filter(F.col("ORIGIN").isNotNull())
    df = df.filter(F.col("DEST").isNotNull())

    logger.info("Loaded %d records with label distribution %s", df.count(), df.groupBy(LABEL_COL).count().collect())
    return df


def compute_class_weights(df) -> Dict[int, float]:
    counts = df.groupBy(LABEL_COL).count().collect()
    total = sum(row["count"] for row in counts)
    n_classes = len(counts)
    weights = {int(row[LABEL_COL]): total / (n_classes * row["count"]) for row in counts}
    logger.info("Class weights: %s", weights)
    return weights


def add_class_weights(df, weights: Dict[int, float]):
    weight_expr = F.when(F.col(LABEL_COL) == 0.0, weights[0]).otherwise(weights[1])
    return df.withColumn(WEIGHT_COL, weight_expr.cast(DoubleType()))


def split_data(df):
    train_df, val_df, test_df = df.randomSplit([TRAIN_RATIO, VAL_RATIO, TEST_RATIO], seed=RANDOM_SEED)
    logger.info("Split sizes – train: %d, val: %d, test: %d", train_df.count(), val_df.count(), test_df.count())
    return train_df, val_df, test_df


def build_preprocessing_stages(categorical_cols: List[str], numeric_cols: List[str]) -> List:
    stages = []
    indexed_cat_cols = []
    for col_name in categorical_cols:
        output_col = f"{col_name}_idx"
        stages.append(StringIndexer(inputCol=col_name, outputCol=output_col, handleInvalid="keep"))
        indexed_cat_cols.append(output_col)

    imputed_num_cols = [f"{c}_imp" for c in numeric_cols]
    stages.append(Imputer(inputCols=numeric_cols, outputCols=imputed_num_cols, strategy="median"))
    stages.append(VectorAssembler(inputCols=indexed_cat_cols + imputed_num_cols, outputCol="raw_features", handleInvalid="keep"))
    stages.append(StandardScaler(inputCol="raw_features", outputCol=FEATURE_COL, withMean=True, withStd=True))
    return stages


def build_lr_pipeline(preprocessing_stages: List) -> Tuple[Pipeline, object]:
    lr = LogisticRegression(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
        family="binomial",
        standardization=False,
    )
    pipeline = Pipeline(stages=preprocessing_stages + [lr])
    param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01]).build()
    return pipeline, param_grid


def build_gbt_pipeline(preprocessing_stages: List) -> Tuple[Pipeline, object]:
    gbt = GBTClassifier(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxIter=50,
        maxDepth=6,
        stepSize=0.1,
        subsamplingRate=0.8,
        seed=RANDOM_SEED,
    )
    pipeline = Pipeline(stages=preprocessing_stages + [gbt])
    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [6])
        .addGrid(gbt.maxIter, [50])
        .addGrid(gbt.stepSize, [0.1])
        .build()
    )
    return pipeline, param_grid


def train_with_cv(pipeline, param_grid, train_df, num_folds: int = 2):
    evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=num_folds,
        seed=RANDOM_SEED,
        parallelism=2,
    )
    logger.info("Starting cross-validated training (%d folds)...", num_folds)
    start = time.time()
    cv_model = cv.fit(train_df)
    logger.info("Training completed in %.1f s.", time.time() - start)
    return cv_model.bestModel


def evaluate_model(model, test_df, model_name: str) -> Dict[str, float]:
    predictions = model.transform(test_df)
    binary_eval = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction")
    auc_roc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})
    auc_pr = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderPR"})
    mc_eval = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction")
    metrics = {
        "model": model_name,
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "f1": round(mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"}), 4),
        "precision": round(mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"}), 4),
        "recall": round(mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"}), 4),
        "accuracy": round(mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"}), 4),
    }
    logger.info("Evaluation results for %s: %s", model_name, metrics)
    return metrics


def save_json(payload: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_pipeline(model, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    model.write().overwrite().save(path)


def run_shap_explainability(model, test_df, model_name: str, output_dir: str, feature_names: List[str]) -> Dict:
    try:
        import shap
    except ImportError:
        logger.warning("[SHAP] shap not installed; skipping.")
        return {}

    logger.info("[SHAP] Running SHAP explainability for %s", model_name)
    try:
        sample_df = test_df.limit(SHAP_SAMPLE_SIZE)
        sample_predictions = model.transform(sample_df)
        sample_pd = sample_predictions.select(F.col(FEATURE_COL), F.col(LABEL_COL)).toPandas()
        X = np.array([row.toArray() for row in sample_pd[FEATURE_COL]])
        explainer = shap.TreeExplainer(model.stages[-1])
        shap_values = explainer.shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        ranked = dict(sorted(zip(feature_names, mean_abs_shap.tolist()), key=lambda x: -x[1]))
        top_10 = {k: round(float(v), 6) for k, v in list(ranked.items())[:10]}
        payload = {
            "model": model_name,
            "sample_size": len(sample_pd),
            "method": "TreeExplainer (SHAP)",
            "top_10": top_10,
        }
        save_json(payload, os.path.join(output_dir, f"shap_{model_name.lower()}.json"))
        return payload
    except Exception as exc:
        logger.warning("[SHAP] Explainability failed for %s: %s", model_name, exc)
        return {}


def run_mode(spark: SparkSession, input_path: str, mode: str, output_root: str, model_root: str, cv_folds: int = 2):
    logger.info("Training %s model", mode)
    df = load_data(spark, input_path)
    weights = compute_class_weights(df)
    df = add_class_weights(df, weights)
    df.cache()

    train_df, val_df, test_df = split_data(df)
    train_df.cache()
    test_df.cache()

    if mode == "full":
        categorical_cols = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "route"]
        numeric_cols = [
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
    else:
        categorical_cols = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "route"]
        numeric_cols = [
            "YEAR",
            "MONTH",
            "DAY_OF_MONTH",
            "DAY_OF_WEEK",
            "DISTANCE",
            "CRS_ELAPSED_TIME",
            "dep_hour",
            "arr_sched_hour",
            "is_weekend",
            "is_holiday_season",
        ]

    preprocessing_stages = build_preprocessing_stages(categorical_cols, numeric_cols)
    feature_names = [f"{c}_imp" for c in numeric_cols] + [f"{c}_idx" for c in categorical_cols]

    gbt_pipeline, gbt_grid = build_gbt_pipeline(preprocessing_stages)
    gbt_model = train_with_cv(gbt_pipeline, gbt_grid, train_df, num_folds=cv_folds)
    gbt_metrics = evaluate_model(gbt_model, test_df, "GBTClassifier")

    model_dir = os.path.join(model_root, mode)
    save_pipeline(gbt_model, os.path.join(model_dir, "gbt_pipeline"))
    shap_payload = run_shap_explainability(gbt_model, test_df, "GBTClassifier", output_root, feature_names)

    lr_pipeline, lr_grid = build_lr_pipeline(preprocessing_stages)
    lr_model = train_with_cv(lr_pipeline, lr_grid, train_df, num_folds=cv_folds)
    lr_metrics = evaluate_model(lr_model, test_df, "LogisticRegression")
    save_pipeline(lr_model, os.path.join(model_dir, "lr_pipeline"))
    run_shap_explainability(lr_model, test_df, "LogisticRegression", output_root, feature_names)

    metrics_payload = {
        "mode": mode,
        "feature_columns": {
            "categorical": categorical_cols,
            "numeric": numeric_cols,
        },
        "metrics": {
            "gbt": gbt_metrics,
            "lr": lr_metrics,
        },
        "shap": {
            "gbt": shap_payload,
        },
    }
    save_json(metrics_payload, os.path.join(output_root, f"{mode}_metrics.json"))
    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare full-feature vs pre-departure flight delay models.")
    parser.add_argument("--input", default="data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2021_12.csv", help="Local CSV or Parquet input path")
    parser.add_argument("--output-root", default="outputs/pre_departure_comparison", help="Directory for metrics and comparison artifacts")
    parser.add_argument("--model-root", default="models/pre_departure_comparison", help="Directory for trained model artifacts")
    parser.add_argument("--cv-folds", type=int, default=2, help="Number of cross-validation folds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.model_root, exist_ok=True)

    spark = build_spark_session()
    try:
        full_payload = run_mode(spark, args.input, "full", args.output_root, args.model_root, cv_folds=args.cv_folds)
        pre_payload = run_mode(spark, args.input, "pre_departure", args.output_root, args.model_root, cv_folds=args.cv_folds)

        comparison_rows = []
        for model_name in ["gbt", "lr"]:
            full_metrics = full_payload["metrics"][model_name]
            pre_metrics = pre_payload["metrics"][model_name]
            comparison_rows.append({
                "model": model_name,
                "mode": "full",
                "auc_roc": full_metrics["auc_roc"],
                "auc_pr": full_metrics["auc_pr"],
                "f1": full_metrics["f1"],
                "precision": full_metrics["precision"],
                "recall": full_metrics["recall"],
                "accuracy": full_metrics["accuracy"],
            })
            comparison_rows.append({
                "model": model_name,
                "mode": "pre_departure",
                "auc_roc": pre_metrics["auc_roc"],
                "auc_pr": pre_metrics["auc_pr"],
                "f1": pre_metrics["f1"],
                "precision": pre_metrics["precision"],
                "recall": pre_metrics["recall"],
                "accuracy": pre_metrics["accuracy"],
            })

        comparison_path = os.path.join(args.output_root, "comparison_metrics.json")
        save_json({"comparison_rows": comparison_rows}, comparison_path)

        import pandas as pd
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_csv = os.path.join(args.output_root, "comparison_metrics.csv")
        comparison_df.to_csv(comparison_csv, index=False)

        logger.info("Comparison artifacts saved to %s", args.output_root)
        logger.info("Models saved to %s", args.model_root)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

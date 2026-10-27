"""
train_chronological.py
----------------------
Train and evaluate GBT and Logistic Regression flight delay models on a
chronological train/validation/test split of the 2021-2023 BTS corpus, in both
the full-feature and pre-departure-only feature regimes.

Unlike train_model.py / train_pre_departure_comparison.py (random row-level
70/15/15 split, single-point "grid"), this script:
  * splits by calendar month -- train = earliest months, validation = the next
    block, test = the most recent block -- so no flight in the test set
    precedes any flight in the training set;
  * computes class weights on the training partition only;
  * selects hyperparameters on the validation partition (AUC-ROC);
  * selects a decision threshold on the validation partition (positive-class
    F1) and reports test metrics at both 0.5 and that threshold;
  * scores the trivial DEP_DELAY > 15 rule on the identical test partition;
  * labels SHAP values in the same order the VectorAssembler builds features.

Usage (inside the spark-master container):
    spark-submit --master spark://spark-master:7077 \
        src/training/train_chronological.py \
        --input hdfs://hdfs-namenode:9000/full_corpus_v2/data/flights \
        --model-root hdfs://hdfs-namenode:9000/models_chrono \
        --metrics-json /tmp/chrono_metrics.json
"""

import argparse
import itertools
import json
import logging
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from train_pre_departure_comparison import (
    LABEL_COL,
    FEATURE_COL,
    WEIGHT_COL,
    RANDOM_SEED,
    DELAY_THRESHOLD,
    build_spark_session,
    compute_class_weights,
    add_class_weights,
    build_preprocessing_stages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_chronological")

CATEGORICAL_COLS = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "route"]
SCHEDULE_NUMERIC_COLS = [
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
POST_DEPARTURE_COLS = ["DEP_DELAY", "DEP_DEL15", "TAXI_OUT"]

GBT_GRID = {"maxDepth": [6, 8], "maxIter": [50, 100]}
LR_GRID = {"regParam": [0.0, 0.001, 0.01]}

PROB_COL = "p_delayed"
SHAP_SAMPLE_SIZE = 2000

# Feature derivation runs as the first stage of the saved pipeline, so batch
# and streaming inference apply exactly the same logic to raw BTS columns.
# Missing numeric values are left for the pipeline's Imputer (median).
FEATURE_SQL = f"""
SELECT *,
       CASE WHEN DEP_DELAY IS NULL THEN NULL
            WHEN DEP_DELAY > {DELAY_THRESHOLD} THEN 1.0 ELSE 0.0 END AS DEP_DEL15,
       CAST(FLOOR(CRS_DEP_TIME / 100) AS DOUBLE) AS dep_hour,
       CAST(FLOOR(CRS_ARR_TIME / 100) AS DOUBLE) AS arr_sched_hour,
       CASE WHEN DAY_OF_WEEK IN (6, 7) THEN 1.0 ELSE 0.0 END AS is_weekend,
       CASE WHEN MONTH IN (11, 12) THEN 1.0 ELSE 0.0 END AS is_holiday_season,
       CONCAT_WS('_', ORIGIN, DEST) AS route
FROM __THIS__
"""
RAW_NUMERIC_COLS = [
    "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "DISTANCE", "CRS_ELAPSED_TIME",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "DEP_DELAY", "TAXI_OUT",
]


def load_raw(spark, input_path: str):
    """Read ingested Parquet and add the label; no feature derivation here (see FEATURE_SQL)."""
    df = spark.read.parquet(input_path).filter(F.col("ARR_DELAY").isNotNull())
    df = df.withColumn(LABEL_COL, (F.col("ARR_DELAY") > DELAY_THRESHOLD).cast("double"))
    for col_name in RAW_NUMERIC_COLS:
        df = df.withColumn(col_name, F.col(col_name).cast("double"))
    return df


# ─── Split ────────────────────────────────────────────────────────────────────


def month_index(year_col: str = "YEAR", month_col: str = "MONTH"):
    return F.col(year_col).cast("int") * 12 + F.col(month_col).cast("int")


def parse_year_month(value: str) -> int:
    year, month = value.split("-")
    return int(year) * 12 + int(month)


def chronological_split(df, val_start: str, test_start: str):
    """Split on calendar month: [.., val_start) / [val_start, test_start) / [test_start, ..)."""
    idx = month_index()
    val_idx, test_idx = parse_year_month(val_start), parse_year_month(test_start)
    train_df = df.filter(idx < val_idx)
    val_df = df.filter((idx >= val_idx) & (idx < test_idx))
    test_df = df.filter(idx >= test_idx)
    return train_df, val_df, test_df


# ─── Models ───────────────────────────────────────────────────────────────────


def gbt_estimator(params: Dict) -> GBTClassifier:
    return GBTClassifier(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        stepSize=0.1,
        subsamplingRate=0.8,
        seed=RANDOM_SEED,
        **params,
    )


def lr_estimator(params: Dict) -> LogisticRegression:
    return LogisticRegression(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxIter=100,
        elasticNetParam=0.0,
        family="binomial",
        standardization=False,
        **params,
    )


def expand_grid(grid: Dict[str, List]) -> List[Dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def with_probability(predictions):
    return predictions.withColumn(PROB_COL, vector_to_array(F.col("probability"))[1])


def auc(predictions, metric: str) -> float:
    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName=metric
    )
    return evaluator.evaluate(predictions)


def select_on_validation(name: str, make_estimator, grid: Dict, stages: List, train_df, val_df):
    """Fit one pipeline per grid point on train; keep the best by validation AUC-ROC."""
    trials = []
    best_model, best_auc = None, -1.0
    for params in expand_grid(grid):
        start = time.time()
        model = Pipeline(stages=stages + [make_estimator(params)]).fit(train_df)
        val_auc = auc(model.transform(val_df), "areaUnderROC")
        elapsed = time.time() - start
        logger.info("[%s] %s → validation AUC-ROC %.4f (%.0f s)", name, params, val_auc, elapsed)
        trials.append({"params": params, "val_auc_roc": round(val_auc, 4), "fit_seconds": round(elapsed, 1)})
        if val_auc > best_auc:
            best_model, best_auc = model, val_auc
    best = max(trials, key=lambda t: t["val_auc_roc"])
    logger.info("[%s] selected %s", name, best["params"])
    return best_model, trials, best["params"]


# ─── Metrics ──────────────────────────────────────────────────────────────────


def confusion_at(scored, threshold: float) -> Dict[str, int]:
    row = scored.select(
        F.sum(F.when((F.col(PROB_COL) > threshold) & (F.col(LABEL_COL) == 1.0), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col(PROB_COL) > threshold) & (F.col(LABEL_COL) == 0.0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col(PROB_COL) <= threshold) & (F.col(LABEL_COL) == 0.0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col(PROB_COL) <= threshold) & (F.col(LABEL_COL) == 1.0), 1).otherwise(0)).alias("fn"),
    ).collect()[0]
    return {k: int(row[k]) for k in ("tp", "fp", "tn", "fn")}


def metrics_from_confusion(cm: Dict[str, int]) -> Dict[str, float]:
    """Positive-class and support-weighted metrics, matching Section V-C of the paper."""
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    n1, n0 = tp + fn, tn + fp
    n = n0 + n1

    def prf(t, f_pos, f_neg):
        p = t / (t + f_pos) if (t + f_pos) else 0.0
        r = t / (t + f_neg) if (t + f_neg) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    p1, r1, f1_1 = prf(tp, fp, fn)
    p0, r0, f1_0 = prf(tn, fn, fp)
    return {
        "accuracy": round((tp + tn) / n, 4),
        "weighted_precision": round((n0 * p0 + n1 * p1) / n, 4),
        "weighted_f1": round((n0 * f1_0 + n1 * f1_1) / n, 4),
        "pos_precision": round(p1, 4),
        "pos_recall": round(r1, 4),
        "pos_f1": round(f1_1, 4),
        "pct_predicted_delayed": round(100 * (tp + fp) / n, 2),
        **cm,
    }


def best_f1_threshold(scored_val) -> float:
    """Pick the threshold (0.01 grid) that maximises positive-class F1 on validation."""
    buckets = (
        scored_val.withColumn("bucket", F.least(F.floor(F.col(PROB_COL) * 100), F.lit(99)).cast("int"))
        .groupBy("bucket", LABEL_COL)
        .count()
        .collect()
    )
    pos = np.zeros(100)
    neg = np.zeros(100)
    for row in buckets:
        (pos if row[LABEL_COL] == 1.0 else neg)[row["bucket"]] += row["count"]
    best_t, best_f1 = 0.5, -1.0
    for b in range(1, 100):
        tp, fp = pos[b:].sum(), neg[b:].sum()
        fn = pos[:b].sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best_t, best_f1 = b / 100, f1
    # Buckets >= b hold p >= b/100; confusion_at's "p > t" differs only at p == t exactly.
    return best_t


def evaluate(model, val_df, test_df, name: str) -> Dict:
    scored_val = with_probability(model.transform(val_df))
    threshold = best_f1_threshold(scored_val)
    scored_test = with_probability(model.transform(test_df)).persist(StorageLevel.DISK_ONLY)
    result = {
        "model": name,
        "auc_roc": round(auc(scored_test, "areaUnderROC"), 4),
        "auc_pr": round(auc(scored_test, "areaUnderPR"), 4),
        "at_0.5": metrics_from_confusion(confusion_at(scored_test, 0.5)),
        "val_selected_threshold": threshold,
        "at_val_threshold": metrics_from_confusion(confusion_at(scored_test, threshold)),
    }
    scored_test.unpersist()
    logger.info("Test results for %s: %s", name, json.dumps(result))
    return result


def trivial_rule_baseline(test_df) -> Dict:
    scored = test_df.withColumn(
        PROB_COL, F.coalesce(F.col("DEP_DELAY") > DELAY_THRESHOLD, F.lit(False)).cast("double")
    )
    return {"model": "DEP_DELAY > 15 rule", **metrics_from_confusion(confusion_at(scored, 0.5))}


# ─── SHAP ─────────────────────────────────────────────────────────────────────


def run_shap(model, test_df, feature_names: List[str], sample_size: int) -> Dict:
    try:
        import shap
    except ImportError:
        logger.warning("[SHAP] shap not installed; skipping.")
        return {}
    total = test_df.count()
    sample = test_df.sample(fraction=min(1.0, 2.0 * sample_size / total), seed=RANDOM_SEED).limit(sample_size)
    pdf = model.transform(sample).select(FEATURE_COL).toPandas()
    X = np.array([row.toArray() for row in pdf[FEATURE_COL]])
    shap_values = shap.TreeExplainer(model.stages[-1]).shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: -x[1])
    return {
        "method": "TreeExplainer (SHAP), mean |SHAP| on log-odds scale",
        "sample_size": len(pdf),
        "ranking": {k: round(v, 6) for k, v in ranked},
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def run_mode(mode: str, train_df, val_df, test_df, model_root: str, shap_sample: int) -> Dict:
    numeric_cols = SCHEDULE_NUMERIC_COLS + (POST_DEPARTURE_COLS if mode == "full" else [])
    stages = [SQLTransformer(statement=FEATURE_SQL)] + build_preprocessing_stages(CATEGORICAL_COLS, numeric_cols)
    # VectorAssembler order in build_preprocessing_stages: indexed categoricals, then imputed numerics.
    feature_names = [f"{c}_idx" for c in CATEGORICAL_COLS] + [f"{c}_imp" for c in numeric_cols]
    logger.info("=== Mode: %s (%d features) ===", mode, len(feature_names))

    gbt_model, gbt_trials, gbt_params = select_on_validation("gbt", gbt_estimator, GBT_GRID, stages, train_df, val_df)
    gbt_result = evaluate(gbt_model, val_df, test_df, "GBTClassifier")
    gbt_model.write().overwrite().save(f"{model_root}/{mode}/gbt_pipeline")

    lr_model, lr_trials, lr_params = select_on_validation("lr", lr_estimator, LR_GRID, stages, train_df, val_df)
    lr_result = evaluate(lr_model, val_df, test_df, "LogisticRegression")
    lr_model.write().overwrite().save(f"{model_root}/{mode}/lr_pipeline")

    shap_payload = {}
    if shap_sample > 0:
        try:
            shap_payload = run_shap(gbt_model, test_df, feature_names, shap_sample)
        except Exception as exc:
            logger.warning("[SHAP] failed for %s: %s", mode, exc)

    return {
        "feature_columns": {"categorical": CATEGORICAL_COLS, "numeric": numeric_cols},
        "gbt": {"selected_params": gbt_params, "grid": gbt_trials, "test": gbt_result},
        "lr": {"selected_params": lr_params, "grid": lr_trials, "test": lr_result},
        "shap_gbt": shap_payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chronological-split training for full and pre-departure models.")
    parser.add_argument("--input", default="hdfs://hdfs-namenode:9000/full_corpus_v2/data/flights")
    parser.add_argument("--model-root", default="hdfs://hdfs-namenode:9000/models_chrono")
    parser.add_argument("--val-start", default="2023-03", help="First month (YYYY-MM) of the validation partition.")
    parser.add_argument("--test-start", default="2023-08", help="First month (YYYY-MM) of the test partition.")
    parser.add_argument("--modes", nargs="+", default=["full", "pre_departure"], choices=["full", "pre_departure"])
    parser.add_argument("--shap-sample", type=int, default=SHAP_SAMPLE_SIZE, help="0 disables SHAP.")
    parser.add_argument(
        "--sample-fraction", type=float, default=1.0,
        help="Row sample applied before splitting, for smoke tests only (default 1.0 = all data).",
    )
    parser.add_argument("--metrics-json", required=True, help="Local path for the JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = build_spark_session("FlightDelay_Chronological")
    run_start = time.time()
    try:
        df = load_raw(spark, args.input)
        if args.sample_fraction < 1.0:
            df = df.sample(fraction=args.sample_fraction, seed=RANDOM_SEED)
        train_df, val_df, test_df = chronological_split(df, args.val_start, args.test_start)

        weights = compute_class_weights(train_df)
        train_df = add_class_weights(train_df, weights).persist(StorageLevel.DISK_ONLY)
        val_df = add_class_weights(val_df, weights).persist(StorageLevel.DISK_ONLY)
        test_df = add_class_weights(test_df, weights).persist(StorageLevel.DISK_ONLY)

        sizes = {"train": train_df.count(), "val": val_df.count(), "test": test_df.count()}
        prevalence = {
            name: round(part.agg(F.avg(LABEL_COL)).collect()[0][0], 4)
            for name, part in (("train", train_df), ("val", val_df), ("test", test_df))
        }
        logger.info("Split sizes %s, delayed rate %s", sizes, prevalence)

        summary = {
            "input": args.input,
            "sample_fraction": args.sample_fraction,
            "split": {
                "type": "chronological (calendar month)",
                "train_months": f"start .. {args.val_start} (exclusive)",
                "val_months": f"{args.val_start} .. {args.test_start} (exclusive)",
                "test_months": f"{args.test_start} .. end",
                "sizes": sizes,
                "delayed_rate": prevalence,
            },
            "class_weights_from_train": {str(k): v for k, v in weights.items()},
            "trivial_baseline_test": trivial_rule_baseline(test_df),
            "modes": {},
        }
        for mode in args.modes:
            summary["modes"][mode] = run_mode(mode, train_df, val_df, test_df, args.model_root, args.shap_sample)
            summary["elapsed_seconds"] = round(time.time() - run_start, 1)
            # Write after every mode so a later failure doesn't lose finished results.
            with open(args.metrics_json, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            logger.info("Results written to %s", args.metrics_json)
    except Exception as exc:
        logger.exception("Chronological training failed: %s", exc)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

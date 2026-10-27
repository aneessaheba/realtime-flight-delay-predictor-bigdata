"""
dp_repeat.py
------------
Repeat the benchmark's differential-privacy release many times to characterise
the deviation it introduces, instead of reporting a single noisy draw.

Uses benchmark.dp_metric unchanged (Laplace mechanism, epsilon and sensitivity
from benchmark.py) on a fixed set of true metric values, and reports per-metric
mean / p95 / max absolute deviation over N independent draws, plus the
analytic expectation E|Laplace(0, b)| = b with b = sensitivity / epsilon.

Usage:
    python src/evaluation/dp_repeat.py \
        --metrics-json logs/train_full_corpus_metrics_20260712_004845.json \
        --model GBTClassifier --draws 1000 --output logs/dp_repeat.json
    python src/evaluation/dp_repeat.py \
        --metrics-json logs/train_chronological_full_corpus.json \
        --model full/gbt/at_val_threshold --draws 1000 --output logs/dp_repeat_chronological.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import DP_EPSILON, DP_SENSITIVITY, dp_metric  # noqa: E402

DP_FIELDS = ["auc_roc", "auc_pr", "f1", "precision", "accuracy", "positive_class_recall"]


CHRONO_FIELDS = ["accuracy", "weighted_precision", "weighted_f1", "pos_precision", "pos_recall", "pos_f1"]


def load_true_metrics(path: str, model: str) -> dict:
    """
    Read the metrics to release. Supports train_model.py's --metrics-json
    ("metrics" list, model = "GBTClassifier") and train_chronological.py's
    output (model = "<mode>/<gbt|lr>/<at_0.5|at_val_threshold>", e.g.
    "full/gbt/at_val_threshold").
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if "modes" in payload:
        mode, key, point = model.split("/")
        test = payload["modes"][mode][key]["test"]
        values = {"auc_roc": test["auc_roc"], "auc_pr": test["auc_pr"]}
        values.update({k: test[point][k] for k in CHRONO_FIELDS})
        return values
    for entry in payload["metrics"]:
        if entry["model"] == model:
            return {k: entry[k] for k in DP_FIELDS if isinstance(entry.get(k), float)}
    raise ValueError(f"Model {model!r} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat the DP metric release N times.")
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--model", default="GBTClassifier")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    true_metrics = load_true_metrics(args.metrics_json, args.model)
    # diffprivlib (and the numpy fallback in dp_metric) draw from numpy's global RNG.
    np.random.seed(args.seed)
    deviations = {k: [] for k in true_metrics}
    max_per_release = []
    for _ in range(args.draws):
        release = {k: abs(dp_metric(v) - v) for k, v in true_metrics.items()}
        for k, d in release.items():
            deviations[k].append(d)
        max_per_release.append(max(release.values()))

    scale = DP_SENSITIVITY / DP_EPSILON
    per_metric = {
        k: {
            "true": true_metrics[k],
            "mean_abs_dev": round(float(np.mean(v)), 5),
            "p95_abs_dev": round(float(np.percentile(v, 95)), 5),
            "max_abs_dev": round(float(np.max(v)), 5),
        }
        for k, v in deviations.items()
    }
    summary = {
        "source": args.metrics_json,
        "model": args.model,
        "mechanism": "Laplace",
        "epsilon_per_metric": DP_EPSILON,
        "metrics_released": len(true_metrics),
        "total_epsilon_sequential_composition": DP_EPSILON * len(true_metrics),
        "sensitivity": DP_SENSITIVITY,
        "laplace_scale_b": scale,
        "analytic_expected_abs_dev": scale,
        "draws": args.draws,
        "seed": args.seed,
        "per_metric": per_metric,
        "max_over_metrics_per_release": {
            "mean": round(float(np.mean(max_per_release)), 5),
            "median": round(float(np.median(max_per_release)), 5),
            "p95": round(float(np.percentile(max_per_release, 95)), 5),
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

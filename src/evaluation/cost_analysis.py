"""
cost_analysis.py
----------------
Cost-sensitive comparison of the full-feature GBT, LR and the trivial
DEP_DELAY > 15 rule on the chronological test partition, from the confusion
matrices saved by train_chronological.py.

With c_FN the cost of a missed delay and c_FP the cost of a false alarm, the
expected cost of an operating point is c_FN * FN + c_FP * FP. Only the ratio
r = c_FN / c_FP matters for ranking, so for every pair of operating points we
report the break-even ratio at which their costs are equal, and the total
cost (in false-alarm units, FN * r + FP) at a few representative ratios.

Usage:
    python src/evaluation/cost_analysis.py \
        --metrics-json logs/train_chronological_full_corpus.json \
        --output logs/cost_analysis_chronological.json
"""

import argparse
import itertools
import json
import os

RATIOS = [1, 2, 3, 5, 10]


def operating_points(payload: dict) -> dict:
    full = payload["modes"]["full"]
    points = {}
    for key, label in (("gbt", "GBT"), ("lr", "LR")):
        test = full[key]["test"]
        points[f"{label} @0.5"] = test["at_0.5"]
        points[f"{label} @{test['val_selected_threshold']} (val-selected)"] = test["at_val_threshold"]
    points["DEP_DELAY > 15 rule"] = payload["trivial_baseline_test"]
    return {name: {"fn": m["fn"], "fp": m["fp"]} for name, m in points.items()}


def break_even(a: dict, b: dict):
    """Ratio r at which a and b cost the same; None if one dominates the other."""
    d_fn, d_fp = a["fn"] - b["fn"], b["fp"] - a["fp"]
    if d_fn == 0 or (d_fn > 0) != (d_fp > 0):
        return None
    return d_fp / d_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-sensitive comparison of operating points.")
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.metrics_json, encoding="utf-8") as fh:
        points = operating_points(json.load(fh))

    pairs = []
    for (name_a, a), (name_b, b) in itertools.combinations(points.items(), 2):
        r = break_even(a, b)
        if r is None:
            dominant = name_a if (a["fn"] <= b["fn"] and a["fp"] <= b["fp"]) else name_b
            note = f"{dominant} has no more FN and no more FP: cheaper at every ratio"
        else:
            fewer_fn = name_a if a["fn"] < b["fn"] else name_b
            other = name_b if fewer_fn == name_a else name_a
            note = f"{fewer_fn} is cheaper when c_FN/c_FP > {r:.2f}; {other} is cheaper below that"
        pairs.append({"a": name_a, "b": name_b, "break_even_ratio": None if r is None else round(r, 3), "summary": note})

    totals = {
        f"r={r}": dict(sorted(
            ((name, p["fn"] * r + p["fp"]) for name, p in points.items()), key=lambda kv: kv[1]
        ))
        for r in RATIOS
    }
    result = {
        "source": args.metrics_json,
        "test_partition": "Aug-Dec 2023 (chronological split)",
        "cost_model": "total = c_FN * FN + c_FP * FP; reported in false-alarm units (c_FP = 1, c_FN = r)",
        "operating_points": points,
        "pairwise_break_even": pairs,
        "total_cost_by_ratio_sorted": totals,
        "cheapest_by_ratio": {k: next(iter(v)) for k, v in totals.items()},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""
lsh_consistency.py
------------------
MinHash-LSH prediction-consistency check on streamed predictions, with the
baselines needed to interpret it.

benchmark.lsh_anomaly_detection reports the share of LSH candidate pairs
(flights sharing carrier/origin/dest tokens) whose predictions disagree. On
its own that rate is not interpretable: the full-feature model's prediction
is driven mainly by departure delay, so two flights on the same route are
expected to disagree whenever their departure delays differ, and two random
flights disagree at rate 2p(1-p) for a predicted-delay share p.

This script uses the same MinHashLSH settings as benchmark.py and reports:
  * disagreement among LSH candidate pairs, after checking each pair's true
    Jaccard similarity against the LSH threshold;
  * the same with a departure-delay band added to the token set
    (similar route AND similar operating state);
  * the random-pair baseline, both analytic 2p(1-p) and empirical.

Usage (inside the spark-master container):
    spark-submit src/evaluation/lsh_consistency.py \
        --predictions hdfs://hdfs-namenode:9000/output/streaming_2024_t0 \
        --output /tmp/lsh_consistency_2024.json
"""

import argparse
import json
import random

from datasketch import MinHash, MinHashLSH
from pyspark.sql import SparkSession

from benchmark import LSH_NUM_PERM, LSH_THRESHOLD

SAMPLE_SIZE = 5000
RANDOM_PAIRS = 200_000
SEED = 42


def delay_band(dep_delay) -> str:
    if dep_delay is None:
        return "dep_unknown"
    if dep_delay <= 0:
        return "dep_early_or_on_time"
    if dep_delay <= 15:
        return "dep_1_15"
    if dep_delay <= 60:
        return "dep_16_60"
    return "dep_over_60"


def disagreement(rows, token_fn) -> dict:
    tokens = [set(token_fn(r)) for r in rows]
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_NUM_PERM)
    sigs = []
    for i, toks in enumerate(tokens):
        m = MinHash(num_perm=LSH_NUM_PERM, seed=SEED)
        for t in toks:
            m.update(t.encode("utf8"))
        lsh.insert(i, m)
        sigs.append(m)

    candidates = set()
    for i, m in enumerate(sigs):
        for j in lsh.query(m):
            if j != i:
                candidates.add((min(i, j), max(i, j)))

    verified = [
        (i, j) for i, j in candidates
        if len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j]) >= LSH_THRESHOLD
    ]
    disagree = sum(rows[i]["prediction"] != rows[j]["prediction"] for i, j in verified)
    return {
        "candidate_pairs": len(candidates),
        "verified_pairs_jaccard_ge_threshold": len(verified),
        "false_candidate_share": round(1 - len(verified) / len(candidates), 4) if candidates else None,
        "disagreeing_pairs": disagree,
        "disagreement_rate": round(disagree / len(verified), 4) if verified else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LSH prediction-consistency check with baselines.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("FlightDelay_LSHConsistency").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    df = spark.read.parquet(args.predictions).select(
        "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_DELAY", "prediction"
    )
    total = df.count()
    # A random sample, not the first rows: limit() alone returns whatever
    # partition happens to be read first.
    rows = [r.asDict() for r in df.sample(fraction=min(1.0, 2.0 * args.sample_size / total), seed=SEED)
            .limit(args.sample_size).collect()]
    spark.stop()

    p = sum(r["prediction"] == 1.0 for r in rows) / len(rows)
    rng = random.Random(SEED)
    rand_pairs = [rng.sample(range(len(rows)), 2) for _ in range(RANDOM_PAIRS)]
    rand_disagree = sum(rows[i]["prediction"] != rows[j]["prediction"] for i, j in rand_pairs) / RANDOM_PAIRS

    result = {
        "predictions": args.predictions,
        "sample_size": len(rows),
        "lsh_threshold": LSH_THRESHOLD,
        "lsh_num_perm": LSH_NUM_PERM,
        "predicted_delayed_share": round(p, 4),
        "baseline_random_pairs": {
            "analytic_2p_1_minus_p": round(2 * p * (1 - p), 4),
            "empirical": round(rand_disagree, 4),
            "pairs": RANDOM_PAIRS,
        },
        "route_tokens": {
            "tokens": "carrier, origin, dest",
            **disagreement(rows, lambda r: [f"c:{r['OP_UNIQUE_CARRIER']}", f"o:{r['ORIGIN']}", f"d:{r['DEST']}"]),
        },
        "route_and_departure_delay_band": {
            "tokens": "carrier, origin, dest, departure-delay band",
            **disagreement(rows, lambda r: [
                f"c:{r['OP_UNIQUE_CARRIER']}", f"o:{r['ORIGIN']}", f"d:{r['DEST']}", delay_band(r["DEP_DELAY"]),
            ]),
        },
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

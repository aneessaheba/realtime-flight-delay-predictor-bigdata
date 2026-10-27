"""
eda_full_corpus.py
------------------
Compute the exploratory-analysis aggregates reported in the paper (class
balance, delay rate by month / day of week / carrier, numeric-feature
correlations with the label) on the full ingested 2021-2023 corpus in Spark,
and write them to JSON. plot_eda_full_corpus.py draws the figures from it.

eda_analysis.py does the same in pandas, which only fits a single-month
sample in memory; the figures it produced were from such a sample.

Usage (inside the spark-master container):
    spark-submit src/training/eda_full_corpus.py \
        --input hdfs://hdfs-namenode:9000/full_corpus_v2/data/flights \
        --output /tmp/eda_full_corpus.json
"""

import argparse
import json
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CORR_COLS = ["DAY_OF_WEEK", "CRS_DEP_TIME", "DEP_DELAY", "TAXI_OUT", "CRS_ELAPSED_TIME", "DISTANCE", "MONTH"]
MIN_CARRIER_FLIGHTS = 100


def rate_table(df, key: str):
    rows = (
        df.groupBy(key)
        .agg(F.count("*").alias("flights"), F.avg("label").alias("delay_rate"))
        .orderBy(key)
        .collect()
    )
    return [
        {key: r[key], "flights": r["flights"], "delay_rate_pct": round(100 * r["delay_rate"], 2)}
        for r in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-corpus EDA aggregates.")
    parser.add_argument("--input", default="hdfs://hdfs-namenode:9000/full_corpus_v2/data/flights")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("FlightDelay_EDA_FullCorpus").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.read.parquet(args.input).filter(F.col("ARR_DELAY").isNotNull())
        df = df.withColumn("label", (F.col("ARR_DELAY") > 15.0).cast("double"))
        for c in CORR_COLS:
            df = df.withColumn(c, F.col(c).cast("double"))
        df = df.persist()

        counts = {int(r["label"]): r["count"] for r in df.groupBy("label").count().collect()}
        total = sum(counts.values())
        carriers = [c for c in rate_table(df, "OP_UNIQUE_CARRIER") if c["flights"] >= MIN_CARRIER_FLIGHTS]

        # Pearson r with the label, pairwise-complete (rows with a null in either column dropped).
        corr = {}
        for c in CORR_COLS + ["label"]:
            corr[c] = {}
            for d in CORR_COLS + ["label"]:
                sub = df.select(c, d).dropna()
                corr[c][d] = round(sub.stat.corr(c, d), 4) if c != d else 1.0

        result = {
            "input": args.input,
            "total_flights": total,
            "class_counts": {"on_time": counts.get(0, 0), "delayed": counts.get(1, 0)},
            "delayed_pct": round(100 * counts.get(1, 0) / total, 2),
            "by_month": rate_table(df, "MONTH"),
            "by_day_of_week": rate_table(df, "DAY_OF_WEEK"),
            "by_carrier": sorted(carriers, key=lambda c: -c["delay_rate_pct"]),
            "by_year": rate_table(df, "YEAR"),
            "correlation": corr,
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k != "correlation"}, indent=2))
    except Exception as exc:
        print(f"EDA failed: {exc}", file=sys.stderr)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

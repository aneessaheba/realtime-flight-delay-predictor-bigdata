"""
kafka_producer.py  –  Kartheek Alluri (Weeks 7-8, swapped from Keon)
Replays 2024 BTS Airline On-Time Performance data as simulated live
flight-departure events, publishing each flight as a JSON message to
the Kafka topic `flight-events`.

Streaming Algorithm: Reservoir Sampling
    When --sample K is provided, an unbiased random sample of exactly K
    rows is drawn from the dataset using Algorithm R (Vitter, 1985) before
    any messages are published. This gives a fixed-size, uniformly random
    subset of the stream without knowing the total size in advance — a
    classic streaming algorithm that runs in O(n) time with O(K) memory.

Usage
-----
# Full dataset (no sampling):
python src/kafka_producer.py \
    --input  data/2024_ontime.parquet \
    --broker localhost:9093 \
    --topic  flight-events \
    --rate   500

# Reservoir-sampled subset of 50,000 rows:
python src/kafka_producer.py \
    --input  data/2024_ontime.parquet \
    --broker localhost:9093 \
    --topic  flight-events \
    --rate   500 \
    --sample 50000

Arguments
---------
--input   Path to the 2024 BTS Parquet file (or a CSV glob pattern).
--broker  Kafka bootstrap server.  Default: localhost:9093 (external listener).
--topic   Kafka topic name.        Default: flight-events
--rate    Target publish rate in events/second.  0 = as fast as possible.
--limit   Stop after N messages.  0 = stream the whole file then loop.
--loop    If set, replay the file indefinitely (useful for benchmark runs).
--sample  If > 0, apply reservoir sampling to draw this many rows uniformly
          at random before publishing. 0 = use all rows.
"""

import argparse
import json
import random
import time
import sys
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

KEEP_COLS = [
    "FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY",
    "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "DISTANCE",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "ARR_DEL15",
]

LEAK_COLS = {"ARR_TIME", "ARR_DELAY", "ARR_DEL15",
             "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
             "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"}


def load_data(input_path: str) -> pd.DataFrame:
    p = Path(input_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(input_path, usecols=lambda c: c in KEEP_COLS, low_memory=False)
    df = df[df["DEP_TIME"].notna()].reset_index(drop=True)
    print(f"[producer] Loaded {len(df):,} rows from {input_path}")
    return df


def reservoir_sample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Algorithm R (Vitter, 1985) — Reservoir Sampling.

    Draws an unbiased simple random sample of exactly k rows from df in a
    single pass, without replacement. Each element i has exactly k/i
    probability of being selected at step i, so the final reservoir is a
    uniformly random subset over all C(n, k) possible subsets.

    This is applied here to draw a fixed-size benchmark sample from the
    2024 BTS dataset without loading or sorting the full file multiple times.

    Time:  O(n)
    Space: O(k)
    """
    n = len(df)
    if k >= n:
        print(f"[producer] Sample size {k:,} >= dataset size {n:,}; using full dataset.")
        return df

    indices = list(range(k))
    for i in range(k, n):
        j = random.randint(0, i)
        if j < k:
            indices[j] = i

    sampled = df.iloc[sorted(indices)].reset_index(drop=True)
    print(
        f"[producer] Reservoir sampling complete: "
        f"{len(sampled):,} rows drawn from {n:,} "
        f"(rate={len(sampled)/n:.2%})"
    )
    return sampled


def row_to_event(row: pd.Series) -> dict:
    record = {}
    label = None
    for col, val in row.items():
        if pd.isna(val):
            record[col] = None
        elif col == "ARR_DEL15":
            label = int(val)
        elif col not in LEAK_COLS:
            record[col] = val if not isinstance(val, float) else round(val, 4)
    record["_event_ts"] = time.time()
    record["_label"] = label
    return record


def build_producer(broker: str) -> KafkaProducer:
    for attempt in range(1, 6):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[broker],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=5,
                linger_ms=10,
                batch_size=65536,
                compression_type="gzip",
            )
            print(f"[producer] Connected to Kafka at {broker}")
            return producer
        except NoBrokersAvailable:
            wait = 2 ** attempt
            print(f"[producer] Broker not ready (attempt {attempt}/5). Retrying in {wait}s...")
            time.sleep(wait)
    print("[producer] Could not connect to Kafka. Exiting.")
    sys.exit(1)


def publish(producer, topic, df, rate, limit, loop):
    interval = (1.0 / rate) if rate > 0 else 0.0
    sent = 0
    errors = 0

    def _send_batch(rows):
        nonlocal sent, errors
        for _, row in rows:
            event = row_to_event(row)
            try:
                producer.send(topic, value=event)
                sent += 1
            except Exception as exc:
                errors += 1
                if errors <= 10:
                    print(f"[producer] Send error: {exc}")
            if interval > 0:
                time.sleep(interval)
            if limit and sent >= limit:
                return True
            if sent % 1000 == 0:
                producer.flush()
                print(f"[producer] Published {sent:,} messages ({errors} errors)", flush=True)
        return False

    try:
        while True:
            stop = _send_batch(df.iterrows())
            producer.flush()
            print(f"[producer] Finished pass. Total sent: {sent:,}")
            if stop or not loop:
                break
    except KeyboardInterrupt:
        print("\n[producer] Interrupted by user.")
    finally:
        producer.flush()
        producer.close()
        print(f"[producer] Done. Sent={sent:,}  Errors={errors}")


def main():
    parser = argparse.ArgumentParser(description="BTS 2024 Kafka producer")
    parser.add_argument("--input",  default="data/2024_ontime.parquet")
    parser.add_argument("--broker", default="localhost:9093")
    parser.add_argument("--topic",  default="flight-events")
    parser.add_argument("--rate",   type=float, default=500.0)
    parser.add_argument("--limit",  type=int,   default=0)
    parser.add_argument("--loop",   action="store_true")
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Reservoir sample size. 0 = use all rows (no sampling)."
    )
    args = parser.parse_args()

    df = load_data(args.input)

    if args.sample > 0:
        df = reservoir_sample(df, args.sample)

    producer = build_producer(args.broker)
    publish(producer, args.topic, df, rate=args.rate, limit=args.limit, loop=args.loop)


if __name__ == "__main__":
    main()
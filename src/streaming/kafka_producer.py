"""
kafka_producer.py
-----------------
Replay 2024 BTS On-Time Performance records as a real-time JSON stream
to a Kafka topic, simulating live flight event ingestion.

Usage:
    python src/streaming/kafka_producer.py \
        --input-path /data/raw/bts/2024 \
        --kafka-bootstrap localhost:9093 \
        --topic flight-events \
        --rate 100
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("kafka_producer")

# ─── Column definitions ───────────────────────────────────────────────────────

FEATURE_COLUMNS = [
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

# Columns to cast to float where possible; rest remain strings
NUMERIC_COLUMNS = {
    "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
    "CRS_DEP_TIME", "DEP_DELAY", "CRS_ARR_TIME", "ARR_DELAY",
    "CRS_ELAPSED_TIME", "DISTANCE", "CARRIER_DELAY", "WEATHER_DELAY",
    "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
}

LOG_INTERVAL = 1000       # print progress every N messages
RECONNECT_ATTEMPTS = 5    # max retries when Kafka is unavailable
RECONNECT_DELAY_S = 5     # seconds between retry attempts


# ─── Kafka producer setup ─────────────────────────────────────────────────────


def build_producer(bootstrap_servers: str, retries: int = RECONNECT_ATTEMPTS) -> KafkaProducer:
    """
    Build and return a KafkaProducer with JSON serialisation.
    Retries if the broker is not yet available.
    """
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=5,
                max_in_flight_requests_per_connection=5,
                linger_ms=10,            # batch small messages for throughput
                batch_size=65536,        # 64 KB batch
                compression_type="gzip",
                max_block_ms=10_000,
            )
            logger.info("Connected to Kafka at %s.", bootstrap_servers)
            return producer
        except NoBrokersAvailable:
            logger.warning(
                "Kafka not available (attempt %d/%d). Retrying in %ds …",
                attempt,
                retries,
                RECONNECT_DELAY_S,
            )
            time.sleep(RECONNECT_DELAY_S)

    raise RuntimeError(
        f"Could not connect to Kafka at {bootstrap_servers} after {retries} attempts."
    )


# ─── CSV reading ──────────────────────────────────────────────────────────────


def discover_csv_files(input_path: str) -> List[Path]:
    """Return all CSV files under input_path, sorted by name."""
    root = Path(input_path)
    if root.is_file() and root.suffix.lower() == ".csv":
        return [root]
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {input_path}")
    logger.info("Found %d CSV file(s) under %s.", len(files), input_path)
    return files


def _safe_cast(value: str, col_name: str) -> Optional[float]:
    """Attempt to cast a string to float; return None if the value is empty/invalid."""
    stripped = value.strip()
    if stripped in ("", "NA", "N/A", "null", "NULL"):
        return None
    try:
        return float(stripped)
    except ValueError:
        return stripped  # keep as string for unexpected values


def row_to_dict(row: Dict[str, str]) -> Optional[Dict]:
    """
    Transform a raw CSV row dict into a clean feature dict.
    Returns None if the row should be skipped (e.g. no ARR_DELAY).
    """
    record: Dict = {}
    for col in FEATURE_COLUMNS:
        raw_val = row.get(col, "")
        if col in NUMERIC_COLUMNS:
            record[col] = _safe_cast(raw_val, col)
        else:
            record[col] = raw_val.strip() if raw_val.strip() else None

    # Add a producer-side timestamp for latency measurement
    record["producer_ts"] = time.time()

    return record


def csv_record_generator(files: List[Path]) -> Generator[Dict, None, None]:
    """Yield one flight record dict per CSV row across all files."""
    for filepath in files:
        logger.info("Reading file: %s", filepath)
        with filepath.open(newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                record = row_to_dict(row)
                if record is not None:
                    yield record


# ─── Rate-limited producer loop ───────────────────────────────────────────────


def delivery_report(future):
    """Called asynchronously when Kafka confirms delivery."""
    try:
        record_metadata = future.get(timeout=10)
        # Uncomment for verbose per-message logging:
        # logger.debug("Delivered to %s [%d] offset %d",
        #              record_metadata.topic, record_metadata.partition,
        #              record_metadata.offset)
    except KafkaError as exc:
        logger.error("Delivery failed: %s", exc)


def produce_records(
    producer: KafkaProducer,
    records: Iterator[Dict],
    topic: str,
    rate: float,
) -> None:
    """
    Publish records to Kafka at the requested rate (messages/second).
    Uses a token-bucket-style approach for accurate pacing.
    """
    interval = 1.0 / rate if rate > 0 else 0.0
    total_sent = 0
    start_time = time.time()
    batch_start = time.time()

    logger.info(
        "Starting to produce to topic '%s' at %.1f msg/s. Press Ctrl+C to stop.",
        topic,
        rate,
    )

    for record in records:
        send_start = time.time()

        # Use ORIGIN as partition key so the same origin always goes to same partition
        key = record.get("ORIGIN") or "UNKNOWN"

        try:
            future = producer.send(topic, key=key, value=record)
            future.add_callback(lambda meta: None)   # fire-and-forget callbacks
            future.add_errback(lambda exc: logger.error("Send error: %s", exc))
        except KafkaError as exc:
            logger.error("Failed to send message: %s", exc)
            continue

        total_sent += 1

        # Progress logging
        if total_sent % LOG_INTERVAL == 0:
            elapsed = time.time() - batch_start
            throughput = LOG_INTERVAL / elapsed if elapsed > 0 else 0
            overall_elapsed = time.time() - start_time
            overall_throughput = total_sent / overall_elapsed if overall_elapsed > 0 else 0
            logger.info(
                "Sent %d messages | Last %d msgs: %.1f msg/s | Overall: %.1f msg/s",
                total_sent,
                LOG_INTERVAL,
                throughput,
                overall_throughput,
            )
            batch_start = time.time()

        # Rate limiting: sleep for the remaining time in the interval
        if interval > 0:
            elapsed_in_cycle = time.time() - send_start
            sleep_time = interval - elapsed_in_cycle
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Flush remaining messages
    logger.info("Flushing remaining messages …")
    producer.flush()

    total_elapsed = time.time() - start_time
    final_throughput = total_sent / total_elapsed if total_elapsed > 0 else 0
    logger.info(
        "Done. Total messages sent: %d in %.1f s (avg %.1f msg/s).",
        total_sent,
        total_elapsed,
        final_throughput,
    )


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kafka producer: replay 2024 BTS flight records as a JSON stream."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Local directory (or file) containing 2024 BTS CSV files.",
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default="localhost:9093",
        help="Kafka bootstrap server address (default: localhost:9093).",
    )
    parser.add_argument(
        "--topic",
        default="flight-events",
        help="Kafka topic name (default: flight-events).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Target publish rate in messages/second (default: 100).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop over the dataset indefinitely (useful for long-running tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Kafka producer starting.")
    logger.info("  Input path       : %s", args.input_path)
    logger.info("  Bootstrap server : %s", args.kafka_bootstrap)
    logger.info("  Topic            : %s", args.topic)
    logger.info("  Rate             : %.1f msg/s", args.rate)
    logger.info("  Loop             : %s", args.loop)

    producer = build_producer(args.kafka_bootstrap)
    csv_files = discover_csv_files(args.input_path)

    try:
        if args.loop:
            iteration = 0
            while True:
                iteration += 1
                logger.info("Starting dataset loop iteration %d …", iteration)
                records = csv_record_generator(csv_files)
                produce_records(producer, records, args.topic, args.rate)
        else:
            records = csv_record_generator(csv_files)
            produce_records(producer, records, args.topic, args.rate)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.exception("Producer error: %s", exc)
        sys.exit(1)
    finally:
        producer.close()
        logger.info("Producer closed.")


if __name__ == "__main__":
    main()

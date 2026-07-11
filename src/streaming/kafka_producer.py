"""
kafka_producer.py
-----------------
CANONICAL Kafka producer — this is the only producer invoked by
scripts/run_pipeline.sh (see the run_kafka_producer step) and the only one
that should be used or modified. An older, unrelated duplicate previously
lived at the repo root (src/kafka_producer.py); it has been moved to
deprecated/kafka_producer.py and is not wired into any entry point. Its
reservoir-sampling logic has been merged into this file (see
reservoir_sample() below), fixed to use a seeded RNG instead of the
unseeded global `random` module.

Replay BTS On-Time Performance records as a real-time JSON stream to a
Kafka topic, simulating live flight event ingestion. Optionally draws a
fixed-size, unbiased sample via streaming reservoir sampling (Algorithm R,
Vitter 1985) before publishing, so a benchmark run doesn't require
replaying an entire year of data.

Usage:
    # Full dataset, no sampling:
    python src/streaming/kafka_producer.py \
        --input-path /data/raw/bts/2024 \
        --kafka-bootstrap localhost:9093 \
        --topic flight-events \
        --rate 100 \
        --sample-size 0

    # Reservoir-sampled subset of 250,000 rows (default), seed=42 (default):
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
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

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

# Raw BTS CSV exports use mixed-case column names (e.g. "CRSDepTime",
# "DepDelay", "Reporting_Airline") rather than the standardized upper-case
# names in FEATURE_COLUMNS above. Without this mapping, every row.get(col)
# lookup below misses and silently defaults to None for every field — the
# row is "successfully" produced but carries no real data. Mirrors
# BTS_COLUMN_MAP in src/ingestion/ingest_bts_to_hdfs.py.
BTS_COLUMN_MAP = {
    "Year":               "YEAR",
    "Month":              "MONTH",
    "DayofMonth":         "DAY_OF_MONTH",
    "DayOfWeek":          "DAY_OF_WEEK",
    "Reporting_Airline":  "OP_UNIQUE_CARRIER",
    "Origin":             "ORIGIN",
    "Dest":               "DEST",
    "CRSDepTime":         "CRS_DEP_TIME",
    "DepDelay":           "DEP_DELAY",
    "CRSArrTime":         "CRS_ARR_TIME",
    "ArrDelay":           "ARR_DELAY",
    "CRSElapsedTime":     "CRS_ELAPSED_TIME",
    "Distance":           "DISTANCE",
    "CarrierDelay":       "CARRIER_DELAY",
    "WeatherDelay":       "WEATHER_DELAY",
    "NASDelay":           "NAS_DELAY",
    "SecurityDelay":      "SECURITY_DELAY",
    "LateAircraftDelay":  "LATE_AIRCRAFT_DELAY",
}

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
DEFAULT_SAMPLE_SIZE = 250_000
DEFAULT_SEED = 42
LOG_DIR = Path("logs")


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
    normalized_row = {BTS_COLUMN_MAP.get(k, k): v for k, v in row.items()}
    record: Dict = {}
    for col in FEATURE_COLUMNS:
        raw_val = normalized_row.get(col, "")
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


# ─── Reservoir sampling (Algorithm R, Vitter 1985) ────────────────────────────


def reservoir_sample(
    records: Iterator[Dict], k: int, seed: int
) -> Tuple[List[Dict], int]:
    """
    Single-pass, streaming reservoir sampling.

    Draws an unbiased simple random sample of exactly min(k, n) rows from
    `records` in one pass, keeping only O(k) items in memory regardless of
    how many rows the source contains — the reservoir never buffers the
    full dataset, unlike a pandas-based "load everything then sample"
    approach. Each element has exactly k/n probability of being in the
    final reservoir.

    Uses a seeded `random.Random(seed)` instance (not the unseeded global
    `random` module) so sampling is reproducible run over run.

    Returns: (sample, total_rows_seen)
    """
    rng = random.Random(seed)
    reservoir: List[Dict] = []
    n = 0
    for record in records:
        n += 1
        if len(reservoir) < k:
            reservoir.append(record)
        else:
            j = rng.randint(1, n)
            if j <= k:
                reservoir[j - 1] = record
    logger.info(
        "Reservoir sampling complete: %d rows drawn from %d (seed=%d, rate=%.2f%%).",
        len(reservoir), n, seed, (len(reservoir) / n * 100.0) if n else 0.0,
    )
    return reservoir, n


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
) -> int:
    """
    Publish records to Kafka at the requested rate (messages/second).
    Uses a token-bucket-style approach for accurate pacing.
    Returns the total number of messages sent.
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
    return total_sent


# ─── Run logging ───────────────────────────────────────────────────────────────


def run_log_path(run_timestamp: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"producer_run_{run_timestamp}.json"


def write_run_log(
    log_path: Path,
    run_timestamp: str,
    input_path: str,
    csv_files: List[Path],
    sample_size: int,
    seed: int,
    source_row_count: int,
    sample_actual_size: int,
    total_sent: int,
    topic: str,
    kafka_bootstrap: str,
    rate: float,
    status: str,
    error_message: Optional[str] = None,
) -> Path:
    """
    Persist producer run parameters/results to logs/producer_run_<ts>.json.

    Called more than once per run (see main()): once right after reservoir
    sampling completes (status="sampling_complete", before the potentially
    long-running publish loop starts) and again in the finally block with
    the final status. This means a run that hangs or is killed mid-publish
    (e.g. a stuck producer.send() due to broker/network trouble, followed by
    an operator `kill`/`docker stop`) still leaves a log behind recording the
    intended sample size, seed, and source row count — the fields needed to
    reproduce or debug the run — rather than losing that information because
    only a single end-of-run write existed. Each call overwrites the same
    file (keyed by run_timestamp) rather than creating a new one, so the log
    reflects the latest known state of this run, not a fragment of it.
    """
    payload = {
        "run_timestamp_utc": run_timestamp,
        "status": status,
        "error_message": error_message,
        "input_path": input_path,
        "csv_files": [str(f) for f in csv_files],
        "requested_sample_size": sample_size,
        "seed": seed,
        "source_row_count": source_row_count,
        "sample_actual_size": sample_actual_size,
        "total_messages_sent": total_sent,
        "topic": topic,
        "kafka_bootstrap": kafka_bootstrap,
        "rate_msg_per_sec": rate,
    }

    with log_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Run log written to %s (status=%s)", log_path, status)
    return log_path


# ─── Entry point ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kafka producer: replay BTS flight records as a JSON stream."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Local directory (or file) containing BTS CSV files.",
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
        help="Loop over the dataset indefinitely (useful for long-running tests). "
             "Not compatible with --sample-size > 0.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Reservoir sample size (default: {DEFAULT_SAMPLE_SIZE:,}). "
             "0 = use the full dataset with no sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reservoir sampling, for reproducibility (default: {DEFAULT_SEED}).",
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
    logger.info("  Sample size      : %s", args.sample_size if args.sample_size > 0 else "disabled (full dataset)")
    logger.info("  Seed             : %d", args.seed)

    if args.loop and args.sample_size > 0:
        logger.warning("--loop with --sample-size > 0 will resample a new reservoir on every iteration.")

    run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_path = run_log_path(run_timestamp)

    def log_now(status: str, error_message: Optional[str] = None) -> None:
        write_run_log(
            log_path=log_path,
            run_timestamp=run_timestamp,
            input_path=args.input_path,
            csv_files=csv_files,
            sample_size=args.sample_size,
            seed=args.seed,
            source_row_count=source_row_count,
            sample_actual_size=sample_actual_size,
            total_sent=total_sent,
            topic=args.topic,
            kafka_bootstrap=args.kafka_bootstrap,
            rate=args.rate,
            status=status,
            error_message=error_message,
        )

    # SIGTERM (what `kill <pid>` and `docker stop` send by default) has no
    # handler in Python by default and kills the process immediately without
    # running any `finally` block, which previously meant a hung or killed
    # run left no log at all. Translating it into a normal Python exception
    # here lets the existing try/except/finally below run as usual, so a
    # `kill`'d run still gets a "killed" status logged. SIGKILL (`kill -9`)
    # cannot be caught by any process and remains unrecoverable — no code
    # can protect against that.
    def _handle_sigterm(signum, frame):
        raise SystemExit(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_sigterm)

    producer = build_producer(args.kafka_bootstrap)
    csv_files = discover_csv_files(args.input_path)

    source_row_count = 0
    sample_actual_size = 0
    total_sent = 0
    status = "success"
    error_message: Optional[str] = None

    try:
        if args.loop:
            iteration = 0
            while True:
                iteration += 1
                logger.info("Starting dataset loop iteration %d …", iteration)
                if args.sample_size > 0:
                    sample, source_row_count = reservoir_sample(
                        csv_record_generator(csv_files), args.sample_size, args.seed
                    )
                    sample_actual_size = len(sample)
                    log_now("sampling_complete")
                    total_sent = produce_records(producer, iter(sample), args.topic, args.rate)
                else:
                    total_sent = produce_records(
                        producer, csv_record_generator(csv_files), args.topic, args.rate
                    )
                    source_row_count = total_sent
        elif args.sample_size > 0:
            sample, source_row_count = reservoir_sample(
                csv_record_generator(csv_files), args.sample_size, args.seed
            )
            sample_actual_size = len(sample)
            # Interim checkpoint: written before the publish loop (which can
            # run for a long time, or hang on broker/network trouble) so the
            # sample size/seed/source count are captured even if the process
            # is killed before publishing finishes.
            log_now("sampling_complete")
            total_sent = produce_records(producer, iter(sample), args.topic, args.rate)
        else:
            total_sent = produce_records(
                producer, csv_record_generator(csv_files), args.topic, args.rate
            )
            source_row_count = total_sent
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        status = "interrupted"
        error_message = "KeyboardInterrupt"
    except SystemExit as exc:
        logger.info("Shutting down: %s", exc)
        status = "killed"
        error_message = str(exc)
    except Exception as exc:
        logger.exception("Producer error: %s", exc)
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            producer.close()
            logger.info("Producer closed.")
        except Exception as exc:
            logger.warning("Error closing producer: %s", exc)
        log_now(status, error_message)

    if status != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()

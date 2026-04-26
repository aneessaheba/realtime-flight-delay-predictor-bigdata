#!/usr/bin/env python3
"""
smoke_test.py  –  Kartheek Alluri (Weeks 7-8)
End-to-end smoke test for the real-time flight delay prediction pipeline.

Usage:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --kafka localhost:9093 --events 100 --wait 60
"""

import argparse
import json
import sys
import time
import random

import requests

try:
    from kafka import KafkaProducer, KafkaAdminClient
    from kafka.errors import NoBrokersAvailable
    KAFKA_OK = True
except ImportError:
    KAFKA_OK = False

try:
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    ARROW_OK = True
except ImportError:
    ARROW_OK = False

TOPIC = "flight-events"
CARRIERS = ["AA", "DL", "UA", "WN", "AS", "B6", "NK", "F9"]
AIRPORTS = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO"]


def ok(msg):   print(f"  \u2713  {msg}")
def fail(msg): print(f"  \u2717  {msg}"); return False
def warn(msg): print(f"  \u26a0  {msg}")


def synthetic_flight(label=None):
    dep_delay = round(random.gauss(5, 30), 1)
    return {
        "FL_DATE": "2024-06-15",
        "OP_UNIQUE_CARRIER": random.choice(CARRIERS),
        "ORIGIN": random.choice(AIRPORTS),
        "DEST":   random.choice(AIRPORTS),
        "CRS_DEP_TIME":        float(random.randint(500, 2200)),
        "DEP_TIME":            float(random.randint(500, 2200)),
        "DEP_DELAY":           dep_delay,
        "CRS_ARR_TIME":        float(random.randint(600, 2359)),
        "CRS_ELAPSED_TIME":    float(random.randint(60, 360)),
        "ACTUAL_ELAPSED_TIME": float(random.randint(60, 360)),
        "AIR_TIME":            float(random.randint(40, 320)),
        "DISTANCE":            float(random.randint(150, 2800)),
        "_event_ts":           time.time(),
        "_label":              label if label is not None else (1 if dep_delay > 15 else 0),
    }


def check_kafka(broker, n_events):
    if not KAFKA_OK:
        warn("kafka-python not installed; skipping Kafka checks.")
        return True
    print("\n[1] Kafka connectivity")
    try:
        admin = KafkaAdminClient(bootstrap_servers=[broker], request_timeout_ms=5000)
        topics = admin.list_topics()
        admin.close()
    except NoBrokersAvailable:
        return fail(f"Cannot reach Kafka at {broker}")
    except Exception as exc:
        return fail(f"Kafka admin error: {exc}")

    if TOPIC not in topics:
        warn(f"Topic '{TOPIC}' not found — will be auto-created.")
    else:
        ok(f"Topic '{TOPIC}' exists.")

    try:
        producer = KafkaProducer(
            bootstrap_servers=[broker],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all", request_timeout_ms=10000,
        )
        for i in range(n_events):
            producer.send(TOPIC, value=synthetic_flight(label=1 if i % 5 == 0 else 0))
        producer.flush()
        producer.close()
        ok(f"Published {n_events} synthetic events to '{TOPIC}'.")
    except Exception as exc:
        return fail(f"Failed to publish test events: {exc}")
    return True


def check_hdfs(hdfs_url):
    print("\n[2] HDFS namenode")
    try:
        resp = requests.get(
            f"{hdfs_url}/jmx?qry=Hadoop:service=NameNode,name=NameNodeStatus",
            timeout=5)
        if resp.status_code == 200:
            state = resp.json().get("beans", [{}])[0].get("State", "unknown")
            ok(f"Namenode state: {state}")
            return True
        return fail(f"Namenode returned HTTP {resp.status_code}")
    except Exception as exc:
        return fail(f"Cannot reach HDFS at {hdfs_url}: {exc}")


def check_spark(spark_url):
    print("\n[3] Spark master")
    try:
        resp = requests.get(f"{spark_url}/json/", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            ok(f"Spark alive. Workers={info.get('aliveworkers',0)}, "
               f"Cores used={info.get('coresused',0)}")
            return True
        return fail(f"Spark master returned HTTP {resp.status_code}")
    except Exception as exc:
        return fail(f"Cannot reach Spark at {spark_url}: {exc}")


def poll_predictions(output_path, wait_s):
    print(f"\n[4] Polling predictions at {output_path}  (timeout={wait_s}s)")
    if not ARROW_OK:
        warn("pyarrow not installed; skipping prediction read.")
        warn(f"Check manually:  hdfs dfs -ls {output_path}")
        return True

    if output_path.startswith("hdfs://"):
        rest = output_path[len("hdfs://"):]
        slash = rest.find("/")
        host_port = rest[:slash]
        fs_path = rest[slash:]
        host, port = (host_port.split(":") + ["9000"])[:2]
        try:
            fs = pafs.HadoopFileSystem(host=host, port=int(port))
        except Exception as exc:
            warn(f"PyArrow HDFS connection failed: {exc}")
            return True
    else:
        fs = pafs.LocalFileSystem()
        fs_path = output_path

    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            selector = pafs.FileSelector(fs_path, recursive=True)
            files = [f.path for f in fs.get_file_info(selector)
                     if f.path.endswith(".parquet")]
            if files:
                ok(f"Found {len(files)} Parquet file(s).")
                return validate_predictions(fs, files)
        except Exception:
            pass
        print(f"  ... waiting ({int(deadline - time.time())}s left)", end="\r")
        time.sleep(5)
    return fail("No prediction files appeared within timeout.")


def validate_predictions(fs, files):
    print("\n[5] Validating predictions")
    try:
        df = pq.read_table(files[0], filesystem=fs).to_pandas()
    except Exception as exc:
        return fail(f"Could not read Parquet: {exc}")

    passed = True
    if "prediction" not in df.columns:
        fail("'prediction' column missing.")
        passed = False
    else:
        unique_preds = set(df["prediction"].dropna().unique())
        if not unique_preds.issubset({0.0, 1.0}):
            fail(f"Unexpected prediction values: {unique_preds}")
            passed = False
        else:
            ok(f"Predictions binary (0/1). Delayed={df['prediction'].sum():.0f}/{len(df)}")

    if "end_to_end_latency_s" in df.columns:
        avg_lat = df["end_to_end_latency_s"].mean()
        max_lat = df["end_to_end_latency_s"].max()
        if avg_lat <= 2.0:
            ok(f"Latency SLA met: avg={avg_lat:.3f}s  max={max_lat:.3f}s  (target<=2s)")
        else:
            warn(f"Latency above target: avg={avg_lat:.3f}s  max={max_lat:.3f}s")

    if "_label" in df.columns and "prediction" in df.columns:
        labelled = df.dropna(subset=["_label", "prediction"])
        if len(labelled) > 0:
            acc = (labelled["prediction"] == labelled["_label"]).mean()
            ok(f"Accuracy on labelled rows: {acc:.4f}  (n={len(labelled)})")

    return passed


def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke test")
    parser.add_argument("--kafka",      default="localhost:9093")
    parser.add_argument("--hdfs-url",   default="http://localhost:9870")
    parser.add_argument("--spark-url",  default="http://localhost:8080")
    parser.add_argument("--output",     default="hdfs://hdfs-namenode:9000/output/streaming_predictions")
    parser.add_argument("--events",     type=int, default=100)
    parser.add_argument("--wait",       type=int, default=60)
    parser.add_argument("--skip-kafka",   action="store_true")
    parser.add_argument("--skip-hdfs",    action="store_true")
    parser.add_argument("--skip-spark",   action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Flight Delay Predictor — End-to-End Smoke Test")
    print("=" * 60)

    results = {
        "kafka":       check_kafka(args.kafka, args.events)       if not args.skip_kafka   else (warn("skipped") or True),
        "hdfs":        check_hdfs(args.hdfs_url)                  if not args.skip_hdfs    else (warn("skipped") or True),
        "spark":       check_spark(args.spark_url)                if not args.skip_spark   else (warn("skipped") or True),
        "predictions": poll_predictions(args.output, args.wait)   if not args.skip_predict else (warn("skipped") or True),
    }

    print("\n" + "=" * 60)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print("=" * 60)
    all_pass = all(results.values())
    print("  SMOKE TEST PASSED" if all_pass else "  SMOKE TEST FAILED")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

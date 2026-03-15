#!/usr/bin/env bash
# run_pipeline.sh
# ---------------
# Full end-to-end pipeline orchestration for the real-time flight delay
# prediction system.
#
# Steps:
#   1. Set up HDFS directories
#   2. Ingest BTS CSV data into HDFS
#   3. Train LR + GBT models
#   4. Start Spark Structured Streaming consumer (background)
#   5. Start Kafka producer (streams 2024 data)
#   6. Run batch inference on test data
#   7. Run benchmark comparison
#
# Usage:
#   bash scripts/run_pipeline.sh [--input-path <path>] [--skip-ingest] \
#                                [--skip-training] [--skip-streaming]
#
# Environment variables (with defaults):
#   BTS_INPUT_PATH      - Local path to BTS CSV files       (./data/raw)
#   HDFS_DATA_PATH      - HDFS path for flight data         (hdfs://namenode:9000/data/flights)
#   HDFS_MODEL_PATH     - HDFS path for models              (hdfs://namenode:9000/models)
#   SPARK_MASTER        - Spark master URL                  (spark://spark-master:7077)
#   KAFKA_BOOTSTRAP     - Kafka bootstrap server            (localhost:9093)
#   KAFKA_TOPIC         - Kafka topic name                  (flight-events)
#   PRODUCER_RATE       - Kafka producer rate (msg/s)       (100)
#   TRAIN_YEARS         - Space-separated list of years     (2018 2019 2020 2021 2022 2023)
#   TEST_YEARS          - Space-separated list of test yrs  (2024)
#   LOG_DIR             - Directory for log files           (./logs)
#   OUTPUT_DIR          - Local output directory            (./output)

set -euo pipefail

# ─── Default configuration ────────────────────────────────────────────────────

BTS_INPUT_PATH="${BTS_INPUT_PATH:-./data/raw}"
HDFS_DATA_PATH="${HDFS_DATA_PATH:-hdfs://namenode:9000/data/flights}"
HDFS_MODEL_PATH="${HDFS_MODEL_PATH:-hdfs://namenode:9000/models}"
HDFS_OUTPUT_PATH="${HDFS_OUTPUT_PATH:-hdfs://namenode:9000/output}"
HDFS_CHECKPOINT_PATH="${HDFS_CHECKPOINT_PATH:-hdfs://namenode:9000/checkpoints/streaming}"
SPARK_MASTER="${SPARK_MASTER:-spark://spark-master:7077}"
KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP:-localhost:9093}"
KAFKA_TOPIC="${KAFKA_TOPIC:-flight-events}"
PRODUCER_RATE="${PRODUCER_RATE:-100}"
TRAIN_YEARS="${TRAIN_YEARS:-2018 2019 2020 2021 2022 2023}"
TEST_YEARS="${TEST_YEARS:-2024}"
LOG_DIR="${LOG_DIR:-./logs}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# spark-submit packages for Kafka connector
KAFKA_PACKAGE="org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"

# Flags
SKIP_INGEST=false
SKIP_TRAINING=false
SKIP_STREAMING=false
SKIP_BENCHMARK=false

# ─── Argument parsing ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-path)   BTS_INPUT_PATH="$2"; shift 2 ;;
        --skip-ingest)  SKIP_INGEST=true; shift ;;
        --skip-training) SKIP_TRAINING=true; shift ;;
        --skip-streaming) SKIP_STREAMING=true; shift ;;
        --skip-benchmark) SKIP_BENCHMARK=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--input-path <path>] [--skip-ingest] [--skip-training]"
            echo "          [--skip-streaming] [--skip-benchmark]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ─── Logging helpers ──────────────────────────────────────────────────────────

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

PIPELINE_LOG="${LOG_DIR}/pipeline_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

log()     { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]    $*"; }
log_step(){ echo ""; echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP]    ══════ $* ══════"; }
log_ok()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*"; }
warn()    { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]    $*" >&2; }
err()     { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR]   $*" >&2; }

# Track timing
STEP_START=0
step_start() { STEP_START=$(date +%s); }
step_end()   {
    local elapsed=$(( $(date +%s) - STEP_START ))
    log_ok "Step completed in ${elapsed}s."
}

# ─── Error handling ───────────────────────────────────────────────────────────

CONSUMER_PID=""

cleanup() {
    local exit_code=$?
    if [[ -n "${CONSUMER_PID}" ]] && kill -0 "${CONSUMER_PID}" 2>/dev/null; then
        log "Stopping streaming consumer (PID ${CONSUMER_PID}) …"
        kill "${CONSUMER_PID}" 2>/dev/null || true
        wait "${CONSUMER_PID}" 2>/dev/null || true
    fi
    if [[ ${exit_code} -ne 0 ]]; then
        err "Pipeline failed with exit code ${exit_code}. Check ${PIPELINE_LOG} for details."
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

# ─── Prerequisite checks ──────────────────────────────────────────────────────

check_prerequisites() {
    log_step "Checking prerequisites"
    step_start

    local missing=()

    command -v spark-submit &>/dev/null || missing+=("spark-submit")
    command -v python3       &>/dev/null || missing+=("python3")
    command -v docker        &>/dev/null || missing+=("docker")
    command -v docker-compose &>/dev/null || command -v docker &>/dev/null || missing+=("docker-compose")

    if [[ ${#missing[@]} -gt 0 ]]; then
        err "Missing required commands: ${missing[*]}"
        err "Please install them and ensure they are on PATH."
        exit 1
    fi

    if [[ ! -d "${BTS_INPUT_PATH}" ]]; then
        warn "BTS input path does not exist: ${BTS_INPUT_PATH}"
        warn "Create the directory and place BTS CSV files there, or set BTS_INPUT_PATH."
    fi

    log "All prerequisites satisfied."
    step_end
}

# ─── Step 1: HDFS setup ───────────────────────────────────────────────────────

setup_hdfs() {
    log_step "Step 1 – Setting up HDFS directories"
    step_start
    bash "$(dirname "$0")/setup_hdfs.sh"
    step_end
}

# ─── Step 2: Data ingestion ───────────────────────────────────────────────────

ingest_data() {
    if [[ "${SKIP_INGEST}" == "true" ]]; then
        log "Skipping data ingestion (--skip-ingest)."
        return
    fi

    log_step "Step 2 – Ingesting BTS CSV data into HDFS"
    step_start

    # shellcheck disable=SC2086
    spark-submit \
        --master "${SPARK_MASTER}" \
        --deploy-mode client \
        --driver-memory 4g \
        --executor-memory 6g \
        --executor-cores 4 \
        --conf spark.sql.shuffle.partitions=200 \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        src/ingestion/ingest_bts_to_hdfs.py \
        --input-path "${BTS_INPUT_PATH}" \
        --hdfs-path "${HDFS_DATA_PATH}" \
        --years ${TRAIN_YEARS} ${TEST_YEARS} \
        2>&1 | tee "${LOG_DIR}/ingest.log"

    log "Ingestion log saved to ${LOG_DIR}/ingest.log"
    step_end
}

# ─── Step 3: Model training ───────────────────────────────────────────────────

train_models() {
    if [[ "${SKIP_TRAINING}" == "true" ]]; then
        log "Skipping model training (--skip-training)."
        return
    fi

    log_step "Step 3 – Training LR and GBT models"
    step_start

    # shellcheck disable=SC2086
    spark-submit \
        --master "${SPARK_MASTER}" \
        --deploy-mode client \
        --driver-memory 4g \
        --executor-memory 8g \
        --executor-cores 4 \
        --conf spark.sql.shuffle.partitions=200 \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        src/training/train_model.py \
        --hdfs-path "${HDFS_DATA_PATH}" \
        --model-path "${HDFS_MODEL_PATH}" \
        --train-years ${TRAIN_YEARS} \
        --cv-folds 5 \
        2>&1 | tee "${LOG_DIR}/training.log"

    log "Training log saved to ${LOG_DIR}/training.log"
    step_end
}

# ─── Step 4: Start streaming consumer ────────────────────────────────────────

start_streaming_consumer() {
    if [[ "${SKIP_STREAMING}" == "true" ]]; then
        log "Skipping streaming consumer (--skip-streaming)."
        return
    fi

    log_step "Step 4 – Starting Spark Structured Streaming consumer (background)"
    step_start

    spark-submit \
        --master "${SPARK_MASTER}" \
        --deploy-mode client \
        --driver-memory 4g \
        --executor-memory 6g \
        --executor-cores 4 \
        --packages "${KAFKA_PACKAGE}" \
        --conf spark.sql.shuffle.partitions=50 \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        src/streaming/streaming_consumer.py \
        --kafka-bootstrap "${KAFKA_BOOTSTRAP}" \
        --topic "${KAFKA_TOPIC}" \
        --model-path "${HDFS_MODEL_PATH}/gbt_pipeline" \
        --output-path "${HDFS_OUTPUT_PATH}/streaming_predictions" \
        --checkpoint-path "${HDFS_CHECKPOINT_PATH}" \
        --trigger-interval "10 seconds" \
        > "${LOG_DIR}/streaming_consumer.log" 2>&1 &

    CONSUMER_PID=$!
    log "Streaming consumer started with PID ${CONSUMER_PID}."
    log "Consumer log: ${LOG_DIR}/streaming_consumer.log"

    # Give the consumer time to initialise before the producer starts
    log "Waiting 30s for consumer to initialise …"
    sleep 30

    if ! kill -0 "${CONSUMER_PID}" 2>/dev/null; then
        err "Streaming consumer process died unexpectedly. Check ${LOG_DIR}/streaming_consumer.log"
        exit 1
    fi

    step_end
}

# ─── Step 5: Run Kafka producer ───────────────────────────────────────────────

run_kafka_producer() {
    if [[ "${SKIP_STREAMING}" == "true" ]]; then
        log "Skipping Kafka producer (--skip-streaming)."
        return
    fi

    log_step "Step 5 – Running Kafka producer (2024 BTS data)"
    step_start

    local producer_input="${BTS_INPUT_PATH}/2024"
    if [[ ! -d "${producer_input}" ]]; then
        producer_input="${BTS_INPUT_PATH}"
        warn "2024 sub-directory not found; using ${producer_input}"
    fi

    python3 src/streaming/kafka_producer.py \
        --input-path "${producer_input}" \
        --kafka-bootstrap "${KAFKA_BOOTSTRAP}" \
        --topic "${KAFKA_TOPIC}" \
        --rate "${PRODUCER_RATE}" \
        2>&1 | tee "${LOG_DIR}/kafka_producer.log"

    log "Producer log saved to ${LOG_DIR}/kafka_producer.log"

    # Allow the consumer time to process the remaining micro-batches
    log "Waiting 60s for consumer to process remaining batches …"
    sleep 60

    # Gracefully stop the streaming consumer
    if [[ -n "${CONSUMER_PID}" ]] && kill -0 "${CONSUMER_PID}" 2>/dev/null; then
        log "Stopping streaming consumer (PID ${CONSUMER_PID}) …"
        kill -TERM "${CONSUMER_PID}" 2>/dev/null || true
        wait "${CONSUMER_PID}" 2>/dev/null || true
        CONSUMER_PID=""
        log "Streaming consumer stopped."
    fi

    step_end
}

# ─── Step 6: Batch inference ──────────────────────────────────────────────────

run_batch_inference() {
    log_step "Step 6 – Running batch inference on test data"
    step_start

    # shellcheck disable=SC2086
    spark-submit \
        --master "${SPARK_MASTER}" \
        --deploy-mode client \
        --driver-memory 4g \
        --executor-memory 6g \
        --executor-cores 4 \
        --conf spark.sql.shuffle.partitions=200 \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        src/batch/batch_inference.py \
        --data-path "${HDFS_DATA_PATH}" \
        --model-path "${HDFS_MODEL_PATH}/gbt_pipeline" \
        --output-path "${HDFS_OUTPUT_PATH}/batch_predictions" \
        --test-years ${TEST_YEARS} \
        --metrics-json "${OUTPUT_DIR}/batch_metrics.json" \
        2>&1 | tee "${LOG_DIR}/batch_inference.log"

    log "Batch inference log saved to ${LOG_DIR}/batch_inference.log"
    step_end
}

# ─── Step 7: Benchmark ────────────────────────────────────────────────────────

run_benchmark() {
    if [[ "${SKIP_BENCHMARK}" == "true" ]]; then
        log "Skipping benchmark (--skip-benchmark)."
        return
    fi

    log_step "Step 7 – Running benchmark comparison"
    step_start

    spark-submit \
        --master "${SPARK_MASTER}" \
        --deploy-mode client \
        --driver-memory 4g \
        --executor-memory 4g \
        --executor-cores 2 \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        src/evaluation/benchmark.py \
        --batch-path "${HDFS_OUTPUT_PATH}/batch_predictions" \
        --streaming-path "${HDFS_OUTPUT_PATH}/streaming_predictions" \
        --report-path "${OUTPUT_DIR}/benchmark_report.json" \
        --spark-master "${SPARK_MASTER}" \
        2>&1 | tee "${LOG_DIR}/benchmark.log"

    log "Benchmark log saved to ${LOG_DIR}/benchmark.log"
    log "Benchmark report saved to ${OUTPUT_DIR}/benchmark_report.json"
    step_end
}

# ─── Summary ─────────────────────────────────────────────────────────────────

print_summary() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  PIPELINE COMPLETED SUCCESSFULLY"
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Logs directory    : ${LOG_DIR}"
    echo "  Output directory  : ${OUTPUT_DIR}"
    echo "  Benchmark report  : ${OUTPUT_DIR}/benchmark_report.json"
    echo "  Pipeline log      : ${PIPELINE_LOG}"
    echo ""
    echo "  Key HDFS paths:"
    echo "    Data            : ${HDFS_DATA_PATH}"
    echo "    Models          : ${HDFS_MODEL_PATH}"
    echo "    Batch output    : ${HDFS_OUTPUT_PATH}/batch_predictions"
    echo "    Streaming output: ${HDFS_OUTPUT_PATH}/streaming_predictions"
    echo "════════════════════════════════════════════════════════════════════"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

PIPELINE_START=$(date +%s)

log "Pipeline started at $(date)"
log "Pipeline log: ${PIPELINE_LOG}"

check_prerequisites
setup_hdfs
ingest_data
train_models
start_streaming_consumer
run_kafka_producer
run_batch_inference
run_benchmark

PIPELINE_END=$(date +%s)
PIPELINE_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
log "Total pipeline duration: ${PIPELINE_ELAPSED}s ($(( PIPELINE_ELAPSED / 60 ))m $(( PIPELINE_ELAPSED % 60 ))s)"

print_summary

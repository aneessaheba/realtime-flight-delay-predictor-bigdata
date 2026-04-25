#!/usr/bin/env bash
# run_feature_pipeline.sh
# -----------------------
# Batch feature engineering pipeline for flight delay prediction.
#
# Steps:
#   1. Reads ingested flight data from HDFS
#   2. Cleans and filters the dataset
#   3. Performs feature engineering and encoding
#   4. Assembles ML-ready feature vectors
#   5. Saves a reusable Spark ML preprocessing pipeline
#   6. Runs EDA summaries on the cleaned dataset
#
# Usage:
#   bash scripts/run_feature_pipeline.sh [input_path] [cleaned_output] \
#                                        [featured_output] [pipeline_output] \
#                                        [eda_output]

set -euo pipefail

INPUT_PATH=${1:-hdfs://hdfs-namenode:9000/data/flights}
CLEANED_OUTPUT=${2:-hdfs://hdfs-namenode:9000/data/flights/cleaned}
FEATURED_OUTPUT=${3:-hdfs://hdfs-namenode:9000/data/flights/featured}
PIPELINE_OUTPUT=${4:-hdfs://hdfs-namenode:9000/models/preprocessing_pipeline}
EDA_OUTPUT=${5:-hdfs://hdfs-namenode:9000/output/eda}

SPARK_MASTER="spark://spark-master:7077"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log "Starting feature preparation..."
log "Input          : ${INPUT_PATH}"
log "Cleaned output : ${CLEANED_OUTPUT}"
log "Featured output: ${FEATURED_OUTPUT}"
log "Pipeline output: ${PIPELINE_OUTPUT}"
log "EDA output     : ${EDA_OUTPUT}"

spark-submit \
  --master "${SPARK_MASTER}" \
  --deploy-mode client \
  --driver-memory 4g \
  --executor-memory 6g \
  --conf spark.hadoop.fs.defaultFS=hdfs://hdfs-namenode:9000 \
  src/training/prepare_features.py \
  --input "${INPUT_PATH}" \
  --cleaned-output "${CLEANED_OUTPUT}" \
  --featured-output "${FEATURED_OUTPUT}" \
  --pipeline-model-output "${PIPELINE_OUTPUT}"

log "Feature preparation completed."
log "Starting EDA..."

spark-submit \
  --master "${SPARK_MASTER}" \
  --deploy-mode client \
  --driver-memory 2g \
  --executor-memory 4g \
  --conf spark.hadoop.fs.defaultFS=hdfs://hdfs-namenode:9000 \
  src/training/eda_report.py \
  --input "${CLEANED_OUTPUT}" \
  --output "${EDA_OUTPUT}"

log "EDA completed."
log "Feature pipeline finished successfully."
log "Outputs:"
log "  Cleaned data  : ${CLEANED_OUTPUT}"
log "  Featured data : ${FEATURED_OUTPUT}"
log "  Pipeline model: ${PIPELINE_OUTPUT}"
log "  EDA reports   : ${EDA_OUTPUT}"

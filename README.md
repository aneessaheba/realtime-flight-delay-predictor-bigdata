# Real-Time Flight Delay Prediction

End-to-end pipeline for flight delay prediction using Apache Spark, Kafka, and HDFS. Historical BTS flight data is ingested and stored in HDFS, used to train ML models (Logistic Regression + GBT), then predictions are served in real time through a Kafka streaming pipeline.

## Stack

- **Storage**: HDFS (Hadoop 3.2, Docker)
- **Processing**: Apache Spark 3.5 (PySpark)
- **Streaming**: Apache Kafka 3.6 + Spark Structured Streaming
- **ML**: Spark MLlib — Logistic Regression (baseline) + Gradient Boosted Trees (primary)
- **Dataset**: BTS Airline On-Time Performance (2018–2024)

## Quick start (local, no Docker)

```bash
chmod +x run_local.sh
./run_local.sh                          # generates 100k synthetic rows + trains
./run_local.sh path/to/bts_data.csv    # uses real BTS data
```

Artifacts written to `models/` and `plots/`.

## Full cluster setup (Docker)

```bash
docker-compose up -d
bash scripts/setup_hdfs.sh        # wait for namenode, create HDFS dirs
bash scripts/run_pipeline.sh      # ingest → train → stream → batch → benchmark
```

### Step-by-step

**1. Ingest BTS CSVs to HDFS**

```bash
python src/ingestion/ingest_bts_to_hdfs.py \
  --input-path data/raw \
  --hdfs-path hdfs://hdfs-namenode:9000/data/flights \
  --years 2018 2019 2020 2021 2022 2023
```

**2. Feature engineering + EDA**

```bash
bash scripts/run_feature_pipeline.sh
```

Or manually inside the Spark container:

```bash
docker exec -it spark-master bash
spark-submit src/training/prepare_features.py \
  --input hdfs://hdfs-namenode:9000/data/flights \
  --cleaned-output hdfs://hdfs-namenode:9000/data/flights/cleaned \
  --featured-output hdfs://hdfs-namenode:9000/data/flights/featured \
  --pipeline-model-output hdfs://hdfs-namenode:9000/models/preprocessing_pipeline

spark-submit src/training/eda_report.py \
  --input hdfs://hdfs-namenode:9000/data/flights/cleaned \
  --output hdfs://hdfs-namenode:9000/output/eda
```

**3. Train models**

```bash
spark-submit src/training/train_model.py
```

Models saved to `hdfs://hdfs-namenode:9000/models/`.

**4. Streaming inference**

```bash
# Start consumer (background)
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  src/streaming/streaming_consumer.py &

# Start producer
python src/streaming/kafka_producer.py
```

**5. Batch inference**

```bash
spark-submit src/batch/batch_inference.py
```

**6. Benchmark**

```bash
python src/evaluation/benchmark.py
```

## HDFS directory structure

```
/data/flights/          ← raw ingested Parquet (partitioned by YEAR/MONTH)
/data/flights/cleaned/  ← cleaned dataset
/data/flights/featured/ ← feature-engineered dataset
/models/                ← trained ML pipelines
/models/preprocessing_pipeline/
/output/streaming_predictions/
/output/batch_predictions/
/output/eda/            ← EDA CSV reports
/checkpoints/streaming/ ← Spark Streaming checkpoints
```

## HDFS verification

```bash
docker exec -it hdfs-namenode bash
hdfs dfs -ls /data/flights
hdfs dfs -ls /models
```

## Dataset

Bureau of Transportation Statistics Airline On-Time Performance dataset.
Download from: https://www.transtats.bts.gov/DL_SelectFields.aspx

Required columns: `YEAR`, `MONTH`, `DAY_OF_MONTH`, `DAY_OF_WEEK`, `OP_UNIQUE_CARRIER`,
`ORIGIN`, `DEST`, `CRS_DEP_TIME`, `DEP_DELAY`, `CRS_ARR_TIME`, `ARR_DELAY`,
`CRS_ELAPSED_TIME`, `DISTANCE`, `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`,
`SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`

## Target

Binary classification: `ARR_DELAY > 15 minutes` → **delayed (1)**, otherwise **on-time (0)**

Mid-term target: GBT F1 ≥ 0.70

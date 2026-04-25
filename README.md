# Real-Time Flight Delay Prediction

SJSU DATA-228 (Big Data Technologies) — Spring 2026
Course project demonstrating HDFS + Apache Kafka + Spark Structured Streaming.

End-to-end pipeline that ingests historical BTS flight data, trains ML models (Logistic Regression + GBT), and benchmarks streaming inference against batch inference on the same data and model.

---

## Team

| Name | Role |
|------|------|
| Anees Saheba Guddi | Evaluation & Reporting — benchmark design, visualizations, final report |
| Kartheek Alluri | Data & Infrastructure — BTS acquisition, HDFS setup |
| Keon Sadeghi | Streaming & Kafka — producer, consumer, integration |
| Manjot Kaur | ML Engineering — EDA, feature engineering, local training |
| Rish Jain | Integration & DevOps — Docker Compose, feature pipeline, end-to-end testing |

---

## What we've built

### Infrastructure
- **Docker Compose cluster** — 7 services: Zookeeper, Kafka 3.6, Spark master + 2 workers (Spark 3.5), HDFS namenode + datanode (Hadoop 3.2)
- **`scripts/setup_hdfs.sh`** — waits for namenode readiness via JMX, creates all required HDFS directories, sets permissions
- **`scripts/run_pipeline.sh`** — end-to-end orchestration: HDFS setup → ingest → train → streaming consumer → Kafka producer → batch inference → benchmark
- **`scripts/run_feature_pipeline.sh`** — runs feature engineering + EDA as a standalone step

### Data ingestion (`src/ingestion/`)
- **`ingest_bts_to_hdfs.py`** — reads raw BTS CSVs, renames mixed-case headers to a consistent schema, filters cancelled/diverted flights, casts types, and writes Parquet partitioned by `YEAR/MONTH` to HDFS

### Feature engineering & EDA (`src/training/`)
- **`prepare_features.py`** — reads ingested Parquet from HDFS, removes data-leakage columns (post-flight actuals), creates derived features (`dep_hour`, `arr_sched_hour`, `is_weekend`, `is_holiday_season`, `route`), handles missing values, encodes categoricals, saves a reusable Spark ML preprocessing pipeline to HDFS
- **`eda_report.py`** — generates CSV reports to HDFS: summary statistics, missing-value counts, label distribution, delay rates by carrier / origin / destination / month / day-of-week / departure hour
- **`eda_analysis.py`** — local pandas + matplotlib/seaborn analysis producing 7 PNG plots: class distribution, arrival delay distribution, delay by month, delay by day of week, delay by carrier, delay by departure hour, correlation heatmap

### Model training (`src/training/`)
- **`train_model.py`** — full Spark MLlib training on HDFS data: StringIndexer → Imputer → VectorAssembler → StandardScaler → Classifier, with `CrossValidator` (5-fold), trains both Logistic Regression (baseline) and Gradient Boosted Trees (primary), serializes both pipelines back to HDFS
- **`train_local.py`** — same pipeline in `local[*]` Spark mode (no Docker needed), 3-fold CV for faster iteration, checks mid-term target (GBT F1 ≥ 0.70), saves to local `models/`
- **`generate_sample_data.py`** — generates 100k-row synthetic BTS-schema CSV for local testing when real data isn't downloaded yet (~20% delayed, realistic class imbalance)

### Streaming inference (`src/streaming/`)
- **`kafka_producer.py`** — loads 2024 BTS data from HDFS, serializes each flight record as JSON, publishes to Kafka topic `flight-events` at configurable rate (simulates real-time arrivals)
- **`streaming_consumer.py`** — Spark Structured Streaming job that reads from Kafka, deserializes, applies the trained GBT pipeline model, writes predictions to HDFS; measures per-batch latency and throughput

### Batch inference & benchmarking (`src/batch/`, `src/evaluation/`)
- **`batch_inference.py`** — loads the same GBT pipeline, scores the full 2024 held-out set in one Spark job, writes predictions to HDFS
- **`benchmark.py`** — reads both streaming and batch output from HDFS, computes and compares: throughput (events/sec), average latency, AUC-ROC, F1, precision, recall; prints side-by-side report

---

## What's done

- [x] Docker Compose cluster — all 7 services with healthchecks
- [x] HDFS directory structure setup script
- [x] End-to-end pipeline orchestration script
- [x] BTS CSV ingestion to HDFS Parquet (with column normalization)
- [x] Feature engineering pipeline (leakage-free, derived features, Spark ML pipeline serialized to HDFS)
- [x] EDA — HDFS CSV reports + local matplotlib plots
- [x] ML training — LR + GBT with CrossValidator, both HDFS and local modes
- [x] Kafka producer (replay 2024 data as real-time stream)
- [x] Spark Structured Streaming consumer with live inference
- [x] Batch inference on held-out 2024 data
- [x] Batch vs. streaming benchmark evaluation
- [x] Local dev mode — one command setup + train without Docker (`run_local.sh`)

## What's remaining

- [ ] Download real BTS data (2018–2024) from transtats.bts.gov
- [ ] Run `docker-compose up` and verify cluster health end-to-end
- [ ] Run full pipeline and capture real benchmark numbers (throughput, latency, accuracy)
- [ ] Final report — analysis, visualizations, batch vs. streaming comparison
- [ ] Final presentation

---

## Stack

| Component | Technology |
|-----------|-----------|
| Storage | HDFS (Hadoop 3.2, Docker) |
| Processing | Apache Spark 3.5 (PySpark) |
| Streaming | Apache Kafka 3.6 + Spark Structured Streaming |
| ML | Spark MLlib — Logistic Regression + Gradient Boosted Trees |
| Orchestration | Docker Compose |
| Dataset | BTS Airline On-Time Performance (2018–2024) |

---

## Quick start (local, no Docker)

```bash
chmod +x run_local.sh
./run_local.sh                          # generates 100k synthetic rows + trains
./run_local.sh path/to/bts_data.csv    # uses real BTS data
```

Artifacts written to `models/` and `plots/`.

---

## Full cluster (Docker)

```bash
docker-compose up -d
bash scripts/setup_hdfs.sh
bash scripts/run_pipeline.sh
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

**3. Train models**
```bash
spark-submit src/training/train_model.py
```

**4. Streaming inference**
```bash
# Consumer (background)
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  src/streaming/streaming_consumer.py &

# Producer
python src/streaming/kafka_producer.py
```

**5. Batch inference + benchmark**
```bash
spark-submit src/batch/batch_inference.py
python src/evaluation/benchmark.py
```

---

## HDFS layout

```
/data/flights/                   ← raw Parquet (YEAR/MONTH partitioned)
/data/flights/cleaned/           ← cleaned dataset
/data/flights/featured/          ← feature-engineered dataset
/models/gbt_pipeline/            ← trained GBT model
/models/lr_pipeline/             ← trained LR model
/models/preprocessing_pipeline/  ← Spark ML preprocessing pipeline
/output/streaming_predictions/   ← streaming inference output
/output/batch_predictions/       ← batch inference output
/output/eda/                     ← EDA CSV reports
/checkpoints/streaming/          ← Spark Streaming checkpoints
```

---

## Dataset

Bureau of Transportation Statistics — Airline On-Time Performance.
Download: https://www.transtats.bts.gov/DL_SelectFields.aspx

- Training: 2018–2023
- Held-out / streaming simulation: 2024
- Target: `ARR_DELAY > 15 minutes` → **delayed (1)**, otherwise **on-time (0)**
- Mid-term target: GBT F1 ≥ 0.70
- Final target: streaming ≥ 500 events/sec, latency < 2s

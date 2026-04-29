# Real-Time Flight Delay Prediction

**SJSU DATA-228 (Big Data Technologies) — Spring 2026**

End-to-end big data pipeline that ingests historical BTS flight records into HDFS, trains Gradient Boosted Tree and Logistic Regression classifiers using Spark MLlib, streams live flight events through Apache Kafka, and benchmarks real-time inference against batch inference — with Differential Privacy, LSH anomaly detection, Bloom Filter deduplication, Reservoir Sampling, Delta Lake, and SHAP explainability.

---

## Team

| Name | Role |
|------|------|
| Anees Saheba Guddi | Evaluation & Reporting — benchmark design, visualizations, final report |
| Kartheek Alluri | Data & Infrastructure — BTS data acquisition, HDFS setup |
| Keon Sadeghi | Streaming & Kafka — producer, consumer, integration |
| Manjot Kaur | ML Engineering — EDA, feature engineering, local training |
| Rish Jain | Integration & DevOps — Docker Compose, feature pipeline, end-to-end testing |

---

## Benchmark Results (Dec 2021 BTS — 537,183 flights, real data)

### Model accuracy

| Metric | GBT (primary) | LR (baseline) | Target |
|--------|:-------------:|:-------------:|:------:|
| AUC-ROC | **0.9369** | 0.9267 | — |
| AUC-PR | **0.8849** | 0.872 | — |
| F1 | **0.9012** | 0.9088 | ≥ 0.70 ✅ |
| Precision | 0.9043 | 0.9089 | — |
| Recall | 0.8993 | 0.9087 | — |
| Accuracy | 0.8993 | 0.9087 | — |

- Class balance: 78.6% on-time, 21.4% delayed (realistic imbalance, class-weighted training)
- Top feature by importance: `DEP_DELAY` (0.787), followed by `DEST` (0.065), `CARRIER` (0.034)
- Best cross-validation AUC-ROC (3-fold): 0.9361

### Throughput & latency

| Mode | Records | Throughput | Notes |
|------|--------:|:----------:|-------|
| Batch inference | 537,183 | **149,188 rec/sec** | Single Spark job, 3.6 s total |
| Streaming inference | 62,815 (90 s window) | **11,648 events/sec peak** | 23× above 500/sec target ✅ |

- Streaming trigger interval: 10 seconds per micro-batch
- Per-batch processing time: 856 – 1,732 ms
- End-to-end latency (producer → prediction written): mean 5,303 ms, p95 9,921 ms
- Kafka producer sustained rate: ~994 msg/sec across 537k records

### Advanced techniques

| Technique | Result |
|-----------|--------|
| LSH Anomaly Detection (MinHashLSH, 128 perms) | 0% anomaly rate across 25M pair comparisons — predictions consistent |
| Differential Privacy (Laplace, ε=1.0) | Metrics published with noise; true AUC-ROC=0.9386 → reported 0.9411 |
| Bloom Filter deduplication | 1% false-positive rate; tracks flight IDs across micro-batches |
| Reservoir Sampling (Algorithm R) | O(n) single-pass, O(k) memory unbiased sampling from dataset |
| SHAP explainability | TreeExplainer run on GBT test set; feature contributions saved |
| Delta Lake output | Streaming predictions written with ACID transactions + time-travel |

---

## Architecture

```
BTS CSV (raw)
     │
     ▼
[ingest_bts_to_hdfs.py]  ──►  HDFS /data/flights/  (Parquet, YEAR/MONTH partitioned)
     │
     ▼
[prepare_features.py]    ──►  HDFS /data/flights/featured/  +  /models/preprocessing_pipeline/
     │
     ▼
[train_model.py]         ──►  HDFS /models/gbt_pipeline/  +  /models/lr_pipeline/
     │
     ├──► Batch path ──────────────────────────────────────────────────────────┐
     │    [batch_inference.py]  ──►  HDFS /output/batch_predictions/           │
     │                                                                         │
     └──► Streaming path                                                       │
          [kafka_producer.py]  ──►  Kafka topic: flight-events                 │
               │                                                               │
               ▼                                                               │
          [streaming_consumer.py]  ──►  HDFS /output/streaming_predictions/   │
               │                                                               │
               └───────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                               [benchmark.py]  ──►  benchmark_report.json
                               (DP + LSH + latency + throughput comparison)
```

---

## Infrastructure

### Docker Compose cluster (6 services)

| Service | Image | Port | Role |
|---------|-------|------|------|
| `kafka` | apache/kafka:3.7.0 | 9092, 9093 | KRaft mode broker+controller (no Zookeeper) |
| `kafka-init` | apache/kafka:3.7.0 | — | One-shot topic creation (`flight-events`, 4 partitions) |
| `hdfs-namenode` | bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8 | 9870, 9000 | HDFS namenode |
| `hdfs-datanode` | bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8 | 9864 | HDFS datanode |
| `spark-master` | apache/spark:3.5.0 | 8080, 7077 | Spark master |
| `spark-worker-1` | apache/spark:3.5.0 | 8081 | 4 cores, 4 GB |
| `spark-worker-2` | apache/spark:3.5.0 | 8082 | 4 cores, 4 GB |

All services have healthchecks and are connected via `flight-net` bridge network.

### HDFS layout

```
/data/flights/                    ← ingested Parquet (partitioned by YEAR/MONTH)
/data/flights/cleaned/            ← cleaned dataset (nulls dropped, types cast)
/data/flights/featured/           ← feature-engineered dataset (vector assembled)
/models/gbt_pipeline/             ← trained GBT PipelineModel
/models/lr_pipeline/              ← trained LR PipelineModel
/models/preprocessing_pipeline/   ← Spark ML preprocessing pipeline
/output/streaming_predictions/    ← streaming inference output (Parquet / Delta)
/output/batch_predictions/        ← batch inference output (Parquet)
/output/eda/                      ← EDA CSV reports
/checkpoints/streaming/           ← Spark Structured Streaming checkpoints
```

---

## Source code

### `src/ingestion/`
- **`ingest_bts_to_hdfs.py`** — reads raw BTS CSVs from local path, normalises mixed-case column headers (`Reporting_Airline` → `OP_UNIQUE_CARRIER`, etc.), filters cancelled/diverted flights (null `ARR_DELAY`), casts types, fills nullable delay columns with 0, and writes Snappy-compressed Parquet to HDFS partitioned by `YEAR`/`MONTH`. Handles both flat and year-subdirectory input layouts.

### `src/training/`
- **`prepare_features.py`** — reads cleaned Parquet from HDFS; removes data-leakage columns (post-flight actuals like `CARRIER_DELAY`, `WEATHER_DELAY`); derives new features: `dep_hour` (hour from `CRS_DEP_TIME`), `arr_sched_hour`, `is_weekend`, `is_holiday_season` (Nov–Jan), `route` (`ORIGIN_DEST`); StringIndexes categoricals; Imputes numerics; VectorAssembles; saves reusable `PipelineModel` to HDFS.
- **`train_model.py`** — full Spark MLlib training: StringIndexer → Imputer → VectorAssembler → StandardScaler → Classifier; class-weighted training to handle 78/22 imbalance; `CrossValidator` (3-fold); trains both GBT (primary) and LR (baseline); logs feature importances; runs SHAP TreeExplainer on test sample; serialises both `PipelineModel`s to HDFS; supports `--sample-fraction` for fast iteration.
- **`train_local.py`** — identical pipeline in `local[*]` Spark mode (no Docker needed); 3-fold CV; saves to local `models/`; checks mid-term target GBT F1 ≥ 0.70.
- **`eda_analysis.py`** — pandas + matplotlib/seaborn; produces 7 PNG plots saved to `plots/`: class distribution, arrival delay distribution, delay rate by month, by day of week, by carrier, by departure hour, and correlation heatmap of numeric features.
- **`eda_report.py`** — PySpark EDA; writes CSV summary reports to HDFS (`/output/eda/`): dataset statistics, missing-value counts, label distribution, delay rates by carrier / origin / destination / month / day-of-week / departure hour.
- **`generate_sample_data.py`** — generates 100k-row synthetic BTS-schema CSV (~20% delayed) for local testing without real data.

### `src/streaming/`
- **`kafka_producer.py`** — reads BTS CSV files from local path; serialises each flight record as JSON (with `producer_ts` timestamp for end-to-end latency measurement); publishes to Kafka topic `flight-events` at a configurable rate (`--rate`); reports throughput every 1,000 messages.
- **`streaming_consumer.py`** — Spark Structured Streaming job; reads JSON from Kafka; parses with a fixed schema; applies the saved GBT `PipelineModel`; attaches `consumer_ts` timestamp; writes predictions to HDFS (`append` mode); logs per-batch throughput, mean/p50/p95/p99 latency, and a sample prediction table.

### `src/` (extended algorithms)
- **`kafka_producer.py`** — extended producer with **Reservoir Sampling** (Algorithm R, Vitter 1985) via `--sample K` flag; draws an unbiased fixed-size sample from the dataset in a single O(n) pass with O(k) memory, then publishes only those K records. Guarantees every record has equal selection probability regardless of dataset size.
- **`spark_streaming_consumer.py`** — extended consumer with:
  - **Bloom Filter deduplication** (pybloom-live `ScalableBloomFilter`, 1% FP rate) — tracks `flight_id` strings across micro-batches to skip already-processed records, preventing double-counting in the streaming output.
  - **Delta Lake output** (`write.format("delta")`) — predictions written with full ACID transaction guarantees, schema enforcement, and time-travel capability on HDFS.

### `src/batch/`
- **`batch_inference.py`** — loads the saved GBT `PipelineModel` and scores the full held-out dataset in a single Spark job; measures total inference time and throughput (records/sec); writes predictions to HDFS; saves metrics JSON locally.

### `src/evaluation/`
- **`benchmark.py`** — reads both batch and streaming predictions from HDFS; computes classification metrics (AUC-ROC, AUC-PR, F1, precision, recall, accuracy); applies **Differential Privacy** (Laplace mechanism, ε=1.0, sensitivity=0.01) to all published metrics; runs **LSH anomaly detection** (datasketch `MinHashLSH`, 128 permutations, Jaccard threshold 0.5) to flag streaming predictions that disagree on label for near-identical carrier/origin/dest combinations; writes full JSON benchmark report.

### `scripts/`
- **`setup_hdfs.sh`** — polls namenode JMX endpoint until state is `active`, then creates all required HDFS directories (`/data/flights`, `/models`, `/output`, `/checkpoints`) and sets permissions.
- **`run_pipeline.sh`** — end-to-end orchestration: HDFS setup → BTS ingestion → feature engineering → model training → start streaming consumer → run Kafka producer → batch inference → benchmark.
- **`run_feature_pipeline.sh`** — standalone feature engineering + EDA step.
- **`smoke_test.py`** — validates Kafka topic reachability, HDFS namenode health, Spark master UI, and presence of prediction output in HDFS.

---

## ML pipeline detail

```
Raw columns (18)
      │
      ├─ StringIndexer: OP_UNIQUE_CARRIER, ORIGIN, DEST  →  *_idx
      ├─ Imputer (median): DEP_DELAY, CRS_ELAPSED_TIME, DISTANCE, CRS_DEP_TIME, DAY_OF_WEEK, MONTH  →  *_imp
      ├─ VectorAssembler  →  features_raw
      ├─ StandardScaler   →  features
      └─ GBTClassifier / LogisticRegression  →  prediction, probability, rawPrediction

Label: ARR_DELAY > 15 minutes → 1 (delayed), else 0 (on-time)
Class weights: on-time 0.636, delayed 2.333  (handles 78/22 imbalance)
Split: 70% train / 15% validation / 15% test  (fixed seed=42)
CV: 3-fold CrossValidator on training set
```

---

## EDA plots (`plots/`)

| File | Description |
|------|-------------|
| `01_class_distribution.png` | Bar chart of on-time vs delayed flight counts |
| `02_delay_by_month.png` | Delay rate (%) by calendar month |
| `03_delay_by_day.png` | Delay rate (%) by day of week (Mon–Sun) |
| `04_delay_by_carrier.png` | Delay rate (%) ranked by airline carrier |
| `05_arr_delay_dist.png` | Histogram of arrival delay minutes (−60 to +300), threshold line at 15 min |
| `06_correlation_heatmap.png` | Pearson correlation matrix of all numeric features + label |
| `07_delay_by_hour.png` | Delay rate (%) by scheduled departure hour (0–23) |

All plots generated from real Dec 2021 BTS data (537,183 flights).

---

## Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Storage | Apache HDFS | Hadoop 3.2.1 |
| Processing | Apache Spark (PySpark) | 3.5.0 |
| Streaming | Apache Kafka (KRaft) | 3.7.0 |
| Stream processing | Spark Structured Streaming | 3.5.0 |
| ML | Spark MLlib — GBT + LR + CrossValidator | 3.5.0 |
| Delta Lake | delta-spark | 2.4.0 |
| Bloom Filter | pybloom-live | 4.0.0 |
| LSH | datasketch MinHashLSH | 1.6.4 |
| Differential Privacy | diffprivlib / numpy Laplace | 0.5.0 |
| Explainability | SHAP TreeExplainer | 0.44.1 |
| Orchestration | Docker Compose | — |
| Dataset | BTS Airline On-Time Performance | Dec 2021 (537k flights) |

---

## Quick start

### Local mode (no Docker)

```bash
chmod +x run_local.sh
./run_local.sh                          # generates 100k synthetic rows, trains GBT + LR
./run_local.sh path/to/bts_data.csv    # uses real BTS CSV
```

Artifacts: `models/` (PipelineModels + metrics.json), `plots/` (7 PNG EDA plots).

Requirements: Java 17, `pip install pyspark pandas numpy matplotlib seaborn shap`.

### Full cluster (Docker)

```bash
# 1. Start all services
docker-compose up -d

# 2. Wait for healthchecks, then create HDFS directories
bash scripts/setup_hdfs.sh

# 3. Run everything end-to-end
bash scripts/run_pipeline.sh
```

### Step-by-step

**1. Ingest BTS CSVs to HDFS**
```bash
spark-submit src/ingestion/ingest_bts_to_hdfs.py \
  --input-path data/raw \
  --hdfs-path hdfs://hdfs-namenode:9000/data/flights \
  --years 2021
```

**2. Feature engineering**
```bash
spark-submit src/training/prepare_features.py \
  --input hdfs://hdfs-namenode:9000/data/flights \
  --cleaned-output hdfs://hdfs-namenode:9000/data/flights/cleaned \
  --featured-output hdfs://hdfs-namenode:9000/data/flights/featured \
  --pipeline-model-output hdfs://hdfs-namenode:9000/models/preprocessing_pipeline
```

**3. Train models**
```bash
spark-submit src/training/train_model.py \
  --hdfs-path hdfs://hdfs-namenode:9000/data/flights/cleaned \
  --model-path hdfs://hdfs-namenode:9000/models \
  --train-years 2021 \
  --cv-folds 3 \
  --skip-shap
```

**4. Streaming inference**
```bash
# Start consumer (background)
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  src/streaming/streaming_consumer.py \
  --kafka-bootstrap kafka:9092 \
  --model-path hdfs://hdfs-namenode:9000/models/gbt_pipeline \
  --output-path hdfs://hdfs-namenode:9000/output/streaming_predictions \
  --checkpoint-path hdfs://hdfs-namenode:9000/checkpoints/streaming \
  --await-termination 120 &

# Start producer
python3 src/streaming/kafka_producer.py \
  --input-path data/raw \
  --kafka-bootstrap kafka:9092 \
  --rate 5000
```

**5. Batch inference**
```bash
spark-submit src/batch/batch_inference.py \
  --data-path hdfs://hdfs-namenode:9000/data/flights/cleaned \
  --model-path hdfs://hdfs-namenode:9000/models/gbt_pipeline \
  --output-path hdfs://hdfs-namenode:9000/output/batch_predictions \
  --test-years 2021
```

**6. Benchmark**
```bash
spark-submit src/evaluation/benchmark.py \
  --batch-path hdfs://hdfs-namenode:9000/output/batch_predictions \
  --streaming-path hdfs://hdfs-namenode:9000/output/streaming_predictions \
  --report-path models/benchmark_report.json
```

---

## Dataset

**Bureau of Transportation Statistics — Airline On-Time Performance**
Download: https://www.transtats.bts.gov/DL_SelectFields.aspx

| Split | Years | Use |
|-------|-------|-----|
| Training | 2018–2023 | Model training + cross-validation |
| Test / streaming | 2021 (Dec) | Batch inference + Kafka streaming simulation |

- **Target label:** `ARR_DELAY > 15 minutes` → delayed (1), otherwise on-time (0)
- **Class balance:** ~78.6% on-time, ~21.4% delayed
- **Mid-term target:** GBT F1 ≥ 0.70 → **achieved: 0.9012**
- **Final target:** streaming ≥ 500 events/sec → **achieved: 11,648 events/sec**

---

## Repository structure

```
realtime-flight-delay-predictor/
├── docker-compose.yml                  ← 6-service cluster definition
├── requirements.txt                    ← Python dependencies
├── run_local.sh                        ← one-command local dev setup
├── data/
│   └── raw/                            ← BTS CSV files (not committed)
├── models/
│   ├── gbt_pipeline/                   ← trained GBT PipelineModel
│   ├── lr_pipeline/                    ← trained LR PipelineModel
│   ├── metrics.json                    ← model + benchmark metrics
│   └── benchmark_report.json          ← full benchmark JSON report
├── plots/
│   ├── 01_class_distribution.png
│   ├── 02_delay_by_month.png
│   ├── 03_delay_by_day.png
│   ├── 04_delay_by_carrier.png
│   ├── 05_arr_delay_dist.png
│   ├── 06_correlation_heatmap.png
│   └── 07_delay_by_hour.png
├── scripts/
│   ├── setup_hdfs.sh                   ← HDFS directory setup
│   ├── run_pipeline.sh                 ← end-to-end orchestration
│   ├── run_feature_pipeline.sh         ← feature engineering only
│   └── smoke_test.py                   ← connectivity + health check
└── src/
    ├── ingestion/
    │   └── ingest_bts_to_hdfs.py
    ├── training/
    │   ├── prepare_features.py
    │   ├── train_model.py
    │   ├── train_local.py
    │   ├── eda_analysis.py
    │   ├── eda_report.py
    │   └── generate_sample_data.py
    ├── streaming/
    │   ├── kafka_producer.py
    │   └── streaming_consumer.py
    ├── batch/
    │   └── batch_inference.py
    ├── evaluation/
    │   └── benchmark.py
    ├── kafka_producer.py               ← extended: Reservoir Sampling
    └── spark_streaming_consumer.py     ← extended: Bloom Filter + Delta Lake
```

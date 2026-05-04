# Real-Time Flight Delay Predictor

**SJSU DATA-228 — Big Data Technologies · Spring 2026**

An end-to-end big data pipeline that ingests 19M+ BTS flight records into HDFS, trains Gradient Boosted Tree and Logistic Regression classifiers with Spark MLlib, streams live flight events through Apache Kafka, performs real-time inference with Spark Structured Streaming, and benchmarks streaming vs. batch — with Differential Privacy, LSH anomaly detection, Bloom Filter deduplication, Reservoir Sampling, and SHAP explainability.

---

## Team

| Name | Role |
|------|------|
| Anees Saheba Guddi | Evaluation & Reporting — benchmark design, DP, LSH, visualizations, final report |
| Kartheek Alluri | Data & Infrastructure — BTS acquisition, HDFS setup, batch inference |
| Keon Sadeghi | Streaming & Kafka — producer, consumer, Bloom filter, Delta Lake |
| Manjot Kaur | ML Engineering — EDA, feature engineering, model training, SHAP |
| Rish Jain | Integration & DevOps — Docker Compose, orchestration, smoke tests |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BTS Flight CSV Data (2018–2024)                      │
│                        transtats.bts.gov  ·  ~35 GB  ·  19M+ records        │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  INGESTION   ingest_bts_to_hdfs.py                                          │
│  · Normalize column names     · Drop cancelled/diverted flights             │
│  · Cast types                 · Fill delay nulls with 0                     │
│  · Create binary label (ARR_DELAY > 15 min)                                 │
│  · Write Snappy Parquet → HDFS partitioned by YEAR / MONTH                  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
┌──────────────────────────┐    ┌─────────────────────────────────────────────┐
│  FEATURE ENGINEERING     │    │  HDFS STORAGE                               │
│  prepare_features.py     │    │                                             │
│  · Remove leakage cols   │    │  /data/flights/        ← raw Parquet        │
│  · Derive: dep_hour,     │    │  /data/flights/cleaned ← cleaned dataset    │
│    is_weekend,           │    │  /data/flights/featured← feature vectors    │
│    is_holiday_season,    │    │  /models/gbt_pipeline/ ← trained GBT model  │
│    route                 │    │  /models/lr_pipeline/  ← trained LR model   │
│  · StringIndex, Impute,  │    │  /output/batch_preds/  ← batch results      │
│    VectorAssemble        │    │  /output/stream_preds/ ← streaming results  │
│  · Save Pipeline → HDFS  │    │  /checkpoints/         ← Spark checkpoints  │
└──────────────┬───────────┘    └─────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL TRAINING   train_model.py  /  train_local.py                         │
│                                                                             │
│  Pipeline stages:                                                           │
│  StringIndexer → Imputer → VectorAssembler → StandardScaler → Classifier   │
│                                                                             │
│  ┌─────────────────────────┐    ┌────────────────────────────────────────┐  │
│  │  Logistic Regression    │    │  Gradient Boosted Trees (primary)      │  │
│  │  Baseline · interpretable│    │  maxDepth=5 · maxIter=50 · GBT F1=0.90│  │
│  └─────────────────────────┘    └────────────────────────────────────────┘  │
│                                                                             │
│  CrossValidator (3-fold) · Class weighting · SHAP explainability            │
│  Serialized full PipelineModel → HDFS                                       │
└──────────┬──────────────────────────────────────┬───────────────────────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────────┐    ┌─────────────────────────────────────────────┐
│  BATCH INFERENCE         │    │  STREAMING PIPELINE                         │
│  batch_inference.py      │    │                                             │
│                          │    │  kafka_producer.py                          │
│  · Load GBT pipeline     │    │  · Read 2024 BTS CSV                        │
│  · Score 537K records    │    │  · Publish JSON to Kafka @ 100–500 msg/s    │
│  · 340K rec/sec          │    │  · Stamp producer_ts for latency tracking   │
│  · Full metrics + CM     │    │  · Reservoir Sampling (Algorithm R)         │
│  · Write → HDFS          │    │                    │                        │
└──────────┬───────────────┘    │                    ▼                        │
           │                    │  Apache Kafka (KRaft · 4 partitions)        │
           │                    │  topic: flight-events                       │
           │                    │                    │                        │
           │                    │                    ▼                        │
           │                    │  streaming_consumer.py                      │
           │                    │  · Spark Structured Streaming               │
           │                    │  · Micro-batch every 10 seconds             │
           │                    │  · model.transform() → predictions          │
           │                    │  · Bloom Filter deduplication               │
           │                    │  · Latency = consumer_ts − producer_ts      │
           │                    │  · Write → HDFS (Delta Lake + ACID)         │
           │                    └──────────────┬──────────────────────────────┘
           │                                   │
           └───────────────────┬───────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  EVALUATION   benchmark.py                                                  │
│                                                                             │
│  · Classification metrics (AUC-ROC, AUC-PR, F1, precision, recall, acc)    │
│  · Streaming latency stats (mean, p50, p90, p95, p99)                       │
│  · Differential Privacy — Laplace mechanism (ε=1.0) on all metrics         │
│  · LSH Anomaly Detection — MinHashLSH (128 perms, Jaccard ≥ 0.5)           │
│  · Batch vs Streaming comparison                                            │
│  → benchmark_report.json                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Results

### Model Performance

| Metric | GBT (primary) | LR (baseline) | Target |
|--------|:---:|:---:|:---:|
| AUC-ROC | **0.9386** | 0.9267 | — |
| AUC-PR | **0.8877** | 0.872 | — |
| F1 | **0.9029** | 0.9088 | ≥ 0.70 ✅ |
| Precision | 0.9058 | 0.9089 | — |
| Recall | 0.9011 | 0.9087 | — |
| Accuracy | 0.9011 | 0.9087 | — |

- **Top feature:** `DEP_DELAY` (importance 0.787) — departure delay is by far the strongest signal
- **Class balance:** 78.6% on-time / 21.4% delayed — handled via inverse-frequency class weighting
- **Best CV AUC-ROC (3-fold):** 0.9361
- **Data:** Dec 2021 BTS — 537,183 flights

### Throughput & Latency

| Mode | Records | Throughput | Status |
|------|--------:|:---:|:---:|
| Batch inference | 537,183 | **149,188 rec/sec** | ✅ |
| Streaming inference | 62,815 | **11,648 events/sec peak** | ✅ (target: 500/sec) |

| Latency Metric | Value |
|---|---|
| Mean end-to-end | 5,303 ms |
| p50 | 5,212 ms |
| p90 | 9,394 ms |
| p95 | 9,921 ms |
| p99 | 10,438 ms |
| Min | 144 ms |
| Max | 10,613 ms |

### Advanced Techniques

| Technique | Library | Result |
|-----------|---------|--------|
| Differential Privacy (Laplace, ε=1.0) | diffprivlib | Metrics published with noise — true AUC-ROC 0.9386 → reported 0.9411 |
| LSH Anomaly Detection (MinHashLSH, 128 perms) | datasketch | 0% anomaly rate across ~25M pair comparisons |
| Bloom Filter deduplication | pybloom-live | 1% false-positive rate; deduplicates flight IDs across micro-batches |
| Reservoir Sampling (Algorithm R) | custom | O(n) single-pass, O(k) memory unbiased sampling |
| SHAP Explainability | shap | TreeExplainer on GBT test set; feature contributions saved to HDFS |
| Delta Lake output | delta-spark | Streaming predictions with ACID transactions + time-travel |

---

## ML Pipeline Detail

```
Raw columns (18 BTS fields)
        │
        ├─ StringIndexer  →  OP_UNIQUE_CARRIER_idx, ORIGIN_idx, DEST_idx
        ├─ Imputer (median)  →  DEP_DELAY_imp, CRS_ELAPSED_TIME_imp,
        │                       DISTANCE_imp, CRS_DEP_TIME_imp,
        │                       DAY_OF_WEEK_imp, MONTH_imp
        ├─ VectorAssembler  →  raw_features
        ├─ StandardScaler   →  features
        └─ GBTClassifier / LogisticRegression
               └─ prediction · probability · rawPrediction

Label:         ARR_DELAY > 15 min → 1 (delayed), else 0 (on-time)
Class weights: on-time 0.636 · delayed 2.333
Split:         70% train / 15% val / 15% test  (seed=42)
CV:            3-fold CrossValidator optimizing AUC-ROC
GBT grid:      maxDepth ∈ {4,5} · maxIter ∈ {30,50} · stepSize ∈ {0.05,0.1}
LR grid:       regParam ∈ {0.001,0.01,0.1} · elasticNetParam ∈ {0.0,0.5}
```

---

## Infrastructure

### Docker Compose (7 services)

| Service | Image | Ports | Role |
|---------|-------|-------|------|
| `kafka` | apache/kafka:3.7.0 | 9092, 9093 | KRaft broker + controller (no Zookeeper) |
| `kafka-init` | apache/kafka:3.7.0 | — | One-shot: creates `flight-events` topic (4 partitions) |
| `hdfs-namenode` | bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8 | 9870, 9000 | HDFS namenode + web UI |
| `hdfs-datanode` | bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8 | 9864 | HDFS datanode |
| `spark-master` | apache/spark:3.5.0 | 8080, 7077 | Spark standalone master |
| `spark-worker-1` | apache/spark:3.5.0 | 8081 | 4 cores · 4 GB RAM |
| `spark-worker-2` | apache/spark:3.5.0 | 8082 | 4 cores · 4 GB RAM |

All services run on the `flight-net` bridge network with healthchecks and `depends_on` ordering.

### Web UIs

| UI | URL |
|----|-----|
| Spark Master | http://localhost:8080 |
| HDFS Namenode | http://localhost:9870 |
| Spark Worker 1 | http://localhost:8081 |
| Spark Worker 2 | http://localhost:8082 |

---

## Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Distributed Storage | Apache HDFS | Hadoop 3.2.1 |
| Distributed Processing | Apache Spark (PySpark) | 3.5.0 |
| Message Broker | Apache Kafka (KRaft) | 3.7.0 |
| Stream Processing | Spark Structured Streaming | 3.5.0 |
| ML | Spark MLlib — GBT + LR + CrossValidator | 3.5.0 |
| Delta Lake | delta-spark | 2.4.0 |
| Differential Privacy | diffprivlib | 0.5.0 |
| LSH | datasketch MinHashLSH | 1.6.4 |
| Bloom Filter | pybloom-live | 4.0.0 |
| Explainability | SHAP TreeExplainer | 0.44.1 |
| Containerization | Docker Compose | — |
| Dataset | BTS Airline On-Time Performance | 2018–2024 |

---

## Dataset

**Bureau of Transportation Statistics — Airline On-Time Performance**
Download: https://www.transtats.bts.gov/DL_SelectFields.aspx

| Split | Years | Purpose |
|-------|-------|---------|
| Training | 2018–2023 | Model training + cross-validation |
| Test / Streaming | 2024 | Batch inference + Kafka replay |

- **Label:** `ARR_DELAY > 15 minutes → 1 (delayed), else 0 (on-time)`
- **Size:** ~35 GB compressed · 19M+ records
- **Class balance:** ~78.6% on-time · ~21.4% delayed
- **Key insight:** `DEP_DELAY` is the strongest predictor (r=0.48 with label). June/July are peak delay months (24–25%). Budget carriers (JetBlue 29.6%, Allegiant 28.8%) delay far more than legacy carriers (Delta 14.3%).

---

## Quick Start

### Local mode (no Docker required)

```bash
# Install dependencies
pip install pyspark pandas numpy matplotlib seaborn shap datasketch diffprivlib

# Train on synthetic data (auto-generated)
python src/training/train_local.py

# Train on real BTS CSV
python src/training/train_local.py --input data/raw/your_bts_file.csv
```

Artifacts saved to `models/` (PipelineModels + metrics.json) and `plots/` (7 EDA PNGs).

### Full cluster (Docker)

```bash
# 1. Start all 7 services
docker compose up -d

# 2. Create HDFS directories
bash scripts/setup_hdfs.sh

# 3. Run full pipeline end-to-end
bash scripts/run_pipeline.sh
```

### Step-by-step

**1 — Ingest BTS data to HDFS**
```bash
spark-submit src/ingestion/ingest_bts_to_hdfs.py \
  --input-path data/raw \
  --hdfs-path hdfs://hdfs-namenode:9000/data/flights \
  --years 2021
```

**2 — Feature engineering**
```bash
spark-submit src/training/prepare_features.py \
  --input hdfs://hdfs-namenode:9000/data/flights \
  --featured-output hdfs://hdfs-namenode:9000/data/flights/featured \
  --pipeline-model-output hdfs://hdfs-namenode:9000/models/preprocessing_pipeline
```

**3 — Train models**
```bash
spark-submit src/training/train_model.py \
  --hdfs-path hdfs://hdfs-namenode:9000/data/flights/cleaned \
  --model-path hdfs://hdfs-namenode:9000/models \
  --train-years 2021 --cv-folds 3
```

**4 — Start streaming consumer**
```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  src/streaming/streaming_consumer.py \
  --kafka-bootstrap kafka:9092 \
  --model-path hdfs://hdfs-namenode:9000/models/gbt_pipeline \
  --output-path hdfs://hdfs-namenode:9000/output/streaming_predictions \
  --checkpoint-path hdfs://hdfs-namenode:9000/checkpoints/streaming \
  --await-termination 120 &
```

**5 — Start Kafka producer**
```bash
python src/streaming/kafka_producer.py \
  --input-path data/raw \
  --kafka-bootstrap localhost:9093 \
  --rate 500
```

**6 — Batch inference**
```bash
spark-submit src/batch/batch_inference.py \
  --data-path hdfs://hdfs-namenode:9000/data/flights/cleaned \
  --model-path hdfs://hdfs-namenode:9000/models/gbt_pipeline \
  --output-path hdfs://hdfs-namenode:9000/output/batch_predictions \
  --test-years 2024
```

**7 — Benchmark**
```bash
spark-submit src/evaluation/benchmark.py \
  --batch-path hdfs://hdfs-namenode:9000/output/batch_predictions \
  --streaming-path hdfs://hdfs-namenode:9000/output/streaming_predictions \
  --report-path models/benchmark_report.json
```

**8 — Smoke test**
```bash
python scripts/smoke_test.py
```

---

## Repository Structure

```
realtime-flight-delay-predictor/
├── docker-compose.yml
├── data/
│   └── raw/                            ← BTS CSV files (not committed)
├── models/
│   ├── gbt_pipeline/                   ← trained GBT PipelineModel
│   ├── lr_pipeline/                    ← trained LR PipelineModel
│   ├── metrics.json                    ← training metrics
│   └── benchmark_report.json          ← full benchmark report
├── plots/
│   ├── 01_class_distribution.png
│   ├── 02_delay_by_month.png
│   ├── 03_delay_by_day.png
│   ├── 04_delay_by_carrier.png
│   ├── 05_arr_delay_dist.png
│   ├── 06_correlation_heatmap.png
│   └── 07_delay_by_hour.png
├── scripts/
│   ├── setup_hdfs.sh                   ← HDFS directory creation
│   ├── run_pipeline.sh                 ← end-to-end orchestration
│   ├── run_feature_pipeline.sh         ← feature engineering only
│   └── smoke_test.py                   ← health checks
└── src/
    ├── ingestion/
    │   └── ingest_bts_to_hdfs.py       ← CSV → HDFS Parquet
    ├── training/
    │   ├── prepare_features.py         ← feature engineering pipeline
    │   ├── train_model.py              ← cluster training (GBT + LR + SHAP)
    │   ├── train_local.py              ← local training (no Docker)
    │   ├── eda_analysis.py             ← pandas/matplotlib EDA plots
    │   ├── eda_report.py               ← PySpark EDA → HDFS reports
    │   └── generate_sample_data.py     ← synthetic data generator
    ├── streaming/
    │   ├── kafka_producer.py           ← Kafka producer + Reservoir Sampling
    │   └── streaming_consumer.py       ← Spark Streaming + Bloom Filter + Delta Lake
    ├── batch/
    │   └── batch_inference.py          ← batch scoring + metrics
    └── evaluation/
        └── benchmark.py               ← DP + LSH + batch vs streaming report
```

---

## Key Design Decisions

**Full Pipeline Serialization**
Both models are saved as complete `PipelineModel` objects — preprocessing stages (StringIndexer, Imputer, StandardScaler) bundled with model weights. This eliminates training/serving feature skew: the streaming consumer loads one object and calls `transform()` with guaranteed identical preprocessing.

**YEAR/MONTH Partitioning**
HDFS Parquet is partitioned by YEAR and MONTH. When batch inference filters to `YEAR=2024`, Spark applies partition pruning and skips six years of data entirely — critical for performance on 35 GB.

**KRaft Mode Kafka**
Runs without Zookeeper, reducing the infrastructure from 8 services to 7 and eliminating a common failure point.

**Class Weighting**
With 78/22 on-time/delayed split, naive training predicts "on-time" for everything and appears 78% accurate. Inverse-frequency class weights force the model to penalize missed delays.

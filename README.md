# Real-Time Flight Delay Prediction

This project builds an end-to-end pipeline for flight delay prediction using Spark, Kafka, and HDFS. The main idea is to use historical BTS flight data for storage and model training, then support real-time scoring later through streaming.

## Historical data pipeline

This part of the project handles the historical storage stage. It takes BTS Airline On-Time Performance data from 2018 through 2023, cleans it with PySpark, converts it to Parquet, and stores it in HDFS partitioned by year and month.

## Dataset

The source data is the Bureau of Transportation Statistics On-Time Performance dataset.

For this stage, the pipeline uses:
- 2018 through 2023
- all 12 months for each year
- 72 CSV files total

## Ingestion command

Run this from the project root:

python src/ingestion/ingest_bts_to_hdfs.py \
  --input-path data/raw \
  --hdfs-path local_parquet_out \
  --years 2018 2019 2020 2021 2022 2023

## Upload the processed output to HDFS

After the local Parquet output is created:

docker exec -it hdfs-namenode bash

Inside the container:
rm -rf /tmp/local_parquet_out
hdfs dfs -rm -r -f /data/processed/bts_parquet
exit

Back in the project root:
docker cp local_parquet_out hdfs-namenode:/tmp/
docker exec -it hdfs-namenode bash

Inside the container again:
hdfs dfs -mkdir -p /data/processed/bts_parquet
hdfs dfs -put /tmp/local_parquet_out/* /data/processed/bts_parquet

## What the ingestion script does

The script:
- reads all raw CSV files from the yearly folders
- keeps the columns needed for modeling
- renames BTS column names into a cleaner schema
- casts columns to the correct types
- handles missing delay values
- creates a binary target where arrival delay greater than 15 minutes is treated as delayed
- writes the output in Parquet format
- partitions the output by YEAR and MONTH

## Output location

The processed dataset is stored in HDFS at:

/data/processed/bts_parquet

The final structure looks like this:

- YEAR=2018/MONTH=1 through MONTH=12
- YEAR=2019/MONTH=1 through MONTH=12
- YEAR=2020/MONTH=1 through MONTH=12
- YEAR=2021/MONTH=1 through MONTH=12
- YEAR=2022/MONTH=1 through MONTH=12
- YEAR=2023/MONTH=1 through MONTH=12

## HDFS verification

To enter the HDFS container:

docker exec -it hdfs-namenode bash

To list the top-level processed output:

hdfs dfs -ls /data/processed/bts_parquet

To inspect one year in order:

hdfs dfs -ls /data/processed/bts_parquet/YEAR=2018 | sort -V

## Batch feature engineering and EDA pipeline

This part of the project handles the batch preprocessing stage. It reads the ingested Parquet dataset from HDFS, performs additional cleaning and feature engineering with PySpark, and writes an ML-ready dataset back to HDFS. It also generates EDA summaries for data validation and reporting.

Input data pipeline

This stage assumes the historical ingestion step has already completed and the Parquet dataset is already available in HDFS.

It reads:
- historical flight data in Parquet format
- partitioned by year and month
- from the ingestion output directory

## Feature engineering command

Run this inside the Spark container.

To enter the Spark container:

docker exec -it spark-master bash

Then run:

/opt/spark/bin/spark-submit
--master spark://spark-master:7077
src/training/prepare_features.py
--input hdfs://hdfs-namenode:9000/user/spark/data/flights_raw
--cleaned-output hdfs://hdfs-namenode:9000/user/spark/processed/cleaned
--featured-output hdfs://hdfs-namenode:9000/user/spark/processed/featured
--pipeline-model-output hdfs://hdfs-namenode:9000/user/spark/models/preprocessing_pipeline

## Run EDA

After feature engineering completes, run:

/opt/spark/bin/spark-submit
--master spark://spark-master:7077
src/training/eda_report.py
--input hdfs://hdfs-namenode:9000/user/spark/processed/cleaned
--output hdfs://hdfs-namenode:9000/user/spark/reports/eda

## What the feature engineering script does

The script:
- reads Parquet data from HDFS
- filters out invalid records such as cancelled flights or rows with missing arrival delay
- creates a binary target where arrival delay greater than 15 minutes is treated as delayed
- generates time-based features such as departure hour and scheduled arrival hour
- generates route-based features from origin and destination
- handles missing values
- encodes categorical columns using StringIndexer and OneHotEncoder
- assembles all features into a single vector column
- saves a reusable Spark ML preprocessing pipeline
- writes cleaned and feature-engineered datasets in Parquet format

## What the EDA script does

The script:
- reads the cleaned dataset from HDFS
- computes summary statistics
- reports label distribution
- reports missing values
- generates grouped summaries such as delay rate by carrier, airport, month, day of week, and departure hour
- writes the EDA outputs to HDFS for inspection

## Output location

The cleaned dataset is stored in HDFS at:

/user/spark/processed/cleaned

The feature-engineered dataset is stored in HDFS at:

/user/spark/processed/featured

The preprocessing pipeline is stored in HDFS at:

/user/spark/models/preprocessing_pipeline

The EDA reports are stored in HDFS at:

/user/spark/reports/eda

## Output structure

The cleaned and feature-engineered datasets are stored in Parquet format.

The final structure looks like this:

/user/spark/processed/cleaned/YEAR=2018 through YEAR=2023
/user/spark/processed/featured/YEAR=2018 through YEAR=2023

Each record in the feature-engineered dataset includes:
- label
- features

## HDFS verification

To enter the HDFS container:

docker exec -it hdfs-namenode bash

To list the processed output:

hdfs dfs -ls /user/spark/processed

To inspect the cleaned dataset:

hdfs dfs -ls /user/spark/processed/cleaned

To inspect the feature-engineered dataset:

hdfs dfs -ls /user/spark/processed/featured

To inspect the preprocessing pipeline:

hdfs dfs -ls /user/spark/models/preprocessing_pipeline

To inspect the EDA output:

hdfs dfs -ls /user/spark/reports/eda

## Final dataset verification

To enter the Spark container:

docker exec -it spark-master bash

Then start PySpark:

/opt/spark/bin/pyspark

Run the following:

df = spark.read.parquet("hdfs://hdfs-namenode:9000/user/spark/processed/featured")
df.select("label", "features").show(5)
df.groupBy("label").count().show()

## Notes
- The feature pipeline should read only the raw ingested dataset and not a folder that also contains processed outputs
- Processed outputs should be stored separately from the raw ingestion directory
- The output directories are overwritten on each run
- This stage produces the final ML-ready dataset used for model training

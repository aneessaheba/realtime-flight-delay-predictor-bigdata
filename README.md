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
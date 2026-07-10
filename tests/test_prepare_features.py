from pyspark.sql import SparkSession

from src.training.prepare_features import get_feature_columns_for_mode


def test_pre_departure_feature_selection_excludes_post_departure_leaks():
    spark = SparkSession.builder.master("local[1]").appName("prepare-features-test").getOrCreate()
    try:
        df = spark.createDataFrame(
            [
                (
                    2021,
                    12,
                    8,
                    3,
                    100.0,
                    90.0,
                    25.0,
                    1.0,
                    18,
                    20,
                    0,
                    1,
                    "AA",
                    "SFO",
                    "LAX",
                    12.0,
                )
            ],
            schema="""YEAR int, MONTH int, DAY_OF_MONTH int, DAY_OF_WEEK int,
            DISTANCE double, CRS_ELAPSED_TIME double, DEP_DELAY double,
            DEP_DEL15 double, dep_hour int, arr_sched_hour int,
            is_weekend int, is_holiday_season int, OP_UNIQUE_CARRIER string,
            ORIGIN string, DEST string, TAXI_OUT double""",
        )

        cat_cols, num_cols = get_feature_columns_for_mode(df, mode="pre_departure")
        assert "DEP_DELAY" not in num_cols
        assert "DEP_DEL15" not in num_cols
        assert "TAXI_OUT" not in num_cols
        assert "dep_hour" in num_cols
        assert "arr_sched_hour" in num_cols
        assert "is_weekend" in num_cols
        assert "is_holiday_season" in num_cols

        cat_cols_full, num_cols_full = get_feature_columns_for_mode(df, mode="full")
        assert "DEP_DELAY" not in num_cols_full
        assert "DEP_DEL15" not in num_cols_full
        assert "TAXI_OUT" not in num_cols_full
    finally:
        spark.stop()

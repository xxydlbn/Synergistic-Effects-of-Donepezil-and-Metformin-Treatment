import pandas as pd
import numpy as np
from pyspark.sql.functions import col, array, when, array_remove, datediff, avg, regexp_replace, regexp_extract, coalesce, month, countDistinct, row_number, split, collect_list, to_date, collect_set, first, count, lit, udf, lag, abs, length, monotonically_increasing_id, expr, round
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, when, count, sum as sum_

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, DateType
from pyspark.sql.functions import udf, col
import datetime

from sklearn.impute import KNNImputer

import seaborn as sns

from pyspark.sql.functions import col, first, when, lit

from pyspark.sql.types import FloatType

from functools import reduce

from pyspark.sql.functions import rand

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c5075c7f-8095-4bd4-b99e-cd94af62128c"),
    No_dementia_by_icd10_ge65_unique_id=Input(rid="ri.foundry.main.dataset.862d30b3-4bfe-4d45-891f-06afe7f51210"),
    U071_pcr_valid_zip=Input(rid="ri.foundry.main.dataset.09792d68-2c5f-4dd9-98e8-9fa94e9d2c22")
)
def No_Dementia_ge65_COVID19(No_dementia_by_icd10_ge65_unique_id, U071_pcr_valid_zip):
    df1 = No_dementia_by_icd10_ge65_unique_id
    df2 = U071_pcr_valid_zip

    df_COVID = df2.withColumnRenamed("condition_start_date", "condition_start_date_COVID")
    df_COVID = df_COVID.withColumnRenamed("data_partner_id", "data_partner_id_COVID")

    # Ensure condition_start_date is a date type
    df_COVID = df_COVID.withColumn("condition_start_date", to_date(col("condition_start_date_COVID")))

    # Define window specification
    windowSpec = Window.partitionBy("person_id").orderBy("condition_start_date")

    # Calculate the difference in days from the previous row
    df_COVID = df_COVID.withColumn("prev_date_diff", 
                    abs(datediff(col("condition_start_date_COVID"), 
                                    lag("condition_start_date_COVID", 1).over(windowSpec))))

    # Filter rows where the difference is null (first occurrence) or greater than 28 days
    filtered_df_COVID = df_COVID.filter((col("prev_date_diff").isNull()) | (col("prev_date_diff") >= 28))

    # Optionally, remove the "prev_date_diff" column if you want to revert to the original DataFrame structure
    filtered_df_COVID = filtered_df_COVID.drop("prev_date_diff")

    # Perform the join
    joined_df = df1.join(filtered_df_COVID, df1["person_id"] == filtered_df_COVID["person_id"], "left")

    # Select the relevant columns (including age, race, and sex from df2)
    result_df = joined_df.select(
        df1["*"],  # All columns from df1
        filtered_df_COVID["condition_start_date_COVID"],  # Add columns from df2 as needed
        filtered_df_COVID["data_partner_id_COVID"]
    )

    # Drop duplicate rows across all columns
    df_no_duplicates = result_df.dropDuplicates()

    return(df_no_duplicates)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.454c9170-7e8a-4d61-81b5-3352d8449301"),
    No_Dementia_ge65_no_COVID19_death=Input(rid="ri.foundry.main.dataset.edf84e56-3002-48e4-9d28-1ccad90f6b92")
)
def No_Dementia_ge65_no_COVID19_clean_death(No_Dementia_ge65_no_COVID19_death):
    df = No_Dementia_ge65_no_COVID19_death
    # Step 1: Count occurrences of each person_id
    person_id_counts = df.groupBy("person_id").count()

    # Step 2: Identify person_id values appearing more than once
    multiple_person_ids = person_id_counts.filter(F.col("count") > 1).select("person_id")

    # Step 3: Separate filtering logic
    # Retain all rows for person_id appearing only once
    single_occurrence_df = df.join(multiple_person_ids, on="person_id", how="left_anti")

    # Retain only rows with non-null death_date for person_id appearing more than once
    multiple_occurrence_df = df.join(multiple_person_ids, on="person_id", how="inner") \
                            .filter(F.col("death_date").isNotNull())

    # Step 4: Union the results
    final_df = single_occurrence_df.union(multiple_occurrence_df)

    final_df = final_df.dropDuplicates(["person_id"])

    return(final_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.edf84e56-3002-48e4-9d28-1ccad90f6b92"),
    No_Dementia_ge65_no_COVID_unique=Input(rid="ri.foundry.main.dataset.9b5dfae0-987a-4337-a468-b0d52fb4f108"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def No_Dementia_ge65_no_COVID19_death(No_Dementia_ge65_no_COVID_unique, death, Person_info_refined):
    df1 = No_Dementia_ge65_no_COVID_unique
    df2 = death
    df3 = Person_info_refined
    columns_to_select2 = (["person_id", "death_date"])
    df2 = df2.select(*columns_to_select2)
    # Assuming df1 and df2 are already defined
    # Perform a left join
    result_df = df1.join(df2, on = 'person_id', how="left")
    result_df = result_df.join(df3, on = 'person_id', how="left")

    result_df = result_df.distinct()

    result_df = result_df.withColumn("age_2020", 2020 - col("year_of_birth"))

    return(result_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9b5dfae0-987a-4337-a468-b0d52fb4f108"),
    No_Dementia_ge65_without_COVID19=Input(rid="ri.foundry.main.dataset.1f18a0c0-0b9c-4e98-b5e9-5f3959c11092")
)
def No_Dementia_ge65_no_COVID_unique(No_Dementia_ge65_without_COVID19):
    df = No_Dementia_ge65_without_COVID19
    df_unique = df.dropDuplicates(["person_id"])

    return df_unique

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.647507c6-897d-4c4f-9dab-9ea5eec4e8a2"),
    No_Dementia_ge65_COVID19=Input(rid="ri.foundry.main.dataset.c5075c7f-8095-4bd4-b99e-cd94af62128c")
)
def No_Dementia_ge65_with_COVID19(No_Dementia_ge65_COVID19):
    df = No_Dementia_ge65_COVID19

    # Filter rows where condition_start_date_COVID is not null
    filtered_df = df.filter(F.col("condition_start_date_COVID").isNotNull())

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.97f18b4d-c0cf-4cd6-904c-c896f82da155"),
    No_Dementia_ge65_with_COVID19=Input(rid="ri.foundry.main.dataset.647507c6-897d-4c4f-9dab-9ea5eec4e8a2")
)
def No_Dementia_ge65_with_primary_COVID19(No_Dementia_ge65_with_COVID19):
    df1 = No_Dementia_ge65_with_COVID19
    # Step 1: Identify the earliest condition_start_date for each person_id
    window_earliest_date = Window.partitionBy("person_id").orderBy("condition_start_date_COVID")
    df1_with_rank = df1.withColumn("row_num", row_number().over(window_earliest_date))

    # Keep only the rows with the earliest condition_start_date
    df1_earliest = df1_with_rank.filter(col("row_num") == 1).drop("row_num")

    return(df1_earliest)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ba335056-dc57-45b9-a2d5-09e6a750b568"),
    No_Dementia_ge65_with_primary_COVID19_death=Input(rid="ri.foundry.main.dataset.865b38d4-2fc8-4a66-aec4-7d01efd9487e")
)
def No_Dementia_ge65_with_primary_COVID19_clean_death(No_Dementia_ge65_with_primary_COVID19_death):
    df = No_Dementia_ge65_with_primary_COVID19_death
    # Step 1: Count occurrences of each person_id
    person_id_counts = df.groupBy("person_id").count()

    # Step 2: Identify person_id values appearing more than once
    multiple_person_ids = person_id_counts.filter(F.col("count") > 1).select("person_id")

    # Step 3: Separate filtering logic
    # Retain all rows for person_id appearing only once
    single_occurrence_df = df.join(multiple_person_ids, on="person_id", how="left_anti")

    # Retain only rows with non-null death_date for person_id appearing more than once
    multiple_occurrence_df = df.join(multiple_person_ids, on="person_id", how="inner") \
                            .filter(F.col("death_date").isNotNull())

    # Step 4: Union the results
    final_df = single_occurrence_df.union(multiple_occurrence_df)

    final_df = final_df.dropDuplicates(["person_id"])

    return(final_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.865b38d4-2fc8-4a66-aec4-7d01efd9487e"),
    No_Dementia_ge65_with_primary_COVID19=Input(rid="ri.foundry.main.dataset.97f18b4d-c0cf-4cd6-904c-c896f82da155"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def No_Dementia_ge65_with_primary_COVID19_death(No_Dementia_ge65_with_primary_COVID19, death, Person_info_refined):
    df1 = No_Dementia_ge65_with_primary_COVID19
    df2 = death
    df3 = Person_info_refined
    columns_to_select2 = (["person_id", "death_date"])
    df2 = df2.select(*columns_to_select2)
    # Assuming df1 and df2 are already defined
    # Perform a left join
    result_df = df1.join(df2, on = 'person_id', how="left")
    result_df = result_df.join(df3, on = 'person_id', how="left")

    result_df = result_df.distinct()

    result_df = result_df.withColumn("age_2020", 2020 - col("year_of_birth"))

    return(result_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1f18a0c0-0b9c-4e98-b5e9-5f3959c11092"),
    No_Dementia_ge65_COVID19=Input(rid="ri.foundry.main.dataset.c5075c7f-8095-4bd4-b99e-cd94af62128c")
)
def No_Dementia_ge65_without_COVID19(No_Dementia_ge65_COVID19):
    df = No_Dementia_ge65_COVID19

    # Filter rows where condition_start_date_COVID is not null
    filtered_df = df.filter(F.col("condition_start_date_COVID").isNull())

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.0abe6e9c-2779-4c59-a190-77f3104bb4b8"),
    No_Dementia_ge65_with_primary_COVID19_clean_death=Input(rid="ri.foundry.main.dataset.ba335056-dc57-45b9-a2d5-09e6a750b568")
)
def day30_death_No_Dementia_ge65_with_primary_COVID19_clean_death(No_Dementia_ge65_with_primary_COVID19_clean_death):
    df = No_Dementia_ge65_with_primary_COVID19_clean_death
    # Add a column to check if death_date is within 30 days after condition_start_date
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) & 
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 30) & 
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Aggregate to count the rows where death_within_30_days == 1 and total rows
    summary_df = df.agg(
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.count("*").alias("total_count")
    )

    # Add a proportion column
    summary_df = summary_df.withColumn(
        "proportion_within_30_days",
        F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.885b5540-48ae-438e-b456-2e21299b9fea"),
    No_Dementia_ge65_with_primary_COVID19_clean_death=Input(rid="ri.foundry.main.dataset.ba335056-dc57-45b9-a2d5-09e6a750b568")
)
def day60_death_No_Dementia_ge65_with_primary_COVID19_clean_death(No_Dementia_ge65_with_primary_COVID19_clean_death):
    df = No_Dementia_ge65_with_primary_COVID19_clean_death
    # Add a column to check if death_date is within 30 days after condition_start_date
    df = df.withColumn(
        "death_within_60_days",
        F.when(
            (F.col("death_date").isNotNull()) & 
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) & 
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Aggregate to count the rows where death_within_30_days == 1 and total rows
    summary_df = df.agg(
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.count("*").alias("total_count")
    )

    # Add a proportion column
    summary_df = summary_df.withColumn(
        "proportion_within_60_days",
        F.col("death_within_60_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.be472cdf-aef5-413c-ba15-70480456d188"),
    No_Dementia_ge65_with_primary_COVID19_clean_death=Input(rid="ri.foundry.main.dataset.ba335056-dc57-45b9-a2d5-09e6a750b568")
)
def death_No_Dementia_ge65_with_primary_COVID19_clean_death(No_Dementia_ge65_with_primary_COVID19_clean_death):
    df = No_Dementia_ge65_with_primary_COVID19_clean_death
    # Calculate non-null count and total count
    non_null_count = df.filter(F.col("death_date").isNotNull()).count()
    total_count = df.count()
    proportion = non_null_count / total_count

    # Ensure type consistency by casting all values to DoubleType
    summary_df = spark.createDataFrame([
        ("Non-null death_date count", float(non_null_count)),
        ("Proportion of non-null death_date", float(proportion))
    ], ["Metric", "Value"])

    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.bc9e9124-0426-4821-a6e4-4137a0abf7f6"),
    No_Dementia_ge65_with_primary_COVID19_clean_death=Input(rid="ri.foundry.main.dataset.ba335056-dc57-45b9-a2d5-09e6a750b568")
)
def summarize_age_count_no_dementia_ge65_COVID(No_Dementia_ge65_with_primary_COVID19_clean_death):
    df = No_Dementia_ge65_with_primary_COVID19_clean_death
    df_age_summary = (
        df.groupBy("age_2020")
        .count()
        .orderBy("age_2020")
    )

    return df_age_summary

@transform_pandas(
    Output(rid="ri.vector.main.execute.6ad58412-d718-487a-b452-c237d59fecd2"),
    No_Dementia_ge65_with_primary_COVID19_clean_death=Input(rid="ri.foundry.main.dataset.ba335056-dc57-45b9-a2d5-09e6a750b568")
)
def summarize_year_count_no_dementia_ge65_COVID(No_Dementia_ge65_with_primary_COVID19_clean_death):
    df = No_Dementia_ge65_with_primary_COVID19_clean_death
    df_year_summary = (
        df.withColumn("year", F.year(F.col("condition_start_date_COVID")))
        .groupBy("year")
        .count()
        .orderBy("year")
    )

    return df_year_summary


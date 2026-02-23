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
    Output(rid="ri.foundry.main.dataset.d643771d-a019-4ba4-be2a-20ba4a8f2c30"),
    With_dementia_no_drug_ge65_without_covid19_death=Input(rid="ri.foundry.main.dataset.df84bdd4-ed77-47c2-a035-690e23947d95")
)
def With_Dementia_ge65_no_COVID19_clean_death(With_dementia_no_drug_ge65_without_covid19_death):
    df = With_dementia_no_drug_ge65_without_covid19_death
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
    Output(rid="ri.foundry.main.dataset.34043435-3ff1-4659-b151-3ef411268a69"),
    With_Dementia_ge65_without_COVID19=Input(rid="ri.foundry.main.dataset.c6cb912b-f9c1-4c7f-aa73-e47daa701ac2")
)
def With_Dementia_ge65_no_COVID_unique(With_Dementia_ge65_without_COVID19):
    df = With_Dementia_ge65_without_COVID19
    df_unique = df.dropDuplicates(["person_id"])

    return df_unique

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c6cb912b-f9c1-4c7f-aa73-e47daa701ac2"),
    With_Dementia_no_drug_ge65_COVID19=Input(rid="ri.foundry.main.dataset.7849aedd-79c2-4e3e-bf6c-5a7bc3e43314")
)
def With_Dementia_ge65_without_COVID19(With_Dementia_no_drug_ge65_COVID19):
    df = With_Dementia_no_drug_ge65_COVID19

    # Filter rows where condition_start_date_COVID is not null
    filtered_df = df.filter(F.col("condition_start_date_COVID").isNull())

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7849aedd-79c2-4e3e-bf6c-5a7bc3e43314"),
    Primary_dementia_ge65_no_4drug_exposure=Input(rid="ri.foundry.main.dataset.6988828b-3141-4053-a8a9-617fb555f562"),
    U071_pcr_valid_zip=Input(rid="ri.foundry.main.dataset.09792d68-2c5f-4dd9-98e8-9fa94e9d2c22")
)
def With_Dementia_no_drug_ge65_COVID19(Primary_dementia_ge65_no_4drug_exposure, U071_pcr_valid_zip):
    df1 = Primary_dementia_ge65_no_4drug_exposure
    df2 = U071_pcr_valid_zip

    df1 = df1.select(df1['person_id'], df1['condition_start_date'])
    df1 = df1.withColumnRenamed("condition_start_date", "condition_start_date_Dementia")
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
    Output(rid="ri.foundry.main.dataset.61f08fed-3805-4978-80cc-68d9ea4fda38"),
    With_Dementia_no_drug_ge65_COVID19=Input(rid="ri.foundry.main.dataset.7849aedd-79c2-4e3e-bf6c-5a7bc3e43314")
)
def With_Dementia_no_drug_ge65_with_COVID19(With_Dementia_no_drug_ge65_COVID19):
    df = With_Dementia_no_drug_ge65_COVID19

    # Filter rows where condition_start_date_COVID is not null
    filtered_df = df.filter(F.col("condition_start_date_COVID").isNotNull())

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f8bd4a9f-b163-4003-bffb-5b7ba1e483fa"),
    With_dementia_no_drug_ge65_with_primary_covid19_clean_death=Input(rid="ri.foundry.main.dataset.c9d3bc7e-af51-4542-ac57-8534047fd7c5")
)
def With_dementia_no_drug_ge65_before_primary_covid19_clean_death(With_dementia_no_drug_ge65_with_primary_covid19_clean_death):
    df = With_dementia_no_drug_ge65_with_primary_covid19_clean_death
    # Filter rows where condition_start_date is earlier than condition_start_date_COVID
    df_filtered = df.filter(col("condition_start_date_Dementia") < col("condition_start_date_COVID"))

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c9d3bc7e-af51-4542-ac57-8534047fd7c5"),
    With_dementia_no_drug_ge65_with_primary_covid19_death=Input(rid="ri.foundry.main.dataset.428f119a-c5d2-424a-8fe7-fb95c72ed4a8")
)
def With_dementia_no_drug_ge65_with_primary_covid19_clean_death(With_dementia_no_drug_ge65_with_primary_covid19_death):
    df = With_dementia_no_drug_ge65_with_primary_covid19_death
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
    Output(rid="ri.foundry.main.dataset.428f119a-c5d2-424a-8fe7-fb95c72ed4a8"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    With_Dementia_no_drug_ge65_with_COVID19=Input(rid="ri.foundry.main.dataset.61f08fed-3805-4978-80cc-68d9ea4fda38"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def With_dementia_no_drug_ge65_with_primary_covid19_death(With_Dementia_no_drug_ge65_with_COVID19, death, Person_info_refined):
    df1 = With_Dementia_no_drug_ge65_with_COVID19
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
    Output(rid="ri.foundry.main.dataset.df84bdd4-ed77-47c2-a035-690e23947d95"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    With_Dementia_ge65_no_COVID_unique=Input(rid="ri.foundry.main.dataset.34043435-3ff1-4659-b151-3ef411268a69"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def With_dementia_no_drug_ge65_without_covid19_death(With_Dementia_ge65_no_COVID_unique, death, Person_info_refined):
    df1 = With_Dementia_ge65_no_COVID_unique
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
    Output(rid="ri.vector.main.execute.465dd993-dd82-4eda-9df8-0945118723ac"),
    With_dementia_no_drug_ge65_before_primary_covid19_clean_death=Input(rid="ri.foundry.main.dataset.f8bd4a9f-b163-4003-bffb-5b7ba1e483fa")
)
def day30_death_With_dementia_no_drug_ge65_with_primary_covid19_clean_death(With_dementia_no_drug_ge65_before_primary_covid19_clean_death):
    df = With_dementia_no_drug_ge65_before_primary_covid19_clean_death
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
    Output(rid="ri.vector.main.execute.d693845c-1bd0-4d83-b931-ba5281a03bef"),
    With_dementia_no_drug_ge65_before_primary_covid19_clean_death=Input(rid="ri.foundry.main.dataset.f8bd4a9f-b163-4003-bffb-5b7ba1e483fa")
)
def day60_death_With_dementia_no_drug_ge65_with_primary_covid19_clean_death(With_dementia_no_drug_ge65_before_primary_covid19_clean_death):
    df = With_dementia_no_drug_ge65_before_primary_covid19_clean_death
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
    Output(rid="ri.vector.main.execute.47c69018-3f60-4cfb-98a9-5a75878f9316"),
    With_dementia_no_drug_ge65_before_primary_covid19_clean_death=Input(rid="ri.foundry.main.dataset.f8bd4a9f-b163-4003-bffb-5b7ba1e483fa")
)
def death_With_dementia_no_drug_ge65_with_primary_covid19_clean_death(With_dementia_no_drug_ge65_before_primary_covid19_clean_death):
    df = With_dementia_no_drug_ge65_before_primary_covid19_clean_death
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
    Output(rid="ri.vector.main.execute.9ad95be0-ee2e-4a1e-8bba-f9c1364b1b59"),
    With_dementia_no_drug_ge65_before_primary_covid19_clean_death=Input(rid="ri.foundry.main.dataset.f8bd4a9f-b163-4003-bffb-5b7ba1e483fa")
)
def summarize_age_count_dementia_no_drug_ge65_COVID(With_dementia_no_drug_ge65_before_primary_covid19_clean_death):
    df = With_dementia_no_drug_ge65_before_primary_covid19_clean_death
    df_age_summary = (
        df.groupBy("age_2020")
        .count()
        .orderBy("age_2020")
    )

    return df_age_summary

@transform_pandas(
    Output(rid="ri.vector.main.execute.f06a0902-691b-4bc4-b18c-49d253deb694"),
    With_dementia_no_drug_ge65_before_primary_covid19_clean_death=Input(rid="ri.foundry.main.dataset.f8bd4a9f-b163-4003-bffb-5b7ba1e483fa")
)
def summarize_year_count_no_dementia_ge65_COVID(With_dementia_no_drug_ge65_before_primary_covid19_clean_death):
    df = With_dementia_no_drug_ge65_before_primary_covid19_clean_death
    df_year_summary = (
        df.withColumn("year", F.year(F.col("condition_start_date_COVID")))
        .groupBy("year")
        .count()
        .orderBy("year")
    )

    return df_year_summary


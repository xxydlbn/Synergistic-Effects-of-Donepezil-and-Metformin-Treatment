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
    Output(rid="ri.foundry.main.dataset.485f1417-e161-4907-bcf3-20c57c3b9188"),
    Dementia_ge65_donepezil_only=Input(rid="ri.foundry.main.dataset.f4dc4c9d-ecfb-4038-911d-4216d2dc89fe"),
    U071_pcr_valid_zip=Input(rid="ri.foundry.main.dataset.09792d68-2c5f-4dd9-98e8-9fa94e9d2c22")
)
def Dementia_ge65_donepezil_5mg_10mg_only_COVID19_drug_later(Dementia_ge65_donepezil_only, U071_pcr_valid_zip):
    df1 = Dementia_ge65_donepezil_only
    df2 = U071_pcr_valid_zip

    df1 = df1.withColumnRenamed("person_id_donepezil_pre", 'person_id')

    df1 = df1.select(df1['person_id'], df1['condition_start_date_Dementia'], df1['drug_exposure_start_date'])

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
    Output(rid="ri.foundry.main.dataset.8ed79219-c2dc-40e2-b428-bec8745cd148"),
    Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.d8246954-41f9-45d6-9211-f3ba8b773d8e")
)
def Dementia_ge65_donepezil_5mg_10mg_only_after_primary_COVID19_clean_death_drug_later(Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_clean_death_drug_later
    # Filter rows where condition_start_date is earlier than condition_start_date_COVID
    df_filtered = df.filter(col("drug_exposure_start_date") < col("condition_start_date_COVID"))

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9ebfeace-4eb0-4f00-a7d6-fa59dfe48117"),
    Dementia_ge65_donepezil_5mg_10mg_only_COVID19_drug_later=Input(rid="ri.foundry.main.dataset.485f1417-e161-4907-bcf3-20c57c3b9188")
)
def Dementia_ge65_donepezil_5mg_10mg_only_with_COVID19_drug_later(Dementia_ge65_donepezil_5mg_10mg_only_COVID19_drug_later):
    df = Dementia_ge65_donepezil_5mg_10mg_only_COVID19_drug_later

    # Filter rows where condition_start_date_COVID is not null
    filtered_df = df.filter(F.col("condition_start_date_COVID").isNotNull())

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d8246954-41f9-45d6-9211-f3ba8b773d8e"),
    Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_death_drug_later=Input(rid="ri.foundry.main.dataset.2276492d-88fb-4536-894c-97d4f2c9864e")
)
def Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_clean_death_drug_later(Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.2276492d-88fb-4536-894c-97d4f2c9864e"),
    Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_drug_later=Input(rid="ri.foundry.main.dataset.112c86ba-663a-47f8-8f43-779a1e6b0902"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_death_drug_later(Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_drug_later
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
    Output(rid="ri.foundry.main.dataset.112c86ba-663a-47f8-8f43-779a1e6b0902"),
    Dementia_ge65_donepezil_5mg_10mg_only_with_COVID19_drug_later=Input(rid="ri.foundry.main.dataset.9ebfeace-4eb0-4f00-a7d6-fa59dfe48117")
)
def Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_drug_later(Dementia_ge65_donepezil_5mg_10mg_only_with_COVID19_drug_later):
    df1 = Dementia_ge65_donepezil_5mg_10mg_only_with_COVID19_drug_later
    # Step 1: Identify the earliest condition_start_date for each person_id
    window_earliest_date = Window.partitionBy("person_id").orderBy("condition_start_date_COVID")
    df1_with_rank = df1.withColumn("row_num", row_number().over(window_earliest_date))

    # Keep only the rows with the earliest condition_start_date
    df1_earliest = df1_with_rank.filter(col("row_num") == 1).drop("row_num")

    return(df1_earliest)

@transform_pandas(
    Output(rid="ri.vector.main.execute.63230297-20ec-4785-8f83-31f6a1529b64"),
    Dementia_ge65_donepezil_5mg_10mg_only_after_primary_COVID19_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.8ed79219-c2dc-40e2-b428-bec8745cd148")
)
def day30_death_Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_clean_death_drug_later(Dementia_ge65_donepezil_5mg_10mg_only_after_primary_COVID19_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_10mg_only_after_primary_COVID19_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.70481d94-a966-4c58-a1f0-b933db8ed020"),
    Dementia_ge65_donepezil_5mg_10mg_only_after_primary_COVID19_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.8ed79219-c2dc-40e2-b428-bec8745cd148")
)
def day60_death_Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_clean_death_drug_later(Dementia_ge65_donepezil_5mg_10mg_only_after_primary_COVID19_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_10mg_only_after_primary_COVID19_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.64d5048b-c147-43b8-bb1d-62bbde3c01f4"),
    Dementia_ge65_donepezil_5mg_10mg_only_after_primary_COVID19_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.8ed79219-c2dc-40e2-b428-bec8745cd148")
)
def death_Dementia_ge65_donepezil_5mg_10mg_only_with_primary_COVID19_clean_death_drug_later(Dementia_ge65_donepezil_5mg_10mg_only_after_primary_COVID19_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_10mg_only_after_primary_COVID19_clean_death_drug_later
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


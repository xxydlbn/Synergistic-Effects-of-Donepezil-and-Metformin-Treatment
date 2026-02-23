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
    Output(rid="ri.foundry.main.dataset.6e52f897-da3c-4cfe-ad99-224ed1a61b14"),
    Dementia_ge65_donepezil_10mg_only_drug_later=Input(rid="ri.foundry.main.dataset.4881cb35-5896-4ed2-8d99-f589ce2ebcc1"),
    U071_pcr_valid_zip=Input(rid="ri.foundry.main.dataset.09792d68-2c5f-4dd9-98e8-9fa94e9d2c22")
)
def Dementia_ge65_donepezil_10mg_only_COVID19_drug_later(Dementia_ge65_donepezil_10mg_only_drug_later, U071_pcr_valid_zip):
    df1 = Dementia_ge65_donepezil_10mg_only_drug_later
    df2 = U071_pcr_valid_zip

    df1 = df1.withColumnRenamed("person_id_donepezil_10", 'person_id')
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
    Output(rid="ri.foundry.main.dataset.a14cbe3a-ede3-492b-adab-93e532a66528"),
    Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.d9e1046a-74dd-4e78-b05b-3c1315f3b183")
)
def Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_clean_death_drug_later
    # Filter rows where condition_start_date is earlier than condition_start_date_COVID
    df_filtered = df.filter(col("drug_exposure_start_date") < col("condition_start_date_COVID"))

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.725d2c09-932a-4474-8eda-cb54f05c3b90"),
    Dementia_ge65_donepezil_10mg_only_COVID19_drug_later=Input(rid="ri.foundry.main.dataset.6e52f897-da3c-4cfe-ad99-224ed1a61b14")
)
def Dementia_ge65_donepezil_10mg_only_with_COVID19_drug_later(Dementia_ge65_donepezil_10mg_only_COVID19_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_COVID19_drug_later

    # Filter rows where condition_start_date_COVID is not null
    filtered_df = df.filter(F.col("condition_start_date_COVID").isNotNull())

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d9e1046a-74dd-4e78-b05b-3c1315f3b183"),
    Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_death_drug_later=Input(rid="ri.foundry.main.dataset.6f7959d6-0166-49f3-8672-07081da5ea34")
)
def Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.6f7959d6-0166-49f3-8672-07081da5ea34"),
    Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_drug_later=Input(rid="ri.foundry.main.dataset.e65160e5-eeb2-406c-a3d4-911f37895895"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_death_drug_later(Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_drug_later
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
    Output(rid="ri.foundry.main.dataset.e65160e5-eeb2-406c-a3d4-911f37895895"),
    Dementia_ge65_donepezil_10mg_only_with_COVID19_drug_later=Input(rid="ri.foundry.main.dataset.725d2c09-932a-4474-8eda-cb54f05c3b90")
)
def Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_drug_later(Dementia_ge65_donepezil_10mg_only_with_COVID19_drug_later):
    df1 = Dementia_ge65_donepezil_10mg_only_with_COVID19_drug_later
    # Step 1: Identify the earliest condition_start_date for each person_id
    window_earliest_date = Window.partitionBy("person_id").orderBy("condition_start_date_COVID")
    df1_with_rank = df1.withColumn("row_num", row_number().over(window_earliest_date))

    # Keep only the rows with the earliest condition_start_date
    df1_earliest = df1_with_rank.filter(col("row_num") == 1).drop("row_num")

    return(df1_earliest)

@transform_pandas(
    Output(rid="ri.vector.main.execute.9c021393-a716-4e3d-b9dd-5175ff76a9aa"),
    Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.a14cbe3a-ede3-492b-adab-93e532a66528")
)
def day30_death_Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.7809f4f5-251a-4b76-a8c6-946a79f7c3ff"),
    Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.a14cbe3a-ede3-492b-adab-93e532a66528")
)
def day60_death_Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.7b5e79e1-9f93-40eb-910c-b01f2a73bd97"),
    Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.a14cbe3a-ede3-492b-adab-93e532a66528")
)
def death_Dementia_ge65_donepezil_10mg_only_with_primary_COVID19_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.957a5818-ccfc-4f8f-8913-1f01cb7b6420"),
    Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.a14cbe3a-ede3-492b-adab-93e532a66528")
)
def summarize_age_count_dementia_donepezil10mg_ge65_COVID(Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_before_primary_COVID19_clean_death_drug_later
    df_age_summary = (
        df.groupBy("age_2020")
        .count()
        .orderBy("age_2020")
    )

    return df_age_summary


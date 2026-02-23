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
    Output(rid="ri.foundry.main.dataset.1a38067b-ffb4-4232-8aa7-3aae47849cb0"),
    Dementia_ge65_donepezil_5mg_only_drug_later=Input(rid="ri.foundry.main.dataset.1b037dee-1606-4625-9dad-4a1ce8ae416b"),
    U071_pcr_valid_zip=Input(rid="ri.foundry.main.dataset.09792d68-2c5f-4dd9-98e8-9fa94e9d2c22")
)
def Dementia_ge65_donepezil_5mg_only_COVID_drug_later(Dementia_ge65_donepezil_5mg_only_drug_later, U071_pcr_valid_zip):
    df1 = Dementia_ge65_donepezil_5mg_only_drug_later
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
    joined_df = df1.join(filtered_df_COVID, df1["person_id_donepezil_5"] == filtered_df_COVID["person_id"], "left")

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
    Output(rid="ri.foundry.main.dataset.67ffb92d-1914-4e77-9b0c-97a70e972bd0"),
    Dementia_ge65_donepezil_5mg_only_COVID_drug_later=Input(rid="ri.foundry.main.dataset.1a38067b-ffb4-4232-8aa7-3aae47849cb0")
)
def Dementia_ge65_donepezil_5mg_only_COVID_primary_matching_drug_later(Dementia_ge65_donepezil_5mg_only_COVID_drug_later):
    df1 = Dementia_ge65_donepezil_5mg_only_COVID_drug_later
    # Step 1: Identify the earliest condition_start_date for each person_id
    window_earliest_date = Window.partitionBy("person_id_donepezil_5").orderBy("condition_start_date_COVID")
    df1_with_rank = df1.withColumn("row_num", row_number().over(window_earliest_date))

    # Keep only the rows with the earliest condition_start_date
    df1_earliest = df1_with_rank.filter(col("row_num") == 1).drop("row_num")

    return(df1_earliest)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3685475e-e15e-4542-94bf-d33c7b96ff7d"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later=Input(rid="ri.foundry.main.dataset.057c01be-ec46-478f-b264-4d97b7c66770")
)
def Dementia_ge65_donepezil_5mg_only_primary_COVID_after_donepezil_drug_later(Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later
    # Filter rows where condition_start_date is earlier than condition_start_date_COVID
    df_filtered = df.filter(col("drug_exposure_start_date") >= col("condition_start_date_COVID"))

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bab32e26-373a-4497-b742-adb18fe4ace4"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later=Input(rid="ri.foundry.main.dataset.b7ab2d6d-59cb-4205-879b-549f03fd82a9")
)
def Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_Dementia_l1y_drug_later(Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later
    # Filter the dataframe by checking if drug_exposure_date is at least 1 year earlier
    # Filter rows where drug_exposure_date is within 1 year before condition_start_date_COVID
    filtered_df = df.filter(
        (F.col("condition_start_date_Dementia") >= F.date_add(F.col("condition_start_date_COVID"), -365)) &  # At least 1 year before
        (F.col("condition_start_date_Dementia") <= F.col("condition_start_date_COVID"))                      # Not after the condition date
    )

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.81af3f8b-0c81-41c9-8741-680dc45788f9"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later=Input(rid="ri.foundry.main.dataset.b7ab2d6d-59cb-4205-879b-549f03fd82a9")
)
def Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_Dementia_m1y_drug_later(Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later
    # Filter the dataframe by checking if drug_exposure_date is at least 1 year earlier
    filtered_df = df.filter(F.col("condition_start_date_Dementia") < F.date_add(F.col("condition_start_date_COVID"), -365))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e298f06f-dd1d-46fc-915d-ba5798d870f4"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_l1y_drug_later=Input(rid="ri.foundry.main.dataset.7837f30c-7679-4c4c-879d-939c6c7de922")
)
def Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_l1m_drug_later(Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_l1y_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_l1y_drug_later
    # Calculate date difference and filter
    filtered_df = df.filter(
        (F.datediff("condition_start_date_COVID", "drug_exposure_start_date") <= 30) &
        (F.datediff("condition_start_date_COVID", "drug_exposure_start_date") > 0)
    )

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7837f30c-7679-4c4c-879d-939c6c7de922"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later=Input(rid="ri.foundry.main.dataset.b7ab2d6d-59cb-4205-879b-549f03fd82a9")
)
def Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_l1y_drug_later(Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later
    # Filter the dataframe by checking if drug_exposure_date is at least 1 year earlier
    # Filter rows where drug_exposure_date is within 1 year before condition_start_date_COVID
    filtered_df = df.filter(
        (F.col("drug_exposure_start_date") >= F.date_add(F.col("condition_start_date_COVID"), -365)) &  # At least 1 year before
        (F.col("drug_exposure_start_date") <= F.col("condition_start_date_COVID"))                      # Not after the condition date
    )

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.cfa1ae82-dc65-4d63-b661-a630377767f1"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later=Input(rid="ri.foundry.main.dataset.b7ab2d6d-59cb-4205-879b-549f03fd82a9")
)
def Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_m1m_for_step8_drug_later(Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later
    # Filter the dataframe by checking if drug_exposure_date is at least 1 year earlier
    filtered_df = df.filter(F.col("drug_exposure_start_date") < F.date_add(F.col("condition_start_date_COVID"), -30))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.374c76cc-1bc2-439d-9e1e-7a3c08c73198"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_l1y_drug_later=Input(rid="ri.foundry.main.dataset.7837f30c-7679-4c4c-879d-939c6c7de922")
)
def Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_m1m_l1y_drug_later(Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_l1y_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_l1y_drug_later
    # Calculate date difference and filter
    filtered_df = df.filter(
        (F.datediff("condition_start_date_COVID", "drug_exposure_start_date") > 30) &
        (F.datediff("condition_start_date_COVID", "drug_exposure_start_date") <= 365)
    )
    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.658eb40c-3487-46e9-ad40-17abebab30d3"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later=Input(rid="ri.foundry.main.dataset.b7ab2d6d-59cb-4205-879b-549f03fd82a9")
)
def Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_donepezil_m1y_drug_later(Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later
    # Filter the dataframe by checking if drug_exposure_date is at least 1 year earlier
    filtered_df = df.filter(F.col("drug_exposure_start_date") < F.date_add(F.col("condition_start_date_COVID"), -365))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b7ab2d6d-59cb-4205-879b-549f03fd82a9"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later=Input(rid="ri.foundry.main.dataset.057c01be-ec46-478f-b264-4d97b7c66770")
)
def Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later(Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later
    # Filter rows where condition_start_date is earlier than condition_start_date_COVID
    df_filtered = df.filter(col("drug_exposure_start_date") < col("condition_start_date_COVID"))

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.057c01be-ec46-478f-b264-4d97b7c66770"),
    Dementia_ge65_donepezil_5mg_only_COVID_primary_matching_drug_later=Input(rid="ri.foundry.main.dataset.67ffb92d-1914-4e77-9b0c-97a70e972bd0")
)
def Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later(Dementia_ge65_donepezil_5mg_only_COVID_primary_matching_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_COVID_primary_matching_drug_later
    # Filter rows where condition_start_date_COVID is null
    df_filtered = df.filter(col("condition_start_date_COVID").isNotNull())

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.11787e2f-503f-4f3f-bdf9-5495ec3d6f1f"),
    Dementia_ge65_donepezil_5mg_only_COVID_primary_matching_drug_later=Input(rid="ri.foundry.main.dataset.67ffb92d-1914-4e77-9b0c-97a70e972bd0")
)
def Dementia_ge65_donepezil_5mg_only_without_COVID_drug_later(Dementia_ge65_donepezil_5mg_only_COVID_primary_matching_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_COVID_primary_matching_drug_later
    # Filter rows where condition_start_date_COVID is null
    df_filtered = df.filter(col("condition_start_date_COVID").isNull())

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.910373dd-2c04-4f6c-ad61-07766254b874"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_after_donepezil_drug_later=Input(rid="ri.foundry.main.dataset.3685475e-e15e-4542-94bf-d33c7b96ff7d"),
    Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later=Input(rid="ri.foundry.main.dataset.b7ab2d6d-59cb-4205-879b-549f03fd82a9")
)
def Grouped_Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later(Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later, Dementia_ge65_donepezil_5mg_only_primary_COVID_after_donepezil_drug_later):
    df1 = Dementia_ge65_donepezil_5mg_only_primary_COVID_before_donepezil_drug_later
    df2 = Dementia_ge65_donepezil_5mg_only_primary_COVID_after_donepezil_drug_later

    # Add the 'group' column to df1
    df1 = df1.withColumn("group", lit("Donepezil_before_COVID"))

    # Add the 'group' column to df2
    df2 = df2.withColumn("group", lit("Donepezil_after_COVID"))

    df_union = df1.union(df2)

    return(df_union)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.44ff9ce9-82d5-484e-9849-34923d5ec9f9"),
    Grouped_Dementia_ge65_donepezil_5mg_only_primary_COVID_info_drug_later=Input(rid="ri.foundry.main.dataset.1d6de546-09ad-433b-b4f2-95bd8a9fed54")
)
def Grouped_Dementia_ge65_donepezil_5mg_only_primary_COVID_info_cleaned_drug_later(Grouped_Dementia_ge65_donepezil_5mg_only_primary_COVID_info_drug_later):
    df = Grouped_Dementia_ge65_donepezil_5mg_only_primary_COVID_info_drug_later
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
    Output(rid="ri.foundry.main.dataset.1d6de546-09ad-433b-b4f2-95bd8a9fed54"),
    Grouped_Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later=Input(rid="ri.foundry.main.dataset.910373dd-2c04-4f6c-ad61-07766254b874"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Grouped_Dementia_ge65_donepezil_5mg_only_primary_COVID_info_drug_later(Grouped_Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later, Person_info_refined, death):
    df1 = Grouped_Dementia_ge65_donepezil_5mg_only_primary_COVID_drug_later
    df2 = death
    df3 = Person_info_refined
    df1 = df1.withColumnRenamed("person_id_donepezil_5", 'person_id')
    columns_to_select2 = (["person_id", "death_date"])
    df2 = df2.select(*columns_to_select2)
    # Assuming df1 and df2 are already defined
    # Perform a left join
    result_df = df1.join(df2, on = 'person_id', how="left")
    result_df = result_df.join(df3, on = 'person_id', how="left")

    result_df = result_df.distinct()

    result_df = result_df.withColumn("age_2020", 2020 - col("year_of_birth"))

    return(result_df)


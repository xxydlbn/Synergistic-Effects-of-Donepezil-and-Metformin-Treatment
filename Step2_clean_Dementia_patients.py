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
    Output(rid="ri.foundry.main.dataset.9f463066-502d-479e-935a-c9f85dfb2044"),
    Dementia_by_ICD10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    Latest_date_no_covid=Input(rid="ri.foundry.main.dataset.a574abba-e673-42c2-a71f-a8d5d898a2b0")
)
def Dementa_no_COVID(Dementia_by_ICD10_ge65, Latest_date_no_covid):
    df1 = Dementia_by_ICD10_ge65
    df2 = Latest_date_no_covid
    # Remove any person_id from df1 that appears in df2
    df1_filtered = df1.join(df2, on="person_id", how="inner")

    return df1_filtered

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    all_condition_occurrence_ge65=Input(rid="ri.foundry.main.dataset.bac29600-a5bf-4692-bf2c-6e72fd65bd9e")
)
def Dementia_by_ICD10_ge65(all_condition_occurrence_ge65):
    df = all_condition_occurrence_ge65
    Dementia_icd10_codes = [
    "F00", "F00.0", "F00.1", "F00.2", "F00.9",
    "F01", "F01.1", "F01.2", "F01.3", "F01.8", "F01.9",
    "F02", "F02.0", "F02.1", "F02.2", "F02.3", "F02.4", "F02.8",
    "F03", "F05.1", "F05.9", "F06.0", "F06.7",
    "F10.7", "G30", "G30.0", "G30.1", "G30.8", "G30.9",
    "G31.0", "G31.1", "G31.8"
    ]

    # Build a regex pattern that matches any of the ICD-10 codes
    regex_pattern = "|".join([f"\\b{code}\\b" for code in Dementia_icd10_codes])  # Use word boundaries for precise matches

    # Filter rows where condition_source_value contains any of the ICD-10 codes
    filtered_df = df.filter(col("condition_source_value").rlike(regex_pattern))

    unique_df = filtered_df.distinct()

    return(unique_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8972f1f2-1bca-4c17-b6a2-4467581fa90c"),
    Dementia_by_ICD10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117")
)
def Dementia_by_ICD10_ge65_unique_ID(Dementia_by_ICD10_ge65):
    df = Dementia_by_ICD10_ge65
    df_unique = df.select(df['person_id']).distinct()

    return(df_unique)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.95e4b3db-4403-4cac-beee-f29fee84e9ee"),
    Dementia_by_ICD10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    all_condition_occurrence_ge65=Input(rid="ri.foundry.main.dataset.bac29600-a5bf-4692-bf2c-6e72fd65bd9e")
)
def No_Dementia_by_ICD10_ge65(all_condition_occurrence_ge65, Dementia_by_ICD10_ge65):
    df = all_condition_occurrence_ge65
    df_dementia = Dementia_by_ICD10_ge65

    df_no_dementia = df.join(df_dementia, on = 'person_id', how = 'left_anti')

    
    return(df_no_dementia)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.862d30b3-4bfe-4d45-891f-06afe7f51210"),
    No_Dementia_by_ICD10_ge65=Input(rid="ri.foundry.main.dataset.95e4b3db-4403-4cac-beee-f29fee84e9ee")
)
def No_Dementia_by_ICD10_ge65_unique_ID(No_Dementia_by_ICD10_ge65):
    df = No_Dementia_by_ICD10_ge65
    df_unique = df.select(df['person_id']).distinct()

    return(df_unique)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e4971317-3011-4aa9-b783-0e80b02c03e0"),
    Latest_date_no_covid=Input(rid="ri.foundry.main.dataset.a574abba-e673-42c2-a71f-a8d5d898a2b0"),
    No_Dementia_by_ICD10_ge65=Input(rid="ri.foundry.main.dataset.95e4b3db-4403-4cac-beee-f29fee84e9ee")
)
def No_dementa_no_COVID(No_Dementia_by_ICD10_ge65, Latest_date_no_covid):
    df1 = No_Dementia_by_ICD10_ge65
    df2 = Latest_date_no_covid
    # Remove any person_id from df1 that appears in df2
    df1_filtered = df1.join(df2, on="person_id", how="inner")

    return df1_filtered

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bac29600-a5bf-4692-bf2c-6e72fd65bd9e"),
    Age_ge65_at_2020=Input(rid="ri.foundry.main.dataset.f5490b6b-4d7c-423c-bb8a-9ffa3c01694e"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.900fa2ad-87ea-4285-be30-c6b5bab60e86")
)
def all_condition_occurrence_ge65(condition_occurrence, Age_ge65_at_2020):
    df1 = condition_occurrence
    df2 = Age_ge65_at_2020

    df_filtered = df1.join(df2, on = 'person_id', how = 'inner')

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5ecf2742-35e2-4629-b946-02326e6875dd"),
    all_condition_occurrence_ge65=Input(rid="ri.foundry.main.dataset.bac29600-a5bf-4692-bf2c-6e72fd65bd9e")
)
def all_condition_occurrence_ge65_unique_ID(all_condition_occurrence_ge65):
    df = all_condition_occurrence_ge65
    df_unique = df.select(df['person_id']).distinct()

    return(df_unique)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.025a8c98-05a8-4005-9a5b-e3851af6b6a2"),
    Dementa_no_COVID=Input(rid="ri.foundry.main.dataset.9f463066-502d-479e-935a-c9f85dfb2044")
)
def cleaned_dementia_no_COVID(Dementa_no_COVID):
    df = Dementa_no_COVID
    # Define window per person_id ordered by ascending date
    w = Window.partitionBy("person_id").orderBy(F.col("condition_start_date").asc())

    df_first = (
        df.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)   # keep only earliest row
        .drop("rn")
    )

    return df_first

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bef29768-578b-4798-aad4-323a20ac529a"),
    No_dementa_no_COVID=Input(rid="ri.foundry.main.dataset.e4971317-3011-4aa9-b783-0e80b02c03e0")
)
def cleaned_no_dementia_no_COVID(No_dementa_no_COVID):
    df = No_dementa_no_COVID
    # Assign row numbers
    # Define window partitioned by person_id, ordered by date descending
    w = Window.partitionBy("person_id").orderBy(F.col("latest_date").desc())
    df_latest = (
        df
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)   # keep only latest row
        .drop("rn")                 # optional: drop helper column
    )

    return df_latest

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8a4bbdc3-c1db-476b-be0a-94fe3dc50d37"),
    Dementia_ge65_all_drug_exposure=Input(rid="ri.foundry.main.dataset.431952b0-0961-4562-be62-9b06c493da17"),
    cleaned_no_dementia_no_COVID=Input(rid="ri.foundry.main.dataset.bef29768-578b-4798-aad4-323a20ac529a")
)
def no_dementia_no_drug_no_COVID_ge65(cleaned_no_dementia_no_COVID, Dementia_ge65_all_drug_exposure):
    df1 = cleaned_no_dementia_no_COVID
    df2 = Dementia_ge65_all_drug_exposure

    # Remove any person_id from df1 that appears in df2
    df1_filtered = df1.join(df2, on="person_id", how="left_anti")

    return df1_filtered

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.25d3973d-8d68-4d4c-b565-5721928219e2"),
    Dementia_by_ICD10_ge65_unique_ID=Input(rid="ri.foundry.main.dataset.8972f1f2-1bca-4c17-b6a2-4467581fa90c"),
    No_Dementia_by_ICD10_ge65_unique_ID=Input(rid="ri.foundry.main.dataset.862d30b3-4bfe-4d45-891f-06afe7f51210")
)
def test_inner(Dementia_by_ICD10_ge65_unique_ID, No_Dementia_by_ICD10_ge65_unique_ID):
    df1 = Dementia_by_ICD10_ge65_unique_ID
    df2 = No_Dementia_by_ICD10_ge65_unique_ID

    df_result = df1.join(df2, on = 'person_id', how = 'inner')

    return(df_result)


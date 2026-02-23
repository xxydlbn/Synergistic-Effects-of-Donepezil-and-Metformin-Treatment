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
    Output(rid="ri.foundry.main.dataset.86d53657-fbb1-4f34-8de2-7579e2de3bf2"),
    Donepezil_name_match_fo=Input(rid="ri.foundry.main.dataset.84c0d2b7-6aa4-433b-9080-d87b04490603"),
    Galantamine_name_match_fo=Input(rid="ri.foundry.main.dataset.0430f633-a704-457b-bb5e-e042fb0753ea"),
    Memantine_name_match_fo=Input(rid="ri.foundry.main.dataset.e4f254fa-8281-496e-8df6-cca11b4e5b8d"),
    Rivastigmine_name_match_fo=Input(rid="ri.foundry.main.dataset.400f716c-b0f6-44b6-b721-b013cebb9f05")
)
def Dementia_drug_exposure_person_id_summary_refined(Donepezil_name_match_fo, Memantine_name_match_fo, Rivastigmine_name_match_fo, Galantamine_name_match_fo):
    df1 = Donepezil_name_match_fo
    df2 = Memantine_name_match_fo
    df3 = Rivastigmine_name_match_fo
    df4 = Galantamine_name_match_fo

    df_union = df1.union(df2)
    df_union = df_union.union(df3)
    df_union = df_union.union(df4)

    return(df_union)
    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.034f074d-eaa2-4d3c-b1f5-693f520bcea1"),
    Dementia_by_icd10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    Dementia_drug_exposure_person_id_summary_refined=Input(rid="ri.foundry.main.dataset.86d53657-fbb1-4f34-8de2-7579e2de3bf2")
)
def Dementia_ge65_all_drug_exposure(Dementia_by_icd10_ge65,Dementia_drug_exposure_person_id_summary_refined):
    df1 = Dementia_by_icd10_ge65
    df2 = Dementia_drug_exposure_person_id_summary_refined

    df1_selected = df1.select(df1['person_id'], df1['condition_start_date'], df1['condition_concept_name'], 
    df1['condition_source_value'], df1['gender_concept_name'], df1['race_concept_name'], df1['age_2020'] )

    df_filtered = df1_selected.join(df2, on = 'person_id', how = 'inner')

    df_filtered = df_filtered.distinct()

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3777ef04-197b-4e75-935a-ef69ab9c7723"),
    Dementia_by_icd10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    Dementia_ge65_all_drug_exposure=Input(rid="ri.foundry.main.dataset.034f074d-eaa2-4d3c-b1f5-693f520bcea1")
)
def Dementia_ge65_no_4drug_exposure(Dementia_by_icd10_ge65, Dementia_ge65_all_drug_exposure):
    df1 = Dementia_by_icd10_ge65
    df2 = Dementia_ge65_all_drug_exposure

    df_ge65_no_drug = df1.join(df2, on = "person_id", how = 'left_anti')

    return(df_ge65_no_drug)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.266f3902-d887-49eb-8264-5b8b7f6ae067"),
    Dementia_ge65_no_4drug_exposure=Input(rid="ri.foundry.main.dataset.3777ef04-197b-4e75-935a-ef69ab9c7723")
)
def Dementia_ge65_no_4drug_exposure_unique_ID(Dementia_ge65_no_4drug_exposure):
    df = Dementia_ge65_no_4drug_exposure
    df_unique = df.select(df['person_id']).distinct()

    return(df_unique)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.511b886c-91e6-4b01-b133-44a5eb545ca0"),
    Dementia_by_icd10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    Dementia_other3_drug_exposure_person_id_summary_refined=Input(rid="ri.foundry.main.dataset.61cfccc7-bae5-4d65-80c7-295ae67e3a56")
)
def Dementia_ge65_other3drugs_exposure(Dementia_other3_drug_exposure_person_id_summary_refined, Dementia_by_icd10_ge65):
    df1 = Dementia_by_icd10_ge65
    df2 = Dementia_other3_drug_exposure_person_id_summary_refined

    df1_selected = df1.select(df1['person_id'], df1['condition_start_date'], df1['condition_concept_name'], 
    df1['condition_source_value'], df1['gender_concept_name'], df1['race_concept_name'], df1['age_2020'] )

    df_filtered = df1_selected.join(df2, on = 'person_id', how = 'inner')

    
    df_filtered = df_filtered.distinct()

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.61cfccc7-bae5-4d65-80c7-295ae67e3a56"),
    Galantamine_name_match_fo=Input(rid="ri.foundry.main.dataset.0430f633-a704-457b-bb5e-e042fb0753ea"),
    Memantine_name_match_fo=Input(rid="ri.foundry.main.dataset.e4f254fa-8281-496e-8df6-cca11b4e5b8d"),
    Rivastigmine_name_match_fo=Input(rid="ri.foundry.main.dataset.400f716c-b0f6-44b6-b721-b013cebb9f05")
)
def Dementia_other3_drug_exposure_person_id_summary_refined(Memantine_name_match_fo, Rivastigmine_name_match_fo, Galantamine_name_match_fo):
    df2 = Memantine_name_match_fo
    df3 = Rivastigmine_name_match_fo
    df4 = Galantamine_name_match_fo

    df_union = df2.union(df3)
    df_union = df_union.union(df4)

    return(df_union)
    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.84c0d2b7-6aa4-433b-9080-d87b04490603"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
def Donepezil_name_match_fo(drug_exposure):
    df = drug_exposure
    # Assuming df is your dataframe
    filtered_df = df.filter((col("drug_concept_name").contains("donepezil")) | 
                            (col("drug_concept_name").contains("Donepezil")))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0430f633-a704-457b-bb5e-e042fb0753ea"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
def Galantamine_name_match_fo(drug_exposure):
    df = drug_exposure
    # Assuming df is your dataframe
    filtered_df = df.filter((col("drug_concept_name").contains("galantamine")) | 
                            (col("drug_concept_name").contains("Galantamine")))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d7cc9076-caaf-4bc8-94ae-b94a2a532512"),
    Donepezil_name_match_fo=Input(rid="ri.foundry.main.dataset.84c0d2b7-6aa4-433b-9080-d87b04490603")
)
def MG10_Donepezil_drug_concept_id(Donepezil_name_match_fo):
    df = Donepezil_name_match_fo
    # List of drug names to filter
    drug_names = [
        "donepezil hydrochloride 10 MG Oral Tablet",
        "donepezil hydrochloride 10 MG Oral Tablet [Aricept]",
        "donepezil hydrochloride 10 MG Disintegrating Oral Tablet"
    ]

    # Filter rows with specific drug names and get distinct drug_concept_id
    unique_drug_concept_ids = (
        df.filter(col("drug_concept_name").isin(drug_names))
        .select("drug_concept_id")
        .distinct()
    )

    return(unique_drug_concept_ids)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e4f254fa-8281-496e-8df6-cca11b4e5b8d"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
def Memantine_name_match_fo(drug_exposure):
    df = drug_exposure
    # Assuming df is your dataframe
    filtered_df = df.filter((col("drug_concept_name").contains("memantine")) | 
                            (col("drug_concept_name").contains("Memantine")))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.689e3026-3574-442c-bdee-1e7344e18a4a"),
    Donepezil_name_match_fo=Input(rid="ri.foundry.main.dataset.84c0d2b7-6aa4-433b-9080-d87b04490603")
)
def Mg5_Donepezil_drug_concept_id(Donepezil_name_match_fo):
    df = Donepezil_name_match_fo
    # List of drug names to filter
    drug_names = [
        "donepezil hydrochloride 5 MG Oral Tablet",
        "donepezil hydrochloride 5 MG Oral Tablet [Aricept]",
        "donepezil hydrochloride 5 MG Disintegrating Oral Tablet"
    ]

    # Filter rows with specific drug names and get distinct drug_concept_id
    unique_drug_concept_ids = (
        df.filter(col("drug_concept_name").isin(drug_names))
        .select("drug_concept_id")
        .distinct()
    )

    return(unique_drug_concept_ids)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8a605301-c580-409f-872c-6c6d4f1cf8da"),
    Dementia_ge65_no_4drug_exposure=Input(rid="ri.foundry.main.dataset.3777ef04-197b-4e75-935a-ef69ab9c7723")
)
def Primary_Dementia_ge65_no_4drug_exposure(Dementia_ge65_no_4drug_exposure):
    df = Dementia_ge65_no_4drug_exposure
    # Define the window spec partitioned by person_id and ordered by condition_start_date
    window_spec = Window.partitionBy("person_id").orderBy("condition_start_date")

    # Add a row_number column to rank rows within each person_id group
    df_ranked = df.withColumn("rank", row_number().over(window_spec))

    # Filter to keep only the rows with rank = 1 (earliest condition_start_date)
    df_earliest = df_ranked.filter(df_ranked["rank"] == 1).drop("rank")

    return(df_earliest)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.400f716c-b0f6-44b6-b721-b013cebb9f05"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
def Rivastigmine_name_match_fo(drug_exposure):
    df = drug_exposure
    # Assuming df is your dataframe
    filtered_df = df.filter((col("drug_concept_name").contains("rivastigmine")) | 
                            (col("drug_concept_name").contains("Rivastigmine")))

    return(filtered_df)


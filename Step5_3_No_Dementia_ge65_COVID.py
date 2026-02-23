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
    Output(rid="ri.foundry.main.dataset.ef2c2215-1c44-4185-8666-e203a5a90c01"),
    Donepezil_name_match_fo=Input(rid="ri.foundry.main.dataset.e483a29f-d28a-48d0-baed-79e310fce3fd"),
    Galantamine_name_match_fo=Input(rid="ri.foundry.main.dataset.4f20b075-5c03-4a70-941e-46248804e125"),
    Memantine_name_match_fo=Input(rid="ri.foundry.main.dataset.2db6286b-b28b-456b-9e90-5ff46e5f9754"),
    Rivastigmine_name_match_fo=Input(rid="ri.foundry.main.dataset.31a34d74-8547-4978-9dd8-6cb81f5c4a8f")
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
    Output(rid="ri.foundry.main.dataset.61f20c2c-6d03-45d1-bf5f-12f4936c7539"),
    Dementia_by_icd10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    Dementia_drug_exposure_person_id_summary_refined=Input(rid="ri.foundry.main.dataset.ef2c2215-1c44-4185-8666-e203a5a90c01")
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
    Output(rid="ri.foundry.main.dataset.d3429dc8-62e8-42f9-990d-e0cd00f1c869"),
    Dementia_by_icd10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    Dementia_ge65_all_drug_exposure=Input(rid="ri.foundry.main.dataset.61f20c2c-6d03-45d1-bf5f-12f4936c7539")
)
def Dementia_ge65_no_4drug_exposure(Dementia_by_icd10_ge65, Dementia_ge65_all_drug_exposure):
    df1 = Dementia_by_icd10_ge65
    df2 = Dementia_ge65_all_drug_exposure

    df_ge65_no_drug = df1.join(df2, on = "person_id", how = 'left_anti')

    return(df_ge65_no_drug)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7811b4d1-d5a2-4b15-a484-b566d9c11f14"),
    Dementia_ge65_no_4drug_exposure=Input(rid="ri.foundry.main.dataset.d3429dc8-62e8-42f9-990d-e0cd00f1c869")
)
def Dementia_ge65_no_4drug_exposure_unique_ID(Dementia_ge65_no_4drug_exposure):
    df = Dementia_ge65_no_4drug_exposure
    df_unique = df.select(df['person_id']).distinct()

    return(df_unique)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.24c15263-5437-4815-8e5c-433e8b0bf143"),
    Dementia_by_icd10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    Dementia_other3_drug_exposure_person_id_summary_refined=Input(rid="ri.foundry.main.dataset.696463b3-357d-4a38-bfec-0b5047fe2081")
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
    Output(rid="ri.foundry.main.dataset.696463b3-357d-4a38-bfec-0b5047fe2081"),
    Galantamine_name_match_fo=Input(rid="ri.foundry.main.dataset.4f20b075-5c03-4a70-941e-46248804e125"),
    Memantine_name_match_fo=Input(rid="ri.foundry.main.dataset.2db6286b-b28b-456b-9e90-5ff46e5f9754"),
    Rivastigmine_name_match_fo=Input(rid="ri.foundry.main.dataset.31a34d74-8547-4978-9dd8-6cb81f5c4a8f")
)
def Dementia_other3_drug_exposure_person_id_summary_refined(Memantine_name_match_fo, Rivastigmine_name_match_fo, Galantamine_name_match_fo):
    df2 = Memantine_name_match_fo
    df3 = Rivastigmine_name_match_fo
    df4 = Galantamine_name_match_fo

    df_union = df2.union(df3)
    df_union = df_union.union(df4)

    return(df_union)
    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.57d648bc-2902-45b1-8823-ac014a1912a3")
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
    Output(rid="ri.foundry.main.dataset.e8eecf9b-2629-416b-9f03-29a3eae491c3")
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
    Output(rid="ri.foundry.main.dataset.7c8a99ef-a1b2-4800-84cc-3e12c9e705d8"),
    Dementia_ge65_no_4drug_exposure=Input(rid="ri.foundry.main.dataset.d3429dc8-62e8-42f9-990d-e0cd00f1c869")
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


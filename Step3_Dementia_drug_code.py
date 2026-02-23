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
    Output(rid="ri.foundry.main.dataset.2c779a01-8068-46d2-b6e1-5f811cc5ba5e"),
    Donepezil_name_match_fo=Input(rid="ri.foundry.main.dataset.708f1239-27a1-4e40-8c0d-0acb8fac89fd"),
    Galantamine_name_match_fo=Input(rid="ri.foundry.main.dataset.6b03cdb3-c09d-4b0b-8be9-0632b97ffd16"),
    Memantine_name_match_fo=Input(rid="ri.foundry.main.dataset.5efd9aaa-68a6-4124-b9d8-b71be0ed2f19"),
    Rivastigmine_name_match_fo=Input(rid="ri.foundry.main.dataset.8e7685a2-18dd-4cee-8e3f-3cf5f5e0207e")
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
    Output(rid="ri.foundry.main.dataset.431952b0-0961-4562-be62-9b06c493da17"),
    Dementia_by_icd10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    Dementia_drug_exposure_person_id_summary_refined=Input(rid="ri.foundry.main.dataset.2c779a01-8068-46d2-b6e1-5f811cc5ba5e")
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
    Output(rid="ri.foundry.main.dataset.2dd8262a-504d-42f5-82af-f74da0c2c133"),
    Dementia_by_icd10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    Dementia_ge65_all_drug_exposure=Input(rid="ri.foundry.main.dataset.431952b0-0961-4562-be62-9b06c493da17")
)
def Dementia_ge65_no_4drug_exposure(Dementia_by_icd10_ge65, Dementia_ge65_all_drug_exposure):
    df1 = Dementia_by_icd10_ge65
    df2 = Dementia_ge65_all_drug_exposure

    df_ge65_no_drug = df1.join(df2, on = "person_id", how = 'left_anti')

    return(df_ge65_no_drug)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e22e0d2a-89cf-4983-9eec-6ba15aabc304"),
    Dementia_ge65_no_4drug_exposure=Input(rid="ri.foundry.main.dataset.2dd8262a-504d-42f5-82af-f74da0c2c133")
)
def Dementia_ge65_no_4drug_exposure_unique_ID(Dementia_ge65_no_4drug_exposure):
    df = Dementia_ge65_no_4drug_exposure
    df_unique = df.select(df['person_id']).distinct()

    return(df_unique)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6c3799ca-e6d9-4d56-8332-a4372fdc1daa"),
    Dementia_by_icd10_ge65=Input(rid="ri.foundry.main.dataset.deba9b6f-3a4f-4a32-9687-6db5a07e3117"),
    Dementia_other3_drug_exposure_person_id_summary_refined=Input(rid="ri.foundry.main.dataset.51391ee5-3daf-4907-ab17-012a83f5d6d2")
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
    Output(rid="ri.foundry.main.dataset.d66cbad0-6b12-4328-bbb6-b9b4d385f641"),
    Cleaned_dementia_no_covid=Input(rid="ri.foundry.main.dataset.025a8c98-05a8-4005-9a5b-e3851af6b6a2"),
    Dementia_drug_exposure_person_id_summary_refined=Input(rid="ri.foundry.main.dataset.2c779a01-8068-46d2-b6e1-5f811cc5ba5e")
)
def Dementia_no_COVID_no_drug(Cleaned_dementia_no_covid, Dementia_drug_exposure_person_id_summary_refined):
    df1 = Cleaned_dementia_no_covid
    df2 = Dementia_drug_exposure_person_id_summary_refined
    # Remove any person_id from df1 that appears in df2
    df1_filtered = df1.join(df2, on="person_id", how="left_anti")

    return df1_filtered
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.51391ee5-3daf-4907-ab17-012a83f5d6d2"),
    Galantamine_name_match_fo=Input(rid="ri.foundry.main.dataset.6b03cdb3-c09d-4b0b-8be9-0632b97ffd16"),
    Memantine_name_match_fo=Input(rid="ri.foundry.main.dataset.5efd9aaa-68a6-4124-b9d8-b71be0ed2f19"),
    Rivastigmine_name_match_fo=Input(rid="ri.foundry.main.dataset.8e7685a2-18dd-4cee-8e3f-3cf5f5e0207e")
)
def Dementia_other3_drug_exposure_person_id_summary_refined(Memantine_name_match_fo, Rivastigmine_name_match_fo, Galantamine_name_match_fo):
    df2 = Memantine_name_match_fo
    df3 = Rivastigmine_name_match_fo
    df4 = Galantamine_name_match_fo

    df_union = df2.union(df3)
    df_union = df_union.union(df4)

    return(df_union)
    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.708f1239-27a1-4e40-8c0d-0acb8fac89fd"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
def Donepezil_name_match_fo(drug_exposure):
    df = drug_exposure
    # Assuming df is your dataframe
    filtered_df = df.filter((col("drug_concept_name").contains("donepezil")) | 
                            (col("drug_concept_name").contains("Donepezil")))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6b03cdb3-c09d-4b0b-8be9-0632b97ffd16"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
def Galantamine_name_match_fo(drug_exposure):
    df = drug_exposure
    # Assuming df is your dataframe
    filtered_df = df.filter((col("drug_concept_name").contains("galantamine")) | 
                            (col("drug_concept_name").contains("Galantamine")))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9471ff27-6d8d-49e2-93bf-1a2a938e262d"),
    Donepezil_name_match_fo=Input(rid="ri.foundry.main.dataset.708f1239-27a1-4e40-8c0d-0acb8fac89fd")
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
    Output(rid="ri.foundry.main.dataset.5efd9aaa-68a6-4124-b9d8-b71be0ed2f19"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
def Memantine_name_match_fo(drug_exposure):
    df = drug_exposure
    # Assuming df is your dataframe
    filtered_df = df.filter((col("drug_concept_name").contains("memantine")) | 
                            (col("drug_concept_name").contains("Memantine")))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e9068519-398f-4762-9648-456a86986e54"),
    Donepezil_name_match_fo=Input(rid="ri.foundry.main.dataset.708f1239-27a1-4e40-8c0d-0acb8fac89fd")
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
    Output(rid="ri.foundry.main.dataset.6988828b-3141-4053-a8a9-617fb555f562"),
    Dementia_ge65_no_4drug_exposure=Input(rid="ri.foundry.main.dataset.2dd8262a-504d-42f5-82af-f74da0c2c133")
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
    Output(rid="ri.foundry.main.dataset.8e7685a2-18dd-4cee-8e3f-3cf5f5e0207e"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
def Rivastigmine_name_match_fo(drug_exposure):
    df = drug_exposure
    # Assuming df is your dataframe
    filtered_df = df.filter((col("drug_concept_name").contains("rivastigmine")) | 
                            (col("drug_concept_name").contains("Rivastigmine")))

    return(filtered_df)


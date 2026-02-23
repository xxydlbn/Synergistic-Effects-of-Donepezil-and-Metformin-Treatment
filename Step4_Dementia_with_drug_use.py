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
    Output(rid="ri.foundry.main.dataset.215e2a90-9583-4ae3-88eb-7e5713b745e3"),
    Dementia_ge65_all_drug_exposure=Input(rid="ri.foundry.main.dataset.431952b0-0961-4562-be62-9b06c493da17")
)
def Dementia_ge65_Donepezil_10mg_exposure_drug_later(Dementia_ge65_all_drug_exposure):
    df = Dementia_ge65_all_drug_exposure
    df = df.withColumnRenamed('condition_start_date', 'condition_start_date_Dementia')
    # Filtering the DataFrame
    filtered_df = df.filter(col("drug_concept_id").isin(['40223768', '40223769', '40223766'
                                                    ]))

    df = filtered_df.select("person_id", 'condition_start_date_Dementia',"drug_exposure_start_date", "drug_concept_id", 'drug_concept_name')
    # Convert condition_start_date from string to date format
    df = df.withColumn("drug_exposure_start_date", to_date(col("drug_exposure_start_date"), "yyyy-MM-dd"))

    df = df.filter(col("condition_start_date_Dementia") <= col("drug_exposure_start_date"))

    # Define a window spec partitioned by person_id and ordered by condition_start_date and drug_exposure_start_date
    window_spec = Window.partitionBy("person_id").orderBy("condition_start_date_Dementia", "drug_exposure_start_date")

    # Add a rank column based on the window spec
    ranked_df = df.withColumn("rank", F.row_number().over(window_spec))

    # Filter rows where rank = 1 (i.e., earliest dates for each person_id)
    earliest_records_df = ranked_df.filter(F.col("rank") == 1).drop("rank")

    df_with_rank = earliest_records_df.withColumnRenamed("person_id", "person_id_donepezil_10")

    return(df_with_rank)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4881cb35-5896-4ed2-8d99-f589ce2ebcc1"),
    Dementia_ge65_Donepezil_mix_use_drug_later=Input(rid="ri.foundry.main.dataset.f02e9068-79b3-45f7-b8cf-ca86df79deb7"),
    Dementia_ge65_Donepezil_only_drug_later=Input(rid="ri.foundry.main.dataset.f4dc4c9d-ecfb-4038-911d-4216d2dc89fe")
)
def Dementia_ge65_Donepezil_10mg_only_drug_later(Dementia_ge65_Donepezil_only_drug_later, Dementia_ge65_Donepezil_mix_use_drug_later):
    df1 = Dementia_ge65_Donepezil_only_drug_later
    df2 = Dementia_ge65_Donepezil_mix_use_drug_later
    df_filtered = df1.join(df2, df1.person_id_donepezil_pre == df2.person_id_donepezil_mix, "left_anti")

    df_filtered = df_filtered.filter(col("drug_concept_id").isin(['40223768', '40223769', '40223766'
                                                    ]))

    df_filtered = df_filtered.withColumnRenamed("person_id_donepezil_pre", "person_id_donepezil_10")

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e8f39d71-4acc-478f-9aff-35c0dbb28658"),
    Dementia_ge65_Donepezil_only_drug_later=Input(rid="ri.foundry.main.dataset.f4dc4c9d-ecfb-4038-911d-4216d2dc89fe")
)
def Dementia_ge65_Donepezil_5mg_10mg_only_drug_later(Dementia_ge65_Donepezil_only_drug_later):
    df = Dementia_ge65_Donepezil_only_drug_later
    df_unique = df.select(df['person_id_donepezil_pre']).distinct()

    return(df_unique)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bc5a25fb-ed67-4256-918d-c3a082d17055"),
    Dementia_ge65_all_drug_exposure=Input(rid="ri.foundry.main.dataset.431952b0-0961-4562-be62-9b06c493da17")
)
def Dementia_ge65_Donepezil_5mg_exposure_drug_later(Dementia_ge65_all_drug_exposure):
    df = Dementia_ge65_all_drug_exposure
    # Filtering the DataFrame
    df = df.withColumnRenamed('condition_start_date', 'condition_start_date_Dementia')
    filtered_df = df.filter(col("drug_concept_id").isin(['40223778', '40223776', '40223779'
                                                    ]))

    df = filtered_df.select("person_id", 'condition_start_date_Dementia',"drug_exposure_start_date", "drug_concept_id", 'drug_concept_name')
    # Convert condition_start_date from string to date format
    df = df.withColumn("drug_exposure_start_date", to_date(col("drug_exposure_start_date"), "yyyy-MM-dd"))

    df = df.filter(col("condition_start_date_Dementia") <= col("drug_exposure_start_date"))

    # Define a window spec partitioned by person_id and ordered by condition_start_date and drug_exposure_start_date
    window_spec = Window.partitionBy("person_id").orderBy("condition_start_date_Dementia", "drug_exposure_start_date")

    # Add a rank column based on the window spec
    ranked_df = df.withColumn("rank", F.row_number().over(window_spec))

    # Filter rows where rank = 1 (i.e., earliest dates for each person_id)
    earliest_records_df = ranked_df.filter(F.col("rank") == 1).drop("rank")

    df_with_rank = earliest_records_df.withColumnRenamed("person_id", "person_id_donepezil_5")

    return(df_with_rank)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1b037dee-1606-4625-9dad-4a1ce8ae416b"),
    Dementia_ge65_Donepezil_mix_use_drug_later=Input(rid="ri.foundry.main.dataset.f02e9068-79b3-45f7-b8cf-ca86df79deb7"),
    Dementia_ge65_Donepezil_only_drug_later=Input(rid="ri.foundry.main.dataset.f4dc4c9d-ecfb-4038-911d-4216d2dc89fe")
)
def Dementia_ge65_Donepezil_5mg_only_drug_later(Dementia_ge65_Donepezil_only_drug_later, Dementia_ge65_Donepezil_mix_use_drug_later):
    df1 = Dementia_ge65_Donepezil_only_drug_later
    df2 = Dementia_ge65_Donepezil_mix_use_drug_later
    df_filtered = df1.join(df2, df1.person_id_donepezil_pre == df2.person_id_donepezil_mix, "left_anti")

    df_filtered = df_filtered.filter(col("drug_concept_id").isin(['40223778', '40223776', '40223779'
                                                    ]))

    df_filtered = df_filtered.withColumnRenamed("person_id_donepezil_pre", "person_id_donepezil_5")

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f02e9068-79b3-45f7-b8cf-ca86df79deb7"),
    Dementia_ge65_Donepezil_only_drug_later=Input(rid="ri.foundry.main.dataset.f4dc4c9d-ecfb-4038-911d-4216d2dc89fe")
)
def Dementia_ge65_Donepezil_mix_use_drug_later(Dementia_ge65_Donepezil_only_drug_later):
    df= Dementia_ge65_Donepezil_only_drug_later
    # Define allowed and disallowed drug concept IDs
    allowed_ids_10 = ['40223768', '40223769', '40223766']
    allowed_ids_5 = ['40223778', '40223776', '40223779']

    # Step 1: Filter rows where drug_concept_id is in allowed_ids_10 and get distinct person_ids
    df_allowed_10 = df.filter(F.col("drug_concept_id").isin(allowed_ids_10)).select("person_id_donepezil_pre").distinct()

    # Step 2: Filter rows where drug_concept_id is in allowed_ids_5 and get distinct person_ids
    df_allowed_5 = df.filter(F.col("drug_concept_id").isin(allowed_ids_5)).select("person_id_donepezil_pre").distinct()

    # Step 3: Find intersection of person_ids from both allowed sets
    intersect_person_ids = df_allowed_10.intersect(df_allowed_5)

    # Step 4: Filter the original DataFrame to include only the intersecting person_ids
    result_df = df.join(intersect_person_ids, on="person_id_donepezil_pre")

    df_result = result_df.withColumnRenamed("person_id_donepezil_pre", "person_id_donepezil_mix")
    return(df_result)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0817fb3a-cddc-4059-aecf-421ab9ed71d7"),
    Dementia_ge65_Donepezil_mix_use_drug_later=Input(rid="ri.foundry.main.dataset.f02e9068-79b3-45f7-b8cf-ca86df79deb7")
)
def Dementia_ge65_Donepezil_mix_use_unique_id_drug_later(Dementia_ge65_Donepezil_mix_use_drug_later):
    df = Dementia_ge65_Donepezil_mix_use_drug_later
    df_unique = df.select(df['person_id_donepezil_mix']).distinct()

    return(df_unique)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f4dc4c9d-ecfb-4038-911d-4216d2dc89fe"),
    Dementia_ge65_Donepezil_10mg_exposure_drug_later=Input(rid="ri.foundry.main.dataset.215e2a90-9583-4ae3-88eb-7e5713b745e3"),
    Dementia_ge65_Donepezil_5mg_exposure_drug_later=Input(rid="ri.foundry.main.dataset.bc5a25fb-ed67-4256-918d-c3a082d17055"),
    Dementia_ge65_other3drugs_exposure=Input(rid="ri.foundry.main.dataset.6c3799ca-e6d9-4d56-8332-a4372fdc1daa")
)
def Dementia_ge65_Donepezil_only_drug_later(Dementia_ge65_Donepezil_5mg_exposure_drug_later,Dementia_ge65_Donepezil_10mg_exposure_drug_later, Dementia_ge65_other3drugs_exposure):
    df1 = Dementia_ge65_Donepezil_5mg_exposure_drug_later
    df2 = Dementia_ge65_Donepezil_10mg_exposure_drug_later
    df3 = Dementia_ge65_other3drugs_exposure

    df1 = df1.withColumnRenamed("person_id_donepezil_5", "person_id_donepezil_pre")
    df2 = df2.withColumnRenamed("person_id_donepezil_10", "person_id_donepezil_pre")
    df_result = df1.union(df2)
    # Perform a left anti join to filter out rows in df1 where person_id_donepezil appears in df2's person_id
    df_filtered = df_result.join(df3, df1.person_id_donepezil_pre == df3.person_id, "left_anti")

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9848403e-ce50-4199-beaf-4935e33b0bd4"),
    Dementia_ge65_Donepezil_10mg_exposure_drug_later=Input(rid="ri.foundry.main.dataset.215e2a90-9583-4ae3-88eb-7e5713b745e3"),
    Dementia_ge65_Donepezil_5mg_exposure_drug_later=Input(rid="ri.foundry.main.dataset.bc5a25fb-ed67-4256-918d-c3a082d17055")
)
def Dementia_ge65_Donepezil_used_drug_later(Dementia_ge65_Donepezil_5mg_exposure_drug_later, Dementia_ge65_Donepezil_10mg_exposure_drug_later):
    df1 = Dementia_ge65_Donepezil_5mg_exposure_drug_later
    df2 = Dementia_ge65_Donepezil_10mg_exposure_drug_later

    df1_id = df1.select(df1['person_id_donepezil_5']).withColumnRenamed("person_id_donepezil_5", "person_id_donepezil")
    df2_id = df2.select(df2['person_id_donepezil_10']).withColumnRenamed("person_id_donepezil_10", "person_id_donepezil")

    df_unique = df1_id.union(df2_id).distinct()

    return(df_unique)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b15e4834-bf18-43a3-8e8d-6d95a6b2a6ca"),
    Dementia_ge65_Donepezil_10mg_exposure_drug_later=Input(rid="ri.foundry.main.dataset.215e2a90-9583-4ae3-88eb-7e5713b745e3"),
    Dementia_ge65_Donepezil_5mg_exposure_drug_later=Input(rid="ri.foundry.main.dataset.bc5a25fb-ed67-4256-918d-c3a082d17055"),
    Dementia_ge65_other3drugs_exposure=Input(rid="ri.foundry.main.dataset.6c3799ca-e6d9-4d56-8332-a4372fdc1daa")
)
def Dementia_ge65_Donepezil_with_other_drugs_drug_later(Dementia_ge65_Donepezil_5mg_exposure_drug_later, Dementia_ge65_Donepezil_10mg_exposure_drug_later, Dementia_ge65_other3drugs_exposure):
    df1 = Dementia_ge65_Donepezil_10mg_exposure_drug_later
    df2 = Dementia_ge65_Donepezil_5mg_exposure_drug_later
    df3 = Dementia_ge65_other3drugs_exposure
    df3 = df3.select(df3["person_id"])
    # Join df1 with df3 on person_id_donepezil_10 = person_id
    df1_join = df1.join(df3, df1.person_id_donepezil_10 == df3.person_id, "inner")

    # Join df2 with df3 on person_id_donepezil_5 = person_id
    df2_join = df2.join(df3, df2.person_id_donepezil_5 == df3.person_id, "inner")

    # Union the two results and drop duplicates
    df_result = df1_join.select("person_id").union(df2_join.select("person_id")).distinct()

    return(df_result)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.39e88c1b-fa1f-46b8-b023-5c5b2d172be3"),
    Dementia_ge65_Donepezil_10mg_exposure_drug_later=Input(rid="ri.foundry.main.dataset.215e2a90-9583-4ae3-88eb-7e5713b745e3"),
    Dementia_ge65_Donepezil_5mg_exposure_drug_later=Input(rid="ri.foundry.main.dataset.bc5a25fb-ed67-4256-918d-c3a082d17055"),
    Dementia_ge65_other3drugs_exposure=Input(rid="ri.foundry.main.dataset.6c3799ca-e6d9-4d56-8332-a4372fdc1daa")
)
def Dementia_ge65_no_donepezil_other3drugs_used_drug_later(Dementia_ge65_other3drugs_exposure, Dementia_ge65_Donepezil_5mg_exposure_drug_later, Dementia_ge65_Donepezil_10mg_exposure_drug_later):
    df1 = Dementia_ge65_other3drugs_exposure
    df2 = Dementia_ge65_Donepezil_5mg_exposure_drug_later
    df3 = Dementia_ge65_Donepezil_10mg_exposure_drug_later

    df_filtered_1 = df1.join(df2, df1.person_id == df2.person_id_donepezil_5, "left_anti")
    df_filtered_2 = df_filtered_1.join(df3, df_filtered_1.person_id == df3.person_id_donepezil_10, "left_anti")

    df_unique_id = df_filtered_2.select(df_filtered_2['person_id']).distinct()

    return(df_unique_id)


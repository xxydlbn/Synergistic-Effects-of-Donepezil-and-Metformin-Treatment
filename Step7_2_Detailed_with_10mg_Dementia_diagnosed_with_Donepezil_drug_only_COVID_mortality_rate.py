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
    Output(rid="ri.foundry.main.dataset.9b5b2413-cf53-492d-b7d7-b940852b5366"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_death_drug_later=Input(rid="ri.foundry.main.dataset.664e3e24-1dea-4c1d-adf6-96c36b61d908")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.664e3e24-1dea-4c1d-adf6-96c36b61d908"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_drug_later=Input(rid="ri.foundry.main.dataset.9346236f-ea42-4dac-b8f5-b738c7336fee"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_drug_later
    df2 = death
    df3 = Person_info_refined
    df1 = df1.withColumnRenamed("person_id_donepezil_10", 'person_id')
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
    Output(rid="ri.foundry.main.dataset.adc10432-5806-4ebd-afb4-e07b1920d013"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_death_drug_later=Input(rid="ri.foundry.main.dataset.9b388d55-2d5a-4b3a-8991-0e08948afbbd")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.9b388d55-2d5a-4b3a-8991-0e08948afbbd"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_drug_later=Input(rid="ri.foundry.main.dataset.4c67ffef-d650-4482-a80c-585755360760"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_drug_later
    df2 = death
    df3 = Person_info_refined
    df1 = df1.withColumnRenamed("person_id_donepezil_10", 'person_id')
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
    Output(rid="ri.foundry.main.dataset.bd8fdc74-be7f-4821-9906-984c14e918af"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_death_drug_later=Input(rid="ri.foundry.main.dataset.569857ec-90ce-47ad-aec4-99a775ae9b4d")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.569857ec-90ce-47ad-aec4-99a775ae9b4d"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_drug_later=Input(rid="ri.foundry.main.dataset.a1fead25-ba0f-4cbe-8037-607054020bba"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_drug_later
    df2 = death
    df3 = Person_info_refined
    df1 = df1.withColumnRenamed("person_id_donepezil_10", 'person_id')
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
    Output(rid="ri.foundry.main.dataset.eb3c3ffc-3faf-437e-870b-b0c4df39f092"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_death_drug_later=Input(rid="ri.foundry.main.dataset.cfe033d5-80cf-4a1b-a0b8-10c075637103")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.cfe033d5-80cf-4a1b-a0b8-10c075637103"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_drug_later=Input(rid="ri.foundry.main.dataset.94bd1045-eb7b-43fe-bc1c-720413817dc7"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_drug_later
    df2 = death
    df3 = Person_info_refined
    df1 = df1.withColumnRenamed("person_id_donepezil_10", 'person_id')
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
    Output(rid="ri.foundry.main.dataset.73ddaa9f-e8ad-4216-9366-ceba98b22a35"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_death_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.a0cf4953-8aec-47b1-9a90-f00b1fdd6ef6")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_death_for_step8_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_death_for_step8_drug_later
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
    Output(rid="ri.foundry.main.dataset.a0cf4953-8aec-47b1-9a90-f00b1fdd6ef6"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.012436e1-73eb-42cd-ad64-fef4acddaccb"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_death_for_step8_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_for_step8_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_for_step8_drug_later
    df2 = death
    df3 = Person_info_refined
    df1 = df1.withColumnRenamed("person_id_donepezil_10", 'person_id')
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
    Output(rid="ri.foundry.main.dataset.5f02793d-e798-4156-add4-a5be7d06d3ad"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_death_drug_later=Input(rid="ri.foundry.main.dataset.8d4ee2b2-ebb2-40ae-a5ad-7c9095262c03")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.8d4ee2b2-ebb2-40ae-a5ad-7c9095262c03"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_drug_later=Input(rid="ri.foundry.main.dataset.3454d70c-0d22-445c-b642-eaeb5efac482"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_drug_later
    df2 = death
    df3 = Person_info_refined
    df1 = df1.withColumnRenamed("person_id_donepezil_10", 'person_id')
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
    Output(rid="ri.foundry.main.dataset.1e15b98b-c557-458d-aae8-248d9beb44da"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_death_drug_later=Input(rid="ri.foundry.main.dataset.c9f537d1-0e4f-4f1a-a3ee-817680c6b914")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.c9f537d1-0e4f-4f1a-a3ee-817680c6b914"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_drug_later=Input(rid="ri.foundry.main.dataset.450195b1-4eae-4309-8ea1-007eaa48d746"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_drug_later
    df2 = death
    df3 = Person_info_refined
    df1 = df1.withColumnRenamed("person_id_donepezil_10", 'person_id')
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
    Output(rid="ri.vector.main.execute.21255906-7dd7-4e61-858f-19c06f8b54ae"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.9b5b2413-cf53-492d-b7d7-b940852b5366")
)
def day30_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.6750252a-24dd-43ad-bc26-629f66665b2f"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.adc10432-5806-4ebd-afb4-e07b1920d013")
)
def day30_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.2b233055-83df-4677-8686-ff2a8a2601e8"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.bd8fdc74-be7f-4821-9906-984c14e918af")
)
def day30_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.c2350fde-af81-4528-ac12-bf5297fdc8eb"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.eb3c3ffc-3faf-437e-870b-b0c4df39f092")
)
def day30_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.9e761ec1-35fe-43b0-a3f0-05c8fc1261ab"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.73ddaa9f-e8ad-4216-9366-ceba98b22a35")
)
def day30_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later
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
    Output(rid="ri.vector.main.execute.1514010b-b9d4-4300-8640-f2a0d3a89826"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.5f02793d-e798-4156-add4-a5be7d06d3ad")
)
def day30_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.204e2682-bd27-4336-be47-ddc10ab1ad53"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.1e15b98b-c557-458d-aae8-248d9beb44da")
)
def day30_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.d5ab17f2-2222-4498-b351-e7f1b1d10f2e"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.9b5b2413-cf53-492d-b7d7-b940852b5366")
)
def day60_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.ce8b084d-056a-4e89-9ea5-2a829291506b"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.adc10432-5806-4ebd-afb4-e07b1920d013")
)
def day60_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.83964445-6055-42b3-9b68-cdbbb83333f8"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.bd8fdc74-be7f-4821-9906-984c14e918af")
)
def day60_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.29f97deb-35a6-448b-82fa-59091d9cab74"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.eb3c3ffc-3faf-437e-870b-b0c4df39f092")
)
def day60_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.3dc9bdd1-7f36-4c3c-92ab-b76556aaf020"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.73ddaa9f-e8ad-4216-9366-ceba98b22a35")
)
def day60_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later
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
    Output(rid="ri.vector.main.execute.9b227174-016a-445f-ac50-629a7a17d0fc"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.5f02793d-e798-4156-add4-a5be7d06d3ad")
)
def day60_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.909cc8e8-0c41-4425-afb8-c7778106da03"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.1e15b98b-c557-458d-aae8-248d9beb44da")
)
def day60_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.6465c7f5-298d-4675-ab06-c67b706e63ba"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.9b5b2413-cf53-492d-b7d7-b940852b5366")
)
def death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.00fc23a5-457a-437b-887f-7a19643f8f53"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.adc10432-5806-4ebd-afb4-e07b1920d013")
)
def death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.6450ac92-a7c3-41c7-a29d-a66154985fa4"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.bd8fdc74-be7f-4821-9906-984c14e918af")
)
def death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.32eb3531-cfc4-4b48-a116-1c896b0de265"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.eb3c3ffc-3faf-437e-870b-b0c4df39f092")
)
def death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.741b58f0-b880-4cf6-992b-1f0c251a843e"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.5f02793d-e798-4156-add4-a5be7d06d3ad")
)
def death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.06dda096-c72b-46bc-8917-3738f900dd10"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.1e15b98b-c557-458d-aae8-248d9beb44da")
)
def death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later(Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later
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


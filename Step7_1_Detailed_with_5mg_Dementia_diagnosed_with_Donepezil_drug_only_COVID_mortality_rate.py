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
    Output(rid="ri.foundry.main.dataset.0a470f4a-d2f5-40e8-82af-1e7068b6da27"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_death_drug_later=Input(rid="ri.foundry.main.dataset.15ff8f43-be42-41ef-b2b1-cf0632ad1644")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.15ff8f43-be42-41ef-b2b1-cf0632ad1644"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_drug_later=Input(rid="ri.foundry.main.dataset.bab32e26-373a-4497-b742-adb18fe4ace4"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_drug_later
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.dc40e974-ea5d-4f8c-a859-e344704fbb32"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_death_drug_later=Input(rid="ri.foundry.main.dataset.c560238d-06d4-4d06-b23d-718e00d746f3")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.c560238d-06d4-4d06-b23d-718e00d746f3"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_drug_later=Input(rid="ri.foundry.main.dataset.81af3f8b-0c81-41c9-8741-680dc45788f9"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_drug_later
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.611a8213-b4a7-4cb4-a1cd-70b1dfe82d4d"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_death_drug_later=Input(rid="ri.foundry.main.dataset.8b2a6b65-334e-4c87-8a2a-e822453a073b")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.8b2a6b65-334e-4c87-8a2a-e822453a073b"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_drug_later=Input(rid="ri.foundry.main.dataset.e298f06f-dd1d-46fc-915d-ba5798d870f4"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_drug_later
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d6995d3e-8ee6-44bd-974e-90fe8f5d4f07"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_death_drug_later=Input(rid="ri.foundry.main.dataset.31ca6839-7594-4e14-9141-760348499431")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.31ca6839-7594-4e14-9141-760348499431"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_drug_later=Input(rid="ri.foundry.main.dataset.7837f30c-7679-4c4c-879d-939c6c7de922"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_drug_later
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fded6d2f-8c49-467e-bee2-4eb93774d848"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_death_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.511b4f5f-8bde-4f32-ac63-83b4223c11ad")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_death_for_step8_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_death_for_step8_drug_later
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
    Output(rid="ri.foundry.main.dataset.511b4f5f-8bde-4f32-ac63-83b4223c11ad"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.cfa1ae82-dc65-4d63-b661-a630377767f1"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_death_for_step8_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_for_step8_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_for_step8_drug_later
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8d931cc8-a716-4eb2-b390-0ab285ad8e88"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_death_drug_later=Input(rid="ri.foundry.main.dataset.f79b78f9-7b17-4f05-902e-a815930ca096")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_deaths_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.f79b78f9-7b17-4f05-902e-a815930ca096"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_drug_later=Input(rid="ri.foundry.main.dataset.374c76cc-1bc2-439d-9e1e-7a3c08c73198"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_drug_later
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.50943ca3-d121-4d19-be2a-63300125f85b"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_death_drug_later=Input(rid="ri.foundry.main.dataset.e851435e-dfca-485f-bdfa-59b8104e0db7")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_death_drug_later
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
    Output(rid="ri.foundry.main.dataset.e851435e-dfca-485f-bdfa-59b8104e0db7"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_drug_later=Input(rid="ri.foundry.main.dataset.658eb40c-3487-46e9-ad40-17abebab30d3"),
    Person_info_refined=Input(rid="ri.foundry.main.dataset.92fba090-cc16-4099-8161-7314daf6447d"),
    death=Input(rid="ri.foundry.main.dataset.d8cc2ad4-215e-4b5d-bc80-80ffb3454875")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_drug_later, death, Person_info_refined):
    df1 = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_drug_later
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

@transform_pandas(
    Output(rid="ri.vector.main.execute.7abbd740-f22e-43e1-b569-1083192dc418"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.0a470f4a-d2f5-40e8-82af-1e7068b6da27")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.8864090a-963a-485d-898a-4c3250827053"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.dc40e974-ea5d-4f8c-a859-e344704fbb32")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.c831101d-7cf7-461f-b671-9396f6f19034"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.611a8213-b4a7-4cb4-a1cd-70b1dfe82d4d")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.8604f9a2-da33-4c7c-b5f0-8255b181b1c2"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.d6995d3e-8ee6-44bd-974e-90fe8f5d4f07")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.cac0e910-0e0b-4bb1-a22f-26ed2f852740"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.fded6d2f-8c49-467e-bee2-4eb93774d848")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later
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
    Output(rid="ri.vector.main.execute.6bcb89be-4767-424d-bb53-157cf6b4efae"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_deaths_drug_later=Input(rid="ri.foundry.main.dataset.8d931cc8-a716-4eb2-b390-0ab285ad8e88")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_deaths_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_deaths_drug_later
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
    Output(rid="ri.vector.main.execute.31dccc08-f3d8-48c9-a05c-cb55af89b91f"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.50943ca3-d121-4d19-be2a-63300125f85b")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.88799e13-21b7-4ae0-8827-c2ddf985e811"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.0a470f4a-d2f5-40e8-82af-1e7068b6da27")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.dc05b3e0-a2bc-48e0-a791-a438751d146c"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.dc40e974-ea5d-4f8c-a859-e344704fbb32")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.ca91c39f-50a8-433e-866a-0d59f92a264d"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.611a8213-b4a7-4cb4-a1cd-70b1dfe82d4d")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.fbcb9be6-47b3-4d3e-8958-072fe2edfdd7"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.d6995d3e-8ee6-44bd-974e-90fe8f5d4f07")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.d5031805-9871-46d4-9108-8dfd5b530657"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.fded6d2f-8c49-467e-bee2-4eb93774d848")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later
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
    Output(rid="ri.vector.main.execute.988b573d-11a6-4fcb-97c0-e23f8a699539"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_deaths_drug_later=Input(rid="ri.foundry.main.dataset.8d931cc8-a716-4eb2-b390-0ab285ad8e88")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_deaths_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_deaths_drug_later
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
    Output(rid="ri.vector.main.execute.02b2a948-80ac-42ec-be7f-d6956f841fd8"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.50943ca3-d121-4d19-be2a-63300125f85b")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.102659c8-a38b-483e-845b-1f9260adab6e"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.0a470f4a-d2f5-40e8-82af-1e7068b6da27")
)
def death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.374d8632-bfa0-4d88-9574-af177f3b4a03"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.dc40e974-ea5d-4f8c-a859-e344704fbb32")
)
def death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_dementia_m1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.9d666c52-a86c-4784-80ac-7a9c67a92047"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.611a8213-b4a7-4cb4-a1cd-70b1dfe82d4d")
)
def death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1m_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.b9e30d7f-37b1-41f4-b5a5-605b7aab2669"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.d6995d3e-8ee6-44bd-974e-90fe8f5d4f07")
)
def death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_l1y_clean_death_drug_later
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
    Output(rid="ri.vector.main.execute.f7e47f17-1084-497f-9944-9a78c5fef708"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_deaths_drug_later=Input(rid="ri.foundry.main.dataset.8d931cc8-a716-4eb2-b390-0ab285ad8e88")
)
def death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_deaths_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_l1y_clean_deaths_drug_later
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
    Output(rid="ri.vector.main.execute.cc41d59d-2685-40ca-b635-69fd3e3fe0e9"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later=Input(rid="ri.foundry.main.dataset.50943ca3-d121-4d19-be2a-63300125f85b")
)
def death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1y_clean_death_drug_later
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


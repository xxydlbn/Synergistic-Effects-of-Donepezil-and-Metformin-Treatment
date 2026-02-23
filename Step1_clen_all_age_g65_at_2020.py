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
    Output(rid="ri.foundry.main.dataset.f5490b6b-4d7c-423c-bb8a-9ffa3c01694e"),
    person_info=Input(rid="ri.foundry.main.dataset.b06a0444-d369-4224-8186-491d286269b4")
)
def age_ge65_at_2020(person_info):
    df = person_info
    df = df.withColumn("age_2020", F.lit(2020) - F.col("year_of_birth"))

    df_filtered = df.filter(F.col("age_2020") >= 65)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b06a0444-d369-4224-8186-491d286269b4"),
    person=Input(rid="ri.foundry.main.dataset.50cae11a-4afb-457d-99d4-55b4bc2cbe66")
)
def person_info(person):
    df = person
    df = df.select(df["person_id"],df["year_of_birth"], df["gender_concept_name"], df["race_concept_name"])

    return(df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b920243c-5c78-40b7-b746-91457361fe92"),
    age_ge65_at_2020=Input(rid="ri.foundry.main.dataset.f5490b6b-4d7c-423c-bb8a-9ffa3c01694e")
)
def unique_id_age_ge65(age_ge65_at_2020):
    df = age_ge65_at_2020
    df_unique = df.select(df['person_id']).distinct()

    return(df_unique)


import pandas as pd
import numpy as np
from pyspark.sql.functions import col, array, when, array_remove, datediff, avg, regexp_replace, regexp_extract, coalesce, month, countDistinct, row_number, split, collect_list, to_date, collect_set, first, count, lit, udf, lag, abs, length, monotonically_increasing_id, expr, round
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import sum as _sum

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StandardScaler
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

from pyspark.sql.functions import date_add, format_string
from pyspark.sql.types import IntegerType
from pyspark.sql import Row

from pyspark.sql.functions import substring

import functools, operator
import statsmodels.api as sm
from pyspark.sql.functions import greatest
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import math
from pyspark.sql import Row
from pyspark.sql import types as T

@transform_pandas(
    Output(rid="ri.vector.main.execute.ef719a65-30b3-4af1-8bc7-f061384e5a0d"),
    Imputed_mean_10mg_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.2eebd8ef-a1cd-4a06-9883-34d8273572cf"),
    Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.95ac1017-70d0-4860-b98c-63d53d6c9b3a")
)
def SMD_10mg_vs_dementia_no_drug(Imputed_mean_10mg_bmi_joint_imputed, Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement):
    df2 = Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement
    mabs = math.fabs 
    df1 = Imputed_mean_10mg_bmi_joint_imputed

    for col_name in df2.columns:
        if col_name.startswith("c_"):
            df2 = df2.withColumnRenamed(col_name, col_name[2:])

    df1 = df1.withColumn("drug_status", lit("1"))
    df2 = (
    df2.withColumnRenamed("race_concept_name", "race_label")
      .withColumnRenamed("gender_concept_name", "sex_label")
    )

    # 1️⃣  columns that are common to both frames
    common_cols = ['person_id', "BMI_max_observed_or_calculated", "age_2020", "DIABETES_indicator", "HYPERTENSION_indicator", "total_number_of_COVID_vaccine_doses", "PAXLOVID_indicator", 'race_label', 'sex_label', 'drug_status']

    # 2️⃣  keep only those columns in df2
    df1_subset = df1.select(*common_cols)
    df2_subset = df2.select(*common_cols)
    
    df_matched = df1_subset.union(df2_subset)

    def smd_numeric(df, col):
        """
        Standardised mean difference for a numeric column (Spark only).
        Returns a Python float.
        """
        stats = (df
                .groupBy("drug_status")
                .agg(F.count("*").alias("n"),
                    F.avg(col).alias("mean"),
                    F.var_pop(col).alias("var"))
                .orderBy("drug_status")        # 0 first, 1 second
                .collect())

        if len(stats) != 2:                       # one group missing
            return float("nan")

        # Row objects ⇒ access by key
        m0, v0 = stats[0]["mean"], stats[0]["var"]
        m1, v1 = stats[1]["mean"], stats[1]["var"]

        pooled_sd = math.sqrt((v0 + v1) / 2.0)
        return 0.0 if pooled_sd == 0 else mabs(m1 - m0) / pooled_sd

    def smd_binary(df, col):
        """
        SMD for a binary (0/1) column.
        """
        props = (df
                .groupBy("drug_status")
                .agg(F.avg(F.col(col).cast("double")).alias("p"))
                .orderBy("drug_status")
                .collect())

        if len(props) != 2:
            return float("nan")

        p0, p1 = props[0]["p"], props[1]["p"]
        p_bar  = (p0 + p1) / 2.0
        denom = math.sqrt(p_bar * (1 - p_bar)) if 0 < p_bar < 1 else 0
        return 0.0 if denom == 0 else mabs(p1 - p0) / denom

    def smd_categorical(df, col):
        """
        Categorical SMD (κ≥3 levels) using the squared‑difference formulation:

            SMD_cat = sqrt( Σ_k  ((p1k − p0k)**2) / (p̄_k * (1 − p̄_k)) )

        Returns
        -------
        global_smd : float
            Overall imbalance across all levels.
        per_level  : list[tuple[level, float]]
            Each level’s individual SMD contribution (diagnostics only).
        """
        # 1️⃣  Counts per treatment × level  →  two columns n0 / n1
        counts = (df
                .groupBy("drug_status", col)
                .agg(F.count("*").alias("n"))
                .groupBy(col)
                .pivot("drug_status", [0, 1])
                .sum("n")
                .fillna(0))

        rows = counts.collect()

        # 2️⃣  Totals to turn counts → proportions
        n0_total = sum(r["0"] for r in rows)
        n1_total = sum(r["1"] for r in rows)

        if n0_total == 0 or n1_total == 0:
            return float("nan"), []          # should not happen after matching

        sum_sq = 0.0
        per_level = []

        for r in rows:
            level = r[col]
            p0 = r["0"] / n0_total
            p1 = r["1"] / n1_total
            p_bar = (p0 + p1) / 2.0
            denom = math.sqrt(p_bar * (1.0 - p_bar)) if 0 < p_bar < 1 else 0.0
            smd_k = 0.0 if denom == 0.0 else mabs(p1 - p0) / denom

            per_level.append((level, smd_k))
            sum_sq += smd_k ** 2

            global_smd = math.sqrt(sum_sq)
        return global_smd, per_level
        
    # example ─ adapt to your variable names
    numeric_cols = ['age_2020', 'BMI_max_observed_or_calculated', 'total_number_of_COVID_vaccine_doses']

    binary_cols  = ['PAXLOVID_indicator', 
    "HYPERTENSION_indicator", "DIABETES_indicator"]

    # new list – every column whose values are strings or ints with ≥ 3 levels
    categorical_cols = ['race_label', 'sex_label']

    # ─── 1.  Prepare an explicit schema  ──────────────────────────────────────────
    balance_schema = T.StructType([
        T.StructField("variable", T.StringType(),  False),
        T.StructField("cov_type", T.StringType(),  False),
        T.StructField("SMD",      T.DoubleType(),  True)
    ])

    # ─── 2.  Collect rows as *tuples* matching that schema  ──────────────────────
    rows = []

    # numeric
    for col in numeric_cols:
        rows.append((col, "numeric", smd_numeric(df_matched, col)))

    # binary
    for col in binary_cols:
        rows.append((col, "binary",  smd_binary(df_matched, col)))

    # categorical
    for col in categorical_cols:
        g_smd, lvl_smds = smd_categorical(df_matched, col)
        rows.append((col, "categorical", g_smd))
        for lvl, smd_k in lvl_smds:                 # optional per‑level rows
            rows.append((f"{col}={lvl}", "cat_level", smd_k))

    # ─── 3.  Create the DataFrame with the schema  ───────────────────────────────
    balance_tbl = (spark.createDataFrame(rows, schema=balance_schema)
                        .orderBy(F.desc("SMD")))

    return balance_tbl

@transform_pandas(
    Output(rid="ri.vector.main.execute.a057424e-af2f-4242-b618-1b86354d3e41"),
    Imputed_mean_10mg_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.2eebd8ef-a1cd-4a06-9883-34d8273572cf"),
    Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.aa80ff10-5005-450e-a798-3f4488808779")
)
def SMD_10mg_vs_no_dementia(Imputed_mean_10mg_bmi_joint_imputed, Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement):
    mabs = math.fabs 
    df1 = Imputed_mean_10mg_bmi_joint_imputed
    df2 = Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement
    for col_name in df2.columns:
        if col_name.startswith("c_"):
            df2 = df2.withColumnRenamed(col_name, col_name[2:])

    df1 = df1.withColumn("drug_status", lit("1"))
    df2 = (
    df2.withColumnRenamed("race_concept_name", "race_label")
      .withColumnRenamed("gender_concept_name", "sex_label")
    )

    # 1️⃣  columns that are common to both frames
    common_cols = ['person_id', "BMI_max_observed_or_calculated", "age_2020", "DIABETES_indicator", "HYPERTENSION_indicator", "total_number_of_COVID_vaccine_doses", "PAXLOVID_indicator", 'race_label', 'sex_label', 'drug_status']

    # 2️⃣  keep only those columns in df2
    df1_subset = df1.select(*common_cols)
    df2_subset = df2.select(*common_cols)
    
    df_matched = df1_subset.union(df2_subset)

    def smd_numeric(df, col):
        """
        Standardised mean difference for a numeric column (Spark only).
        Returns a Python float.
        """
        stats = (df
                .groupBy("drug_status")
                .agg(F.count("*").alias("n"),
                    F.avg(col).alias("mean"),
                    F.var_pop(col).alias("var"))
                .orderBy("drug_status")        # 0 first, 1 second
                .collect())

        if len(stats) != 2:                       # one group missing
            return float("nan")

        # Row objects ⇒ access by key
        m0, v0 = stats[0]["mean"], stats[0]["var"]
        m1, v1 = stats[1]["mean"], stats[1]["var"]

        pooled_sd = math.sqrt((v0 + v1) / 2.0)
        return 0.0 if pooled_sd == 0 else mabs(m1 - m0) / pooled_sd

    def smd_binary(df, col):
        """
        SMD for a binary (0/1) column.
        """
        props = (df
                .groupBy("drug_status")
                .agg(F.avg(F.col(col).cast("double")).alias("p"))
                .orderBy("drug_status")
                .collect())

        if len(props) != 2:
            return float("nan")

        p0, p1 = props[0]["p"], props[1]["p"]
        p_bar  = (p0 + p1) / 2.0
        denom = math.sqrt(p_bar * (1 - p_bar)) if 0 < p_bar < 1 else 0
        return 0.0 if denom == 0 else mabs(p1 - p0) / denom

    def smd_categorical(df, col):
        """
        Categorical SMD (κ≥3 levels) using the squared‑difference formulation:

            SMD_cat = sqrt( Σ_k  ((p1k − p0k)**2) / (p̄_k * (1 − p̄_k)) )

        Returns
        -------
        global_smd : float
            Overall imbalance across all levels.
        per_level  : list[tuple[level, float]]
            Each level’s individual SMD contribution (diagnostics only).
        """
        # 1️⃣  Counts per treatment × level  →  two columns n0 / n1
        counts = (df
                .groupBy("drug_status", col)
                .agg(F.count("*").alias("n"))
                .groupBy(col)
                .pivot("drug_status", [0, 1])
                .sum("n")
                .fillna(0))

        rows = counts.collect()

        # 2️⃣  Totals to turn counts → proportions
        n0_total = sum(r["0"] for r in rows)
        n1_total = sum(r["1"] for r in rows)

        if n0_total == 0 or n1_total == 0:
            return float("nan"), []          # should not happen after matching

        sum_sq = 0.0
        per_level = []

        for r in rows:
            level = r[col]
            p0 = r["0"] / n0_total
            p1 = r["1"] / n1_total
            p_bar = (p0 + p1) / 2.0
            denom = math.sqrt(p_bar * (1.0 - p_bar)) if 0 < p_bar < 1 else 0.0
            smd_k = 0.0 if denom == 0.0 else mabs(p1 - p0) / denom

            per_level.append((level, smd_k))
            sum_sq += smd_k ** 2

            global_smd = math.sqrt(sum_sq)
        return global_smd, per_level
        
    # example ─ adapt to your variable names
    numeric_cols = ['age_2020', 'BMI_max_observed_or_calculated', 'total_number_of_COVID_vaccine_doses']

    binary_cols  = ['PAXLOVID_indicator', 
    "HYPERTENSION_indicator", "DIABETES_indicator"]

    # new list – every column whose values are strings or ints with ≥ 3 levels
    categorical_cols = ['race_label', 'sex_label']

    # ─── 1.  Prepare an explicit schema  ──────────────────────────────────────────
    balance_schema = T.StructType([
        T.StructField("variable", T.StringType(),  False),
        T.StructField("cov_type", T.StringType(),  False),
        T.StructField("SMD",      T.DoubleType(),  True)
    ])

    # ─── 2.  Collect rows as *tuples* matching that schema  ──────────────────────
    rows = []

    # numeric
    for col in numeric_cols:
        rows.append((col, "numeric", smd_numeric(df_matched, col)))

    # binary
    for col in binary_cols:
        rows.append((col, "binary",  smd_binary(df_matched, col)))

    # categorical
    for col in categorical_cols:
        g_smd, lvl_smds = smd_categorical(df_matched, col)
        rows.append((col, "categorical", g_smd))
        for lvl, smd_k in lvl_smds:                 # optional per‑level rows
            rows.append((f"{col}={lvl}", "cat_level", smd_k))

    # ─── 3.  Create the DataFrame with the schema  ───────────────────────────────
    balance_tbl = (spark.createDataFrame(rows, schema=balance_schema)
                        .orderBy(F.desc("SMD")))

    return balance_tbl

@transform_pandas(
    Output(rid="ri.vector.main.execute.fd1b2873-3ee1-48a4-9e41-98ca540c7644"),
    Imputed_mean_5mg_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.68ccc96c-c8ea-480a-8cf8-4dbf136184af"),
    Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.d3d86760-e498-4acd-9aa1-1cba0024317a")
)
def SMD_5mg_vs_dementia_no_drug(Imputed_mean_5mg_bmi_joint_imputed, Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement):
    df2 = Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement
    mabs = math.fabs 
    df1 = Imputed_mean_5mg_bmi_joint_imputed

    for col_name in df2.columns:
        if col_name.startswith("c_"):
            df2 = df2.withColumnRenamed(col_name, col_name[2:])

    df1 = df1.withColumn("drug_status", lit("1"))
    df2 = (
    df2.withColumnRenamed("race_concept_name", "race_label")
      .withColumnRenamed("gender_concept_name", "sex_label")
    )

    # 1️⃣  columns that are common to both frames
    common_cols = ['person_id', "BMI_max_observed_or_calculated", "age_2020", "DIABETES_indicator", "HYPERTENSION_indicator", "total_number_of_COVID_vaccine_doses", "PAXLOVID_indicator", 'race_label', 'sex_label', 'drug_status']

    # 2️⃣  keep only those columns in df2
    df1_subset = df1.select(*common_cols)
    df2_subset = df2.select(*common_cols)
    
    df_matched = df1_subset.union(df2_subset)

    def smd_numeric(df, col):
        """
        Standardised mean difference for a numeric column (Spark only).
        Returns a Python float.
        """
        stats = (df
                .groupBy("drug_status")
                .agg(F.count("*").alias("n"),
                    F.avg(col).alias("mean"),
                    F.var_pop(col).alias("var"))
                .orderBy("drug_status")        # 0 first, 1 second
                .collect())

        if len(stats) != 2:                       # one group missing
            return float("nan")

        # Row objects ⇒ access by key
        m0, v0 = stats[0]["mean"], stats[0]["var"]
        m1, v1 = stats[1]["mean"], stats[1]["var"]

        pooled_sd = math.sqrt((v0 + v1) / 2.0)
        return 0.0 if pooled_sd == 0 else mabs(m1 - m0) / pooled_sd

    def smd_binary(df, col):
        """
        SMD for a binary (0/1) column.
        """
        props = (df
                .groupBy("drug_status")
                .agg(F.avg(F.col(col).cast("double")).alias("p"))
                .orderBy("drug_status")
                .collect())

        if len(props) != 2:
            return float("nan")

        p0, p1 = props[0]["p"], props[1]["p"]
        p_bar  = (p0 + p1) / 2.0
        denom = math.sqrt(p_bar * (1 - p_bar)) if 0 < p_bar < 1 else 0
        return 0.0 if denom == 0 else mabs(p1 - p0) / denom

    def smd_categorical(df, col):
        """
        Categorical SMD (κ≥3 levels) using the squared‑difference formulation:

            SMD_cat = sqrt( Σ_k  ((p1k − p0k)**2) / (p̄_k * (1 − p̄_k)) )

        Returns
        -------
        global_smd : float
            Overall imbalance across all levels.
        per_level  : list[tuple[level, float]]
            Each level’s individual SMD contribution (diagnostics only).
        """
        # 1️⃣  Counts per treatment × level  →  two columns n0 / n1
        counts = (df
                .groupBy("drug_status", col)
                .agg(F.count("*").alias("n"))
                .groupBy(col)
                .pivot("drug_status", [0, 1])
                .sum("n")
                .fillna(0))

        rows = counts.collect()

        # 2️⃣  Totals to turn counts → proportions
        n0_total = sum(r["0"] for r in rows)
        n1_total = sum(r["1"] for r in rows)

        if n0_total == 0 or n1_total == 0:
            return float("nan"), []          # should not happen after matching

        sum_sq = 0.0
        per_level = []

        for r in rows:
            level = r[col]
            p0 = r["0"] / n0_total
            p1 = r["1"] / n1_total
            p_bar = (p0 + p1) / 2.0
            denom = math.sqrt(p_bar * (1.0 - p_bar)) if 0 < p_bar < 1 else 0.0
            smd_k = 0.0 if denom == 0.0 else mabs(p1 - p0) / denom

            per_level.append((level, smd_k))
            sum_sq += smd_k ** 2

            global_smd = math.sqrt(sum_sq)
        return global_smd, per_level
        
    # example ─ adapt to your variable names
    numeric_cols = ['age_2020', 'BMI_max_observed_or_calculated', 'total_number_of_COVID_vaccine_doses']

    binary_cols  = ['PAXLOVID_indicator', 
    "HYPERTENSION_indicator", "DIABETES_indicator"]

    # new list – every column whose values are strings or ints with ≥ 3 levels
    categorical_cols = ['race_label', 'sex_label']

    # ─── 1.  Prepare an explicit schema  ──────────────────────────────────────────
    balance_schema = T.StructType([
        T.StructField("variable", T.StringType(),  False),
        T.StructField("cov_type", T.StringType(),  False),
        T.StructField("SMD",      T.DoubleType(),  True)
    ])

    # ─── 2.  Collect rows as *tuples* matching that schema  ──────────────────────
    rows = []

    # numeric
    for col in numeric_cols:
        rows.append((col, "numeric", smd_numeric(df_matched, col)))

    # binary
    for col in binary_cols:
        rows.append((col, "binary",  smd_binary(df_matched, col)))

    # categorical
    for col in categorical_cols:
        g_smd, lvl_smds = smd_categorical(df_matched, col)
        rows.append((col, "categorical", g_smd))
        for lvl, smd_k in lvl_smds:                 # optional per‑level rows
            rows.append((f"{col}={lvl}", "cat_level", smd_k))

    # ─── 3.  Create the DataFrame with the schema  ───────────────────────────────
    balance_tbl = (spark.createDataFrame(rows, schema=balance_schema)
                        .orderBy(F.desc("SMD")))

    return balance_tbl

@transform_pandas(
    Output(rid="ri.vector.main.execute.78971a54-e127-4ef5-a139-e34ed193e8b3"),
    Imputed_mean_5mg_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.68ccc96c-c8ea-480a-8cf8-4dbf136184af"),
    Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.54d6cc13-fdf5-4dcf-af26-1399a10e114b")
)
def SMD_5mg_vs_no_dementia(Imputed_mean_5mg_bmi_joint_imputed, Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement):
    mabs = math.fabs 
    df1 = Imputed_mean_5mg_bmi_joint_imputed
    df2 = Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement
    for col_name in df2.columns:
        if col_name.startswith("c_"):
            df2 = df2.withColumnRenamed(col_name, col_name[2:])

    df1 = df1.withColumn("drug_status", lit("1"))
    df2 = (
    df2.withColumnRenamed("race_concept_name", "race_label")
      .withColumnRenamed("gender_concept_name", "sex_label")
    )

    # 1️⃣  columns that are common to both frames
    common_cols = ['person_id', "BMI_max_observed_or_calculated", "age_2020", "DIABETES_indicator", "HYPERTENSION_indicator", "total_number_of_COVID_vaccine_doses", "PAXLOVID_indicator", 'race_label', 'sex_label', 'drug_status']

    # 2️⃣  keep only those columns in df2
    df1_subset = df1.select(*common_cols)
    df2_subset = df2.select(*common_cols)
    
    df_matched = df1_subset.union(df2_subset)

    def smd_numeric(df, col):
        """
        Standardised mean difference for a numeric column (Spark only).
        Returns a Python float.
        """
        stats = (df
                .groupBy("drug_status")
                .agg(F.count("*").alias("n"),
                    F.avg(col).alias("mean"),
                    F.var_pop(col).alias("var"))
                .orderBy("drug_status")        # 0 first, 1 second
                .collect())

        if len(stats) != 2:                       # one group missing
            return float("nan")

        # Row objects ⇒ access by key
        m0, v0 = stats[0]["mean"], stats[0]["var"]
        m1, v1 = stats[1]["mean"], stats[1]["var"]

        pooled_sd = math.sqrt((v0 + v1) / 2.0)
        return 0.0 if pooled_sd == 0 else mabs(m1 - m0) / pooled_sd

    def smd_binary(df, col):
        """
        SMD for a binary (0/1) column.
        """
        props = (df
                .groupBy("drug_status")
                .agg(F.avg(F.col(col).cast("double")).alias("p"))
                .orderBy("drug_status")
                .collect())

        if len(props) != 2:
            return float("nan")

        p0, p1 = props[0]["p"], props[1]["p"]
        p_bar  = (p0 + p1) / 2.0
        denom = math.sqrt(p_bar * (1 - p_bar)) if 0 < p_bar < 1 else 0
        return 0.0 if denom == 0 else mabs(p1 - p0) / denom

    def smd_categorical(df, col):
        """
        Categorical SMD (κ≥3 levels) using the squared‑difference formulation:

            SMD_cat = sqrt( Σ_k  ((p1k − p0k)**2) / (p̄_k * (1 − p̄_k)) )

        Returns
        -------
        global_smd : float
            Overall imbalance across all levels.
        per_level  : list[tuple[level, float]]
            Each level’s individual SMD contribution (diagnostics only).
        """
        # 1️⃣  Counts per treatment × level  →  two columns n0 / n1
        counts = (df
                .groupBy("drug_status", col)
                .agg(F.count("*").alias("n"))
                .groupBy(col)
                .pivot("drug_status", [0, 1])
                .sum("n")
                .fillna(0))

        rows = counts.collect()

        # 2️⃣  Totals to turn counts → proportions
        n0_total = sum(r["0"] for r in rows)
        n1_total = sum(r["1"] for r in rows)

        if n0_total == 0 or n1_total == 0:
            return float("nan"), []          # should not happen after matching

        sum_sq = 0.0
        per_level = []

        for r in rows:
            level = r[col]
            p0 = r["0"] / n0_total
            p1 = r["1"] / n1_total
            p_bar = (p0 + p1) / 2.0
            denom = math.sqrt(p_bar * (1.0 - p_bar)) if 0 < p_bar < 1 else 0.0
            smd_k = 0.0 if denom == 0.0 else mabs(p1 - p0) / denom

            per_level.append((level, smd_k))
            sum_sq += smd_k ** 2

            global_smd = math.sqrt(sum_sq)
        return global_smd, per_level
        
    # example ─ adapt to your variable names
    numeric_cols = ['age_2020', 'BMI_max_observed_or_calculated', 'total_number_of_COVID_vaccine_doses']

    binary_cols  = ['PAXLOVID_indicator', 
    "HYPERTENSION_indicator", "DIABETES_indicator"]

    # new list – every column whose values are strings or ints with ≥ 3 levels
    categorical_cols = ['race_label', 'sex_label']

    # ─── 1.  Prepare an explicit schema  ──────────────────────────────────────────
    balance_schema = T.StructType([
        T.StructField("variable", T.StringType(),  False),
        T.StructField("cov_type", T.StringType(),  False),
        T.StructField("SMD",      T.DoubleType(),  True)
    ])

    # ─── 2.  Collect rows as *tuples* matching that schema  ──────────────────────
    rows = []

    # numeric
    for col in numeric_cols:
        rows.append((col, "numeric", smd_numeric(df_matched, col)))

    # binary
    for col in binary_cols:
        rows.append((col, "binary",  smd_binary(df_matched, col)))

    # categorical
    for col in categorical_cols:
        g_smd, lvl_smds = smd_categorical(df_matched, col)
        rows.append((col, "categorical", g_smd))
        for lvl, smd_k in lvl_smds:                 # optional per‑level rows
            rows.append((f"{col}={lvl}", "cat_level", smd_k))

    # ─── 3.  Create the DataFrame with the schema  ───────────────────────────────
    balance_tbl = (spark.createDataFrame(rows, schema=balance_schema)
                        .orderBy(F.desc("SMD")))

    return balance_tbl

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.95ac1017-70d0-4860-b98c-63d53d6c9b3a"),
    propensity_score_10mg_vs_dementia_no_drug=Input(rid="ri.foundry.main.dataset.5844b1c4-245d-4670-b929-d85dd8da6086")
)
def Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement(
    propensity_score_10mg_vs_dementia_no_drug,
    k_neighbors: int = 10
):
    """
    For each treated unit (drug_status == '1'):
      1) Find up to K nearest control units by |propensity_score difference|.
      2) Among those K, pick the one with the smallest |age_2020 difference|.
      3) Match without replacement (a control can be used at most once).

    Returns a DataFrame of matched pairs with treated/control columns side-by-side.
    """

    # Work on a copy
    df = propensity_score_10mg_vs_dementia_no_drug.copy()

    # Split
    treated = df[df['drug_status'] == '1'].copy()
    control = df[df['drug_status'] == '0'].copy()

    # Basic checks
    if treated.empty:
        raise ValueError("The treated group is empty. Please check your 'drug_status' values.")
    if control.empty:
        raise ValueError("The control group is empty. Please check your 'drug_status' values.")
    if 'propensity_score' not in df.columns:
        raise ValueError("Missing 'propensity_score' column.")
    if 'age_2020' not in df.columns:
        raise ValueError("Missing 'age_2020' column (required for the second-stage selection).")

    # Ensure numeric types
    treated['propensity_score'] = treated['propensity_score'].astype(float)
    control['propensity_score'] = control['propensity_score'].astype(float)
    treated['age_2020'] = treated['age_2020'].astype(float)
    control['age_2020'] = control['age_2020'].astype(float)

    used_control_indices = set()
    pairs = []

    # Greedy loop over treated units (you can sort treated by PS to stabilize if desired)
    for t_idx, t_row in treated.iterrows():
        # Controls still available
        available = control.loc[~control.index.isin(used_control_indices)].copy()
        if available.empty:
            break

        # Stage 1: KNN on propensity score
        available['ps_distance'] = np.abs(available['propensity_score'] - t_row['propensity_score'])
        # Take up to k nearest by ps_distance
        k = min(k_neighbors, len(available))
        knn_ps = available.nsmallest(k, 'ps_distance').copy()

        # Stage 2: among those K, choose closest by age_2020
        knn_ps['age_distance'] = np.abs(knn_ps['age_2020'] - t_row['age_2020'])
        best_ctrl_idx = knn_ps['age_distance'].idxmin()
        best_ctrl = knn_ps.loc[best_ctrl_idx]

        # Mark chosen control as used
        used_control_indices.add(best_ctrl_idx)

        # Build a pair record (treated + control, with suffixes)
        pair_record = {
            # linkage
            'treated_index': t_idx,
            'control_index': best_ctrl_idx,
            # diagnostics
            'propensity_distance': float(best_ctrl['ps_distance']),
            'age_distance': float(best_ctrl['age_distance']),
        }

        # Add treated columns with prefix t_
        for col in df.columns:
            pair_record[f"t_{col}"] = t_row[col]
        # Add control columns with prefix c_
        for col in df.columns:
            pair_record[f"c_{col}"] = best_ctrl[col]

        pairs.append(pair_record)

    # Return as a DataFrame
    if pairs:
        result = pd.DataFrame(pairs)
    else:
        # No matches found (e.g., control empty)
        result = pd.DataFrame()

    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.aa80ff10-5005-450e-a798-3f4488808779"),
    propensity_score_10mg_vs_no_dementia_BMI_imputed=Input(rid="ri.foundry.main.dataset.fd5e447d-ddff-428c-82d8-47ed4a53aaea")
)
def Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement(
    propensity_score_10mg_vs_no_dementia_BMI_imputed,
    k_neighbors: int = 10
):
    """
    For each treated unit (drug_status == '1'):
      1) Find up to K nearest control units by |propensity_score difference|.
      2) Among those K, pick the one with the smallest |age_2020 difference|.
      3) Match without replacement (a control can be used at most once).

    Returns a DataFrame of matched pairs with treated/control columns side-by-side.
    """

    # Work on a copy
    df = propensity_score_10mg_vs_no_dementia_BMI_imputed.copy()

    # Split
    treated = df[df['drug_status'] == '1'].copy()
    control = df[df['drug_status'] == '0'].copy()

    # Basic checks
    if treated.empty:
        raise ValueError("The treated group is empty. Please check your 'drug_status' values.")
    if control.empty:
        raise ValueError("The control group is empty. Please check your 'drug_status' values.")
    if 'propensity_score' not in df.columns:
        raise ValueError("Missing 'propensity_score' column.")
    if 'age_2020' not in df.columns:
        raise ValueError("Missing 'age_2020' column (required for the second-stage selection).")

    # Ensure numeric types
    treated['propensity_score'] = treated['propensity_score'].astype(float)
    control['propensity_score'] = control['propensity_score'].astype(float)
    treated['age_2020'] = treated['age_2020'].astype(float)
    control['age_2020'] = control['age_2020'].astype(float)

    used_control_indices = set()
    pairs = []

    # Greedy loop over treated units (you can sort treated by PS to stabilize if desired)
    for t_idx, t_row in treated.iterrows():
        # Controls still available
        available = control.loc[~control.index.isin(used_control_indices)].copy()
        if available.empty:
            break

        # Stage 1: KNN on propensity score
        available['ps_distance'] = np.abs(available['propensity_score'] - t_row['propensity_score'])
        # Take up to k nearest by ps_distance
        k = min(k_neighbors, len(available))
        knn_ps = available.nsmallest(k, 'ps_distance').copy()

        # Stage 2: among those K, choose closest by age_2020
        knn_ps['age_distance'] = np.abs(knn_ps['age_2020'] - t_row['age_2020'])
        best_ctrl_idx = knn_ps['age_distance'].idxmin()
        best_ctrl = knn_ps.loc[best_ctrl_idx]

        # Mark chosen control as used
        used_control_indices.add(best_ctrl_idx)

        # Build a pair record (treated + control, with suffixes)
        pair_record = {
            # linkage
            'treated_index': t_idx,
            'control_index': best_ctrl_idx,
            # diagnostics
            'propensity_distance': float(best_ctrl['ps_distance']),
            'age_distance': float(best_ctrl['age_distance']),
        }

        # Add treated columns with prefix t_
        for col in df.columns:
            pair_record[f"t_{col}"] = t_row[col]
        # Add control columns with prefix c_
        for col in df.columns:
            pair_record[f"c_{col}"] = best_ctrl[col]

        pairs.append(pair_record)

    # Return as a DataFrame
    if pairs:
        result = pd.DataFrame(pairs)
    else:
        # No matches found (e.g., control empty)
        result = pd.DataFrame()

    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d3d86760-e498-4acd-9aa1-1cba0024317a"),
    propensity_score_5mg_vs_dementia_no_drug=Input(rid="ri.foundry.main.dataset.fb7dc071-6f8c-48a0-b777-a4d6267c6d94")
)
def Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement(
    propensity_score_5mg_vs_dementia_no_drug,
    k_neighbors: int = 10
):
    """
    For each treated unit (drug_status == '1'):
      1) Find up to K nearest control units by |propensity_score difference|.
      2) Among those K, pick the one with the smallest |age_2020 difference|.
      3) Match without replacement (a control can be used at most once).

    Returns a DataFrame of matched pairs with treated/control columns side-by-side.
    """

    # Work on a copy
    df = propensity_score_5mg_vs_dementia_no_drug.copy()

    # Split
    treated = df[df['drug_status'] == '1'].copy()
    control = df[df['drug_status'] == '0'].copy()

    # Basic checks
    if treated.empty:
        raise ValueError("The treated group is empty. Please check your 'drug_status' values.")
    if control.empty:
        raise ValueError("The control group is empty. Please check your 'drug_status' values.")
    if 'propensity_score' not in df.columns:
        raise ValueError("Missing 'propensity_score' column.")
    if 'age_2020' not in df.columns:
        raise ValueError("Missing 'age_2020' column (required for the second-stage selection).")

    # Ensure numeric types
    treated['propensity_score'] = treated['propensity_score'].astype(float)
    control['propensity_score'] = control['propensity_score'].astype(float)
    treated['age_2020'] = treated['age_2020'].astype(float)
    control['age_2020'] = control['age_2020'].astype(float)

    used_control_indices = set()
    pairs = []

    # Greedy loop over treated units (you can sort treated by PS to stabilize if desired)
    for t_idx, t_row in treated.iterrows():
        # Controls still available
        available = control.loc[~control.index.isin(used_control_indices)].copy()
        if available.empty:
            break

        # Stage 1: KNN on propensity score
        available['ps_distance'] = np.abs(available['propensity_score'] - t_row['propensity_score'])
        # Take up to k nearest by ps_distance
        k = min(k_neighbors, len(available))
        knn_ps = available.nsmallest(k, 'ps_distance').copy()

        # Stage 2: among those K, choose closest by age_2020
        knn_ps['age_distance'] = np.abs(knn_ps['age_2020'] - t_row['age_2020'])
        best_ctrl_idx = knn_ps['age_distance'].idxmin()
        best_ctrl = knn_ps.loc[best_ctrl_idx]

        # Mark chosen control as used
        used_control_indices.add(best_ctrl_idx)

        # Build a pair record (treated + control, with suffixes)
        pair_record = {
            # linkage
            'treated_index': t_idx,
            'control_index': best_ctrl_idx,
            # diagnostics
            'propensity_distance': float(best_ctrl['ps_distance']),
            'age_distance': float(best_ctrl['age_distance']),
        }

        # Add treated columns with prefix t_
        for col in df.columns:
            pair_record[f"t_{col}"] = t_row[col]
        # Add control columns with prefix c_
        for col in df.columns:
            pair_record[f"c_{col}"] = best_ctrl[col]

        pairs.append(pair_record)

    # Return as a DataFrame
    if pairs:
        result = pd.DataFrame(pairs)
    else:
        # No matches found (e.g., control empty)
        result = pd.DataFrame()

    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4fe6f219-853c-4316-a933-91e0fc77693d"),
    propensity_score_5mg_vs_no_dementia_BMI_imputed=Input(rid="ri.foundry.main.dataset.ac015d94-ff04-4c1f-8b77-dc1fdae7b5c1")
)
def Sampling_propensity_score_5mg_vs_no_dementia_KNN2(propensity_score_5mg_vs_no_dementia_BMI_imputed):
    df = propensity_score_5mg_vs_no_dementia_BMI_imputed
    # Step 1: Separate treated and control units.
    treated = df[df['drug_status'] == '1'].copy()
    control = df[df['drug_status'] == '0'].copy()

    # Check if either group is empty
    if treated.empty:
        raise ValueError("The treated group is empty. Please check your 'drug_status' values.")
    if control.empty:
        raise ValueError("The control group is empty. Please check your 'drug_status' values.")

    # Optional: Use the logit of the propensity score if that works better
    # For this example, we'll use the propensity score directly.
    treated_ps = treated['propensity_score'].values.reshape(-1, 1)
    control_ps = control['propensity_score'].values.reshape(-1, 1)

    # Step 2: Fit the KNN model on the control group.
    knn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    knn.fit(control_ps)

    # Step 3: Find the nearest control unit for each treated unit.
    distances, indices = knn.kneighbors(treated_ps)

    # Step 4: Extract the matched control units.
    matched_control = control.iloc[indices.flatten()].copy()
    matched_control['match_distance'] = distances.flatten()

    # Step 5: Combine the treated units and their matched controls into a single matched sample.
    matched_sample = pd.concat([treated, matched_control], ignore_index=True)

    return(matched_control)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.68382f14-ee0e-4c66-ae26-31aacdcd2987"),
    propensity_score_5mg_vs_no_dementia_BMI_imputed=Input(rid="ri.foundry.main.dataset.ac015d94-ff04-4c1f-8b77-dc1fdae7b5c1")
)
def Sampling_propensity_score_5mg_vs_no_dementia_KNN3(propensity_score_5mg_vs_no_dementia_BMI_imputed):
    df = propensity_score_5mg_vs_no_dementia_BMI_imputed
    # Step 1: Separate treated and control units.
    treated = df[df['drug_status'] == '1'].copy()
    control = df[df['drug_status'] == '0'].copy()

    # Check if either group is empty
    if treated.empty:
        raise ValueError("The treated group is empty. Please check your 'drug_status' values.")
    if control.empty:
        raise ValueError("The control group is empty. Please check your 'drug_status' values.")

    # Optional: Use the logit of the propensity score if that works better
    # For this example, we'll use the propensity score directly.
    treated_ps = treated['propensity_score'].values.reshape(-1, 1)
    control_ps = control['propensity_score'].values.reshape(-1, 1)

    # Step 2: Fit the KNN model on the control group.
    knn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
    knn.fit(control_ps)

    # Step 3: Find the nearest control unit for each treated unit.
    distances, indices = knn.kneighbors(treated_ps)

    # Step 4: Extract the matched control units.
    matched_control = control.iloc[indices.flatten()].copy()
    matched_control['match_distance'] = distances.flatten()

    # Step 5: Combine the treated units and their matched controls into a single matched sample.
    matched_sample = pd.concat([treated, matched_control], ignore_index=True)

    return(matched_control)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7e895d36-c61e-4fb3-a18a-133b2a336c8a"),
    propensity_score_5mg_vs_no_dementia_BMI_imputed=Input(rid="ri.foundry.main.dataset.ac015d94-ff04-4c1f-8b77-dc1fdae7b5c1")
)
def Sampling_propensity_score_5mg_vs_no_dementia_KNN4(propensity_score_5mg_vs_no_dementia_BMI_imputed):
    df = propensity_score_5mg_vs_no_dementia_BMI_imputed
    # Step 1: Separate treated and control units.
    treated = df[df['drug_status'] == '1'].copy()
    control = df[df['drug_status'] == '0'].copy()

    # Check if either group is empty
    if treated.empty:
        raise ValueError("The treated group is empty. Please check your 'drug_status' values.")
    if control.empty:
        raise ValueError("The control group is empty. Please check your 'drug_status' values.")

    # Optional: Use the logit of the propensity score if that works better
    # For this example, we'll use the propensity score directly.
    treated_ps = treated['propensity_score'].values.reshape(-1, 1)
    control_ps = control['propensity_score'].values.reshape(-1, 1)

    # Step 2: Fit the KNN model on the control group.
    knn = NearestNeighbors(n_neighbors=4, algorithm='ball_tree')
    knn.fit(control_ps)

    # Step 3: Find the nearest control unit for each treated unit.
    distances, indices = knn.kneighbors(treated_ps)

    # Step 4: Extract the matched control units.
    matched_control = control.iloc[indices.flatten()].copy()
    matched_control['match_distance'] = distances.flatten()

    # Step 5: Combine the treated units and their matched controls into a single matched sample.
    matched_sample = pd.concat([treated, matched_control], ignore_index=True)

    return(matched_control)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f9bbb990-db82-43d4-9536-9aed9131ad34"),
    propensity_score_5mg_vs_no_dementia_BMI_imputed=Input(rid="ri.foundry.main.dataset.ac015d94-ff04-4c1f-8b77-dc1fdae7b5c1")
)
def Sampling_propensity_score_5mg_vs_no_dementia_KNN5(propensity_score_5mg_vs_no_dementia_BMI_imputed):
    df = propensity_score_5mg_vs_no_dementia_BMI_imputed
    # Step 1: Separate treated and control units.
    treated = df[df['drug_status'] == '1'].copy()
    control = df[df['drug_status'] == '0'].copy()

    # Check if either group is empty
    if treated.empty:
        raise ValueError("The treated group is empty. Please check your 'drug_status' values.")
    if control.empty:
        raise ValueError("The control group is empty. Please check your 'drug_status' values.")

    # Optional: Use the logit of the propensity score if that works better
    # For this example, we'll use the propensity score directly.
    treated_ps = treated['propensity_score'].values.reshape(-1, 1)
    control_ps = control['propensity_score'].values.reshape(-1, 1)

    # Step 2: Fit the KNN model on the control group.
    knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knn.fit(control_ps)

    # Step 3: Find the nearest control unit for each treated unit.
    distances, indices = knn.kneighbors(treated_ps)

    # Step 4: Extract the matched control units.
    matched_control = control.iloc[indices.flatten()].copy()
    matched_control['match_distance'] = distances.flatten()

    # Step 5: Combine the treated units and their matched controls into a single matched sample.
    matched_sample = pd.concat([treated, matched_control], ignore_index=True)

    return(matched_control)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.54d6cc13-fdf5-4dcf-af26-1399a10e114b"),
    propensity_score_5mg_vs_no_dementia_BMI_imputed=Input(rid="ri.foundry.main.dataset.ac015d94-ff04-4c1f-8b77-dc1fdae7b5c1")
)
def Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement(
    propensity_score_5mg_vs_no_dementia_BMI_imputed,
    k_neighbors: int = 10
):
    """
    For each treated unit (drug_status == '1'):
      1) Find up to K nearest control units by |propensity_score difference|.
      2) Among those K, pick the one with the smallest |age_2020 difference|.
      3) Match without replacement (a control can be used at most once).

    Returns a DataFrame of matched pairs with treated/control columns side-by-side.
    """

    # Work on a copy
    df = propensity_score_5mg_vs_no_dementia_BMI_imputed.copy()

    # Split
    treated = df[df['drug_status'] == '1'].copy()
    control = df[df['drug_status'] == '0'].copy()

    # Basic checks
    if treated.empty:
        raise ValueError("The treated group is empty. Please check your 'drug_status' values.")
    if control.empty:
        raise ValueError("The control group is empty. Please check your 'drug_status' values.")
    if 'propensity_score' not in df.columns:
        raise ValueError("Missing 'propensity_score' column.")
    if 'age_2020' not in df.columns:
        raise ValueError("Missing 'age_2020' column (required for the second-stage selection).")

    # Ensure numeric types
    treated['propensity_score'] = treated['propensity_score'].astype(float)
    control['propensity_score'] = control['propensity_score'].astype(float)
    treated['age_2020'] = treated['age_2020'].astype(float)
    control['age_2020'] = control['age_2020'].astype(float)

    used_control_indices = set()
    pairs = []

    # Greedy loop over treated units (you can sort treated by PS to stabilize if desired)
    for t_idx, t_row in treated.iterrows():
        # Controls still available
        available = control.loc[~control.index.isin(used_control_indices)].copy()
        if available.empty:
            break

        # Stage 1: KNN on propensity score
        available['ps_distance'] = np.abs(available['propensity_score'] - t_row['propensity_score'])
        # Take up to k nearest by ps_distance
        k = min(k_neighbors, len(available))
        knn_ps = available.nsmallest(k, 'ps_distance').copy()

        # Stage 2: among those K, choose closest by age_2020
        knn_ps['age_distance'] = np.abs(knn_ps['age_2020'] - t_row['age_2020'])
        best_ctrl_idx = knn_ps['age_distance'].idxmin()
        best_ctrl = knn_ps.loc[best_ctrl_idx]

        # Mark chosen control as used
        used_control_indices.add(best_ctrl_idx)

        # Build a pair record (treated + control, with suffixes)
        pair_record = {
            # linkage
            'treated_index': t_idx,
            'control_index': best_ctrl_idx,
            # diagnostics
            'propensity_distance': float(best_ctrl['ps_distance']),
            'age_distance': float(best_ctrl['age_distance']),
        }

        # Add treated columns with prefix t_
        for col in df.columns:
            pair_record[f"t_{col}"] = t_row[col]
        # Add control columns with prefix c_
        for col in df.columns:
            pair_record[f"c_{col}"] = best_ctrl[col]

        pairs.append(pair_record)

    # Return as a DataFrame
    if pairs:
        result = pd.DataFrame(pairs)
    else:
        # No matches found (e.g., control empty)
        result = pd.DataFrame()

    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.58738439-6206-4f4d-ab80-749a8fb35eec"),
    Sampling_propensity_score_5mg_vs_no_dementia_KNN2=Input(rid="ri.foundry.main.dataset.4fe6f219-853c-4316-a933-91e0fc77693d")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later_KNN2(Sampling_propensity_score_5mg_vs_no_dementia_KNN2):
    df = Sampling_propensity_score_5mg_vs_no_dementia_KNN2
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
        100*F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.d1d4b038-fe14-4c1b-a9cf-8741a124f35b"),
    Sampling_propensity_score_5mg_vs_no_dementia_KNN3=Input(rid="ri.foundry.main.dataset.68382f14-ee0e-4c66-ae26-31aacdcd2987")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later_KNN3(Sampling_propensity_score_5mg_vs_no_dementia_KNN3):
    df = Sampling_propensity_score_5mg_vs_no_dementia_KNN3
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
        100*F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.7feb7c3c-74e8-46f8-922e-bf1dae3ef8cc"),
    Sampling_propensity_score_5mg_vs_no_dementia_KNN4=Input(rid="ri.foundry.main.dataset.7e895d36-c61e-4fb3-a18a-133b2a336c8a")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later_KNN4(Sampling_propensity_score_5mg_vs_no_dementia_KNN4):
    df = Sampling_propensity_score_5mg_vs_no_dementia_KNN4
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
        100*F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.df80e27a-42b6-47c9-9c79-332e3ca8801d"),
    Sampling_propensity_score_5mg_vs_no_dementia_KNN5=Input(rid="ri.foundry.main.dataset.f9bbb990-db82-43d4-9536-9aed9131ad34")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later_KNN5(Sampling_propensity_score_5mg_vs_no_dementia_KNN5):
    df = Sampling_propensity_score_5mg_vs_no_dementia_KNN5
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
        100*F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.f459c004-6e0a-426c-84df-6420805dff5f"),
    Sampling_propensity_score_5mg_vs_no_dementia_KNN2=Input(rid="ri.foundry.main.dataset.4fe6f219-853c-4316-a933-91e0fc77693d")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later_KNN2(Sampling_propensity_score_5mg_vs_no_dementia_KNN2):
    df = Sampling_propensity_score_5mg_vs_no_dementia_KNN2
    # Add a column to check if death_date is within 30 days after condition_start_date
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) & 
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) & 
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
        100*F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.4a52f05b-bbc1-4cad-9037-5f1794a80b63"),
    Sampling_propensity_score_5mg_vs_no_dementia_KNN3=Input(rid="ri.foundry.main.dataset.68382f14-ee0e-4c66-ae26-31aacdcd2987")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later_KNN3(Sampling_propensity_score_5mg_vs_no_dementia_KNN3):
    df = Sampling_propensity_score_5mg_vs_no_dementia_KNN3
    # Add a column to check if death_date is within 30 days after condition_start_date
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) & 
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) & 
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
        100*F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.2a17100a-9762-494c-ab54-19698481112b"),
    Sampling_propensity_score_5mg_vs_no_dementia_KNN4=Input(rid="ri.foundry.main.dataset.7e895d36-c61e-4fb3-a18a-133b2a336c8a")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later_KNN4(Sampling_propensity_score_5mg_vs_no_dementia_KNN4):
    df = Sampling_propensity_score_5mg_vs_no_dementia_KNN4
    # Add a column to check if death_date is within 30 days after condition_start_date
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) & 
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) & 
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
        100*F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.5d18c844-d4a4-4379-a392-5de1a423f74c"),
    Sampling_propensity_score_5mg_vs_no_dementia_KNN5=Input(rid="ri.foundry.main.dataset.f9bbb990-db82-43d4-9536-9aed9131ad34")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_drug_later_KNN5(Sampling_propensity_score_5mg_vs_no_dementia_KNN5):
    df = Sampling_propensity_score_5mg_vs_no_dementia_KNN5
    # Add a column to check if death_date is within 30 days after condition_start_date
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) & 
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) & 
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
        100*F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.a9aebcf2-6a8f-494a-b84a-458ab1de7b88"),
    Sampling_propensity_score_5mg_vs_no_dementia_KNN2=Input(rid="ri.foundry.main.dataset.4fe6f219-853c-4316-a933-91e0fc77693d")
)
def death_summary_by_age_KNN2_no_dementia(Sampling_propensity_score_5mg_vs_no_dementia_KNN2):
    df = Sampling_propensity_score_5mg_vs_no_dementia_KNN2

    # Create columns to check if death_date is within 30 days or 60 days after condition_start_date_COVID
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 30) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    df = df.withColumn(
        "death_within_60_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups with the last group capturing all individuals 85 and older
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Group by the defined age ranges and compute the summaries
    summary_df = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    )

    # Add proportion columns (as percentages)
    summary_df = summary_df.withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    )

    return summary_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.db71ee58-e256-4f7e-a59b-6eea94f9f0b0"),
    Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.d3d86760-e498-4acd-9aa1-1cba0024317a")
)
def death_summary_by_age_KNN5_nonreplacement_dementia_no_drug_all_groups_BMI_joint_imputed(Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement):
    df = Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement

    for col_name in df.columns:
        if col_name.startswith("c_"):
            df = df.withColumnRenamed(col_name, col_name[2:])

    # Create columns to check if death_date is within 30 days or 60 days after condition_start_date_COVID
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 30) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    df = df.withColumn(
        "death_within_60_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups for specific age groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Compute summary for each age group
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    )

    # Compute overall summary for all ages
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine the age-specific summaries with the overall summary
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.8b53d7d9-9265-485e-aadc-e3f204bf8f57"),
    Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.95ac1017-70d0-4860-b98c-63d53d6c9b3a")
)
def death_summary_by_age_KNN5_nonreplacement_dementia_no_drug_all_groups_BMI_joint_imputed_10(Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement):
    df = Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement

    for col_name in df.columns:
        if col_name.startswith("c_"):
            df = df.withColumnRenamed(col_name, col_name[2:])

    # Create columns to check if death_date is within 30 days or 60 days after condition_start_date_COVID
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 30) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    df = df.withColumn(
        "death_within_60_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups for specific age groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Compute summary for each age group
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    )

    # Compute overall summary for all ages
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine the age-specific summaries with the overall summary
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.4764d2a3-9dcf-47f3-8579-77dd4f71e799"),
    Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.54d6cc13-fdf5-4dcf-af26-1399a10e114b")
)
def death_summary_by_age_KNN5_nonreplacement_no_dementia_all_groups_BMI_joint_imputed(Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement):
    df = Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement

    for col_name in df.columns:
        if col_name.startswith("c_"):
            df = df.withColumnRenamed(col_name, col_name[2:])

    # Create columns to check if death_date is within 30 days or 60 days after condition_start_date_COVID
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 30) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    df = df.withColumn(
        "death_within_60_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups for specific age groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Compute summary for each age group
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    )

    # Compute overall summary for all ages
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine the age-specific summaries with the overall summary
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.9509ac35-c0ed-4b16-9726-18904b29d833"),
    Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.aa80ff10-5005-450e-a798-3f4488808779")
)
def death_summary_by_age_KNN5_nonreplacement_no_dementia_all_groups_BMI_joint_imputed_10(Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement):
    df = Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement

    for col_name in df.columns:
        if col_name.startswith("c_"):
            df = df.withColumnRenamed(col_name, col_name[2:])

    # Create columns to check if death_date is within 30 days or 60 days after condition_start_date_COVID
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 30) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    df = df.withColumn(
        "death_within_60_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups for specific age groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Compute summary for each age group
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    )

    # Compute overall summary for all ages
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine the age-specific summaries with the overall summary
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.78beb264-d851-49c5-a853-6f9c531c5054"),
    Imputed_mean_10mg_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.2eebd8ef-a1cd-4a06-9883-34d8273572cf")
)
def death_summary_by_age_dementia_10mg_only_BMI_imputed(Imputed_mean_10mg_bmi_joint_imputed):
    df = Imputed_mean_10mg_bmi_joint_imputed

    # Create columns to check if death_date is within 30 days or 60 days after condition_start_date_COVID
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 30) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    df = df.withColumn(
        "death_within_60_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups for specific age groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Compute summary for each age group
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    )

    # Compute overall summary for all ages
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine the age-specific summaries with the overall summary
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.vector.main.execute.af943fd0-8593-424a-8b41-f3ede5e5f0a0"),
    Imputed_mean_5mg_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.68ccc96c-c8ea-480a-8cf8-4dbf136184af")
)
def death_summary_by_age_dementia_5mg_only_BMI_imputed(Imputed_mean_5mg_bmi_joint_imputed):
    df = Imputed_mean_5mg_bmi_joint_imputed

    # Create columns to check if death_date is within 30 days or 60 days after condition_start_date_COVID
    df = df.withColumn(
        "death_within_30_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 30) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    df = df.withColumn(
        "death_within_60_days",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) <= 60) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups for specific age groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Compute summary for each age group
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    )

    # Compute overall summary for all ages
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine the age-specific summaries with the overall summary
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5844b1c4-245d-4670-b929-d85dd8da6086"),
    Join_10mg_with_dementia_no_drug_drug_later_bmi_both_imputed_for_psm_matching_joint_imputed=Input(rid="ri.foundry.main.dataset.8b635183-a966-4056-b51d-3629601c51c9")
)
def propensity_score_10mg_vs_dementia_no_drug(Join_10mg_with_dementia_no_drug_drug_later_bmi_both_imputed_for_psm_matching_joint_imputed):
    df = Join_10mg_with_dementia_no_drug_drug_later_bmi_both_imputed_for_psm_matching_joint_imputed
    # Assume df is your original DataFrame.
    # One-hot encode the categorical variables.
    categorical_vars = ['gender_concept_name', 'race_concept_name', 'DIABETES_indicator', 'HYPERTENSION_indicator', 'PAXLOVID_indicator']
    df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

    # Clean dummy variable names (remove spaces and special characters)
    df_encoded.columns = [col.replace(" ", "_").replace("-", "_") for col in df_encoded.columns]

    # Convert continuous columns to numeric (if not already)
    df_encoded['age_2020'] = pd.to_numeric(df_encoded['age_2020'], errors='coerce')
    df_encoded['BMI_max_observed_or_calculated'] = pd.to_numeric(df_encoded['BMI_max_observed_or_calculated'], errors='coerce')
    df_encoded['total_number_of_COVID_vaccine_doses'] = pd.to_numeric(df_encoded['total_number_of_COVID_vaccine_doses'], errors='coerce')

    # Scale continuous variables to help with numerical stability
    scaler = StandardScaler()
    df_encoded[['age_2020', 'BMI_max_observed_or_calculated']] = scaler.fit_transform(
        df_encoded[['age_2020', 'BMI_max_observed_or_calculated']]
    )

    # --- Build the Design Matrix ---

    # Define predictor columns: continuous variables and dummy variables.
    predictor_columns = ['age_2020', 'BMI_max_observed_or_calculated', 'total_number_of_COVID_vaccine_doses'] + [
        col for col in df_encoded.columns 
        if any(prefix in col for prefix in ['gender_concept_name_', 'race_concept_name_', 
                                            'DIABETES_indicator_', 'HYPERTENSION_indicator_', "PAXLOVID_indicator"])
    ]

    # Build design matrix X and ensure it is numeric
    X = df_encoded[predictor_columns].apply(pd.to_numeric, errors='coerce')
    X = sm.add_constant(X, has_constant='add')  # add intercept term

    # Treatment variable (ensure it is numeric)
    y = pd.to_numeric(df_encoded['drug_status'], errors='coerce')

    # Drop rows with any missing values in X or y
    data = pd.concat([X, y], axis=1).dropna()
    X_clean = data[X.columns]
    y_clean = data['drug_status']

    # --- Fit Regularized Logistic Regression ---

    # Use fit_regularized to help overcome numerical issues.
    # Adjust alpha (penalty strength) and maxiter as needed.
    result = sm.Logit(y_clean, X_clean).fit_regularized(method='l1', alpha=0.01, maxiter=1000)

    # --- Generate Propensity Scores ---

    # Calculate predicted propensity scores
    propensity_scores = result.predict(X_clean)

    # Clip predicted probabilities to avoid exact 0 or 1 (prevents division-by-zero in logit)
    propensity_scores = np.clip(propensity_scores, 1e-10, 1 - 1e-10)

    # Compute logit of the propensity scores
    logit_propensity = np.log(propensity_scores / (1 - propensity_scores))

    # Save the results back into the original dataframe.
    # (Here, we assume that the indices of 'data' match those in 'df'.)
    df.loc[data.index, 'propensity_score'] = propensity_scores
    df.loc[data.index, 'logit_propensity'] = logit_propensity
    
    return(df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fd5e447d-ddff-428c-82d8-47ed4a53aaea"),
    Join_10mg_no_dementia_drug_later_both_bmi_imputed_for_psm_joint_imputed=Input(rid="ri.foundry.main.dataset.44147aa5-4f9b-456c-84bb-23829b1910cd")
)
def propensity_score_10mg_vs_no_dementia_BMI_imputed(Join_10mg_no_dementia_drug_later_both_bmi_imputed_for_psm_joint_imputed):
    df = Join_10mg_no_dementia_drug_later_both_bmi_imputed_for_psm_joint_imputed
    # Assume df is your original DataFrame.
    # One-hot encode the categorical variables.
    categorical_vars = ['gender_concept_name', 'race_concept_name', 'DIABETES_indicator', 'HYPERTENSION_indicator', 'PAXLOVID_indicator']
    df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

    # Clean dummy variable names (remove spaces and special characters)
    df_encoded.columns = [col.replace(" ", "_").replace("-", "_") for col in df_encoded.columns]

    # Convert continuous columns to numeric (if not already)
    df_encoded['age_2020'] = pd.to_numeric(df_encoded['age_2020'], errors='coerce')
    df_encoded['BMI_max_observed_or_calculated'] = pd.to_numeric(df_encoded['BMI_max_observed_or_calculated'], errors='coerce')
    df_encoded['total_number_of_COVID_vaccine_doses'] = pd.to_numeric(df_encoded['total_number_of_COVID_vaccine_doses'], errors='coerce')

    # Scale continuous variables to help with numerical stability
    scaler = StandardScaler()
    df_encoded[['age_2020', 'BMI_max_observed_or_calculated']] = scaler.fit_transform(
        df_encoded[['age_2020', 'BMI_max_observed_or_calculated']]
    )

    # --- Build the Design Matrix ---

    # Define predictor columns: continuous variables and dummy variables.
    predictor_columns = ['age_2020', 'BMI_max_observed_or_calculated', 'total_number_of_COVID_vaccine_doses'] + [
        col for col in df_encoded.columns 
        if any(prefix in col for prefix in ['gender_concept_name_', 'race_concept_name_', 
                                            'DIABETES_indicator_', 'HYPERTENSION_indicator_', "PAXLOVID_indicator"])
    ]

    # Build design matrix X and ensure it is numeric
    X = df_encoded[predictor_columns].apply(pd.to_numeric, errors='coerce')
    X = sm.add_constant(X, has_constant='add')  # add intercept term

    # Treatment variable (ensure it is numeric)
    y = pd.to_numeric(df_encoded['drug_status'], errors='coerce')

    # Drop rows with any missing values in X or y
    data = pd.concat([X, y], axis=1).dropna()
    X_clean = data[X.columns]
    y_clean = data['drug_status']

    # --- Fit Regularized Logistic Regression ---

    # Use fit_regularized to help overcome numerical issues.
    # Adjust alpha (penalty strength) and maxiter as needed.
    result = sm.Logit(y_clean, X_clean).fit_regularized(method='l1', alpha=0.01, maxiter=1000)

    # --- Generate Propensity Scores ---

    # Calculate predicted propensity scores
    propensity_scores = result.predict(X_clean)

    # Clip predicted probabilities to avoid exact 0 or 1 (prevents division-by-zero in logit)
    propensity_scores = np.clip(propensity_scores, 1e-10, 1 - 1e-10)

    # Compute logit of the propensity scores
    logit_propensity = np.log(propensity_scores / (1 - propensity_scores))

    # Save the results back into the original dataframe.
    # (Here, we assume that the indices of 'data' match those in 'df'.)
    df.loc[data.index, 'propensity_score'] = propensity_scores
    df.loc[data.index, 'logit_propensity'] = logit_propensity
    
    return(df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fb7dc071-6f8c-48a0-b777-a4d6267c6d94"),
    Join_5mg_with_dementia_no_drug_drug_later_bmi_both_imputed_for_psm_matching_joint_imputed=Input(rid="ri.foundry.main.dataset.d9dbdf73-816d-49db-acd8-7e02cae26eee")
)
def propensity_score_5mg_vs_dementia_no_drug(Join_5mg_with_dementia_no_drug_drug_later_bmi_both_imputed_for_psm_matching_joint_imputed):
    df = Join_5mg_with_dementia_no_drug_drug_later_bmi_both_imputed_for_psm_matching_joint_imputed
    # Assume df is your original DataFrame.
    # One-hot encode the categorical variables.
    categorical_vars = ['gender_concept_name', 'race_concept_name', 'DIABETES_indicator', 'HYPERTENSION_indicator', 'PAXLOVID_indicator']
    df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

    # Clean dummy variable names (remove spaces and special characters)
    df_encoded.columns = [col.replace(" ", "_").replace("-", "_") for col in df_encoded.columns]

    # Convert continuous columns to numeric (if not already)
    df_encoded['age_2020'] = pd.to_numeric(df_encoded['age_2020'], errors='coerce')
    df_encoded['BMI_max_observed_or_calculated'] = pd.to_numeric(df_encoded['BMI_max_observed_or_calculated'], errors='coerce')
    df_encoded['total_number_of_COVID_vaccine_doses'] = pd.to_numeric(df_encoded['total_number_of_COVID_vaccine_doses'], errors='coerce')

    # Scale continuous variables to help with numerical stability
    scaler = StandardScaler()
    df_encoded[['age_2020', 'BMI_max_observed_or_calculated']] = scaler.fit_transform(
        df_encoded[['age_2020', 'BMI_max_observed_or_calculated']]
    )

    # --- Build the Design Matrix ---

    # Define predictor columns: continuous variables and dummy variables.
    predictor_columns = ['age_2020', 'BMI_max_observed_or_calculated', 'total_number_of_COVID_vaccine_doses'] + [
        col for col in df_encoded.columns 
        if any(prefix in col for prefix in ['gender_concept_name_', 'race_concept_name_', 
                                            'DIABETES_indicator_', 'HYPERTENSION_indicator_', "PAXLOVID_indicator"])
    ]

    # Build design matrix X and ensure it is numeric
    X = df_encoded[predictor_columns].apply(pd.to_numeric, errors='coerce')
    X = sm.add_constant(X, has_constant='add')  # add intercept term

    # Treatment variable (ensure it is numeric)
    y = pd.to_numeric(df_encoded['drug_status'], errors='coerce')

    # Drop rows with any missing values in X or y
    data = pd.concat([X, y], axis=1).dropna()
    X_clean = data[X.columns]
    y_clean = data['drug_status']

    # --- Fit Regularized Logistic Regression ---

    # Use fit_regularized to help overcome numerical issues.
    # Adjust alpha (penalty strength) and maxiter as needed.
    result = sm.Logit(y_clean, X_clean).fit_regularized(method='l1', alpha=0.01, maxiter=1000)

    # --- Generate Propensity Scores ---

    # Calculate predicted propensity scores
    propensity_scores = result.predict(X_clean)

    # Clip predicted probabilities to avoid exact 0 or 1 (prevents division-by-zero in logit)
    propensity_scores = np.clip(propensity_scores, 1e-10, 1 - 1e-10)

    # Compute logit of the propensity scores
    logit_propensity = np.log(propensity_scores / (1 - propensity_scores))

    # Save the results back into the original dataframe.
    # (Here, we assume that the indices of 'data' match those in 'df'.)
    df.loc[data.index, 'propensity_score'] = propensity_scores
    df.loc[data.index, 'logit_propensity'] = logit_propensity
    
    return(df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ac015d94-ff04-4c1f-8b77-dc1fdae7b5c1"),
    Join_5mg_no_dementia_drug_later_both_bmi_imputed_for_psm_joint_imputed=Input(rid="ri.foundry.main.dataset.66dd4260-b7f8-4cff-be46-b0ca11a34ffa")
)
def propensity_score_5mg_vs_no_dementia_BMI_imputed(Join_5mg_no_dementia_drug_later_both_bmi_imputed_for_psm_joint_imputed):
    df = Join_5mg_no_dementia_drug_later_both_bmi_imputed_for_psm_joint_imputed
    # Assume df is your original DataFrame.
    # One-hot encode the categorical variables.
    categorical_vars = ['gender_concept_name', 'race_concept_name', 'DIABETES_indicator', 'HYPERTENSION_indicator', 'PAXLOVID_indicator']
    df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

    # Clean dummy variable names (remove spaces and special characters)
    df_encoded.columns = [col.replace(" ", "_").replace("-", "_") for col in df_encoded.columns]

    # Convert continuous columns to numeric (if not already)
    df_encoded['age_2020'] = pd.to_numeric(df_encoded['age_2020'], errors='coerce')
    df_encoded['BMI_max_observed_or_calculated'] = pd.to_numeric(df_encoded['BMI_max_observed_or_calculated'], errors='coerce')
    df_encoded['total_number_of_COVID_vaccine_doses'] = pd.to_numeric(df_encoded['total_number_of_COVID_vaccine_doses'], errors='coerce')

    # Scale continuous variables to help with numerical stability
    scaler = StandardScaler()
    df_encoded[['age_2020', 'BMI_max_observed_or_calculated']] = scaler.fit_transform(
        df_encoded[['age_2020', 'BMI_max_observed_or_calculated']]
    )

    # --- Build the Design Matrix ---

    # Define predictor columns: continuous variables and dummy variables.
    predictor_columns = ['age_2020', 'BMI_max_observed_or_calculated', 'total_number_of_COVID_vaccine_doses'] + [
        col for col in df_encoded.columns 
        if any(prefix in col for prefix in ['gender_concept_name_', 'race_concept_name_', 
                                            'DIABETES_indicator_', 'HYPERTENSION_indicator_', "PAXLOVID_indicator"])
    ]

    # Build design matrix X and ensure it is numeric
    X = df_encoded[predictor_columns].apply(pd.to_numeric, errors='coerce')
    X = sm.add_constant(X, has_constant='add')  # add intercept term

    # Treatment variable (ensure it is numeric)
    y = pd.to_numeric(df_encoded['drug_status'], errors='coerce')

    # Drop rows with any missing values in X or y
    data = pd.concat([X, y], axis=1).dropna()
    X_clean = data[X.columns]
    y_clean = data['drug_status']

    # --- Fit Regularized Logistic Regression ---

    # Use fit_regularized to help overcome numerical issues.
    # Adjust alpha (penalty strength) and maxiter as needed.
    result = sm.Logit(y_clean, X_clean).fit_regularized(method='l1', alpha=0.01, maxiter=1000)

    # --- Generate Propensity Scores ---

    # Calculate predicted propensity scores
    propensity_scores = result.predict(X_clean)

    # Clip predicted probabilities to avoid exact 0 or 1 (prevents division-by-zero in logit)
    propensity_scores = np.clip(propensity_scores, 1e-10, 1 - 1e-10)

    # Compute logit of the propensity scores
    logit_propensity = np.log(propensity_scores / (1 - propensity_scores))

    # Save the results back into the original dataframe.
    # (Here, we assume that the indices of 'data' match those in 'df'.)
    df.loc[data.index, 'propensity_score'] = propensity_scores
    df.loc[data.index, 'logit_propensity'] = logit_propensity
    
    return(df)


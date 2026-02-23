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
    Output(rid="ri.foundry.main.dataset.97bf7c0d-6a1d-4088-ad7c-24e455b1700a"),
    Imputed_mean_10mg_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.2eebd8ef-a1cd-4a06-9883-34d8273572cf"),
    Metformin_name_match_fo=Input(rid="ri.foundry.main.dataset.66638b01-90dd-4f1d-b6db-b9786d08268f")
)
def Dementia_10mg_match_Metformin_BMI_joint_imputed(Metformin_name_match_fo, Imputed_mean_10mg_bmi_joint_imputed):
    df1 = Imputed_mean_10mg_bmi_joint_imputed
    df2 = Metformin_name_match_fo
    df2_renamed = df2.withColumnRenamed("drug_exposure_start_date", "drug_exposure_start_date_Metformin") \
                    .select("person_id", "drug_exposure_start_date_Metformin")

    df_joined = df1.join(df2_renamed, on="person_id", how="left")

    return(df_joined)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.dacba14c-6a95-4641-98ae-da7771ada5b2"),
    Dementia_10mg_match_Metformin_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.97bf7c0d-6a1d-4088-ad7c-24e455b1700a")
)
def Dementia_10mg_no_Metformin_earliest_BMI_joint_imputed_10(Dementia_10mg_match_Metformin_BMI_joint_imputed):
    df_joined = Dementia_10mg_match_Metformin_BMI_joint_imputed

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNull())

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.16d83fc9-f823-4d38-be11-4283f26e2486"),
    Dementia_10mg_no_Metformin_earliest_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.dacba14c-6a95-4641-98ae-da7771ada5b2")
)
def Dementia_10mg_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed(Dementia_10mg_no_Metformin_earliest_BMI_joint_imputed_10):
    df = Dementia_10mg_no_Metformin_earliest_BMI_joint_imputed_10
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.1637b429-1627-485d-8d1a-bf4bbb38ccc9"),
    Dementia_10mg_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.16d83fc9-f823-4d38-be11-4283f26e2486")
)
def Dementia_10mg_no_Metformin_earliest_with_diabetes_30_death_rate(Dementia_10mg_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed):
    df = Dementia_10mg_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.833e820b-d083-41b9-9491-ab6ce9de2aad"),
    Dementia_10mg_no_Metformin_earliest_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.dacba14c-6a95-4641-98ae-da7771ada5b2")
)
def Dementia_10mg_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed(Dementia_10mg_no_Metformin_earliest_BMI_joint_imputed_10):
    df = Dementia_10mg_no_Metformin_earliest_BMI_joint_imputed_10
    df_filtered = df.filter(col("DIABETES_indicator") == 0)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.c635735c-2467-4186-88ee-8e81035ff471"),
    Dementia_10mg_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.833e820b-d083-41b9-9491-ab6ce9de2aad")
)
def Dementia_10mg_no_Metformin_earliest_without_diabetes_30_death_rate(Dementia_10mg_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed):
    df = Dementia_10mg_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.d408d7c6-d266-4211-8165-40223fcec62d"),
    Dementia_10mg_match_Metformin_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.97bf7c0d-6a1d-4088-ad7c-24e455b1700a")
)
def Dementia_10mg_with_Metformin_earliest_BMI_joint_imputed_10(Dementia_10mg_match_Metformin_BMI_joint_imputed):
    df_joined = Dementia_10mg_match_Metformin_BMI_joint_imputed

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNotNull())

    # Order by person_id, condition_start_date_COVID, and drug_exposure_start_date_Metformin in ascending order
    df_ordered = df_filtered.orderBy(["person_id", "condition_start_date_COVID", "drug_exposure_start_date_Metformin"], ascending=[True, True, True])

    # Keep only the first occurrence (earliest date) per person_id
    df_earliest = df_ordered.dropDuplicates(["person_id"])

    
    return(df_earliest)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.113fa0ad-0782-41ca-a263-419a1f22bf60"),
    Dementia_10mg_with_Metformin_earliest_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.d408d7c6-d266-4211-8165-40223fcec62d")
)
def Dementia_10mg_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed(Dementia_10mg_with_Metformin_earliest_BMI_joint_imputed_10):
    df = Dementia_10mg_with_Metformin_earliest_BMI_joint_imputed_10
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.ec66fd65-f36c-48d7-b346-085278d1b5ba"),
    Dementia_10mg_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.113fa0ad-0782-41ca-a263-419a1f22bf60")
)
def Dementia_10mg_with_Metformin_earliest_with_diabetes_death_rate(Dementia_10mg_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed):
    df = Dementia_10mg_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.c8466722-6da4-4e9d-9a18-ae42ad7df48c"),
    Metformin_name_match_fo=Input(rid="ri.foundry.main.dataset.66638b01-90dd-4f1d-b6db-b9786d08268f"),
    Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement_cleaned=Input(rid="ri.foundry.main.dataset.e641c3c2-f7b0-4384-be5e-e3e3acd5dd77")
)
def Dementia_no_drug_match_Metformin_BMI_joint_imputed_10(Metformin_name_match_fo, Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement_cleaned):
    df1 = Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement_cleaned
    df2 = Metformin_name_match_fo
    df2_renamed = df2.withColumnRenamed("drug_exposure_start_date", "drug_exposure_start_date_Metformin") \
                    .select("person_id", "drug_exposure_start_date_Metformin")

    df_joined = df1.join(df2_renamed, on="person_id", how="left")

    return(df_joined)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.928bae53-f6b8-488e-8702-0abf42042f6b"),
    Dementia_no_drug_match_Metformin_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.c8466722-6da4-4e9d-9a18-ae42ad7df48c")
)
def Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed_10(Dementia_no_drug_match_Metformin_BMI_joint_imputed_10):
    df_joined = Dementia_no_drug_match_Metformin_BMI_joint_imputed_10

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNull())

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.37658cff-ca6c-4f74-b3bd-8c71e0a89310"),
    Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.928bae53-f6b8-488e-8702-0abf42042f6b")
)
def Dementia_no_drug_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed(Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed_10):
    df = Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed_10
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.a9e8ef51-528c-422f-8621-f1c777e2161a"),
    Dementia_no_drug_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.37658cff-ca6c-4f74-b3bd-8c71e0a89310")
)
def Dementia_no_drug_no_Metformin_earliest_with_diabetes_30_death_rate(Dementia_no_drug_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed):
    df = Dementia_no_drug_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.fa6f653a-7d40-4458-816a-296fb8e3001c"),
    Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.928bae53-f6b8-488e-8702-0abf42042f6b")
)
def Dementia_no_drug_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed(Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed_10):
    df = Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed_10
    df_filtered = df.filter(col("DIABETES_indicator") == 0)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.0d27c8bd-ed4e-427a-953a-54e02a71ca0a"),
    Dementia_no_drug_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.fa6f653a-7d40-4458-816a-296fb8e3001c")
)
def Dementia_no_drug_no_Metformin_earliest_without_diabetes_30_death_rate(Dementia_no_drug_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed):
    df = Dementia_no_drug_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.f0c2dc7c-dbe1-4c83-ad77-24aa99d95b7b"),
    Dementia_no_drug_match_Metformin_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.c8466722-6da4-4e9d-9a18-ae42ad7df48c")
)
def Dementia_no_drug_with_Metformin_earliest_BMI_joint_imputed_10(Dementia_no_drug_match_Metformin_BMI_joint_imputed_10):
    df_joined = Dementia_no_drug_match_Metformin_BMI_joint_imputed_10

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNotNull())

    # Order by person_id, condition_start_date_COVID, and drug_exposure_start_date_Metformin in ascending order
    df_ordered = df_filtered.orderBy(["person_id", "condition_start_date_COVID", "drug_exposure_start_date_Metformin"], ascending=[True, True, True])

    # Keep only the first occurrence (earliest date) per person_id
    df_earliest = df_ordered.dropDuplicates(["person_id"])

    
    return(df_earliest)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d3e9f1c3-4197-4491-a831-1213165767a7"),
    Dementia_no_drug_with_Metformin_earliest_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.f0c2dc7c-dbe1-4c83-ad77-24aa99d95b7b")
)
def Dementia_no_drug_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed(Dementia_no_drug_with_Metformin_earliest_BMI_joint_imputed_10):
    df = Dementia_no_drug_with_Metformin_earliest_BMI_joint_imputed_10
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.260cd4c6-c653-4c37-bbbd-0cb0753cebfb"),
    Dementia_no_drug_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.d3e9f1c3-4197-4491-a831-1213165767a7")
)
def Dementia_no_drug_with_Metformin_earliest_with_diabetes_death_rate(Dementia_no_drug_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed):
    df = Dementia_no_drug_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.66638b01-90dd-4f1d-b6db-b9786d08268f"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
def Metformin_name_match_fo(drug_exposure):
    df = drug_exposure
    # Assuming df is your dataframe
    filtered_df = df.filter((col("drug_concept_name").contains("Metformin")) | 
                            (col("drug_concept_name").contains("metformin")))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.c3fac2ef-e6f0-4deb-98a1-d1290c0b8ed0"),
    No_Dementia_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.051d10cf-070b-4d8f-bc2e-bd7b844b33ea")
)
def No_Dementia_no_Metformin_earliest_no_diabetes_death_rate(No_Dementia_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed):
    df = No_Dementia_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.df8050d3-9188-4b79-b829-bfe1e1d85f9b"),
    No_dementia_no_Metformin_earliest_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.6790d4f2-3777-4810-a970-e11d14da3be3")
)
def No_Dementia_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed(No_dementia_no_Metformin_earliest_BMI_joint_imputed_10):
    df = No_dementia_no_Metformin_earliest_BMI_joint_imputed_10
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.b75017ff-8258-45c9-9abb-103835f56eab"),
    No_Dementia_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.df8050d3-9188-4b79-b829-bfe1e1d85f9b")
)
def No_Dementia_no_Metformin_earliest_with_diabetes_death_rate(No_Dementia_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed):
    df = No_Dementia_no_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.051d10cf-070b-4d8f-bc2e-bd7b844b33ea"),
    No_dementia_no_Metformin_earliest_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.6790d4f2-3777-4810-a970-e11d14da3be3")
)
def No_Dementia_no_Metformin_earliest_without_diabetes_10group_BMI_joint_imputed(No_dementia_no_Metformin_earliest_BMI_joint_imputed_10):
    df = No_dementia_no_Metformin_earliest_BMI_joint_imputed_10
    df_filtered = df.filter(col("DIABETES_indicator") == 0)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.28f80d5a-994e-4557-b943-36253d4d1e0a"),
    Metformin_name_match_fo=Input(rid="ri.foundry.main.dataset.66638b01-90dd-4f1d-b6db-b9786d08268f"),
    Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement_cleaned=Input(rid="ri.foundry.main.dataset.dbb1bded-2314-404f-a3d1-2b82ae688130")
)
def No_dementia_match_Metformin_BMI_joint_imputed_10(Metformin_name_match_fo, Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement_cleaned):
    df1 = Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement_cleaned
    df2 = Metformin_name_match_fo
    df2_renamed = df2.withColumnRenamed("drug_exposure_start_date", "drug_exposure_start_date_Metformin") \
                    .select("person_id", "drug_exposure_start_date_Metformin")

    df_joined = df1.join(df2_renamed, on="person_id", how="left")

    return(df_joined)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6790d4f2-3777-4810-a970-e11d14da3be3"),
    No_dementia_match_Metformin_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.28f80d5a-994e-4557-b943-36253d4d1e0a")
)
def No_dementia_no_Metformin_earliest_BMI_joint_imputed_10(No_dementia_match_Metformin_BMI_joint_imputed_10):
    df_joined = No_dementia_match_Metformin_BMI_joint_imputed_10

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNull())

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.dd555266-0b89-40f0-981b-c0fe143c2bb8"),
    No_dementia_match_Metformin_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.28f80d5a-994e-4557-b943-36253d4d1e0a")
)
def No_dementia_with_Metformin_earliest_BMI_joint_imputed_10(No_dementia_match_Metformin_BMI_joint_imputed_10):
    df_joined = No_dementia_match_Metformin_BMI_joint_imputed_10

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNotNull())

    # Order by person_id, condition_start_date_COVID, and drug_exposure_start_date_Metformin in ascending order
    df_ordered = df_filtered.orderBy(["person_id", "condition_start_date_COVID", "drug_exposure_start_date_Metformin"], ascending=[True, True, True])

    # Keep only the first occurrence (earliest date) per person_id
    df_earliest = df_ordered.dropDuplicates(["person_id"])

    
    return(df_earliest)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a26553e3-2642-404e-9863-edb843de03fd"),
    No_dementia_with_Metformin_earliest_BMI_joint_imputed_10=Input(rid="ri.foundry.main.dataset.dd555266-0b89-40f0-981b-c0fe143c2bb8")
)
def No_dementia_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed(No_dementia_with_Metformin_earliest_BMI_joint_imputed_10):
    df = No_dementia_with_Metformin_earliest_BMI_joint_imputed_10
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.56f30541-4d39-46ac-bd31-8f395828e1b9"),
    No_dementia_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.a26553e3-2642-404e-9863-edb843de03fd")
)
def No_dementia_with_Metformin_earliest_with_diabetes_death_rate(No_dementia_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed):
    df = No_dementia_with_Metformin_earliest_with_diabetes_10group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.e641c3c2-f7b0-4384-be5e-e3e3acd5dd77"),
    Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.95ac1017-70d0-4860-b98c-63d53d6c9b3a")
)
def Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement_cleaned(Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement):
    df1 = Sampling_propensity_score_10mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement
    for col_name in df1.columns:
        if col_name.startswith("c_"):
            df1 = df1.withColumnRenamed(col_name, col_name[2:])
    # Suppose df is your DataFrame
    cols_to_keep = [c for c in df1.columns if not c.startswith("t_")]

    df1 = df1.select(*cols_to_keep)

    return df1

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.dbb1bded-2314-404f-a3d1-2b82ae688130"),
    Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.aa80ff10-5005-450e-a798-3f4488808779")
)
def Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement_cleaned(Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement):
    df1 = Sampling_propensity_score_10mg_vs_no_dementia_knn5_then_age_match_nonreplacement
    for col_name in df1.columns:
        if col_name.startswith("c_"):
            df1 = df1.withColumnRenamed(col_name, col_name[2:])
    # Suppose df is your DataFrame
    cols_to_keep = [c for c in df1.columns if not c.startswith("t_")]

    df1 = df1.select(*cols_to_keep)

    return df1


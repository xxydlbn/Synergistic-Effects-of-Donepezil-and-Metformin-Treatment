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
    Output(rid="ri.foundry.main.dataset.b9d6f5ed-3497-466f-b169-92f93cde9636"),
    Imputed_mean_5mg_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.68ccc96c-c8ea-480a-8cf8-4dbf136184af"),
    Metformin_name_match_fo=Input(rid="ri.foundry.main.dataset.ae359ac1-3cb0-4ac2-a221-eb857d066894")
)
def Dementia_5mg_match_Metformin_BMI_joint_imputed(Metformin_name_match_fo, Imputed_mean_5mg_bmi_joint_imputed):
    df1 = Imputed_mean_5mg_bmi_joint_imputed
    df2 = Metformin_name_match_fo
    df2_renamed = df2.withColumnRenamed("drug_exposure_start_date", "drug_exposure_start_date_Metformin") \
                    .select("person_id", "drug_exposure_start_date_Metformin")

    df_joined = df1.join(df2_renamed, on="person_id", how="left")

    return(df_joined)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f83b2570-87a7-42b4-a7ee-0d9301ddbe54"),
    Dementia_5mg_match_Metformin_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.b9d6f5ed-3497-466f-b169-92f93cde9636")
)
def Dementia_5mg_no_Metformin_earliest_BMI_joint_imputed(Dementia_5mg_match_Metformin_BMI_joint_imputed):
    df_joined = Dementia_5mg_match_Metformin_BMI_joint_imputed

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNull())

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.b2811b10-509d-4854-b87e-58e1f3ea5d45"),
    Dementia_5mg_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.326474ba-5756-4e44-be47-472ad15ee297")
)
def Dementia_5mg_no_Metformin_earliest_with_diabetes_30_death_rate(Dementia_5mg_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed):
    df = Dementia_5mg_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.326474ba-5756-4e44-be47-472ad15ee297"),
    Dementia_5mg_no_Metformin_earliest_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.f83b2570-87a7-42b4-a7ee-0d9301ddbe54")
)
def Dementia_5mg_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed(Dementia_5mg_no_Metformin_earliest_BMI_joint_imputed):
    df = Dementia_5mg_no_Metformin_earliest_BMI_joint_imputed
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.b3185588-e95b-4ff7-a960-166492d8465b"),
    Dementia_5mg_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.5ca69544-6c20-4aa5-9269-0adc4ae735b2")
)
def Dementia_5mg_no_Metformin_earliest_without_diabetes_30_death_rate(Dementia_5mg_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed):
    df = Dementia_5mg_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.5ca69544-6c20-4aa5-9269-0adc4ae735b2"),
    Dementia_5mg_no_Metformin_earliest_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.f83b2570-87a7-42b4-a7ee-0d9301ddbe54")
)
def Dementia_5mg_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed(Dementia_5mg_no_Metformin_earliest_BMI_joint_imputed):
    df = Dementia_5mg_no_Metformin_earliest_BMI_joint_imputed
    df_filtered = df.filter(col("DIABETES_indicator") == 0)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e5ef29de-b220-4ebe-af1f-77811fa03ba0"),
    Dementia_5mg_match_Metformin_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.b9d6f5ed-3497-466f-b169-92f93cde9636")
)
def Dementia_5mg_with_Metformin_earliest_BMI_joint_imputed(Dementia_5mg_match_Metformin_BMI_joint_imputed):
    df_joined = Dementia_5mg_match_Metformin_BMI_joint_imputed

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNotNull())

    # Order by person_id, condition_start_date_COVID, and drug_exposure_start_date_Metformin in ascending order
    df_ordered = df_filtered.orderBy(["person_id", "condition_start_date_COVID", "drug_exposure_start_date_Metformin"], ascending=[True, True, True])

    # Keep only the first occurrence (earliest date) per person_id
    df_earliest = df_ordered.dropDuplicates(["person_id"])

    
    return(df_earliest)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b467f2e4-51c6-40b0-bb78-a8ad6e2ed9f4"),
    Dementia_5mg_with_Metformin_earliest_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.e5ef29de-b220-4ebe-af1f-77811fa03ba0")
)
def Dementia_5mg_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed(Dementia_5mg_with_Metformin_earliest_BMI_joint_imputed):
    df = Dementia_5mg_with_Metformin_earliest_BMI_joint_imputed
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.28d41ae9-b17c-44ce-b51e-8b7c57de2937"),
    Dementia_5mg_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.b467f2e4-51c6-40b0-bb78-a8ad6e2ed9f4")
)
def Dementia_5mg_with_Metformin_earliest_with_diabetes_death_rate(Dementia_5mg_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed):
    df = Dementia_5mg_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.9d5c102f-939e-4e82-9605-8f3a827db2ed"),
    Metformin_name_match_fo=Input(rid="ri.foundry.main.dataset.ae359ac1-3cb0-4ac2-a221-eb857d066894"),
    Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement_cleaned=Input(rid="ri.foundry.main.dataset.6e64aa51-0a7c-4de6-b7c5-76bccab36dc0")
)
def Dementia_no_drug_match_Metformin_BMI_joint_imputed(Metformin_name_match_fo, Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement_cleaned):
    df1 = Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement_cleaned
    df2 = Metformin_name_match_fo
    df2_renamed = df2.withColumnRenamed("drug_exposure_start_date", "drug_exposure_start_date_Metformin") \
                    .select("person_id", "drug_exposure_start_date_Metformin")

    df_joined = df1.join(df2_renamed, on="person_id", how="left")

    return(df_joined)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.64a362ce-337c-4121-9a01-0caf43eccccd"),
    Dementia_no_drug_match_Metformin_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.9d5c102f-939e-4e82-9605-8f3a827db2ed")
)
def Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed(Dementia_no_drug_match_Metformin_BMI_joint_imputed):
    df_joined = Dementia_no_drug_match_Metformin_BMI_joint_imputed

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNull())

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.abe14be5-3e87-4b7c-b610-540eafdbc10e"),
    Dementia_no_drug_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.692a8ef7-7d8b-48c8-bb4c-c308a6b1223f")
)
def Dementia_no_drug_no_Metformin_earliest_with_diabetes_30_death_rate(Dementia_no_drug_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed):
    df = Dementia_no_drug_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.692a8ef7-7d8b-48c8-bb4c-c308a6b1223f"),
    Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.64a362ce-337c-4121-9a01-0caf43eccccd")
)
def Dementia_no_drug_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed(Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed):
    df = Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.79b1b26f-253f-4356-99f4-bf7aeaea9503"),
    Dementia_no_drug_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.371f6f07-e3e3-4229-abb9-e49344d761af")
)
def Dementia_no_drug_no_Metformin_earliest_without_diabetes_30_death_rate(Dementia_no_drug_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed):
    df = Dementia_no_drug_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.371f6f07-e3e3-4229-abb9-e49344d761af"),
    Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.64a362ce-337c-4121-9a01-0caf43eccccd")
)
def Dementia_no_drug_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed(Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed):
    df = Dementia_no_drug_no_Metformin_earliest_BMI_joint_imputed
    df_filtered = df.filter(col("DIABETES_indicator") == 0)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.39cb116f-2f83-49cb-9e37-634e8977f7ff"),
    Dementia_no_drug_match_Metformin_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.9d5c102f-939e-4e82-9605-8f3a827db2ed")
)
def Dementia_no_drug_with_Metformin_earliest_BMI_joint_imputed(Dementia_no_drug_match_Metformin_BMI_joint_imputed):
    df_joined = Dementia_no_drug_match_Metformin_BMI_joint_imputed

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNotNull())

    # Order by person_id, condition_start_date_COVID, and drug_exposure_start_date_Metformin in ascending order
    df_ordered = df_filtered.orderBy(["person_id", "condition_start_date_COVID", "drug_exposure_start_date_Metformin"], ascending=[True, True, True])

    # Keep only the first occurrence (earliest date) per person_id
    df_earliest = df_ordered.dropDuplicates(["person_id"])

    
    return(df_earliest)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4ac80d9d-15a1-4117-9f2b-d15aebad7b9b"),
    Dementia_no_drug_with_Metformin_earliest_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.39cb116f-2f83-49cb-9e37-634e8977f7ff")
)
def Dementia_no_drug_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed(Dementia_no_drug_with_Metformin_earliest_BMI_joint_imputed):
    df = Dementia_no_drug_with_Metformin_earliest_BMI_joint_imputed
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.e772d573-8783-4691-b94d-be631e161767"),
    Dementia_no_drug_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.4ac80d9d-15a1-4117-9f2b-d15aebad7b9b")
)
def Dementia_no_drug_with_Metformin_earliest_with_diabetes_death_rate(Dementia_no_drug_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed):
    df = Dementia_no_drug_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.ae359ac1-3cb0-4ac2-a221-eb857d066894"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
def Metformin_name_match_fo(drug_exposure):
    df = drug_exposure
    # Assuming df is your dataframe
    filtered_df = df.filter((col("drug_concept_name").contains("Metformin")) | 
                            (col("drug_concept_name").contains("metformin")))

    return(filtered_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.8095ab44-1df3-49e6-b015-34285900ff2b"),
    No_Dementia_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.a73016e7-d8cc-4303-83af-a017578c4e73")
)
def No_Dementia_no_Metformin_earliest_no_diabetes_death_rate_(No_Dementia_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed):
    df = No_Dementia_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.378023de-a92f-47f7-b90b-dd28409f0d76"),
    No_dementia_no_Metformin_earliest_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.bda3dc2d-04b8-4c16-807d-869bf49bdc83")
)
def No_Dementia_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed(No_dementia_no_Metformin_earliest_BMI_joint_imputed):
    df = No_dementia_no_Metformin_earliest_BMI_joint_imputed
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.e0f47f37-7983-46ce-81b8-9fdc7a386b15"),
    No_Dementia_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.378023de-a92f-47f7-b90b-dd28409f0d76")
)
def No_Dementia_no_Metformin_earliest_with_diabetes_death_rate(No_Dementia_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed):
    df = No_Dementia_no_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.a73016e7-d8cc-4303-83af-a017578c4e73"),
    No_dementia_no_Metformin_earliest_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.bda3dc2d-04b8-4c16-807d-869bf49bdc83")
)
def No_Dementia_no_Metformin_earliest_without_diabetes_5group_BMI_joint_imputed(No_dementia_no_Metformin_earliest_BMI_joint_imputed):
    df = No_dementia_no_Metformin_earliest_BMI_joint_imputed
    df_filtered = df.filter(col("DIABETES_indicator") == 0)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ef5acbd8-c756-4878-a0c7-271b9f33b69e"),
    Metformin_name_match_fo=Input(rid="ri.foundry.main.dataset.ae359ac1-3cb0-4ac2-a221-eb857d066894"),
    Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement_cleaned=Input(rid="ri.foundry.main.dataset.96f0f26e-84a4-46ca-8aef-95abebb54be3")
)
def No_dementia_match_Metformin_BMI_joint_imputed(Metformin_name_match_fo, Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement_cleaned):
    df1 = Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement_cleaned
    df2 = Metformin_name_match_fo
    df2_renamed = df2.withColumnRenamed("drug_exposure_start_date", "drug_exposure_start_date_Metformin") \
                    .select("person_id", "drug_exposure_start_date_Metformin")

    df_joined = df1.join(df2_renamed, on="person_id", how="left")

    return(df_joined)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bda3dc2d-04b8-4c16-807d-869bf49bdc83"),
    No_dementia_match_Metformin_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.ef5acbd8-c756-4878-a0c7-271b9f33b69e")
)
def No_dementia_no_Metformin_earliest_BMI_joint_imputed(No_dementia_match_Metformin_BMI_joint_imputed):
    df_joined = No_dementia_match_Metformin_BMI_joint_imputed

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNull())

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0080a2fa-bd98-4c36-88e4-ba870174571b"),
    No_dementia_match_Metformin_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.ef5acbd8-c756-4878-a0c7-271b9f33b69e")
)
def No_dementia_with_Metformin_earliest_BMI_joint_imputed(No_dementia_match_Metformin_BMI_joint_imputed):
    df_joined = No_dementia_match_Metformin_BMI_joint_imputed

    # Filter out rows where drug_exposure_start_date_Metformin is null
    df_filtered = df_joined.filter(df_joined["drug_exposure_start_date_Metformin"].isNotNull())

    # Order by person_id, condition_start_date_COVID, and drug_exposure_start_date_Metformin in ascending order
    df_ordered = df_filtered.orderBy(["person_id", "condition_start_date_COVID", "drug_exposure_start_date_Metformin"], ascending=[True, True, True])

    # Keep only the first occurrence (earliest date) per person_id
    df_earliest = df_ordered.dropDuplicates(["person_id"])

    
    return(df_earliest)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c4e43a16-ee23-481e-bf4a-0e4f0294b52e"),
    No_dementia_with_Metformin_earliest_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.0080a2fa-bd98-4c36-88e4-ba870174571b")
)
def No_dementia_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed(No_dementia_with_Metformin_earliest_BMI_joint_imputed):
    df = No_dementia_with_Metformin_earliest_BMI_joint_imputed
    df_filtered = df.filter(col("DIABETES_indicator") == 1)

    return(df_filtered)

@transform_pandas(
    Output(rid="ri.vector.main.execute.75a14bd3-5c50-4191-8387-e1f11e925ab8"),
    No_dementia_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.c4e43a16-ee23-481e-bf4a-0e4f0294b52e")
)
def No_dementia_with_Metformin_earliest_with_diabetes_death_rate(No_dementia_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed):
    df = No_dementia_with_Metformin_earliest_with_diabetes_5group_BMI_joint_imputed
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
    Output(rid="ri.foundry.main.dataset.6e64aa51-0a7c-4de6-b7c5-76bccab36dc0"),
    Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.d3d86760-e498-4acd-9aa1-1cba0024317a")
)
def Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement_cleaned(Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement):
    df1 = Sampling_propensity_score_5mg_vs_dementia_no_drug_knn5_then_age_match_nonreplacement
    for col_name in df1.columns:
        if col_name.startswith("c_"):
            df1 = df1.withColumnRenamed(col_name, col_name[2:])
    # Suppose df is your DataFrame
    cols_to_keep = [c for c in df1.columns if not c.startswith("t_")]

    df1 = df1.select(*cols_to_keep)

    return df1

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.96f0f26e-84a4-46ca-8aef-95abebb54be3"),
    Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement=Input(rid="ri.foundry.main.dataset.54d6cc13-fdf5-4dcf-af26-1399a10e114b")
)
def Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement_cleaned(Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement):
    df1 = Sampling_propensity_score_5mg_vs_no_dementia_knn5_then_age_match_nonreplacement
    for col_name in df1.columns:
        if col_name.startswith("c_"):
            df1 = df1.withColumnRenamed(col_name, col_name[2:])
    # Suppose df is your DataFrame
    cols_to_keep = [c for c in df1.columns if not c.startswith("t_")]

    df1 = df1.select(*cols_to_keep)

    return df1


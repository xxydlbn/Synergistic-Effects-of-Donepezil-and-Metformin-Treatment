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
    Output(rid="ri.foundry.main.dataset.b2399564-b7e7-426e-8f51-0c0169f1123d"),
    Dementia_drug_no_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.798e4531-9ef0-4212-ae93-48e3dbaec2a8"),
    Dementia_drug_no_metformin_earliest_without_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.89004968-9005-4e75-a2ac-c76dd2c62273"),
    Dementia_drug_with_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.09c371c4-2237-4790-b5ae-77070fed8786")
)
def Backup_1_DDDM(Dementia_drug_no_metformin_earliest_without_diabetes_5and10_, Dementia_drug_no_metformin_earliest_with_diabetes_5and10_, Dementia_drug_with_metformin_earliest_with_diabetes_5and10_):
    df1 = Dementia_drug_no_metformin_earliest_without_diabetes_5and10_
    df2 = Dementia_drug_no_metformin_earliest_with_diabetes_5and10_
    df3 = Dementia_drug_with_metformin_earliest_with_diabetes_5and10_

    # Add metformin_status column to each df
    df1 = df1.withColumn("metformin_status", F.lit("no Metformin"))
    df2 = df2.withColumn("metformin_status", F.lit("no Metformin"))
    df3 = df3.withColumn("metformin_status", F.lit("with Metformin"))

    df1 = df1.withColumn("diabetes_status", F.lit("no Diabetes"))
    df2 = df2.withColumn("diabetes_status", F.lit("with Diabetes"))
    df3 = df3.withColumn("diabetes_status", F.lit("with Diabetes"))

    df1 = df1.withColumn("dementia_status", F.lit("with Dementia"))
    df2 = df2.withColumn("dementia_status", F.lit("with Dementia"))
    df3 = df3.withColumn("dementia_status", F.lit("with Dementia"))

    df1 = df1.withColumn("donepezil_status", F.lit("with Donepezil"))
    df2 = df2.withColumn("donepezil_status", F.lit("with Donepezil"))
    df3 = df3.withColumn("donepezil_status", F.lit("with Donepezil")) 

    # Now union them
    final_df = df1.unionByName(df2).unionByName(df3)

    return final_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4c1937e1-36c3-440d-a29d-609b3b5490b9"),
    Dementia_no_drug_no_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.e8274a6a-e1e6-4caa-b937-466c303e6c82"),
    Dementia_no_drug_no_metformin_earliest_without_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.e073d5c6-cd95-4cb2-98f3-eaee27c90d8d"),
    Dementia_no_drug_with_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.3a95fc19-568d-4d36-942b-ab3c201ab14b")
)
def Backup_2_DDDM(Dementia_no_drug_no_metformin_earliest_without_diabetes_5and10_, Dementia_no_drug_with_metformin_earliest_with_diabetes_5and10_, Dementia_no_drug_no_metformin_earliest_with_diabetes_5and10_):

    df1 = Dementia_no_drug_no_metformin_earliest_without_diabetes_5and10_
    df2 = Dementia_no_drug_no_metformin_earliest_with_diabetes_5and10_
    df3 = Dementia_no_drug_with_metformin_earliest_with_diabetes_5and10_

    # Add metformin_status column to each df
    df1 = df1.withColumn("metformin_status", F.lit("no Metformin"))
    df2 = df2.withColumn("metformin_status", F.lit("no Metformin"))
    df3 = df3.withColumn("metformin_status", F.lit("with Metformin"))

    df1 = df1.withColumn("diabetes_status", F.lit("no Diabetes"))
    df2 = df2.withColumn("diabetes_status", F.lit("with Diabetes"))
    df3 = df3.withColumn("diabetes_status", F.lit("with Diabetes"))

    df1 = df1.withColumn("dementia_status", F.lit("with Dementia"))
    df2 = df2.withColumn("dementia_status", F.lit("with Dementia"))
    df3 = df3.withColumn("dementia_status", F.lit("with Dementia"))

    df1 = df1.withColumn("donepezil_status", F.lit("no Donepezil"))
    df2 = df2.withColumn("donepezil_status", F.lit("no Donepezil"))
    df3 = df3.withColumn("donepezil_status", F.lit("no Donepezil")) 

        # Now union them
    final_df = df1.unionByName(df2).unionByName(df3)

    return final_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.feb9d87c-a7ab-4a21-b7bc-4e0b79abafb0"),
    No_dementia_no_metformin_earliest_no_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.efc89462-a6d7-46ab-bf18-0d6c435a73d5"),
    No_dementia_no_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.c5486587-4e26-4fb8-ae82-bc6f70f36a02"),
    No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_=Input(rid="ri.foundry.main.dataset.1715a2b4-f7d5-4be6-afd4-54618da2713a")
)
def Backup_3_DDDM(No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_, No_dementia_no_metformin_earliest_with_diabetes_5and10_, No_dementia_no_metformin_earliest_no_diabetes_5and10_):
    df1 = No_dementia_no_metformin_earliest_no_diabetes_5and10_
    df2 = No_dementia_no_metformin_earliest_with_diabetes_5and10_
    df3 = No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_

    # Add metformin_status column to each df
    df1 = df1.withColumn("metformin_status", F.lit("no Metformin"))
    df2 = df2.withColumn("metformin_status", F.lit("no Metformin"))
    df3 = df3.withColumn("metformin_status", F.lit("with Metformin"))

    df1 = df1.withColumn("diabetes_status", F.lit("no Diabetes"))
    df2 = df2.withColumn("diabetes_status", F.lit("with Diabetes"))
    df3 = df3.withColumn("diabetes_status", F.lit("with Diabetes"))

    df1 = df1.withColumn("dementia_status", F.lit("no Dementia"))
    df2 = df2.withColumn("dementia_status", F.lit("no Dementia"))
    df3 = df3.withColumn("dementia_status", F.lit("no Dementia"))

    df1 = df1.withColumn("donepezil_status", F.lit("no Donepezil"))
    df2 = df2.withColumn("donepezil_status", F.lit("no Donepezil"))
    df3 = df3.withColumn("donepezil_status", F.lit("no Donepezil")) 
        # Now union them
    final_df = df1.unionByName(df2).unionByName(df3)

    return final_df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.781be2ca-cad9-4dc6-98ae-d34ab60ce525"),
    Dementia_drug_no_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.798e4531-9ef0-4212-ae93-48e3dbaec2a8"),
    Dementia_drug_no_metformin_earliest_without_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.89004968-9005-4e75-a2ac-c76dd2c62273"),
    Dementia_drug_with_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.09c371c4-2237-4790-b5ae-77070fed8786")
)
def Dementia_drug_all_withwithout_diabetes_withwithout_metformin_BMI_joint_imputed(Dementia_drug_no_metformin_earliest_without_diabetes_5and10_, Dementia_drug_no_metformin_earliest_with_diabetes_5and10_, Dementia_drug_with_metformin_earliest_with_diabetes_5and10_):
    df1 = Dementia_drug_no_metformin_earliest_without_diabetes_5and10_
    df2 = Dementia_drug_no_metformin_earliest_with_diabetes_5and10_
    df3 = Dementia_drug_with_metformin_earliest_with_diabetes_5and10_

    df_union = df1.union(df2)
    df_union = df_union.union(df3)

    return(df_union)

@transform_pandas(
    Output(rid="ri.vector.main.execute.6bf3fb14-f680-4c4d-a903-632fb77d3262"),
    Dementia_drug_all_withwithout_diabetes_withwithout_metformin_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.781be2ca-cad9-4dc6-98ae-d34ab60ce525")
)
def Dementia_drug_all_withwithout_diabetes_withwithout_metformin_death_rate_BMi_joint_imputed(Dementia_drug_all_withwithout_diabetes_withwithout_metformin_BMI_joint_imputed):
    df = Dementia_drug_all_withwithout_diabetes_withwithout_metformin_BMI_joint_imputed
    # Create columns for 30d, 60d, and anytime death
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

    # No time limit: any death after COVID diagnosis date
    df = df.withColumn(
        "death_anytime",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Summary by age
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    )

    # Overall summary
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.798e4531-9ef0-4212-ae93-48e3dbaec2a8"),
    Dementia_10mg_no_metformin_earliest_with_diabetes_10group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.16d83fc9-f823-4d38-be11-4283f26e2486"),
    Dementia_5mg_no_metformin_earliest_with_diabetes_5group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.326474ba-5756-4e44-be47-472ad15ee297")
)
def Dementia_drug_no_metformin_earliest_with_diabetes_5and10_(Dementia_5mg_no_metformin_earliest_with_diabetes_5group_bmi_joint_imputed,Dementia_10mg_no_metformin_earliest_with_diabetes_10group_bmi_joint_imputed):
    df1 = Dementia_5mg_no_metformin_earliest_with_diabetes_5group_bmi_joint_imputed
    df2 = Dementia_10mg_no_metformin_earliest_with_diabetes_10group_bmi_joint_imputed

    df1 = df1.withColumn("donepezil_dose", F.lit("5mg"))
    df2 = df2.withColumn("donepezil_dose", F.lit("10mg"))

    df_union = df1.union(df2)

    return(df_union)
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.91a2a161-37fc-4294-8eeb-b55ebd38fd13"),
    Dementia_drug_no_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.798e4531-9ef0-4212-ae93-48e3dbaec2a8")
)
def Dementia_drug_no_metformin_earliest_with_diabetes_5and10_death_rate(Dementia_drug_no_metformin_earliest_with_diabetes_5and10_):
    df = Dementia_drug_no_metformin_earliest_with_diabetes_5and10_
    # Create columns for 30d, 60d, and anytime death
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

    # No time limit: any death after COVID diagnosis date
    df = df.withColumn(
        "death_anytime",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Summary by age
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    )

    # Overall summary
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.89004968-9005-4e75-a2ac-c76dd2c62273"),
    Dementia_10mg_no_metformin_earliest_without_diabetes_10group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.833e820b-d083-41b9-9491-ab6ce9de2aad"),
    Dementia_5mg_no_metformin_earliest_without_diabetes_5group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.5ca69544-6c20-4aa5-9269-0adc4ae735b2")
)
def Dementia_drug_no_metformin_earliest_without_diabetes_5and10_(Dementia_5mg_no_metformin_earliest_without_diabetes_5group_bmi_joint_imputed, Dementia_10mg_no_metformin_earliest_without_diabetes_10group_bmi_joint_imputed):
    df1 = Dementia_5mg_no_metformin_earliest_without_diabetes_5group_bmi_joint_imputed
    df2 = Dementia_10mg_no_metformin_earliest_without_diabetes_10group_bmi_joint_imputed

    df1 = df1.withColumn("donepezil_dose", F.lit("5mg"))
    df2 = df2.withColumn("donepezil_dose", F.lit("10mg"))

    df_union = df1.union(df2)

    return(df_union)

@transform_pandas(
    Output(rid="ri.vector.main.execute.4119141c-b57f-4af8-a6c8-12d6da6b083a"),
    Dementia_drug_no_metformin_earliest_without_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.89004968-9005-4e75-a2ac-c76dd2c62273")
)
def Dementia_drug_no_metformin_earliest_without_diabetes_5and10_death_rate(Dementia_drug_no_metformin_earliest_without_diabetes_5and10_):
    df = Dementia_drug_no_metformin_earliest_without_diabetes_5and10_
    # Create columns for 30d, 60d, and anytime death
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

    # No time limit: any death after COVID diagnosis date
    df = df.withColumn(
        "death_anytime",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Summary by age
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    )

    # Overall summary
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.09c371c4-2237-4790-b5ae-77070fed8786"),
    Dementia_10mg_with_metformin_earliest_with_diabetes_10group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.113fa0ad-0782-41ca-a263-419a1f22bf60"),
    Dementia_5mg_with_metformin_earliest_with_diabetes_5group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.b467f2e4-51c6-40b0-bb78-a8ad6e2ed9f4")
)
def Dementia_drug_with_metformin_earliest_with_diabetes_5and10_(Dementia_5mg_with_metformin_earliest_with_diabetes_5group_bmi_joint_imputed, Dementia_10mg_with_metformin_earliest_with_diabetes_10group_bmi_joint_imputed):
    df1 = Dementia_5mg_with_metformin_earliest_with_diabetes_5group_bmi_joint_imputed
    df2 = Dementia_10mg_with_metformin_earliest_with_diabetes_10group_bmi_joint_imputed

    df1 = df1.withColumn("donepezil_dose", F.lit("5mg"))
    df2 = df2.withColumn("donepezil_dose", F.lit("10mg"))

    df_union = df1.union(df2)

    return(df_union)

@transform_pandas(
    Output(rid="ri.vector.main.execute.99f9bc37-19a6-4a3f-a0dc-3b340e0bc4bb"),
    Dementia_drug_with_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.09c371c4-2237-4790-b5ae-77070fed8786")
)
def Dementia_drug_with_metformin_earliest_with_diabetes_5and10_death_rate(Dementia_drug_with_metformin_earliest_with_diabetes_5and10_):
    df = Dementia_drug_with_metformin_earliest_with_diabetes_5and10_
    # Create columns for 30d, 60d, and anytime death
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

    # No time limit: any death after COVID diagnosis date
    df = df.withColumn(
        "death_anytime",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Summary by age
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    )

    # Overall summary
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e8274a6a-e1e6-4caa-b937-466c303e6c82"),
    Dementia_no_drug_no_metformin_earliest_with_diabetes_10group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.37658cff-ca6c-4f74-b3bd-8c71e0a89310"),
    Dementia_no_drug_no_metformin_earliest_with_diabetes_5group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.692a8ef7-7d8b-48c8-bb4c-c308a6b1223f")
)
def Dementia_no_drug_no_metformin_earliest_with_diabetes_5and10_(Dementia_no_drug_no_metformin_earliest_with_diabetes_5group_bmi_joint_imputed, Dementia_no_drug_no_metformin_earliest_with_diabetes_10group_bmi_joint_imputed):
    df1 = Dementia_no_drug_no_metformin_earliest_with_diabetes_5group_bmi_joint_imputed
    df2 = Dementia_no_drug_no_metformin_earliest_with_diabetes_10group_bmi_joint_imputed
    df_union = df1.union(df2)

    return(df_union)
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.008f5866-ba48-4433-833b-8198924fdf54"),
    Dementia_no_drug_no_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.e8274a6a-e1e6-4caa-b937-466c303e6c82")
)
def Dementia_no_drug_no_metformin_earliest_with_diabetes_5and10_death_rate(Dementia_no_drug_no_metformin_earliest_with_diabetes_5and10_):
    df = Dementia_no_drug_no_metformin_earliest_with_diabetes_5and10_
    # Create columns for 30d, 60d, and anytime death
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

    # No time limit: any death after COVID diagnosis date
    df = df.withColumn(
        "death_anytime",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Summary by age
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    )

    # Overall summary
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e073d5c6-cd95-4cb2-98f3-eaee27c90d8d"),
    Dementia_no_drug_no_metformin_earliest_without_diabetes_10group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.fa6f653a-7d40-4458-816a-296fb8e3001c"),
    Dementia_no_drug_no_metformin_earliest_without_diabetes_5group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.371f6f07-e3e3-4229-abb9-e49344d761af")
)
def Dementia_no_drug_no_metformin_earliest_without_diabetes_5and10_(Dementia_no_drug_no_metformin_earliest_without_diabetes_5group_bmi_joint_imputed, Dementia_no_drug_no_metformin_earliest_without_diabetes_10group_bmi_joint_imputed):
    df1 = Dementia_no_drug_no_metformin_earliest_without_diabetes_5group_bmi_joint_imputed
    df2 = Dementia_no_drug_no_metformin_earliest_without_diabetes_10group_bmi_joint_imputed
    df_union = df1.union(df2)

    return(df_union)

@transform_pandas(
    Output(rid="ri.vector.main.execute.8368852b-034a-490f-be16-18672d7a557c"),
    Dementia_no_drug_no_metformin_earliest_without_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.e073d5c6-cd95-4cb2-98f3-eaee27c90d8d")
)
def Dementia_no_drug_no_metformin_earliest_without_diabetes_5and10_death_rate(Dementia_no_drug_no_metformin_earliest_without_diabetes_5and10_):
    df = Dementia_no_drug_no_metformin_earliest_without_diabetes_5and10_
    # Create columns for 30d, 60d, and anytime death
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

    # No time limit: any death after COVID diagnosis date
    df = df.withColumn(
        "death_anytime",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Summary by age
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    )

    # Overall summary
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3a95fc19-568d-4d36-942b-ab3c201ab14b"),
    Dementia_no_drug_with_metformin_earliest_with_diabetes_10group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.d3e9f1c3-4197-4491-a831-1213165767a7"),
    Dementia_no_drug_with_metformin_earliest_with_diabetes_5group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.4ac80d9d-15a1-4117-9f2b-d15aebad7b9b")
)
def Dementia_no_drug_with_metformin_earliest_with_diabetes_5and10_(Dementia_no_drug_with_metformin_earliest_with_diabetes_5group_bmi_joint_imputed, Dementia_no_drug_with_metformin_earliest_with_diabetes_10group_bmi_joint_imputed):
    df1 = Dementia_no_drug_with_metformin_earliest_with_diabetes_5group_bmi_joint_imputed
    df2 = Dementia_no_drug_with_metformin_earliest_with_diabetes_10group_bmi_joint_imputed

    df_union = df1.union(df2)

    return(df_union)

@transform_pandas(
    Output(rid="ri.vector.main.execute.5ba90355-8ca1-4835-a6ce-770529c08bd1"),
    Dementia_no_drug_with_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.3a95fc19-568d-4d36-942b-ab3c201ab14b")
)
def Dementia_no_drug_with_metformin_earliest_with_diabetes_5and10_death_rate(Dementia_no_drug_with_metformin_earliest_with_diabetes_5and10_):
    df = Dementia_no_drug_with_metformin_earliest_with_diabetes_5and10_
    # Create columns for 30d, 60d, and anytime death
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

    # No time limit: any death after COVID diagnosis date
    df = df.withColumn(
        "death_anytime",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Summary by age
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    )

    # Overall summary
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.efc89462-a6d7-46ab-bf18-0d6c435a73d5"),
    No_dementia_no_metformin_earliest_without_diabetes_10group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.051d10cf-070b-4d8f-bc2e-bd7b844b33ea"),
    No_dementia_no_metformin_earliest_without_diabetes_5group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.a73016e7-d8cc-4303-83af-a017578c4e73")
)
def No_dementia_no_metformin_earliest_no_diabetes_5and10_(No_dementia_no_metformin_earliest_without_diabetes_5group_bmi_joint_imputed, No_dementia_no_metformin_earliest_without_diabetes_10group_bmi_joint_imputed):
    df1 = No_dementia_no_metformin_earliest_without_diabetes_5group_bmi_joint_imputed
    df2 = No_dementia_no_metformin_earliest_without_diabetes_10group_bmi_joint_imputed

    df_union = df1.union(df2)

    return(df_union)

@transform_pandas(
    Output(rid="ri.vector.main.execute.116cb75a-b9de-4b0a-ab01-b6866ae8caf1"),
    No_dementia_no_metformin_earliest_no_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.efc89462-a6d7-46ab-bf18-0d6c435a73d5")
)
def No_dementia_no_metformin_earliest_no_diabetes_5and10_death_rate(No_dementia_no_metformin_earliest_no_diabetes_5and10_):
    df = No_dementia_no_metformin_earliest_no_diabetes_5and10_
    
    # Create columns for 30d, 60d, and anytime death
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

    # No time limit: any death after COVID diagnosis date
    df = df.withColumn(
        "death_anytime",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Summary by age
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    )

    # Overall summary
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c5486587-4e26-4fb8-ae82-bc6f70f36a02"),
    No_dementia_no_metformin_earliest_with_diabetes_10group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.df8050d3-9188-4b79-b829-bfe1e1d85f9b"),
    No_dementia_no_metformin_earliest_with_diabetes_5group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.378023de-a92f-47f7-b90b-dd28409f0d76")
)
def No_dementia_no_metformin_earliest_with_diabetes_5and10_(No_dementia_no_metformin_earliest_with_diabetes_5group_bmi_joint_imputed, No_dementia_no_metformin_earliest_with_diabetes_10group_bmi_joint_imputed):
    df1 = No_dementia_no_metformin_earliest_with_diabetes_5group_bmi_joint_imputed
    df2 = No_dementia_no_metformin_earliest_with_diabetes_10group_bmi_joint_imputed
    df_union = df1.union(df2)

    return(df_union)

@transform_pandas(
    Output(rid="ri.vector.main.execute.06eef0f4-35a0-4299-a07a-7dfe589d9056"),
    No_dementia_no_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.c5486587-4e26-4fb8-ae82-bc6f70f36a02")
)
def No_dementia_no_metformin_earliest_with_diabetes_5and10_death_rate(No_dementia_no_metformin_earliest_with_diabetes_5and10_):
    df = No_dementia_no_metformin_earliest_with_diabetes_5and10_
    # Create columns for 30d, 60d, and anytime death
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

    # No time limit: any death after COVID diagnosis date
    df = df.withColumn(
        "death_anytime",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Summary by age
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    )

    # Overall summary
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1715a2b4-f7d5-4be6-afd4-54618da2713a"),
    No_dementia_with_metformin_earliest_with_diabetes_10group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.a26553e3-2642-404e-9863-edb843de03fd"),
    No_dementia_with_metformin_earliest_with_diabetes_5group_bmi_joint_imputed=Input(rid="ri.foundry.main.dataset.c4e43a16-ee23-481e-bf4a-0e4f0294b52e")
)
def No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_(No_dementia_with_metformin_earliest_with_diabetes_5group_bmi_joint_imputed, No_dementia_with_metformin_earliest_with_diabetes_10group_bmi_joint_imputed):
    df1 = No_dementia_with_metformin_earliest_with_diabetes_5group_bmi_joint_imputed
    df2 = No_dementia_with_metformin_earliest_with_diabetes_10group_bmi_joint_imputed

    df_union = df1.union(df2)

    return(df_union)

@transform_pandas(
    Output(rid="ri.vector.main.execute.2bbb5d33-f2e3-4c4b-b5c9-42f521a07666"),
    No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_=Input(rid="ri.foundry.main.dataset.1715a2b4-f7d5-4be6-afd4-54618da2713a")
)
def No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_death_rate(No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_):
    df = No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_
    # Create columns for 30d, 60d, and anytime death
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

    # No time limit: any death after COVID diagnosis date
    df = df.withColumn(
        "death_anytime",
        F.when(
            (F.col("death_date").isNotNull()) &
            (F.datediff(F.col("death_date"), F.col("condition_start_date_COVID")) >= 0),
            1
        ).otherwise(0)
    )

    # Define age range groups
    df = df.withColumn(
        "age_range",
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-75")
         .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-85")
         .when(F.col("age_2020") >= 85, ">=85")
         .otherwise("Other")
    )

    # Summary by age
    summary_by_age = df.groupBy("age_range").agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    )

    # Overall summary
    overall_summary = df.agg(
        F.count("*").alias("total_count"),
        F.sum("death_within_30_days").alias("death_within_30_days_count"),
        F.sum("death_within_60_days").alias("death_within_60_days_count"),
        F.sum("death_anytime").alias("death_anytime_count")
    ).withColumn(
        "proportion_within_30_days", F.col("death_within_30_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_within_60_days", F.col("death_within_60_days_count") / F.col("total_count") * 100
    ).withColumn(
        "proportion_anytime", F.col("death_anytime_count") / F.col("total_count") * 100
    ).withColumn("age_range", F.lit("All Ages"))

    # Combine
    summary_df = summary_by_age.unionByName(overall_summary)

    return summary_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6f8ea4e3-7743-420c-82df-d342d4d1a717"),
    Dementia_drug_no_metformin_earliest_without_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.89004968-9005-4e75-a2ac-c76dd2c62273"),
    Dementia_drug_with_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.09c371c4-2237-4790-b5ae-77070fed8786")
)
def required_comp_1(Dementia_drug_no_metformin_earliest_without_diabetes_5and10_, Dementia_drug_with_metformin_earliest_with_diabetes_5and10_):
    df1=Dementia_drug_no_metformin_earliest_without_diabetes_5and10_
    df2=Dementia_drug_with_metformin_earliest_with_diabetes_5and10_

    # Round up age_2020 first
    df1_rounded = df1.withColumn("age_roundup", F.ceil(F.col("age_2020")))
    df2_rounded = df2.withColumn("age_roundup", F.ceil(F.col("age_2020")))

    # Count by rounded age and sex
    df1_result = (
        df1_rounded.groupBy("age_roundup", "sex_label")
                .count()
                .orderBy("age_roundup", "sex_label")
    )

    df2_result = (
        df2_rounded.groupBy("age_roundup", "sex_label")
                .count()
                .orderBy("age_roundup", "sex_label")
    )

    # Add dataset labels
    df1_labeled = df1_result.withColumn("dataset", F.lit("df1"))
    df2_labeled = df2_result.withColumn("dataset", F.lit("df2"))

    # Merge
    final_result = df1_labeled.unionByName(df2_labeled)

    return final_result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f59e6b65-bc2d-4db5-a133-cfdfff75e5f9"),
    No_dementia_no_metformin_earliest_no_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.efc89462-a6d7-46ab-bf18-0d6c435a73d5"),
    No_dementia_no_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.c5486587-4e26-4fb8-ae82-bc6f70f36a02"),
    No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_=Input(rid="ri.foundry.main.dataset.1715a2b4-f7d5-4be6-afd4-54618da2713a")
)
def required_comp_2(No_dementia_no_metformin_earliest_no_diabetes_5and10_, No_dementia_no_metformin_earliest_with_diabetes_5and10_, No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_):
    df1 = No_dementia_with_metformin_earliest_with_diabetes_10group_5and10_
    df2 = No_dementia_no_metformin_earliest_no_diabetes_5and10_
    df3 = No_dementia_no_metformin_earliest_with_diabetes_5and10_

    # Round up age_2020 first
    df1_rounded = df1.withColumn("age_roundup", F.ceil(F.col("age_2020")))
    df2_rounded = df2.withColumn("age_roundup", F.ceil(F.col("age_2020")))
    df3_rounded = df3.withColumn("age_roundup", F.ceil(F.col("age_2020")))

    # Count by rounded age and sex
    df1_result = (
        df1_rounded.groupBy("age_roundup", "gender_concept_name")
                .count()
                .orderBy("age_roundup", "gender_concept_name")
    )

    df2_result = (
        df2_rounded.groupBy("age_roundup", "gender_concept_name")
                .count()
                .orderBy("age_roundup", "gender_concept_name")
    )

    df3_result = (
        df3_rounded.groupBy("age_roundup", "gender_concept_name")
                .count()
                .orderBy("age_roundup", "gender_concept_name")
    )

    # Add dataset labels
    df1_labeled = df1_result.withColumn("dataset", F.lit("df1"))
    df2_labeled = df2_result.withColumn("dataset", F.lit("df2"))
    df3_labeled = df3_result.withColumn("dataset", F.lit("df3"))

    # Merge
    final_result = (
        df1_labeled
            .unionByName(df2_labeled)
            .unionByName(df3_labeled)
    )

    return final_result
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5390c54d-765d-49d8-adfc-2b2e44ba3457"),
    Dementia_drug_with_metformin_earliest_with_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.09c371c4-2237-4790-b5ae-77070fed8786"),
    No_dementia_no_metformin_earliest_no_diabetes_5and10_=Input(rid="ri.foundry.main.dataset.efc89462-a6d7-46ab-bf18-0d6c435a73d5")
)
def required_comp_3(Dementia_drug_with_metformin_earliest_with_diabetes_5and10_, No_dementia_no_metformin_earliest_no_diabetes_5and10_):
    df1=Dementia_drug_with_metformin_earliest_with_diabetes_5and10_
    df2=No_dementia_no_metformin_earliest_no_diabetes_5and10_

    # Round up age_2020 first
    df1_rounded = df1.withColumn("age_roundup", F.ceil(F.col("age_2020")))
    df2_rounded = df2.withColumn("age_roundup", F.ceil(F.col("age_2020")))

    # Rename df2's gender_concept_name → sex_label
    df2_renamed = df2_rounded.withColumnRenamed("gender_concept_name", "sex_label")

    # Count by rounded age and sex
    df1_result = (
        df1_rounded.groupBy("age_roundup", "sex_label")
                .count()
                .orderBy("age_roundup", "sex_label")
    )

    df2_result = (
        df2_renamed.groupBy("age_roundup", "sex_label")
                .count()
                .orderBy("age_roundup", "sex_label")
    )

    # Add dataset labels
    df1_labeled = df1_result.withColumn("dataset", F.lit("df1"))
    df2_labeled = df2_result.withColumn("dataset", F.lit("df2"))

    # Merge
    final_result = df1_labeled.unionByName(df2_labeled)

    return final_result


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

from pyspark.sql.functions import mean, stddev, count
from pyspark.sql.functions import col, row_number, broadcast
from pyspark.sql import DataFrame

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

from pyspark.ml.classification import LogisticRegression

from pyspark.ml.feature import OneHotEncoder, StringIndexer

import statsmodels.api as sm

from sklearn.neighbors import NearestNeighbors


from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import LogisticRegression

from pyspark.sql.functions import broadcast
from scipy.spatial import cKDTree

@transform_pandas(
    Output(rid="ri.vector.main.execute.f04290ba-6324-4e38-b359-ae1a1cbd8287"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.fded6d2f-8c49-467e-bee2-4eb93774d848")
)
def Dementia_drug_COVID_gap_day_summary_5mg(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later):
    df = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later
    # Calculate the gap in days between the dates
    df = df.withColumn("gap_dementia_covid", 
                    F.datediff(F.col("condition_start_date_COVID"), F.col("condition_start_date_Dementia")))
    df = df.withColumn("gap_drug_covid", 
                    F.datediff(F.col("condition_start_date_COVID"), F.col("drug_exposure_start_date")))

    # Show overall summary statistics for the gaps
    df.select("gap_dementia_covid", "gap_drug_covid").describe().show()

    # Create an age group column
    df = df.withColumn("age_group", 
        F.when((F.col("age_2020") >= 65) & (F.col("age_2020") < 75), "65-74")
        .when((F.col("age_2020") >= 75) & (F.col("age_2020") < 85), "75-84")
        .when(F.col("age_2020") >= 85, "85+")
    )

    # Group by age group and compute summary statistics
    summary_by_age = df.groupBy("age_group").agg(
        F.count("*").alias("count"),
        F.avg("gap_dementia_covid").alias("avg_gap_dementia_covid"),
        F.min("gap_dementia_covid").alias("min_gap_dementia_covid"),
        F.max("gap_dementia_covid").alias("max_gap_dementia_covid"),
        F.avg("gap_drug_covid").alias("avg_gap_drug_covid"),
        F.min("gap_drug_covid").alias("min_gap_drug_covid"),
        F.max("gap_drug_covid").alias("max_gap_drug_covid")
    )

    return(summary_by_age)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992"),
    Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.73ddaa9f-e8ad-4216-9366-ceba98b22a35"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.fded6d2f-8c49-467e-bee2-4eb93774d848"),
    Logic_Liaison_All_patients_summary_facts_table_lds=Input(rid="ri.foundry.main.dataset.80175e0f-69da-41e2-8065-2c9a7d3bc571")
)
def Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later, Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later, Logic_Liaison_All_patients_summary_facts_table_lds):
    df1 = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later
    df2 = Logic_Liaison_All_patients_summary_facts_table_lds
    df3 = Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later
    df1 = df1.withColumn("drug_dose", F.lit(5))
    df3 = df3.withColumn("drug_dose", F.lit(10))

    df_union = df1.union(df3)

    # Perform the join
    df_result = df_union.join(df2, on="person_id", how="left")

    # Create DIABETES_indicator based on the maximum value of the two columns
    df_result = df_result.withColumn(
        "DIABETES_indicator", 
        F.greatest(F.col("DIABETESCOMPLICATED_indicator"), F.col("DIABETESUNCOMPLICATED_indicator"))
    )

    # Drop the original two columns if they are no longer needed
    df_result = df_result.drop("DIABETESCOMPLICATED_indicator", "DIABETESUNCOMPLICATED_indicator")

    df_result = df_result.select("person_id", "condition_start_date_COVID", "death_date", "age_2020", "BMI_max_observed_or_calculated", "gender_concept_name", "race_concept_name", "DIABETES_indicator", "HYPERTENSION_indicator", "total_number_of_COVID_vaccine_doses", "PAXLOVID_indicator",'drug_dose')

    return(df_result)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4b69afb1-3419-4965-a6a6-b1771aa45731"),
    Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later=Input(rid="ri.foundry.main.dataset.fded6d2f-8c49-467e-bee2-4eb93774d848"),
    Logic_Liaison_All_patients_summary_facts_table_lds=Input(rid="ri.foundry.main.dataset.80175e0f-69da-41e2-8065-2c9a7d3bc571")
)
def Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_most_propensity_score(Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later, Logic_Liaison_All_patients_summary_facts_table_lds):
    df1 = Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_drug_later
    df2 = Logic_Liaison_All_patients_summary_facts_table_lds

    # Perform the join
    df_result = df1.join(df2, on="person_id", how="left")

    df_result = df_result.dropna(subset=["BMI_max_observed_or_calculated"])

    # Create DIABETES_indicator based on the maximum value of the two columns
    df_result = df_result.withColumn(
        "DIABETES_indicator", 
        F.greatest(F.col("DIABETESCOMPLICATED_indicator"), F.col("DIABETESUNCOMPLICATED_indicator"))
    )

    # Drop the original two columns if they are no longer needed
    df_result = df_result.drop("DIABETESCOMPLICATED_indicator", "DIABETESUNCOMPLICATED_indicator")

    # List of comorbidity indicator columns
    comorbidity_columns = [
        "CARDIOMYOPATHIES_indicator",
        "HEMIPLEGIAORPARAPLEGIA_indicator",
        "CEREBROVASCULARDISEASE_indicator",
        "HIVINFECTION_indicator",
        "CHRONICLUNGDISEASE_indicator",
        "HYPERTENSION_indicator",
        "CONGESTIVEHEARTFAILURE_indicator",
        "METASTATICSOLIDTUMORCANCERS_indicator",
        "CORONARYARTERYDISEASE_indicator",
        "MYOCARDIALINFARCTION_indicator",
        "DEMENTIA_indicator",
        "OTHERIMMUNOCOMPROMISED_indicator",
        "DEPRESSION_indicator",
        "PEPTICULCER_indicator",
        "DIABETES_indicator",
        "PERIPHERALVASCULARDISEASE_indicator",
        "DOWNSYNDROME_indicator",
        "PSYCHOSIS_indicator",
        "HEARTFAILURE_indicator",
        "PULMONARYEMBOLISM_indicator",
        "SICKLECELLDISEASE_indicator",
        "SUBSTANCEABUSE_indicator",
        "THALASSEMIA_indicator"
    ]

    # Sum all indicator columns to create comorbidity_score
    df_result = df_result.withColumn("comorbidity_score", sum(F.col(c) for c in comorbidity_columns))

    #df_result = df_result.withColumn(
    #"BMI_max_observed_or_calculated",
    #when(col("BMI_max_observed_or_calculated") > 60, 60).otherwise(col("BMI_max_observed_or_calculated"))
    #)
    # List of disease-related indicator columns
    disease_columns = [
        "KIDNEYDISEASE_indicator",
        "MALIGNANTCANCER_indicator",
        "TUBERCULOSIS_indicator",
        "MILDLIVERDISEASE_indicator",
        "MODERATESEVERELIVERDISEASE_indicator",
        "SUBSTANCEUSEDISORDER_indicator",
        "RHEUMATOLOGICDISEASE_indicator"
    ]

    # Sum all disease indicator columns to create disease_score
    df_result = df_result.withColumn("disease_score", sum(F.col(c) for c in disease_columns))

    return(df_result)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f035da4b-6813-440d-8feb-1a0cebb0dcae"),
    Union_encode_categorical_data_5mg10mg_BMI=Input(rid="ri.foundry.main.dataset.01c8a150-aee6-4316-ba41-fa8a5b424c07")
)
def Iterative_imputation_5mg10mg_BMI(Union_encode_categorical_data_5mg10mg_BMI):
    df = Union_encode_categorical_data_5mg10mg_BMI

    def Iterative_imp(df_pandas, M=4):
        df_original = df_pandas.copy()

        # Step 0: Preserve ID columns — don't use them in imputation
        id_cols = ["person_id"]  # adjust as needed
        id_data = df_original[id_cols]

        # Step 1: Identify OHE columns
        race_cols     = [col for col in df_original.columns if col.startswith("race_concept_name_ohe_")]
        sex_cols      = [col for col in df_original.columns if col.startswith("gender_concept_name_ohe_")]

        # Step 2: Specify variables to impute
        mi_cols = race_cols + sex_cols + ["BMI_max_observed_or_calculated"]

        # Auxiliary variables (already numeric)
        aux_cols = [
            "age_2020", "DIABETES_indicator", "HYPERTENSION_indicator",
            "total_number_of_COVID_vaccine_doses", "PAXLOVID_indicator",'drug_dose'
        ]

        all_mi_cols = mi_cols + aux_cols

        # Step 3: Setup Iterative Imputer
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            sample_posterior=True,
            imputation_order="ascending",
            initial_strategy="mean"
        )

        # Step 4: Perform M imputations
        imputed_sets = []
        for m in range(M):
            seed = 1000 + m
            np.random.seed(seed)
            imp.random_state = seed

            df_imp = df_original.copy()

            # Perform imputation only on selected columns
            imputed_values = imp.fit_transform(df_imp[all_mi_cols])
            df_imp_imputed = pd.DataFrame(imputed_values, columns=all_mi_cols)

            # Add back ID columns and imp_id
            df_imp_imputed[id_cols] = df_original[id_cols].values
            df_imp_imputed["imp_id"] = m + 1

            imputed_sets.append(df_imp_imputed)

        # Step 5: Combine into long format
        df_MI_long = pd.concat(imputed_sets, ignore_index=True)

        return df_MI_long

    df_imputed = Iterative_imp(df)

    return df_imputed

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c92d1156-e733-4826-b785-38fa513f9b81"),
    Logic_Liaison_All_patients_summary_facts_table_lds=Input(rid="ri.foundry.main.dataset.80175e0f-69da-41e2-8065-2c9a7d3bc571"),
    No_dementia_ge65_with_primary_covid19_clean_death=Input(rid="ri.foundry.main.dataset.ba335056-dc57-45b9-a2d5-09e6a750b568")
)
def No_dementia_ge65_with_primary_covid19_clean_death_more_info_checked(No_dementia_ge65_with_primary_covid19_clean_death, Logic_Liaison_All_patients_summary_facts_table_lds):
    df1 = No_dementia_ge65_with_primary_covid19_clean_death
    df2 = Logic_Liaison_All_patients_summary_facts_table_lds
        # Perform the join
    df_result = df1.join(df2, on="person_id", how="left")

    # Create DIABETES_indicator based on the maximum value of the two columns
    df_result = df_result.withColumn(
        "DIABETES_indicator", 
        F.greatest(F.col("DIABETESCOMPLICATED_indicator"), F.col("DIABETESUNCOMPLICATED_indicator"))
    )

    # Drop the original two columns if they are no longer needed
    df_result = df_result.drop("DIABETESCOMPLICATED_indicator", "DIABETESUNCOMPLICATED_indicator")

    # List of comorbidity indicator columns
    comorbidity_columns = [
        "CARDIOMYOPATHIES_indicator",
        "HEMIPLEGIAORPARAPLEGIA_indicator",
        "CEREBROVASCULARDISEASE_indicator",
        "HIVINFECTION_indicator",
        "CHRONICLUNGDISEASE_indicator",
        "HYPERTENSION_indicator",
        "CONGESTIVEHEARTFAILURE_indicator",
        "METASTATICSOLIDTUMORCANCERS_indicator",
        "CORONARYARTERYDISEASE_indicator",
        "MYOCARDIALINFARCTION_indicator",
        "DEMENTIA_indicator",
        "OTHERIMMUNOCOMPROMISED_indicator",
        "DEPRESSION_indicator",
        "PEPTICULCER_indicator",
        "DIABETES_indicator",
        "PERIPHERALVASCULARDISEASE_indicator",
        "DOWNSYNDROME_indicator",
        "PSYCHOSIS_indicator",
        "HEARTFAILURE_indicator",
        "PULMONARYEMBOLISM_indicator",
        "SICKLECELLDISEASE_indicator",
        "SUBSTANCEABUSE_indicator",
        "THALASSEMIA_indicator"
    ]

    # Sum all indicator columns to create comorbidity_score
    df_result = df_result.withColumn("comorbidity_score", sum(F.col(c) for c in comorbidity_columns))

    #df_result = df_result.withColumn(
    #"BMI_max_observed_or_calculated",
    #when(col("BMI_max_observed_or_calculated") > 60, 60).otherwise(col("BMI_max_observed_or_calculated"))
    #)
    # List of disease-related indicator columns
    disease_columns = [
        "KIDNEYDISEASE_indicator",
        "MALIGNANTCANCER_indicator",
        "TUBERCULOSIS_indicator",
        "MILDLIVERDISEASE_indicator",
        "MODERATESEVERELIVERDISEASE_indicator",
        "SUBSTANCEUSEDISORDER_indicator",
        "RHEUMATOLOGICDISEASE_indicator"
    ]

    # Sum all disease indicator columns to create disease_score
    df_result = df_result.withColumn("disease_score", sum(F.col(c) for c in disease_columns))

    return(df_result)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f8de7ced-5dab-4298-a213-7e47bad1ae70"),
    Logic_Liaison_All_patients_summary_facts_table_lds=Input(rid="ri.foundry.main.dataset.80175e0f-69da-41e2-8065-2c9a7d3bc571"),
    No_dementia_ge65_no_covid19_clean_death=Input(rid="ri.foundry.main.dataset.454c9170-7e8a-4d61-81b5-3352d8449301")
)
def No_dementia_no_drug_no_COVID_ge65_clean_death_more_info_checked(No_dementia_ge65_no_covid19_clean_death, Logic_Liaison_All_patients_summary_facts_table_lds):
    df1 = No_dementia_ge65_no_covid19_clean_death
    df2 = Logic_Liaison_All_patients_summary_facts_table_lds
    df_result = df1.join(df2, on="person_id", how="left")

    df_result = df_result.dropna(subset=["BMI_max_observed_or_calculated"])
    # Create DIABETES_indicator based on the maximum value of the two columns
    df_result = df_result.withColumn(
        "DIABETES_indicator", 
        F.greatest(F.col("DIABETESCOMPLICATED_indicator"), F.col("DIABETESUNCOMPLICATED_indicator"))
    )

    # Drop the original two columns if they are no longer needed
    df_result = df_result.drop("DIABETESCOMPLICATED_indicator", "DIABETESUNCOMPLICATED_indicator")

    # List of comorbidity indicator columns
    comorbidity_columns = [
        "CARDIOMYOPATHIES_indicator",
        "HEMIPLEGIAORPARAPLEGIA_indicator",
        "CEREBROVASCULARDISEASE_indicator",
        "HIVINFECTION_indicator",
        "CHRONICLUNGDISEASE_indicator",
        "HYPERTENSION_indicator",
        "CONGESTIVEHEARTFAILURE_indicator",
        "METASTATICSOLIDTUMORCANCERS_indicator",
        "CORONARYARTERYDISEASE_indicator",
        "MYOCARDIALINFARCTION_indicator",
        "DEMENTIA_indicator",
        "OTHERIMMUNOCOMPROMISED_indicator",
        "DEPRESSION_indicator",
        "PEPTICULCER_indicator",
        "DIABETES_indicator",
        "PERIPHERALVASCULARDISEASE_indicator",
        "DOWNSYNDROME_indicator",
        "PSYCHOSIS_indicator",
        "HEARTFAILURE_indicator",
        "PULMONARYEMBOLISM_indicator",
        "SICKLECELLDISEASE_indicator",
        "SUBSTANCEABUSE_indicator",
        "THALASSEMIA_indicator"
    ]

    # Sum all indicator columns to create comorbidity_score
    df_result = df_result.withColumn("comorbidity_score", sum(F.col(c) for c in comorbidity_columns))

    #df_result = df_result.withColumn(
    #"BMI_max_observed_or_calculated",
    #when(col("BMI_max_observed_or_calculated") > 60, 60).otherwise(col("BMI_max_observed_or_calculated"))
    #)
    # List of disease-related indicator columns
    disease_columns = [
        "KIDNEYDISEASE_indicator",
        "MALIGNANTCANCER_indicator",
        "TUBERCULOSIS_indicator",
        "MILDLIVERDISEASE_indicator",
        "MODERATESEVERELIVERDISEASE_indicator",
        "SUBSTANCEUSEDISORDER_indicator",
        "RHEUMATOLOGICDISEASE_indicator"
    ]

    # Sum all disease indicator columns to create disease_score
    df_result = df_result.withColumn("disease_score", sum(F.col(c) for c in disease_columns))

    return(df_result)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.01c8a150-aee6-4316-ba41-fa8a5b424c07"),
    mg5mg10_union_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.e8b1b227-4221-4552-bc95-6fe6dc99660e")
)
def Union_encode_categorical_data_5mg10mg_BMI(mg5mg10_union_for_BMI_imputation):
    df_original = mg5mg10_union_for_BMI_imputation
    
    def clean_missing(df):
        missing_pattern = r'(?i)^\s*(unknown|no matching concept|null|none|n/?a|not available|missing|undefined)\s*$'
        
        for col in df.columns:
            df = df.withColumn(
                col,
                F.when(
                    F.col(col).rlike(missing_pattern) | F.col(col).rlike(r'^\s*$'),
                    None
                ).otherwise(F.col(col))
            )
        return df
    df_cleaned = clean_missing(df_original)

    def one_hot_encode_pyspark(df, categorical_cols, keep_original=True, drop_last=True):
        """
        Perform one-hot encoding in PySpark for given categorical columns.
        
        Args:
            df: Spark DataFrame
            categorical_cols: List of column names to one-hot encode
            keep_original: Whether to retain original categorical columns
            drop_last: Drop the last category to avoid collinearity
        
        Returns:
            Spark DataFrame with one-hot encoded columns added
        """
        indexers = [
            StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep")
            for col in categorical_cols
        ]

        encoders = [
            OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_ohe", dropLast=drop_last)
            for col in categorical_cols
        ]

        pipeline = Pipeline(stages=indexers + encoders)
        model = pipeline.fit(df)
        df_transformed = model.transform(df)

        # Optionally drop original and/or index columns
        cols_to_drop = []
        if not keep_original:
            cols_to_drop += categorical_cols
        cols_to_drop += [col + "_idx" for col in categorical_cols]

        df_result = df_transformed.drop(*cols_to_drop)
        return df_result

    categorical_cols = ["gender_concept_name", "race_concept_name"]
    df_encoded = one_hot_encode_pyspark(df_cleaned, categorical_cols)

    for col in ["gender_concept_name_ohe", "race_concept_name_ohe"]:
        num_categories = df_encoded.select(col).head()[0].size
        df_encoded = df_encoded.withColumn(col + "_arr", vector_to_array(col))
        for i in range(num_categories):
            df_encoded = df_encoded.withColumn(f"{col}_{i}", F.col(f"{col}_arr")[i])
        df_encoded = df_encoded.drop(col, col + "_arr")

    return df_encoded

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.aabeef27-e155-45b0-b0a0-decf8daedc9b"),
    Logic_Liaison_All_patients_summary_facts_table_lds=Input(rid="ri.foundry.main.dataset.80175e0f-69da-41e2-8065-2c9a7d3bc571"),
    With_dementia_no_drug_ge65_before_primary_covid19_clean_death=Input(rid="ri.foundry.main.dataset.f8bd4a9f-b163-4003-bffb-5b7ba1e483fa")
)
def With_dementia_no_drug_ge65_before_primary_covid19_clean_death_more_info_checked(With_dementia_no_drug_ge65_before_primary_covid19_clean_death, Logic_Liaison_All_patients_summary_facts_table_lds):
    df1 = With_dementia_no_drug_ge65_before_primary_covid19_clean_death
    df2 = Logic_Liaison_All_patients_summary_facts_table_lds
    df_result = df1.join(df2, on="person_id", how="left")

    # Create DIABETES_indicator based on the maximum value of the two columns
    df_result = df_result.withColumn(
        "DIABETES_indicator", 
        F.greatest(F.col("DIABETESCOMPLICATED_indicator"), F.col("DIABETESUNCOMPLICATED_indicator"))
    )

    # Drop the original two columns if they are no longer needed
    df_result = df_result.drop("DIABETESCOMPLICATED_indicator", "DIABETESUNCOMPLICATED_indicator")

    # List of comorbidity indicator columns
    comorbidity_columns = [
        "CARDIOMYOPATHIES_indicator",
        "HEMIPLEGIAORPARAPLEGIA_indicator",
        "CEREBROVASCULARDISEASE_indicator",
        "HIVINFECTION_indicator",
        "CHRONICLUNGDISEASE_indicator",
        "HYPERTENSION_indicator",
        "CONGESTIVEHEARTFAILURE_indicator",
        "METASTATICSOLIDTUMORCANCERS_indicator",
        "CORONARYARTERYDISEASE_indicator",
        "MYOCARDIALINFARCTION_indicator",
        "DEMENTIA_indicator",
        "OTHERIMMUNOCOMPROMISED_indicator",
        "DEPRESSION_indicator",
        "PEPTICULCER_indicator",
        "DIABETES_indicator",
        "PERIPHERALVASCULARDISEASE_indicator",
        "DOWNSYNDROME_indicator",
        "PSYCHOSIS_indicator",
        "HEARTFAILURE_indicator",
        "PULMONARYEMBOLISM_indicator",
        "SICKLECELLDISEASE_indicator",
        "SUBSTANCEABUSE_indicator",
        "THALASSEMIA_indicator"
    ]

    # Sum all indicator columns to create comorbidity_score
    df_result = df_result.withColumn("comorbidity_score", sum(F.col(c) for c in comorbidity_columns))

    #df_result = df_result.withColumn(
    #"BMI_max_observed_or_calculated",
    #when(col("BMI_max_observed_or_calculated") > 60, 60).otherwise(col("BMI_max_observed_or_calculated"))
    #)
    # List of disease-related indicator columns
    disease_columns = [
        "KIDNEYDISEASE_indicator",
        "MALIGNANTCANCER_indicator",
        "TUBERCULOSIS_indicator",
        "MILDLIVERDISEASE_indicator",
        "MODERATESEVERELIVERDISEASE_indicator",
        "SUBSTANCEUSEDISORDER_indicator",
        "RHEUMATOLOGICDISEASE_indicator"
    ]

    # Sum all disease indicator columns to create disease_score
    df_result = df_result.withColumn("disease_score", sum(F.col(c) for c in disease_columns))

    return(df_result)

@transform_pandas(
    Output(rid="ri.vector.main.execute.011888dd-3203-4ca0-8fe4-e38a123cf891"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992")
)
def day30_death_Dementia_ge65_donepezil_10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_all(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation):
    df = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation

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
    Output(rid="ri.vector.main.execute.620ea412-9c9c-44e0-9f1a-d4b8da21b7ff"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation):
    df = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation
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
    Output(rid="ri.vector.main.execute.5745a27f-5bbd-4f7d-82c4-7fb497df2740"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_65_75(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation):
    df = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation

    df = df.filter((col("age_2020") >= 65) & (col("age_2020") < 75))

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
    Output(rid="ri.vector.main.execute.c8bf51b4-7822-450f-a0bc-b9104eefb320"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_75_85(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation):
    df = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation
    # Add a column to check if death_date is within 30 days after condition_start_date

    df = df.filter((col("age_2020") >= 75) & (col("age_2020") < 85))

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
    Output(rid="ri.vector.main.execute.8a8a9b83-613b-48ff-a19c-bd2d0905821a"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992")
)
def day30_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_85(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation):
    df = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation

    df = df.filter((col("age_2020") >= 85))

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
    Output(rid="ri.vector.main.execute.d85fbf59-842d-426c-ae01-96c5272bc3d8"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation):
    df = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation
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
    Output(rid="ri.vector.main.execute.d92065e3-072e-47a8-a442-03c0a9dd7c30"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_65_75(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation):
    df = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation

    df = df.filter((col("age_2020") >= 65) & (col("age_2020") < 75))

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
        F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.3515ec0e-d7d6-4fff-96cb-0045c1b3999e"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_75_85(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation):
    df = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation

    df = df.filter((col("age_2020") >= 75) & (col("age_2020") < 85))

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
        F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.vector.main.execute.56a8a2b2-d37c-4198-a660-d67e01d68680"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992")
)
def day60_death_Dementia_ge65_donepezil_5mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_85(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation):
    df = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation

    df = df.filter((col("age_2020") >= 85))

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
        F.col("death_within_30_days_count") / F.col("total_count")
    )
    return(summary_df)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.22c0dcd6-3479-418d-ac29-b3a0854df46c"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992")
)
def encode_categorical_data_5mg_BMI(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation):
    df_original = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation
    
    def clean_missing(df):
        missing_pattern = r'(?i)^\s*(unknown|no matching concept|null|none|n/?a|not available|missing|undefined)\s*$'
        
        for col in df.columns:
            df = df.withColumn(
                col,
                F.when(
                    F.col(col).rlike(missing_pattern) | F.col(col).rlike(r'^\s*$'),
                    None
                ).otherwise(F.col(col))
            )
        return df
    df_cleaned = clean_missing(df_original)

    def one_hot_encode_pyspark(df, categorical_cols, keep_original=True, drop_last=True):
        """
        Perform one-hot encoding in PySpark for given categorical columns.
        
        Args:
            df: Spark DataFrame
            categorical_cols: List of column names to one-hot encode
            keep_original: Whether to retain original categorical columns
            drop_last: Drop the last category to avoid collinearity
        
        Returns:
            Spark DataFrame with one-hot encoded columns added
        """
        indexers = [
            StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep")
            for col in categorical_cols
        ]

        encoders = [
            OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_ohe", dropLast=drop_last)
            for col in categorical_cols
        ]

        pipeline = Pipeline(stages=indexers + encoders)
        model = pipeline.fit(df)
        df_transformed = model.transform(df)

        # Optionally drop original and/or index columns
        cols_to_drop = []
        if not keep_original:
            cols_to_drop += categorical_cols
        cols_to_drop += [col + "_idx" for col in categorical_cols]

        df_result = df_transformed.drop(*cols_to_drop)
        return df_result

    categorical_cols = ["gender_concept_name", "race_concept_name"]
    df_encoded = one_hot_encode_pyspark(df_cleaned, categorical_cols)

    for col in ["gender_concept_name_ohe", "race_concept_name_ohe"]:
        num_categories = df_encoded.select(col).head()[0].size
        df_encoded = df_encoded.withColumn(col + "_arr", vector_to_array(col))
        for i in range(num_categories):
            df_encoded = df_encoded.withColumn(f"{col}_{i}", F.col(f"{col}_arr")[i])
        df_encoded = df_encoded.drop(col, col + "_arr")

    return df_encoded

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2eebd8ef-a1cd-4a06-9883-34d8273572cf"),
    imputed_mean_5mg10mg_BMI_wuth_drug=Input(rid="ri.foundry.main.dataset.e87d0951-70cb-4686-99cc-7dcb2197aba3")
)
def imputed_mean_10mg_BMI_joint_imputed(imputed_mean_5mg10mg_BMI_wuth_drug):
    df1 = imputed_mean_5mg10mg_BMI_wuth_drug
    df1 = (
        df1.filter(F.col("drug_dose_y") == 10)  # keep only rows where drug_dose_y is 5 or 10
        .drop("df_belong")                           # remove the column afterward
    )

    return df1

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.506b7d66-b8b5-45d6-abe6-586b94df5067"),
    Iterative_imputation_5mg10mg_BMI=Input(rid="ri.foundry.main.dataset.f035da4b-6813-440d-8feb-1a0cebb0dcae"),
    mg5mg10_union_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.e8b1b227-4221-4552-bc95-6fe6dc99660e")
)
def imputed_mean_5mg10mg_BMI(Iterative_imputation_5mg10mg_BMI, mg5mg10_union_for_BMI_imputation):
    df_imputed = Iterative_imputation_5mg10mg_BMI
    df_date = mg5mg10_union_for_BMI_imputation
    # Define ID and control columns
    id_cols = ["person_id"]
    control_cols = id_cols + ["imp_id"]

    # Dynamically detect imputed columns
    imputed_cols = [col for col in df_imputed.columns if col not in control_cols]

    # One-hot column groups
    race_ohe_cols = [col for col in imputed_cols if col.startswith("race_concept_name_ohe_")]
    sex_ohe_cols  = [col for col in imputed_cols if col.startswith("gender_concept_name_ohe_")]

    # Other numeric columns (excluding OHE)
    numeric_cols = [
        col for col in imputed_cols
        if col not in race_ohe_cols + sex_ohe_cols
    ]
        
    df_numeric_point = (
        df_imputed
        .groupby(id_cols)[numeric_cols]
        .mean()
        .reset_index()
    )

    def decode_ohe_row(row, ohe_cols, label_map):
        values = row[ohe_cols].values
        max_idx = np.argmax(values)
        return label_map[max_idx]

    # Define the first specific mappings
    race_label_map = {
        0: "White",
        1: "Black or African American",
        2: "Asian",
        3: "Other",
        4: "American Indian or Alaska Native",
        5: "Native Hawaiian or Other Pacific Islander",
        6: "Hispanic or Latino"
    }

    # Add keys 7 through 33, all mapping to "Other"
    race_label_map.update({i: "Other" for i in range(7, 34)})

    sex_label_map = {0: "female", 1: "male"} 

    # Apply row-wise decoding
    df_imputed["race_label"] = df_imputed[race_ohe_cols].apply(
        lambda row: decode_ohe_row(row, race_ohe_cols, race_label_map),
        axis=1
    )
    df_imputed["sex_label"] = df_imputed[sex_ohe_cols].apply(
        lambda row: decode_ohe_row(row, sex_ohe_cols, sex_label_map),
        axis=1
    )

    def majority_vote(df_long, id_cols, cat_col):
        vote_counts = (
            df_long
            .groupby(id_cols + [cat_col])
            .size()
            .reset_index(name="count")
        )
        vote_sorted = vote_counts.sort_values(
            by=["count", cat_col],
            ascending=[False, True]
        )
        majority_df = vote_sorted.drop_duplicates(subset=id_cols).drop(columns=["count"])
        return majority_df

    # Majority vote for each categorical label
    df_race_majority     = majority_vote(df_imputed, id_cols, "race_label")
    df_sex_majority      = majority_vote(df_imputed, id_cols, "sex_label")

    # Merge everything
    df_point_est_final = (
        df_numeric_point
        .merge(df_race_majority,     on="person_id", how="left")
        .merge(df_sex_majority,      on="person_id", how="left")
    )

    # Select only the needed columns from df2
    df_date_selected = df_date[["person_id", "condition_start_date_COVID", "death_date", "df_belong",'drug_dose']]

    # Merge on person_id
    df_merged = df_point_est_final.merge(df_date_selected, on="person_id", how="left")

    return df_merged

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e87d0951-70cb-4686-99cc-7dcb2197aba3"),
    imputed_mean_5mg10mg_BMI=Input(rid="ri.foundry.main.dataset.506b7d66-b8b5-45d6-abe6-586b94df5067")
)
def imputed_mean_5mg10mg_BMI_wuth_drug(imputed_mean_5mg10mg_BMI):
    df1 = imputed_mean_5mg10mg_BMI
    df1 = (
        df1.filter(F.col("drug_dose_y").isin([5, 10]))  # keep only rows where drug_dose_y is 5 or 10
        .drop("df_belong")                           # remove the column afterward
    )

    return (df1)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.68ccc96c-c8ea-480a-8cf8-4dbf136184af"),
    imputed_mean_5mg10mg_BMI_wuth_drug=Input(rid="ri.foundry.main.dataset.e87d0951-70cb-4686-99cc-7dcb2197aba3")
)
def imputed_mean_5mg_BMI_joint_imputed(imputed_mean_5mg10mg_BMI_wuth_drug):
    df1 = imputed_mean_5mg10mg_BMI_wuth_drug
    df1 = (
        df1.filter(F.col("drug_dose_y") == 5)  # keep only rows where drug_dose_y is 5 or 10
        .drop("df_belong")                           # remove the column afterward
    )

    return df1

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d615d522-fbae-4c1e-86db-953d13dd4900"),
    imputed_mean_5mg10mg_BMI=Input(rid="ri.foundry.main.dataset.506b7d66-b8b5-45d6-abe6-586b94df5067")
)
def imputed_mean_BMI_wuth_drug_Dementia_no_drug(imputed_mean_5mg10mg_BMI):
    df1 = imputed_mean_5mg10mg_BMI
    df1 = (
        df1.filter(
            (F.col("df_belong") == "C") & (F.col("drug_dose_y") == 0)
        )  # keep only rows where df_belong == "B" AND drug_dose_y == 0
        .drop("df_belong")  # remove the column afterward
    )

    return (df1)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.19ee68fe-5dfa-4689-bdf8-2c3fc146911e"),
    imputed_mean_5mg10mg_BMI=Input(rid="ri.foundry.main.dataset.506b7d66-b8b5-45d6-abe6-586b94df5067")
)
def imputed_mean_BMI_wuth_drug_no_dementia(imputed_mean_5mg10mg_BMI):
    df1 = imputed_mean_5mg10mg_BMI
    df1 = (
        df1.filter(
            (F.col("df_belong") == "B") & (F.col("drug_dose_y") == 0)
        )  # keep only rows where df_belong == "B" AND drug_dose_y == 0
        .drop("df_belong")  # remove the column afterward
    )

    return (df1)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.44147aa5-4f9b-456c-84bb-23829b1910cd"),
    imputed_mean_10mg_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.2eebd8ef-a1cd-4a06-9883-34d8273572cf"),
    imputed_mean_BMI_wuth_drug_no_dementia=Input(rid="ri.foundry.main.dataset.19ee68fe-5dfa-4689-bdf8-2c3fc146911e")
)
def join_10mg_no_dementia_drug_later_both_BMI_imputed_for_PSM_joint_imputed(imputed_mean_10mg_BMI_joint_imputed, imputed_mean_BMI_wuth_drug_no_dementia):
    df1 = imputed_mean_10mg_BMI_joint_imputed
    df2 = imputed_mean_BMI_wuth_drug_no_dementia

    df1 = df1.withColumnRenamed("race_label", "race_concept_name")
    df1 = df1.withColumnRenamed("sex_label", "gender_concept_name")
    df2 = df2.withColumnRenamed("race_label", "race_concept_name")
    df2 = df2.withColumnRenamed("sex_label", "gender_concept_name")

    # Add a new column "drug_status" to both dataframes
    df1 = df1.withColumn("drug_status", lit("1"))
    df2 = df2.withColumn("drug_status", lit("0"))

    # Get the columns of df2
    df1_cols = df1.columns

    # Select only those columns in df1 that are present in df2
    df2_trimmed = df2.select(df1_cols)

    # Perform the union operation
    df_union = df2_trimmed.union(df1)

    return(df_union)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8b635183-a966-4056-b51d-3629601c51c9"),
    imputed_mean_10mg_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.2eebd8ef-a1cd-4a06-9883-34d8273572cf"),
    imputed_mean_BMI_wuth_drug_Dementia_no_drug=Input(rid="ri.foundry.main.dataset.d615d522-fbae-4c1e-86db-953d13dd4900")
)
def join_10mg_with_dementia_no_drug_drug_later_BMI_both_imputed_for_PSM_matching_joint_imputed(imputed_mean_10mg_BMI_joint_imputed, imputed_mean_BMI_wuth_drug_Dementia_no_drug):
    df1 = imputed_mean_10mg_BMI_joint_imputed
    df2 = imputed_mean_BMI_wuth_drug_Dementia_no_drug
    df1 = df1.withColumnRenamed("race_label", "race_concept_name")
    df1 = df1.withColumnRenamed("sex_label", "gender_concept_name")

    # Add a new column "drug_status" to both dataframes
    df1 = df1.withColumnRenamed("race_label", "race_concept_name")
    df1 = df1.withColumnRenamed("sex_label", "gender_concept_name")
    df2 = df2.withColumnRenamed("race_label", "race_concept_name")
    df2 = df2.withColumnRenamed("sex_label", "gender_concept_name")

    # Add a new column "drug_status" to both dataframes
    df1 = df1.withColumn("drug_status", lit("1"))
    df2 = df2.withColumn("drug_status", lit("0"))

    # Get the columns of df2
    df1_cols = df1.columns

    # Select only those columns in df1 that are present in df2
    df2_trimmed = df2.select(df1_cols)

    # Perform the union operation
    df_union = df2_trimmed.union(df1)

    return(df_union)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.66dd4260-b7f8-4cff-be46-b0ca11a34ffa"),
    imputed_mean_5mg_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.68ccc96c-c8ea-480a-8cf8-4dbf136184af"),
    imputed_mean_BMI_wuth_drug_no_dementia=Input(rid="ri.foundry.main.dataset.19ee68fe-5dfa-4689-bdf8-2c3fc146911e")
)
def join_5mg_no_dementia_drug_later_both_BMI_imputed_for_PSM_joint_imputed(imputed_mean_5mg_BMI_joint_imputed, imputed_mean_BMI_wuth_drug_no_dementia):
    df1 = imputed_mean_5mg_BMI_joint_imputed
    df2 = imputed_mean_BMI_wuth_drug_no_dementia

    df1 = df1.withColumnRenamed("race_label", "race_concept_name")
    df1 = df1.withColumnRenamed("sex_label", "gender_concept_name")
    df2 = df2.withColumnRenamed("race_label", "race_concept_name")
    df2 = df2.withColumnRenamed("sex_label", "gender_concept_name")

    # Add a new column "drug_status" to both dataframes
    df1 = df1.withColumn("drug_status", lit("1"))
    df2 = df2.withColumn("drug_status", lit("0"))

    # Get the columns of df2
    df1_cols = df1.columns

    # Select only those columns in df1 that are present in df2
    df2_trimmed = df2.select(df1_cols)

    # Perform the union operation
    df_union = df2_trimmed.union(df1)

    return(df_union)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d9dbdf73-816d-49db-acd8-7e02cae26eee"),
    imputed_mean_5mg_BMI_joint_imputed=Input(rid="ri.foundry.main.dataset.68ccc96c-c8ea-480a-8cf8-4dbf136184af"),
    imputed_mean_BMI_wuth_drug_Dementia_no_drug=Input(rid="ri.foundry.main.dataset.d615d522-fbae-4c1e-86db-953d13dd4900")
)
def join_5mg_with_dementia_no_drug_drug_later_BMI_both_imputed_for_PSM_matching_joint_imputed(imputed_mean_5mg_BMI_joint_imputed, imputed_mean_BMI_wuth_drug_Dementia_no_drug):
    df1 = imputed_mean_5mg_BMI_joint_imputed
    df2 = imputed_mean_BMI_wuth_drug_Dementia_no_drug
    df1 = df1.withColumnRenamed("race_label", "race_concept_name")
    df1 = df1.withColumnRenamed("sex_label", "gender_concept_name")

    # Add a new column "drug_status" to both dataframes
    df1 = df1.withColumnRenamed("race_label", "race_concept_name")
    df1 = df1.withColumnRenamed("sex_label", "gender_concept_name")
    df2 = df2.withColumnRenamed("race_label", "race_concept_name")
    df2 = df2.withColumnRenamed("sex_label", "gender_concept_name")

    # Add a new column "drug_status" to both dataframes
    df1 = df1.withColumn("drug_status", lit("1"))
    df2 = df2.withColumn("drug_status", lit("0"))

    # Get the columns of df2
    df1_cols = df1.columns

    # Select only those columns in df1 that are present in df2
    df2_trimmed = df2.select(df1_cols)

    # Perform the union operation
    df_union = df2_trimmed.union(df1)

    return(df_union)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e8b1b227-4221-4552-bc95-6fe6dc99660e"),
    Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation=Input(rid="ri.foundry.main.dataset.8ca4f955-8c53-48ec-a076-68dc007ab992"),
    No_dementia_ge65_with_primary_covid19_clean_death_more_info_checked=Input(rid="ri.foundry.main.dataset.c92d1156-e733-4826-b785-38fa513f9b81"),
    With_dementia_no_drug_ge65_before_primary_covid19_clean_death_more_info_checked=Input(rid="ri.foundry.main.dataset.aabeef27-e155-45b0-b0a0-decf8daedc9b")
)
def mg5mg10_union_for_BMI_imputation(Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation, No_dementia_ge65_with_primary_covid19_clean_death_more_info_checked, With_dementia_no_drug_ge65_before_primary_covid19_clean_death_more_info_checked):
    df1 = Dementia_ge65_donepezil_5mg10mg_only_primary_covid_before_donepezil_donepezil_m1m_clean_death_for_step8_more_info_drug_later_for_diabetes_for_BMI_imputation
    df2 = No_dementia_ge65_with_primary_covid19_clean_death_more_info_checked
    df3 = With_dementia_no_drug_ge65_before_primary_covid19_clean_death_more_info_checked
    df2 = df2.withColumn("drug_dose", F.lit(0))
    df3 = df3.withColumn("drug_dose", F.lit(0))
    # Step 1. Get df1's columns
    cols = df1.columns

    # Step 2. Select only those columns from df2 and df3
    df2_sel = df2.select([c for c in cols if c in df2.columns])
    df3_sel = df3.select([c for c in cols if c in df3.columns])

    # Step 3. Add the source indicator column
    df1_tag = df1.withColumn("df_belong", F.lit("A"))
    df2_tag = df2_sel.withColumn("df_belong", F.lit("B"))
    df3_tag = df3_sel.withColumn("df_belong", F.lit("C"))

    # Step 4. Union them together
    df_all = df1_tag.unionByName(df2_tag).unionByName(df3_tag)

    return(df_all)


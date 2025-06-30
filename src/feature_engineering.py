"""
Feature engineering functions.
"""

import polars as pl
from config import TARGET, ID, NUM_COLS, CAT_COLS


def clean_target_variable(df, target_col=TARGET, method='log_transform'):
    """Remove rows with null values in target variable and apply transformations."""
    global TARGET
    
    initial_count = df.shape[0]
    df_cleaned = df.filter(pl.col(target_col).is_not_null())
    removed_count = initial_count - df_cleaned.shape[0]

    if removed_count > 0:
        print(f"Removed {removed_count} rows with null {target_col} values")

    # Remove outlier (salary of 350 for a junior analyst - assumed error)
    df_cleaned = df_cleaned.filter(pl.col(target_col) > 400)

    if method == 'log_transform':
        print(f"Applying log1p transformation to '{target_col}'")
        df_cleaned = df_cleaned.with_columns(
            (pl.col(target_col) + 1).log().alias(f'{TARGET}_log')
        )
        TARGET = f'{TARGET}_log'
        print(f"A new column '{target_col}_log' has been created. Use this for training.")
        print("Remember to inverse transform your predictions (exp(pred) - 1) later.")
        
    elif method == 'capping':
        lower_percentile = 0.01
        upper_percentile = 0.99
        lower_cap = df_cleaned.select(pl.col(target_col).quantile(lower_percentile)).item()
        upper_cap = df_cleaned.select(pl.col(target_col).quantile(upper_percentile)).item()

        print(f"Capping '{target_col}' values between {lower_cap:.2f} and {upper_cap:.2f}.")
        df_cleaned = df_cleaned.with_columns(
            pl.col(target_col).clip(lower_bound=lower_cap, upper_bound=upper_cap).alias(target_col)
        )
        print(f"Values outside this range have been capped.")
    else:
        print(f"Warning: Unknown outlier handling method '{method}'. No outlier treatment applied.")

    return df_cleaned


def engineer_age_features(df, pred=False):
    """Engineer age-related features."""
    df = df.with_columns(
        pl.when(pl.col('Age').is_not_null()).then(
            pl.when(pl.col('Age') < 30).then(pl.lit('young'))
            .when(pl.col('Age') < 45).then(pl.lit('mid'))
            .otherwise(pl.lit('old')))
            .otherwise(None).alias('Age_bin')
    )

    if not pred:
        CAT_COLS.append('Age_bin')

    return df


def engineer_gender_features(df, pred=False):
    """Engineer gender-related features."""
    # Placeholder for future gender feature engineering
    return df


def engineer_education_features(df, pred=False):
    """Engineer education-related features."""
    # Placeholder for future education feature engineering
    return df


def engineer_job_title_features(df, pred=False):
    """Engineer job title-related features."""
    # Create job title categories
    df = df.with_columns(
        pl.when(pl.col('Job Title').str.contains('Director')).then(pl.lit('Director'))
        .when(pl.col('Job Title').str.contains('Manager')).then(pl.lit('Manager'))
        .when(pl.col('Job Title').str.contains('Coordinator')).then(pl.lit('Coordinator'))
        .when(pl.col('Job Title').str.contains('Analyst')).then(pl.lit('Analyst'))
        .when(pl.col('Job Title').str.contains('CEO')).then(pl.lit('Executive'))
        .when(pl.col('Job Title').str.contains('Chief')).then(pl.lit('Executive'))
        .when(pl.col('Job Title').str.contains('VP')).then(pl.lit('Executive'))
        .otherwise(pl.lit('Other')).alias('Job_Title_Category')
    )

    # Create job level features
    df = df.with_columns(
        pl.when(pl.col('Job Title').str.contains('Senior')).then(pl.lit('Senior'))
        .when(pl.col('Job Title').str.contains('Junior')).then(pl.lit('Junior'))
        .otherwise(pl.lit('Other')).alias('Job_Level')
    )

    # Remove original Job Title and update column lists
    if not pred:
        df = df.drop('Job Title')
        CAT_COLS.remove('Job Title')
        CAT_COLS.extend(['Job_Title_Category', 'Job_Level'])
    
    return df


def engineer_experience_features(df, pred=False):
    """Engineer years of experience-related features."""
    df = df.with_columns(
        pl.when(pl.col('Years of Experience') < 1).then(pl.lit('junior'))
        .when(pl.col('Years of Experience') < 3).then(pl.lit('mid'))
        .otherwise(pl.lit('senior')).alias('Experience_bin')
    )
    if not pred:
        CAT_COLS.append('Experience_bin')
    return df


def create_interaction_features(df, pred=False):
    """Create interaction features."""
    # Education-Age aggregations
    ed_age = df.group_by('Education Level').agg(
        pl.col('Age').mean().alias('Education_Age_mean'),
        pl.col('Age').std().alias('Education_Age_std')
    )
    
    df = df.join(ed_age, on='Education Level', how='left')        
    if not pred:       
        NUM_COLS.extend(['Education_Age_mean', 'Education_Age_std'])

    # Experience-Age ratio
    df = df.with_columns(
        (pl.col('Years of Experience') / pl.col('Age')).alias('YoE_Age_ratio')
    )
    if not pred:
        NUM_COLS.append('YoE_Age_ratio')
    
    # Experience squared and interaction with age
    df = df.with_columns(
        (pl.col('Years of Experience') * pl.col('Years of Experience')).alias('YoE_squared'),
        (pl.col('Years of Experience') * pl.col('Age')).alias('YoE_Age_interaction')
    )
    if not pred:
        NUM_COLS.extend(['YoE_squared', 'YoE_Age_interaction'])

    return df


def perform_feature_engineering(df, target_method='log_transform'):
    """Perform all feature engineering steps."""
    # Clean target variable
    df = clean_target_variable(df, method=target_method)
    
    # Engineer features
    df = engineer_age_features(df)
    df = engineer_gender_features(df)
    df = engineer_education_features(df)
    df = engineer_job_title_features(df)
    df = engineer_experience_features(df)
    df = create_interaction_features(df)

    # Extract IDs and remove from dataframe
    ids = df[ID]
    df = df.drop(ID)
    
    return df, ids

def perform_feature_engineering_preds(df):
    """Perform all feature engineering steps."""

    # Engineer features
    df = engineer_age_features(df, True)
    df = engineer_gender_features(df, True)
    df = engineer_education_features(df, True)
    df = engineer_job_title_features(df, True)
    df = engineer_experience_features(df, True)
    df = create_interaction_features(df, True)

    
    return df

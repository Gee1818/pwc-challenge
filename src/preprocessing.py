"""
Data preprocessing pipeline functions.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from config import NUM_COLS, CAT_COLS


def create_preprocessing_pipeline():
    """Create sklearn preprocessing pipeline."""
    # Numerical pipeline
    num_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    num_pipeline = Pipeline([
        ('num_imputer', num_imputer),
        ('scaler', scaler)
    ])
    
    # Categorical pipeline
    cat_imputer = SimpleImputer(strategy='most_frequent')
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_pipeline = Pipeline([
        ('cat_imputer', cat_imputer),
        ('ohe_encoder', ohe_encoder)
    ])

    # Column transformer
    pipeline = ColumnTransformer([
        ('num', num_pipeline, NUM_COLS),
        ('cat_ohe', cat_pipeline, CAT_COLS),
    ])
    
    return pipeline
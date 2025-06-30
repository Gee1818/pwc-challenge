"""
LGBM regression model training script.
"""

import warnings
import numpy as np
import polars as pl
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor

# Import custom modules
from config import SEED, TARGET, N_FOLDS
from data_utils import load_data_environment, join_salary_people
from feature_engineering import perform_feature_engineering
from preprocessing import create_preprocessing_pipeline
from evaluation_utils import (
    cross_validation, 
    evaluate_model_seeds, 
    bootstrap_confidence_intervals,
    print_metrics_with_ci,
    plot_error_vs_predicted
)
from model_utils import save_model_and_pipeline, save_oof_predictions

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    """Main training pipeline for LGBM model."""
    print("Starting LGBM model training pipeline...")
    
    # Load data
    print("Loading data...")
    df_salary_raw, df_people_raw, df_descriptions_raw = load_data_environment()
    df_salary_people_raw = join_salary_people(df_salary_raw, df_people_raw)
    
    # Feature engineering 
    print("Performing feature engineering...")
    df_salary_people, ids = perform_feature_engineering(df_salary_people_raw, target_method='no_transform')
    
    # Prepare data
    X = df_salary_people.drop(TARGET)
    y = df_salary_people[TARGET]
    
    print(f"Data shape: {X.shape}")
    print(f"Target: {TARGET}")

    # Best params from optim notebook
    best_params = {'n_estimators': 314, 'max_depth': 5, 'learning_rate': 0.05996856373000153, 'feature_fraction': 0.6750935464649083, 'min_data_in_leaf': 30, 'max_bin': 654, 'num_leaves': 32}

    # Train LGBM model
    print(f"\nTraining LGBM model with params={best_params}...")
    model_lgbm = LGBMRegressor(**best_params, linear_tree=True, random_state=SEED, n_jobs=-1, verbose=-1)
    results_lgbm, importances_lgbm, oof_preds_lgbm = cross_validation(model_lgbm, X, y)

    # Evaluate model stability across seeds
    print("\nEvaluating model stability across different seeds...")
    lgbm_seed_results = evaluate_model_seeds(model_lgbm, X, y)
    print(lgbm_seed_results)
    
    # Bootstrap confidence intervals
    print("\nCalculating bootstrap confidence intervals...")
    bootstrap_ci = bootstrap_confidence_intervals(np.array(y), oof_preds_lgbm)
    print_metrics_with_ci(bootstrap_ci)
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    preprocessing_pipeline = create_preprocessing_pipeline()
    X_train = preprocessing_pipeline.fit_transform(X)
    
    final_lgbm_model = LGBMRegressor(**best_params, linear_tree=True, random_state=SEED, n_jobs=-1, verbose=-1)
    final_lgbm_model.fit(X_train, y)
    
    # Save model and predictions
    save_model_and_pipeline(final_lgbm_model, preprocessing_pipeline, 'lgbm')
    save_oof_predictions(ids, oof_preds_lgbm, 'lgbm')
    
    print("\nLGBM model training completed successfully!")
    
    return

if __name__ == "__main__":
    main()

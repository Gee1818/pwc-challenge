"""
Random Forest log regression model training script.
"""

import warnings
import numpy as np
import polars as pl
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

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
    """Main training pipeline for Random Forest model with log transformed target."""
    print("Starting rf_log model training pipeline...")
    
    # Load data
    print("Loading data...")
    df_salary_raw, df_people_raw, df_descriptions_raw = load_data_environment()
    df_salary_people_raw = join_salary_people(df_salary_raw, df_people_raw)
    
    # Feature engineering 
    print("Performing feature engineering...")
    df_salary_people, ids = perform_feature_engineering(df_salary_people_raw, target_method='log_transform')
    
    # Prepare data
    X = df_salary_people.drop(TARGET)
    y = df_salary_people[f'{TARGET}_log']
    y_base = df_salary_people[TARGET]

    
    print(f"Data shape: {X.shape}")
    print(f"Target: {TARGET}")

    # Best params from optim notebook
    best_params = {'n_estimators': 310, 'max_depth':12, 'min_samples_split': 3, 'min_samples_leaf': 1}
            
    # Train rf model
    print(f"\nTraining rf_log model with params={best_params}...")
    model_rf = RandomForestRegressor(**best_params, random_state=SEED)
    results_rf, importances_rf, oof_preds_rf_log = cross_validation(model_rf, X, y)
    oof_preds_rf_log = np.expm1(oof_preds_rf_log)

    # Evaluate model stability across seeds
    print("\nEvaluating model stability across different seeds...")
    rf_seed_results = evaluate_model_seeds(model_rf, X, y)
    print(rf_seed_results)
    
    # Bootstrap confidence intervals
    print("\nCalculating bootstrap confidence intervals...")
    bootstrap_ci = bootstrap_confidence_intervals(np.array(y_base), oof_preds_rf_log)
    print_metrics_with_ci(bootstrap_ci)
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    preprocessing_pipeline = create_preprocessing_pipeline()
    X_train = preprocessing_pipeline.fit_transform(X)
    
    final_rf_log_model = RandomForestRegressor(**best_params, random_state=SEED)
    final_rf_log_model.fit(X_train, y)
    
    # Save model and predictions
    save_model_and_pipeline(final_rf_log_model, preprocessing_pipeline, 'rf_log')
    save_oof_predictions(ids, oof_preds_rf_log, 'rf_log')
    
    print("\nRandom Forest with log target model training completed successfully!")
    
    return

if __name__ == "__main__":
    main()

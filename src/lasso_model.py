"""
Lasso regression model training script.
"""

import warnings
import numpy as np
import polars as pl
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LassoCV

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
    """Main training pipeline for Lasso model."""
    print("Starting Lasso model training pipeline...")
    
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

    # Optimal alpha from optim notebook
    optimal_alpha = 136
        
    # Train Lasso model
    print(f"\nTraining Lasso model with alpha={optimal_alpha:.4f}...")
    model_lasso = Lasso(alpha=optimal_alpha, random_state=SEED)
    results_lasso, importances_lasso, oof_preds_lasso = cross_validation(model_lasso, X, y)

    # Evaluate model stability across seeds
    print("\nEvaluating model stability across different seeds...")
    lasso_seed_results = evaluate_model_seeds(model_lasso, X, y)
    print(lasso_seed_results)
    
    # Bootstrap confidence intervals
    print("\nCalculating bootstrap confidence intervals...")
    bootstrap_ci = bootstrap_confidence_intervals(np.array(y), oof_preds_lasso)
    print_metrics_with_ci(bootstrap_ci)
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    preprocessing_pipeline = create_preprocessing_pipeline()
    X_train = preprocessing_pipeline.fit_transform(X)
    
    final_lasso_model = Lasso(alpha=optimal_alpha, random_state=SEED)
    final_lasso_model.fit(X_train, y)
    
    # Save model and predictions
    save_model_and_pipeline(final_lasso_model, preprocessing_pipeline, 'lasso')
    save_oof_predictions(ids, oof_preds_lasso, 'lasso')
    
    print("\nLasso model training completed successfully!")
    
    return

if __name__ == "__main__":
    main()

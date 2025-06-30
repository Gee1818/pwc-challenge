"""
Model-specific utilities and export functions.
"""

import joblib
import polars as pl
from config import MODELS_PATH, OOF_PATH


def save_model_and_pipeline(model, pipeline, model_name):
    """Save trained model and preprocessing pipeline."""
    model_path = f'{MODELS_PATH}final_{model_name}_model.pkl'
    pipeline_path = f'{MODELS_PATH}{model_name}_pipeline.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipeline_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Pipeline saved to: {pipeline_path}")


def save_oof_predictions(ids, predictions, model_name):
    """Save out-of-fold predictions to CSV."""
    oof_path = f'{OOF_PATH}oof_preds_{model_name}.csv'
    
    pl.DataFrame({
        'id': ids,
        'predicted': predictions
    }).write_csv(oof_path)
    
    print(f"OOF predictions saved to: {oof_path}")


def load_model_and_pipeline(model_name):
    """Load trained model and preprocessing pipeline."""
    model_path = f'{MODELS_PATH}final_{model_name}_model.pkl'
    pipeline_path = f'{MODELS_PATH}{model_name}_pipeline.pkl'
    
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    
    return model, pipeline

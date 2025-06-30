"""
Model evaluation and cross-validation utilities.
"""

import numpy as np
import polars as pl
import plotly.express as px
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, r2_score
from config import SEED, N_FOLDS, TARGET
from preprocessing import create_preprocessing_pipeline


def metrics(y_true, y_pred):
    """Calculate RMSE and R2 metrics."""
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'r2': r2}


def bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for RMSE and R2."""
    n_samples = len(y_true)
    bootstrap_rmse = []
    bootstrap_r2 = []

    np.random.seed(SEED)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        rmse_boot = root_mean_squared_error(y_true_boot, y_pred_boot)
        r2_boot = r2_score(y_true_boot, y_pred_boot)

        bootstrap_rmse.append(rmse_boot)
        bootstrap_r2.append(r2_boot)

    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100

    rmse_ci = np.percentile(bootstrap_rmse, [lower_percentile, upper_percentile])
    r2_ci = np.percentile(bootstrap_r2, [lower_percentile, upper_percentile])

    return {
        'rmse': {
            'mean': np.mean(bootstrap_rmse),
            'std': np.std(bootstrap_rmse),
            'lower_ci': rmse_ci[0],
            'upper_ci': rmse_ci[1]
        },
        'r2': {
            'mean': np.mean(bootstrap_r2),
            'std': np.std(bootstrap_r2),
            'lower_ci': r2_ci[0],
            'upper_ci': r2_ci[1]
        }
    }


def print_metrics_with_ci(results):
    """Print metrics with confidence intervals."""
    print("\nBootstrap Confidence Intervals for Out-of-Fold Predictions:")
    print("=" * 60)
    print(f"RMSE: {results['rmse']['mean']:.2f}")
    print(f"      95% CI: [{results['rmse']['lower_ci']:.2f}, {results['rmse']['upper_ci']:.2f}]")
    print(f"R²:   {results['r2']['mean']:.3f}")
    print(f"      95% CI: [{results['r2']['lower_ci']:.3f}, {results['r2']['upper_ci']:.3f}]")


def feature_importance(pipeline, model):
    """Get feature importance from the model."""
    importances = []
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        importances = np.zeros(len(pipeline.get_feature_names_out()))
    
    fe_impo = pl.DataFrame({
        'feature': pipeline.get_feature_names_out(),
        'importance': importances
    }).sort('importance', descending=True)
    
    return fe_impo


def cross_validation(model, X, y, seed=SEED, verbose=True, preprocessing_func=None):
    """Perform cross-validation with a given model."""
    if preprocessing_func is None:
        preprocessing_func = create_preprocessing_pipeline
        
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    results = []
    importances_table = None
    oof_preds = np.zeros(len(y))

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index].clone(), X[val_index].clone()
        y_train, y_val = y[train_index].clone(), y[val_index].clone()

        # Create preprocessing pipeline
        preprocessing_pipeline = preprocessing_func()

        X_train = preprocessing_pipeline.fit_transform(X_train)
        X_val = preprocessing_pipeline.transform(X_val)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        oof_preds[val_index] = y_pred

        result = metrics(y_val, y_pred)
        results.append(result)

        # Get feature importance for this fold
        fe_impo = feature_importance(preprocessing_pipeline, model)

        # Rename the importance column to include fold number
        fe_impo = fe_impo.with_columns(
            pl.col('importance').alias(f'fold_{fold + 1}')
        ).drop('importance')

        # Initialize or join the importances table
        if importances_table is None:
            importances_table = fe_impo
        else:
            importances_table = importances_table.join(
                fe_impo, on='feature', how='inner'
            )

        if verbose:
            print(f"Fold {fold + 1} - RMSE: {result['rmse']:.2f}, R2: {result['r2']:.2f}")

    # Add summary statistics across folds
    fold_columns = [f'fold_{i+1}' for i in range(N_FOLDS)]

    importances_table = importances_table.with_columns([
        pl.mean_horizontal([pl.col(col).round(2) for col in fold_columns]).alias('mean_importance'),
        pl.concat_list([pl.col(col).round(2) for col in fold_columns]).list.std().alias('std_importance')
    ]).drop(fold_columns)

    importances_table = importances_table.with_columns(
        (pl.col("mean_importance") / pl.col("mean_importance").sum() * 100).alias("mean_importance_pct").round(1),
    )

    # Round numeric columns to 2 decimals
    numeric_cols = ['mean_importance', 'std_importance']
    importances_table = importances_table.with_columns([
        pl.col(col).round(2) for col in numeric_cols
    ])

    # Sort by mean importance
    importances_table = importances_table.sort('mean_importance', descending=True)

    if verbose:
        avg_rmse = np.mean([result['rmse'] for result in results])
        avg_r2 = np.mean([result['r2'] for result in results])
        print(f"\nAverage RMSE across folds: {avg_rmse:.2f}")
        print(f"Average R2 across folds: {avg_r2:.2f}")

    return results, importances_table, oof_preds


def evaluate_model_seeds(model, X, y, seeds=None, preprocessing_func=None):
    """Evaluate model with multiple seeds to assess stability."""
    if seeds is None:
        seeds = np.random.RandomState(SEED).choice(range(10000), size=5, replace=False).tolist()
    
    results = []
    for s in seeds:
        seed_results, _, _ = cross_validation(model, X, y, s, verbose=False, preprocessing_func=preprocessing_func)

        avg_rmse = np.mean([result['rmse'] for result in seed_results])
        avg_r2 = np.mean([result['r2'] for result in seed_results])

        results.append({
            'seed': s,
            'rmse': avg_rmse,
            'r2': avg_r2
        })

    eval_results = pl.DataFrame(results)
    eval_results = eval_results.with_columns(
        pl.col('seed').cast(pl.String).alias('seed'))

    # Calculate means and add summary row
    mean_rmse = eval_results['rmse'].mean()
    mean_r2 = eval_results['r2'].mean()

    summary_row = pl.DataFrame({
        'seed': 'Mean',
        'rmse': mean_rmse,
        'r2': mean_r2
    })

    eval_results = eval_results.vstack(summary_row)
    return eval_results


def plot_error_vs_predicted(oof_preds, y, ids, title_suffix=""):
    """Plot predicted vs actual values."""
    errors = pl.DataFrame({
        'id': ids,
        'actual': y,
        'predicted': oof_preds
    })
    
    errors = errors.with_columns(
        ((pl.col('predicted') - pl.col('actual')) ** 2).alias('squared_error')
    )

    rmse_value = np.sqrt(errors['squared_error'].mean())

    fig = px.scatter(
        errors.to_pandas(),
        x='predicted',
        y='actual',
        hover_data=['id'],
        labels={'predicted': 'Predicted Values', 'actual': 'Real Values'},
        title=f'Real vs Predicted Values{title_suffix}\nRMSE: {rmse_value:.2f}'
    )
    
    # Add 45-degree line
    max_val = max(errors['predicted'].max(), errors['actual'].max())
    min_val = min(errors['predicted'].min(), errors['actual'].min())

    fig.add_shape(
        type='line',
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color='Red', width=2, dash='dash')
    )
    
    fig.update_layout(
        width=800,
        height=800,
        legend_title_text='',
        showlegend=False
    )
    
    fig.show()

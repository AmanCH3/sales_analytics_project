# src/ensemble_model.py
"""
Ensemble model combining Prophet, XGBoost, and LightGBM predictions
Uses weighted average based on individual model performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------
# Paths
# ---------------------------------------------
PROPHET_FORECAST = Path("data/processed/prophet_forecast.csv")
XGBOOST_FORECAST = Path("data/processed/xgboost_forecast.csv")
LIGHTGBM_FORECAST = Path("data/processed/lightgbm_forecast.csv")
ENSEMBLE_FORECAST = Path("data/processed/ensemble_forecast.csv")

PROPHET_METADATA = Path("models/prophet_metadata.json")
XGBOOST_METADATA = Path("models/xgboost_metadata.json")
LIGHTGBM_METADATA = Path("models/lightgbm_metadata.json")
ENSEMBLE_METADATA = Path("models/ensemble_metadata.json")

# ---------------------------------------------
# Load Model Metrics
# ---------------------------------------------
def load_model_metrics():
    """Load test metrics from all models"""
    metrics = {}
    
    for name, path in [
        ("Prophet", PROPHET_METADATA),
        ("XGBoost", XGBOOST_METADATA),
        ("LightGBM", LIGHTGBM_METADATA)
    ]:
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                # Get test metrics
                test_metrics = data.get('test_metrics', {})
                metrics[name] = {
                    'MAE': test_metrics.get('MAE', float('inf')),
                    'RMSE': test_metrics.get('RMSE', float('inf')),
                    'MAPE': test_metrics.get('MAPE', float('inf'))
                }
        else:
            print(f"⚠️  Warning: {name} metadata not found at {path}")
            metrics[name] = {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf')}
    
    return metrics

# ---------------------------------------------
# Calculate Optimal Weights
# ---------------------------------------------
def calculate_weights(metrics, method='inverse_mae'):
    """
    Calculate ensemble weights based on model performance
    
    Methods:
    - 'inverse_mae': Weights inversely proportional to MAE (lower MAE = higher weight)
    - 'inverse_rmse': Weights inversely proportional to RMSE
    - 'equal': Equal weights for all models
    """
    
    if method == 'equal':
        weights = {'Prophet': 1/3, 'XGBoost': 1/3, 'LightGBM': 1/3}
        return weights
    
    # Use inverse of error metric
    metric_key = 'MAE' if method == 'inverse_mae' else 'RMSE'
    
    # Calculate inverse weights
    inverse_weights = {}
    for model, model_metrics in metrics.items():
        error = model_metrics[metric_key]
        if error == float('inf') or error == 0:
            inverse_weights[model] = 0
        else:
            inverse_weights[model] = 1 / error
    
    # Normalize to sum to 1
    total = sum(inverse_weights.values())
    if total == 0:
        # Fallback to equal weights
        weights = {'Prophet': 1/3, 'XGBoost': 1/3, 'LightGBM': 1/3}
    else:
        weights = {model: w / total for model, w in inverse_weights.items()}
    
    return weights

# ---------------------------------------------
# Create Ensemble Forecast
# ---------------------------------------------
def create_ensemble(weights=None, method='inverse_mae'):
    """
    Create ensemble forecast from individual model predictions
    """
    print("="*60)
    print("ENSEMBLE FORECASTING MODEL")
    print("="*60)
    
    # Load forecasts
    print("\n[1/4] Loading individual model forecasts...")
    forecasts = {}
    
    for name, path in [
        ("Prophet", PROPHET_FORECAST),
        ("XGBoost", XGBOOST_FORECAST),
        ("LightGBM", LIGHTGBM_FORECAST)
    ]:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["ds"])
            forecasts[name] = df
            print(f"  ✓ Loaded {name}: {len(df)} predictions")
        else:
            print(f"  ✗ {name} forecast not found at {path}")
            return None
    
    # Verify all forecasts have same dates
    dates_prophet = set(forecasts["Prophet"]["ds"])
    dates_xgboost = set(forecasts["XGBoost"]["ds"])
    dates_lightgbm = set(forecasts["LightGBM"]["ds"])
    
    if not (dates_prophet == dates_xgboost == dates_lightgbm):
        print("⚠️  Warning: Forecasts have different date ranges!")
    
    # Load metrics and calculate weights
    print("\n[2/4] Calculating ensemble weights...")
    metrics = load_model_metrics()
    
    if weights is None:
        weights = calculate_weights(metrics, method=method)
    
    print(f"\nEnsemble weights ({method}):")
    for model, weight in weights.items():
        print(f"  {model:10s}: {weight:.4f} ({weight*100:.1f}%)")
    
    # Create ensemble predictions
    print("\n[3/4] Combining predictions...")
    
    # Merge all forecasts
    df_ensemble = forecasts["Prophet"][["ds"]].copy()
    
    df_ensemble["prophet_pred"] = forecasts["Prophet"]["yhat"].values
    df_ensemble["xgboost_pred"] = forecasts["XGBoost"]["yhat"].values
    df_ensemble["lightgbm_pred"] = forecasts["LightGBM"]["yhat"].values
    
    # Calculate weighted average
    df_ensemble["yhat"] = (
        weights["Prophet"] * df_ensemble["prophet_pred"] +
        weights["XGBoost"] * df_ensemble["xgboost_pred"] +
        weights["LightGBM"] * df_ensemble["lightgbm_pred"]
    )
    
    # Calculate prediction intervals (average of individual intervals)
    df_ensemble["yhat_lower"] = (
        weights["Prophet"] * forecasts["Prophet"]["yhat_lower"].values +
        weights["XGBoost"] * forecasts["XGBoost"]["yhat_lower"].values +
        weights["LightGBM"] * forecasts["LightGBM"]["yhat_lower"].values
    )
    
    df_ensemble["yhat_upper"] = (
        weights["Prophet"] * forecasts["Prophet"]["yhat_upper"].values +
        weights["XGBoost"] * forecasts["XGBoost"]["yhat_upper"].values +
        weights["LightGBM"] * forecasts["LightGBM"]["yhat_upper"].values
    )
    
    # Save ensemble forecast
    print("\n[4/4] Saving ensemble forecast...")
    output_df = df_ensemble[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    output_df.to_csv(ENSEMBLE_FORECAST, index=False)
    print(f"✓ Saved ensemble forecast to {ENSEMBLE_FORECAST}")
    
    # Also save detailed version with individual predictions
    detailed_path = Path("data/processed/ensemble_detailed.csv")
    df_ensemble.to_csv(detailed_path, index=False)
    print(f"✓ Saved detailed forecast to {detailed_path}")
    
    # Save metadata
    metadata = {
        "model": "Ensemble",
        "timestamp": pd.Timestamp.now().isoformat(),
        "method": method,
        "weights": weights,
        "individual_metrics": metrics,
        "forecast_days": len(output_df)
    }
    
    with open(ENSEMBLE_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {ENSEMBLE_METADATA}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("ENSEMBLE FORECAST SUMMARY")
    print(f"{'='*60}")
    print(f"Number of predictions: {len(output_df)}")
    print(f"Average daily forecast: {output_df['yhat'].mean():,.2f} NPR")
    print(f"Total forecast: {output_df['yhat'].sum():,.2f} NPR")
    print(f"Min daily forecast: {output_df['yhat'].min():,.2f} NPR")
    print(f"Max daily forecast: {output_df['yhat'].max():,.2f} NPR")
    
    # Prediction variance
    print(f"\nPrediction Statistics:")
    print(f"  Prophet avg:  {df_ensemble['prophet_pred'].mean():,.2f} NPR")
    print(f"  XGBoost avg:  {df_ensemble['xgboost_pred'].mean():,.2f} NPR")
    print(f"  LightGBM avg: {df_ensemble['lightgbm_pred'].mean():,.2f} NPR")
    print(f"  Ensemble avg: {df_ensemble['yhat'].mean():,.2f} NPR")
    
    # Model agreement (standard deviation across models)
    df_ensemble['std_across_models'] = df_ensemble[['prophet_pred', 'xgboost_pred', 'lightgbm_pred']].std(axis=1)
    avg_disagreement = df_ensemble['std_across_models'].mean()
    print(f"\nAverage model disagreement: {avg_disagreement:,.2f} NPR")
    print(f"(Lower is better - indicates models agree more)")
    
    print(f"{'='*60}\n")
    
    return output_df, weights, metrics

# ---------------------------------------------
# Compare All Models
# ---------------------------------------------
def compare_models():
    """Generate comparison report of all models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON REPORT")
    print("="*60)
    
    metrics = load_model_metrics()
    
    # Create comparison table
    comparison_data = []
    for model, model_metrics in metrics.items():
        comparison_data.append({
            'Model': model,
            'MAE': model_metrics['MAE'],
            'RMSE': model_metrics['RMSE'],
            'MAPE': model_metrics['MAPE']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('MAE')
    
    print("\nTest Set Performance:")
    print(df_comparison.to_string(index=False))
    
    # Best model
    best_model = df_comparison.iloc[0]['Model']
    print(f"\n🏆 Best performing model (by MAE): {best_model}")
    
    # Save comparison
    comparison_path = Path("data/processed/model_comparison.csv")
    df_comparison.to_csv(comparison_path, index=False)
    print(f"\n✓ Saved comparison to {comparison_path}")
    
    return df_comparison

# ---------------------------------------------
# Main Execution
# ---------------------------------------------
def run(method='inverse_mae', custom_weights=None):
    """
    Create ensemble forecast
    
    Parameters:
    - method: 'inverse_mae', 'inverse_rmse', or 'equal'
    - custom_weights: dict with keys 'Prophet', 'XGBoost', 'LightGBM'
    """
    
    # Create comparison report
    comparison = compare_models()
    
    # Create ensemble
    ensemble_df, weights, metrics = create_ensemble(
        weights=custom_weights,
        method=method
    )
    
    return ensemble_df, weights, metrics, comparison

if __name__ == "__main__":
    # Run with automatic weight calculation
    ensemble_df, weights, metrics, comparison = run(method='inverse_mae')
    
    # Or use custom weights:
    # custom_weights = {'Prophet': 0.3, 'XGBoost': 0.4, 'LightGBM': 0.3}
    # ensemble_df, weights, metrics, comparison = run(custom_weights=custom_weights)
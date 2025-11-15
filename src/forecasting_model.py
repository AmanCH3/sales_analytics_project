# src/forecasting_model_improved.py
"""
Improved Prophet model with:
- Train/test validation
- Cross-validation
- Better seasonality handling
- Model evaluation metrics
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from pathlib import Path
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------
# Paths
# ---------------------------------------------
FEATURE_PATH = Path("data/processed/featurized_sales_data.csv")
MODEL_PATH = Path("models/prophet_model.pkl")
FORECAST_CSV = Path("data/processed/prophet_forecast.csv")
METADATA_PATH = Path("models/prophet_metadata.json")
TARGET = "revenue_npr"
FORECAST_DAYS = 90

# ---------------------------------------------
# Model Evaluation
# ---------------------------------------------
def evaluate_model(y_true, y_pred, model_name="Prophet"):
    """Calculate comprehensive evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    
    metrics = {
        "model": model_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape)
    }
    
    print(f"\n{'='*50}")
    print(f"{model_name} Model Evaluation Metrics")
    print(f"{'='*50}")
    print(f"MAE:  {mae:,.2f} NPR")
    print(f"RMSE: {rmse:,.2f} NPR")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"{'='*50}\n")
    
    return metrics

def remove_outliers(df, column, n_std=3):
    """Remove outliers using z-score method"""
    mean = df[column].mean()
    std = df[column].std()
    df_clean = df[np.abs(df[column] - mean) <= n_std * std].copy()
    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"Removed {removed} outliers from {column}")
    return df_clean

# ---------------------------------------------
# Prepare Data for Prophet
# ---------------------------------------------
def prepare_prophet_data(df, test_size=0.2):
    """Prepare data with train/test split"""
    # Aggregate to daily
    df_daily = df.groupby("date").agg({
        TARGET: "sum",
        "rainfall_mm": "mean",
        "temperature": "mean",
        "discount_percent": "mean",
        "customer_traffic": "sum",
        "festival_season": "max"
    }).reset_index()
    
    # Remove outliers
    df_daily = remove_outliers(df_daily, TARGET, n_std=3)
    
    # Rename for Prophet
    df_prophet = df_daily.rename(columns={"date": "ds", TARGET: "y"})
    
    # Prepare regressors
    regressor_cols = ["rainfall_mm", "temperature", "discount_percent", "customer_traffic", "festival_season"]
    df_regressors = df_prophet[["ds"] + regressor_cols].copy()
    
    # Train/test split (time-based)
    split_index = int(len(df_prophet) * (1 - test_size))
    train_df = df_prophet.iloc[:split_index][["ds", "y"]].copy()
    test_df = df_prophet.iloc[split_index:][["ds", "y"]].copy()
    
    return df_prophet, df_regressors, train_df, test_df

# ---------------------------------------------
# Train Prophet Model with Enhanced Settings
# ---------------------------------------------
def train_prophet_model(train_df, df_regressors, test_df=None):
    """Train Prophet with optimized settings"""
    print("\nInitializing Prophet model with enhanced settings...")
    
    # Initialize Prophet with better parameters
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Better for data with trends
        changepoint_prior_scale=0.3,  # Control flexibility (0.05 is default)
        seasonality_prior_scale=10.0,  # Control seasonality strength
        interval_width=0.95  # 95% confidence intervals
    )
    
    # Add custom seasonalities
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Add regressors
    regressor_cols = [col for col in df_regressors.columns if col != "ds"]
    for col in regressor_cols:
        m.add_regressor(col)
        print(f"  Added regressor: {col}")
    
    # Merge regressors with training data
    train_with_regressors = train_df.merge(df_regressors, on="ds", how="left")
    
    print(f"\nTraining Prophet on {len(train_with_regressors)} days...")
    m.fit(train_with_regressors)
    print("✓ Training complete")
    
    # Evaluate on test set if provided
    test_metrics = None
    if test_df is not None and len(test_df) > 0:
        print("\nEvaluating on test set...")
        test_with_regressors = test_df.merge(df_regressors, on="ds", how="left")
        forecast_test = m.predict(test_with_regressors)
        
        y_true = test_df['y'].values
        y_pred = forecast_test['yhat'].values
        test_metrics = evaluate_model(y_true, y_pred, "Prophet")
    
    return m, test_metrics

# ---------------------------------------------
# Prophet Cross-Validation
# ---------------------------------------------
# def perform_cross_validation(model, df_prophet, df_regressors):
    """Perform time-series cross-validation"""
    print("\nPerforming cross-validation...")
    
    try:
        # Merge regressors
        df_full = df_prophet.merge(df_regressors, on="ds", how="left")
        
        # Retrain on full data for CV
        regressor_cols = [col for col in df_regressors.columns if col != "ds"]
        m_cv = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05 ,
        )
        m_cv.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        for col in regressor_cols:
            m_cv.add_regressor(col)
        
        m_cv.fit(df_full)
        
        # Cross-validation
        # Initial: minimum data needed to train
        # Period: how often to re-evaluate
        # Horizon: forecast period to evaluate
        df_cv = cross_validation(
            m_cv,
            initial='180 days',
            period='30 days',
            horizon='60 days'
        )
        
        # Calculate metrics
        df_metrics = performance_metrics(df_cv)
        
        print("\nCross-Validation Results:")
        print(f"  Average MAE:  {df_metrics['mae'].mean():,.2f} NPR")
        print(f"  Average RMSE: {df_metrics['rmse'].mean():,.2f} NPR")
        print(f"  Average MAPE: {df_metrics['mape'].mean():.4f}")
        
        cv_metrics = {
            "cv_mae_mean": float(df_metrics['mae'].mean()),
            "cv_rmse_mean": float(df_metrics['rmse'].mean()),
            "cv_mape_mean": float(df_metrics['mape'].mean())
        }
        
        return cv_metrics
    
    except Exception as e:
        print(f"⚠️  Cross-validation failed: {e}")
        return None
def perform_cross_validation(model, df_prophet, df_regressors):
    """Perform time-series cross-validation"""
    print("\nPerforming cross-validation...")
    
    try:
        # [FIX] REMOVED this problematic merge:
        # df_full = df_prophet.merge(df_regressors, on="ds", how="left")
        
        # We still need df_regressors to get the column names
        regressor_cols = [col for col in df_regressors.columns if col != "ds"]
        
        # Retrain on full data for CV
        m_cv = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        m_cv.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        for col in regressor_cols:
            m_cv.add_regressor(col)
        
        # [FIX] Fit directly on df_prophet, which has the correct columns
        m_cv.fit(df_prophet) 
        
        # Cross-validation
        df_cv = cross_validation(
            m_cv,
            initial='180 days',
            period='30 days',
            horizon='60 days'
        )
        
        # Calculate metrics
        df_metrics = performance_metrics(df_cv)
        
        print("\nCross-Validation Results:")
        print(f"  Average MAE:  {df_metrics['mae'].mean():,.2f} NPR")
        print(f"  Average RMSE: {df_metrics['rmse'].mean():,.2f} NPR")
        print(f"  Average MAPE: {df_metrics['mape'].mean():.4f}")
        
        cv_metrics = {
            "cv_mae_mean": float(df_metrics['mae'].mean()),
            "cv_rmse_mean": float(df_metrics['rmse'].mean()),
            "cv_mape_mean": float(df_metrics['mape'].mean())
        }
        
        return cv_metrics
    
    except Exception as e:
        print(f"⚠️  Cross-validation failed: {e}")
        return None
# ---------------------------------------------
# Generate Future Forecast
# ---------------------------------------------
def generate_forecast(model, df_regressors, periods=90):
    """Generate future forecast with regressors"""
    print(f"\nGenerating {periods}-day forecast...")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Extend regressors for future dates
    last_date = df_regressors['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=periods)
    
    # Use seasonal averages for future regressors
    future_regressors = []
    for future_date in future_dates:
        same_month = df_regressors[df_regressors['ds'].dt.month == future_date.month]
        
        if len(same_month) > 0:
            future_regressors.append({
                'ds': future_date,
                'rainfall_mm': same_month['rainfall_mm'].mean(),
                'temperature': same_month['temperature'].mean(),
                'customer_traffic': same_month['customer_traffic'].mean(),
                'discount_percent': 0,  # Conservative
                'festival_season': 0
            })
        else:
            # Fallback to overall mean
            future_regressors.append({
                'ds': future_date,
                'rainfall_mm': df_regressors['rainfall_mm'].mean(),
                'temperature': df_regressors['temperature'].mean(),
                'customer_traffic': df_regressors['customer_traffic'].mean(),
                'discount_percent': 0,
                'festival_season': 0
            })
    
    df_future_regressors = pd.DataFrame(future_regressors)
    
    # Combine historical and future regressors
    all_regressors = pd.concat([df_regressors, df_future_regressors], ignore_index=True)
    
    # Merge with future dataframe
    future_with_regressors = future.merge(all_regressors, on="ds", how="left")
    future_with_regressors = future_with_regressors.ffill().bfill()
    
    # Generate forecast
    forecast = model.predict(future_with_regressors)
    
    print("✓ Forecast generated")
    
    return forecast

# ---------------------------------------------
# Main Execution
# ---------------------------------------------
def run(test_size=0.2, perform_cv=True):
    print("="*60)
    print("IMPROVED PROPHET FORECASTING MODEL")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading featurized data...")
    df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
    
    # Prepare data
    print("\n[2/6] Preparing Prophet data with train/test split...")
    df_prophet, df_regressors, train_df, test_df = prepare_prophet_data(df, test_size)
    
    print(f"Training set: {len(train_df)} days ({train_df['ds'].min()} to {train_df['ds'].max()})")
    print(f"Test set: {len(test_df)} days ({test_df['ds'].min()} to {test_df['ds'].max()})")
    
    # Train model
    print("\n[3/6] Training Prophet model...")
    model, test_metrics = train_prophet_model(train_df, df_regressors, test_df)
    
    # Cross-validation
    cv_metrics = None
    if perform_cv:
        print("\n[4/6] Performing cross-validation...")
        cv_metrics = perform_cross_validation(model, df_prophet, df_regressors)
    else:
        print("\n[4/6] Skipping cross-validation (set perform_cv=True to enable)")
    
    # Retrain on full data for production
    print("\n[5/6] Retraining on full dataset for production...")
    full_data = df_prophet[["ds", "y"]].merge(df_regressors, on="ds", how="left")
    
    regressor_cols = [col for col in df_regressors.columns if col != "ds"]
    m_prod = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.5
    )
    m_prod.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    for col in regressor_cols:
        m_prod.add_regressor(col)
    
    m_prod.fit(full_data)
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m_prod, MODEL_PATH)
    print(f"✓ Saved model to {MODEL_PATH}")
    
    # Generate forecast
    print(f"\n[6/6] Generating {FORECAST_DAYS}-day future forecast...")
    forecast = generate_forecast(m_prod, df_regressors, FORECAST_DAYS)
    
    # Save forecast
    forecast_output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(FORECAST_DAYS)
    forecast_output.to_csv(FORECAST_CSV, index=False)
    print(f"✓ Saved forecast to {FORECAST_CSV}")
    
    # Save metadata
    metadata = {
        "model": "Prophet",
        "timestamp": pd.Timestamp.now().isoformat(),
        "test_metrics": test_metrics if test_metrics else {},
        "cv_metrics": cv_metrics if cv_metrics else {},
        "parameters": {
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.05,
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "monthly_seasonality": True
        },
        "regressors": regressor_cols,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "forecast_days": FORECAST_DAYS
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {METADATA_PATH}")
    
    # Summary
    print(f"\n{'='*60}")
    print("FORECAST SUMMARY")
    print(f"{'='*60}")
    future_forecast = forecast_output['yhat'].values
    print(f"Average daily forecast: {np.mean(future_forecast):,.2f} NPR")
    print(f"Total forecast ({FORECAST_DAYS} days): {np.sum(future_forecast):,.2f} NPR")
    print(f"Min daily forecast: {np.min(future_forecast):,.2f} NPR")
    print(f"Max daily forecast: {np.max(future_forecast):,.2f} NPR")
    print(f"{'='*60}\n")
    
    return m_prod, test_metrics, forecast_output

if __name__ == "__main__":
    # Set perform_cv=False to skip cross-validation (faster)
    model, metrics, forecast = run(test_size=0.5, perform_cv=True)
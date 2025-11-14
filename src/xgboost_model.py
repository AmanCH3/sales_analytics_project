# src/xgboost_model_improved.py
"""
Improved XGBoost model with:
- Train/test split and validation
- Advanced feature engineering
- Hyperparameter tuning
- Model evaluation metrics
- Better future regressor forecasting
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import joblib
import json
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------
# Paths
# ---------------------------------------------
FEATURE_PATH = Path("data/processed/featurized_sales_data.csv")
MODEL_PATH = Path("models/xgboost_model.pkl")
FORECAST_CSV = Path("data/processed/xgboost_forecast.csv")
METADATA_PATH = Path("models/xgboost_metadata.json")
TARGET = "revenue_npr"
FORECAST_DAYS = 90

# ---------------------------------------------
# Advanced Feature Engineering
# ---------------------------------------------
def create_advanced_features(df):
    """Create comprehensive time-series features"""
    df_feat = df.copy()
    df_feat['date'] = pd.to_datetime(df_feat['date'])
    
    # Basic time features
    df_feat['dayofweek'] = df_feat['date'].dt.dayofweek
    df_feat['day_of_month'] = df_feat['date'].dt.day
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['year'] = df_feat['date'].dt.year
    df_feat['quarter'] = df_feat['date'].dt.quarter
    df_feat['weekofyear'] = df_feat['date'].dt.isocalendar().week.astype(int)
    df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)
    df_feat['is_month_start'] = df_feat['date'].dt.is_month_start.astype(int)
    df_feat['is_month_end'] = df_feat['date'].dt.is_month_end.astype(int)
    
    # Cyclical encoding (better for capturing seasonality)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['dayofweek'] / 7)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['dayofweek'] / 7)
    
    # Multiple lag features
    for lag in [1, 3, 7, 14, 21, 30]:
        df_feat[f'lag_{lag}'] = df_feat[TARGET].shift(lag)
    
    # Rolling statistics (multiple windows)
    for window in [7, 14, 30]:
        df_feat[f'rolling_mean_{window}'] = df_feat[TARGET].shift(1).rolling(window).mean()
        df_feat[f'rolling_std_{window}'] = df_feat[TARGET].shift(1).rolling(window).std()
        df_feat[f'rolling_min_{window}'] = df_feat[TARGET].shift(1).rolling(window).min()
        df_feat[f'rolling_max_{window}'] = df_feat[TARGET].shift(1).rolling(window).max()
    
    # Exponential moving averages
    df_feat['ema_7'] = df_feat[TARGET].shift(1).ewm(span=7, adjust=False).mean()
    df_feat['ema_30'] = df_feat[TARGET].shift(1).ewm(span=30, adjust=False).mean()
    
    # Interaction features
    if 'discount_percent' in df_feat.columns and 'customer_traffic' in df_feat.columns:
        df_feat['discount_x_traffic'] = df_feat['discount_percent'] * df_feat['customer_traffic']
    if 'temperature' in df_feat.columns and 'rainfall_mm' in df_feat.columns:
        df_feat['temp_x_rainfall'] = df_feat['temperature'] * df_feat['rainfall_mm']
    
    # Fill NaNs
    return df_feat.bfill().ffill().fillna(0)

# ---------------------------------------------
# Model Evaluation
# ---------------------------------------------
def evaluate_model(y_true, y_pred, model_name="XGBoost"):
    """Calculate comprehensive evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Avoid division by zero
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

# ---------------------------------------------
# Better Future Regressor Forecasting
# ---------------------------------------------
def forecast_external_regressors(history_df, forecast_days):
    """Forecast external variables using seasonal averages"""
    future_regressors = []
    last_date = history_df['date'].max()
    
    for i in range(1, forecast_days + 1):
        next_date = last_date + pd.DateOffset(days=i)
        
        # Use historical averages for the same month and day of week
        same_month = history_df[history_df['date'].dt.month == next_date.month]
        same_dow = same_month[same_month['date'].dt.dayofweek == next_date.dayofweek]
        
        # If we have data for this specific day pattern, use it; otherwise use month average
        ref_data = same_dow if len(same_dow) > 0 else same_month
        
        future_regressors.append({
            'date': next_date,
            'rainfall_mm': ref_data['rainfall_mm'].mean() if len(ref_data) > 0 else history_df['rainfall_mm'].mean(),
            'temperature': ref_data['temperature'].mean() if len(ref_data) > 0 else history_df['temperature'].mean(),
            'customer_traffic': ref_data['customer_traffic'].mean() if len(ref_data) > 0 else history_df['customer_traffic'].mean(),
            'discount_percent': 0,  # Conservative assumption
            'festival_season': 0  # Update manually for known festivals
        })
    
    return pd.DataFrame(future_regressors)

# ---------------------------------------------
# Remove Outliers
# ---------------------------------------------
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
# Main Training & Forecasting with Validation
# ---------------------------------------------
def run(tune_hyperparameters=False, test_size=0.2):
    print("="*60)
    print("IMPROVED XGBOOST FORECASTING MODEL")
    print("="*60)
    
    # Load and aggregate data
    print("\n[1/7] Loading featurized data...")
    df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
    
    df_daily = df.groupby("date").agg({
        TARGET: "sum",
        "rainfall_mm": "mean",
        "temperature": "mean",
        "discount_percent": "mean",
        "customer_traffic": "sum",
        "festival_season": "max"
    }).reset_index()
    
    # Remove outliers
    print("\n[2/7] Removing outliers...")
    df_daily = remove_outliers(df_daily, TARGET, n_std=3)
    
    # Create advanced features
    print("\n[3/7] Creating advanced time-series features...")
    df_featured = create_advanced_features(df_daily)
    
    # Define all features
    FEATURES = [
        # External regressors
        "rainfall_mm", "temperature", "discount_percent", "customer_traffic", "festival_season",
        # Time features
        "dayofweek", "day_of_month", "month", "year", "quarter", "weekofyear",
        "is_weekend", "is_month_start", "is_month_end",
        # Cyclical features
        "month_sin", "month_cos", "day_sin", "day_cos",
        # Lag features
        "lag_1", "lag_3", "lag_7", "lag_14", "lag_21", "lag_30",
        # Rolling features
        "rolling_mean_7", "rolling_std_7", "rolling_min_7", "rolling_max_7",
        "rolling_mean_14", "rolling_std_14", "rolling_min_14", "rolling_max_14",
        "rolling_mean_30", "rolling_std_30", "rolling_min_30", "rolling_max_30",
        # EMA features
        "ema_7", "ema_30",
        # Interaction features
        "discount_x_traffic", "temp_x_rainfall"
    ]
    
    # Train/Test Split (time-based)
    print("\n[4/7] Splitting data into train/test sets...")
    split_index = int(len(df_featured) * (1 - test_size))
    train_df = df_featured.iloc[:split_index].copy()
    test_df = df_featured.iloc[split_index:].copy()
    
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    
    print(f"Training set: {len(train_df)} days ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Test set: {len(test_df)} days ({test_df['date'].min()} to {test_df['date'].max()})")
    
    # Train Model
    print("\n[5/7] Training XGBoost model...")
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning (this may take a while)...")
        param_grid = {
            'max_depth': [5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [500, 1000],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        base_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
    else:
        # Use good default parameters
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train, verbose=False)
        best_params = model.get_params()
    
    # Evaluate on test set
    print("\n[6/7] Evaluating model on test set...")
    y_pred_test = model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_pred_test, "XGBoost")
    
    # Also evaluate on train set to check for overfitting
    y_pred_train = model.predict(X_train)
    train_metrics = evaluate_model(y_train, y_pred_train, "XGBoost (Train)")
    
    # Check for overfitting
    if train_metrics['MAE'] < test_metrics['MAE'] * 0.5:
        print("⚠️  WARNING: Possible overfitting detected!")
        print(f"Train MAE ({train_metrics['MAE']:.2f}) is much lower than Test MAE ({test_metrics['MAE']:.2f})")
    
    # Feature importance
    feature_importance = dict(zip(FEATURES, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Most Important Features:")
    for i, (feat, importance) in enumerate(top_features, 1):
        print(f"{i:2d}. {feat:25s}: {importance:.4f}")
    
    # Retrain on full data for production
    print("\n[7/7] Retraining on full dataset for production...")
    X_full = df_featured[FEATURES]
    y_full = df_featured[TARGET]
    model.fit(X_full, y_full, verbose=False)
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✓ Saved model to {MODEL_PATH}")
    
    # Save metadata
    metadata = {
        "model": "XGBoost",
        "timestamp": pd.Timestamp.now().isoformat(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "parameters": {k: str(v) for k, v in best_params.items()},
        "features": FEATURES,
        "top_10_features": {k: float(v) for k, v in top_features},
        "train_size": len(train_df),
        "test_size": len(test_df),
        "forecast_days": FORECAST_DAYS
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {METADATA_PATH}")
    
    # Generate Future Forecast
    print(f"\n{'='*60}")
    print(f"GENERATING {FORECAST_DAYS}-DAY FUTURE FORECAST")
    print(f"{'='*60}")
    
    # Forecast external regressors
    print("\nForecasting external regressors...")
    future_regressors = forecast_external_regressors(df_featured, FORECAST_DAYS)
    
    # Recursive forecasting
    history_df = df_featured.copy()
    future_predictions = []
    last_date = history_df['date'].max()
    
    print("Generating recursive predictions...")
    for i in range(FORECAST_DAYS):
        if (i + 1) % 30 == 0:
            print(f"  Generated {i + 1}/{FORECAST_DAYS} predictions...")
        
        next_date = last_date + pd.DateOffset(days=i + 1)
        
        # Get regressor values
        regressors = future_regressors[future_regressors['date'] == next_date].iloc[0]
        
        # Create features for next day
        features_dict = {
            # External regressors
            "rainfall_mm": regressors['rainfall_mm'],
            "temperature": regressors['temperature'],
            "discount_percent": regressors['discount_percent'],
            "customer_traffic": regressors['customer_traffic'],
            "festival_season": regressors['festival_season'],
            
            # Time features
            "dayofweek": next_date.dayofweek,
            "day_of_month": next_date.day,
            "month": next_date.month,
            "year": next_date.year,
            "quarter": next_date.quarter,
            "weekofyear": int(next_date.isocalendar().week),
            "is_weekend": int(next_date.dayofweek >= 5),
            "is_month_start": int(next_date.is_month_start),
            "is_month_end": int(next_date.is_month_end),
            
            # Cyclical features
            "month_sin": np.sin(2 * np.pi * next_date.month / 12),
            "month_cos": np.cos(2 * np.pi * next_date.month / 12),
            "day_sin": np.sin(2 * np.pi * next_date.dayofweek / 7),
            "day_cos": np.cos(2 * np.pi * next_date.dayofweek / 7),
            
            # Lag features
            "lag_1": history_df[TARGET].iloc[-1],
            "lag_3": history_df[TARGET].iloc[-3],
            "lag_7": history_df[TARGET].iloc[-7],
            "lag_14": history_df[TARGET].iloc[-14],
            "lag_21": history_df[TARGET].iloc[-21],
            "lag_30": history_df[TARGET].iloc[-30],
            
            # Rolling features (7-day)
            "rolling_mean_7": history_df[TARGET].iloc[-7:].mean(),
            "rolling_std_7": history_df[TARGET].iloc[-7:].std(),
            "rolling_min_7": history_df[TARGET].iloc[-7:].min(),
            "rolling_max_7": history_df[TARGET].iloc[-7:].max(),
            
            # Rolling features (14-day)
            "rolling_mean_14": history_df[TARGET].iloc[-14:].mean(),
            "rolling_std_14": history_df[TARGET].iloc[-14:].std(),
            "rolling_min_14": history_df[TARGET].iloc[-14:].min(),
            "rolling_max_14": history_df[TARGET].iloc[-14:].max(),
            
            # Rolling features (30-day)
            "rolling_mean_30": history_df[TARGET].iloc[-30:].mean(),
            "rolling_std_30": history_df[TARGET].iloc[-30:].std(),
            "rolling_min_30": history_df[TARGET].iloc[-30:].min(),
            "rolling_max_30": history_df[TARGET].iloc[-30:].max(),
            
            # EMA features
            "ema_7": history_df[TARGET].iloc[-7:].ewm(span=7, adjust=False).mean().iloc[-1],
            "ema_30": history_df[TARGET].iloc[-30:].ewm(span=30, adjust=False).mean().iloc[-1],
            
            # Interaction features
            "discount_x_traffic": regressors['discount_percent'] * regressors['customer_traffic'],
            "temp_x_rainfall": regressors['temperature'] * regressors['rainfall_mm']
        }
        
        # Predict
        X_new = pd.DataFrame([features_dict])[FEATURES]
        pred = model.predict(X_new)[0]
        future_predictions.append(max(pred, 0))  # Ensure non-negative
        
        # Append to history
        new_row = features_dict.copy()
        new_row['date'] = next_date
        new_row[TARGET] = pred
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    
    print(f"✓ Generated all {FORECAST_DAYS} predictions")
    
    # Save forecast
    future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=FORECAST_DAYS)
    df_forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions,
        'yhat_lower': future_predictions,  # Placeholder
        'yhat_upper': future_predictions   # Placeholder
    })
    
    df_forecast.to_csv(FORECAST_CSV, index=False)
    print(f"✓ Saved forecast to {FORECAST_CSV}")
    
    print(f"\n{'='*60}")
    print("FORECAST SUMMARY")
    print(f"{'='*60}")
    print(f"Average daily forecast: {np.mean(future_predictions):,.2f} NPR")
    print(f"Total forecast ({FORECAST_DAYS} days): {np.sum(future_predictions):,.2f} NPR")
    print(f"Min daily forecast: {np.min(future_predictions):,.2f} NPR")
    print(f"Max daily forecast: {np.max(future_predictions):,.2f} NPR")
    print(f"{'='*60}\n")
    
    return model, test_metrics, df_forecast

if __name__ == "__main__":
    # Set tune_hyperparameters=True for the first run (takes longer)
    # Set it to False for subsequent runs to use good defaults
    model, metrics, forecast = run(tune_hyperparameters=False, test_size=0.2)
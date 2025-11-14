# src/lightgbm_model_improved.py
"""
Improved LightGBM model with all enhancements
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
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
MODEL_PATH = Path("models/lightgbm_model.pkl")
FORECAST_CSV = Path("data/processed/lightgbm_forecast.csv")
METADATA_PATH = Path("models/lightgbm_metadata.json")
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
    
    # Cyclical encoding
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['dayofweek'] / 7)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['dayofweek'] / 7)
    
    # Multiple lag features
    for lag in [1, 3, 7, 14, 21, 30]:
        df_feat[f'lag_{lag}'] = df_feat[TARGET].shift(lag)
    
    # Rolling statistics
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
    
    return df_feat.bfill().ffill().fillna(0)

def evaluate_model(y_true, y_pred, model_name="LightGBM"):
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

def forecast_external_regressors(history_df, forecast_days):
    """Forecast external variables using seasonal averages"""
    future_regressors = []
    last_date = history_df['date'].max()
    
    for i in range(1, forecast_days + 1):
        next_date = last_date + pd.DateOffset(days=i)
        same_month = history_df[history_df['date'].dt.month == next_date.month]
        same_dow = same_month[same_month['date'].dt.dayofweek == next_date.dayofweek]
        ref_data = same_dow if len(same_dow) > 0 else same_month
        
        future_regressors.append({
            'date': next_date,
            'rainfall_mm': ref_data['rainfall_mm'].mean() if len(ref_data) > 0 else history_df['rainfall_mm'].mean(),
            'temperature': ref_data['temperature'].mean() if len(ref_data) > 0 else history_df['temperature'].mean(),
            'customer_traffic': ref_data['customer_traffic'].mean() if len(ref_data) > 0 else history_df['customer_traffic'].mean(),
            'discount_percent': 0,
            'festival_season': 0
        })
    
    return pd.DataFrame(future_regressors)

def remove_outliers(df, column, n_std=3):
    """Remove outliers using z-score method"""
    mean = df[column].mean()
    std = df[column].std()
    df_clean = df[np.abs(df[column] - mean) <= n_std * std].copy()
    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"Removed {removed} outliers from {column}")
    return df_clean

def run(tune_hyperparameters=False, test_size=0.2):
    print("="*60)
    print("IMPROVED LIGHTGBM FORECASTING MODEL")
    print("="*60)
    
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
    
    print("\n[2/7] Removing outliers...")
    df_daily = remove_outliers(df_daily, TARGET, n_std=3)
    
    print("\n[3/7] Creating advanced time-series features...")
    df_featured = create_advanced_features(df_daily)
    
    FEATURES = [
        "rainfall_mm", "temperature", "discount_percent", "customer_traffic", "festival_season",
        "dayofweek", "day_of_month", "month", "year", "quarter", "weekofyear",
        "is_weekend", "is_month_start", "is_month_end",
        "month_sin", "month_cos", "day_sin", "day_cos",
        "lag_1", "lag_3", "lag_7", "lag_14", "lag_21", "lag_30",
        "rolling_mean_7", "rolling_std_7", "rolling_min_7", "rolling_max_7",
        "rolling_mean_14", "rolling_std_14", "rolling_min_14", "rolling_max_14",
        "rolling_mean_30", "rolling_std_30", "rolling_min_30", "rolling_max_30",
        "ema_7", "ema_30",
        "discount_x_traffic", "temp_x_rainfall"
    ]
    
    CATEGORICAL_FEATURES = ["dayofweek", "month", "year", "quarter", "festival_season", 
                           "is_weekend", "is_month_start", "is_month_end"]
    
    print("\n[4/7] Splitting data into train/test sets...")
    split_index = int(len(df_featured) * (1 - test_size))
    train_df = df_featured.iloc[:split_index].copy()
    test_df = df_featured.iloc[split_index:].copy()
    
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    
    # Convert categorical features
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    print(f"Training set: {len(train_df)} days")
    print(f"Test set: {len(test_df)} days")
    
    print("\n[5/7] Training LightGBM model...")
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'num_leaves': [31, 50, 70],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [500, 1000],
            'min_child_samples': [20, 30, 50],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        base_model = lgb.LGBMRegressor(objective='regression_l1', n_jobs=-1, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
    else:
        model = lgb.LGBMRegressor(
            objective='regression_l1',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=50,
            min_child_samples=30,
            subsample=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
        best_params = model.get_params()
    
    print("\n[6/7] Evaluating model on test set...")
    y_pred_test = model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_pred_test, "LightGBM")
    
    y_pred_train = model.predict(X_train)
    train_metrics = evaluate_model(y_train, y_pred_train, "LightGBM (Train)")
    
    if train_metrics['MAE'] < test_metrics['MAE'] * 0.5:
        print("⚠️  WARNING: Possible overfitting detected!")
    
    feature_importance = dict(zip(FEATURES, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Most Important Features:")
    for i, (feat, importance) in enumerate(top_features, 1):
        print(f"{i:2d}. {feat:25s}: {importance:.4f}")
    
    print("\n[7/7] Retraining on full dataset for production...")
    X_full = df_featured[FEATURES]
    y_full = df_featured[TARGET]
    
    for col in CATEGORICAL_FEATURES:
        if col in X_full.columns:
            X_full[col] = X_full[col].astype('category')
    
    model.fit(X_full, y_full, categorical_feature=CATEGORICAL_FEATURES)
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✓ Saved model to {MODEL_PATH}")
    
    metadata = {
        "model": "LightGBM",
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
    
    print(f"\n{'='*60}")
    print(f"GENERATING {FORECAST_DAYS}-DAY FUTURE FORECAST")
    print(f"{'='*60}")
    
    future_regressors = forecast_external_regressors(df_featured, FORECAST_DAYS)
    history_df = df_featured.copy()
    future_predictions = []
    last_date = history_df['date'].max()
    
    print("Generating recursive predictions...")
    for i in range(FORECAST_DAYS):
        if (i + 1) % 30 == 0:
            print(f"  Generated {i + 1}/{FORECAST_DAYS} predictions...")
        
        next_date = last_date + pd.DateOffset(days=i + 1)
        regressors = future_regressors[future_regressors['date'] == next_date].iloc[0]
        
        features_dict = {
            "rainfall_mm": regressors['rainfall_mm'],
            "temperature": regressors['temperature'],
            "discount_percent": regressors['discount_percent'],
            "customer_traffic": regressors['customer_traffic'],
            "festival_season": regressors['festival_season'],
            "dayofweek": next_date.dayofweek,
            "day_of_month": next_date.day,
            "month": next_date.month,
            "year": next_date.year,
            "quarter": next_date.quarter,
            "weekofyear": int(next_date.isocalendar().week),
            "is_weekend": int(next_date.dayofweek >= 5),
            "is_month_start": int(next_date.is_month_start),
            "is_month_end": int(next_date.is_month_end),
            "month_sin": np.sin(2 * np.pi * next_date.month / 12),
            "month_cos": np.cos(2 * np.pi * next_date.month / 12),
            "day_sin": np.sin(2 * np.pi * next_date.dayofweek / 7),
            "day_cos": np.cos(2 * np.pi * next_date.dayofweek / 7),
            "lag_1": history_df[TARGET].iloc[-1],
            "lag_3": history_df[TARGET].iloc[-3],
            "lag_7": history_df[TARGET].iloc[-7],
            "lag_14": history_df[TARGET].iloc[-14],
            "lag_21": history_df[TARGET].iloc[-21],
            "lag_30": history_df[TARGET].iloc[-30],
            "rolling_mean_7": history_df[TARGET].iloc[-7:].mean(),
            "rolling_std_7": history_df[TARGET].iloc[-7:].std(),
            "rolling_min_7": history_df[TARGET].iloc[-7:].min(),
            "rolling_max_7": history_df[TARGET].iloc[-7:].max(),
            "rolling_mean_14": history_df[TARGET].iloc[-14:].mean(),
            "rolling_std_14": history_df[TARGET].iloc[-14:].std(),
            "rolling_min_14": history_df[TARGET].iloc[-14:].min(),
            "rolling_max_14": history_df[TARGET].iloc[-14:].max(),
            "rolling_mean_30": history_df[TARGET].iloc[-30:].mean(),
            "rolling_std_30": history_df[TARGET].iloc[-30:].std(),
            "rolling_min_30": history_df[TARGET].iloc[-30:].min(),
            "rolling_max_30": history_df[TARGET].iloc[-30:].max(),
            "ema_7": history_df[TARGET].iloc[-7:].ewm(span=7, adjust=False).mean().iloc[-1],
            "ema_30": history_df[TARGET].iloc[-30:].ewm(span=30, adjust=False).mean().iloc[-1],
            "discount_x_traffic": regressors['discount_percent'] * regressors['customer_traffic'],
            "temp_x_rainfall": regressors['temperature'] * regressors['rainfall_mm']
        }
        
        X_new = pd.DataFrame([features_dict])[FEATURES]
        for col in CATEGORICAL_FEATURES:
            if col in X_new.columns:
                X_new[col] = X_new[col].astype('category')
        
        pred = model.predict(X_new)[0]
        future_predictions.append(max(pred, 0))
        
        new_row = features_dict.copy()
        new_row['date'] = next_date
        new_row[TARGET] = pred
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    
    print(f"✓ Generated all {FORECAST_DAYS} predictions")
    
    future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=FORECAST_DAYS)
    df_forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions,
        'yhat_lower': future_predictions,
        'yhat_upper': future_predictions
    })
    
    df_forecast.to_csv(FORECAST_CSV, index=False)
    print(f"✓ Saved forecast to {FORECAST_CSV}")
    
    print(f"\n{'='*60}")
    print("FORECAST SUMMARY")
    print(f"{'='*60}")
    print(f"Average daily forecast: {np.mean(future_predictions):,.2f} NPR")
    print(f"Total forecast ({FORECAST_DAYS} days): {np.sum(future_predictions):,.2f} NPR")
    print(f"{'='*60}\n")
    
    return model, test_metrics, df_forecast

if __name__ == "__main__":
    model, metrics, forecast = run(tune_hyperparameters=False, test_size=0.2)
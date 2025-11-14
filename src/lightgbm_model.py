# src/lightgbm_model.py
"""
Train a LightGBM model on ALL data and generate a 90-day
recursive forecast for the future.
"""

import pandas as pd
import lightgbm as lgb
from pathlib import Path
import joblib
import numpy as np

# ---------------------------------------------
# Paths
# ---------------------------------------------
FEATURE_PATH = Path("data/processed/featurized_sales_data.csv")
MODEL_PATH = Path("models/lightgbm_model.pkl")
FORECAST_CSV = Path("data/processed/lightgbm_forecast.csv")
TARGET = "revenue_npr"
FORECAST_DAYS = 90 # How many days into the future to predict

# ---------------------------------------------
# Feature Engineering
# ---------------------------------------------
def create_time_series_features(df):
    """Create time-series features based on the 'date' column"""
    df_feat = df.copy()
    df_feat['date'] = pd.to_datetime(df_feat['date'])
    df_feat['dayofweek'] = df_feat['date'].dt.dayofweek
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['year'] = df_feat['date'].dt.year
    df_feat['quarter'] = df_feat['date'].dt.quarter
    # .astype(int) is important for lightgbm
    df_feat['weekofyear'] = df_feat['date'].dt.isocalendar().week.astype(int)
    
    # Lag features (sales 1 week ago, 2 weeks ago)
    df_feat['lag_7'] = df_feat[TARGET].shift(7)
    df_feat['lag_14'] = df_feat[TARGET].shift(14)
    
    # Rolling mean
    df_feat['rolling_mean_7'] = df_feat[TARGET].shift(1).rolling(window=7).mean()
    
    # bfill to handle NaNs from initial shifts
    # ffill for any remaining NaNs, then 0 for any still left
    return df_feat.bfill().ffill().fillna(0)

# ---------------------------------------------
# Main Training & Forecasting
# ---------------------------------------------
def run():
    print("Loading featurized data...")
    # Load base data
    df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
    
    # Aggregate to daily sales, similar to Prophet model
    df_daily = df.groupby("date").agg({
        TARGET: "sum",
        "rainfall_mm": "mean",
        "temperature": "mean",
        "discount_percent": "mean",
        "customer_traffic": "sum",
        "festival_season": "max"
    }).reset_index()

    print("Creating time-series features for training...")
    df_featured = create_time_series_features(df_daily)
    
    # Define features and target
    FEATURES = [
        "rainfall_mm", "temperature", "discount_percent", "customer_traffic", 
        "festival_season", "dayofweek", "month", "year", "quarter", 
        "weekofyear", "lag_7", "lag_14", "rolling_mean_7"
    ]
    
    # LightGBM works best with categorical features declared
    CATEGORICAL_FEATURES = ["dayofweek", "month", "year", "quarter", "festival_season"]

    # --- Train Model on ALL Data ---
    X_train = df_featured[FEATURES]
    y_train = df_featured[TARGET]
    
    # Convert categorical features to 'category' dtype for LightGBM
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            
    print(f"Training model on ALL data ({len(df_featured)} days)...")

    # --- Train Model ---
    model = lgb.LGBMRegressor(
        objective='regression_l1', # MAE is often more robust to outliers
        n_estimators=1000,
        learning_rate=0.01,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        categorical_feature=CATEGORICAL_FEATURES
    )
    
    # --- Save Model ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved LightGBM model to {MODEL_PATH}")

    # --- Recursive Forecasting Loop ---
    print(f"Generating {FORECAST_DAYS}-day recursive forecast...")
    
    # Start with a copy of the full historical data
    history_df = df_featured.copy()
    future_predictions = []

    # Get the last known values for external regressors to carry forward
    last_known_rainfall = history_df['rainfall_mm'].iloc[-1]
    last_known_temp = history_df['temperature'].iloc[-1]
    last_known_traffic = history_df['customer_traffic'].iloc[-1]
    
    last_date = history_df['date'].max()
    
    for i in range(1, FORECAST_DAYS + 1):
        # 1. Create the date for the next day
        next_date = last_date + pd.DateOffset(days=i)
        
        # 2. Get lag/rolling features from the *current* history
        lag_7 = history_df[TARGET].iloc[-7]
        lag_14 = history_df[TARGET].iloc[-14]
        rolling_mean_7 = history_df[TARGET].iloc[-7:].mean()
        
        # 3. Create date features for the new date
        dayofweek = next_date.dayofweek
        month = next_date.month
        year = next_date.year
        quarter = next_date.quarter
        weekofyear = next_date.isocalendar().week
        
        # 4. Create feature vector (dictionary) for the new prediction
        features_dict = {
            # Assumed external features
            "rainfall_mm": last_known_rainfall, # Assumption
            "temperature": last_known_temp,     # Assumption
            "discount_percent": 0,                # Assumption: No future discount
            "customer_traffic": last_known_traffic, # Assumption
            "festival_season": 0,                 # Assumption: No future festival
            
            # Date features
            "dayofweek": dayofweek,
            "month": month,
            "year": year,
            "quarter": quarter,
            "weekofyear": int(weekofyear),
            
            # Lag/Rolling features
            "lag_7": lag_7,
            "lag_14": lag_14,
            "rolling_mean_7": rolling_mean_7
        }

        # 5. Convert to DataFrame in the correct feature order
        X_new = pd.DataFrame([features_dict])[FEATURES]
        # Convert categorical features to 'category' dtype
        for col in CATEGORICAL_FEATURES:
            if col in X_new.columns:
                X_new[col] = X_new[col].astype('category')
        
        # 6. Predict
        pred = model.predict(X_new)[0]
        future_predictions.append(pred)
        
        # 7. IMPORTANT: Append this new prediction to history
        # so it can be used for the *next* day's lag features
        new_row = features_dict.copy()
        new_row['date'] = next_date
        new_row[TARGET] = pred
        
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)

    # --- Save Future Forecast ---
    future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=FORECAST_DAYS)
    
    df_forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions,
        'yhat_lower': future_predictions, # Add dummy columns for streamlit app
        'yhat_upper': future_predictions
    })
    
    df_forecast.to_csv(FORECAST_CSV, index=False)
    print(f"Saved LightGBM *future* forecast to {FORECAST_CSV}")
    
    return model

if __name__ == "__main__":
    run()
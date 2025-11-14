# src/forecasting_model.py
"""
Train a Prophet forecasting model with regressors and save the model.
Also provides a helper to produce future forecasts.
"""

import pandas as pd
from prophet import Prophet
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

FEATURE_PATH = Path("data/processed/featurized_sales_data.csv")
MODEL_PATH = Path("models/prophet_model.pkl")
FORECAST_CSV = Path("data/processed/prophet_forecast.csv")

def prepare_prophet_df(df: pd.DataFrame, target="revenue_npr", group_by=None):
    # group_by: if None aggregate across dataset; else e.g., 'product_id' or 'store_id'
    if group_by:
        g = df.groupby(["date", group_by])[target].sum().reset_index().rename(columns={group_by: "group"})
        # create simple pivot or keep group column; for now return aggregated series across group
        return g
    else:
        df_agg = df.groupby("date")[target].sum().reset_index().rename(columns={"date":"ds", target:"y"})
        return df_agg

def train_global_model(df_prophet: pd.DataFrame, add_regressors_df=None, periods=90):
    # df_prophet: must have columns ds (datetime) and y (target)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    # add regressors if provided (must align by ds)
    if add_regressors_df is not None:
        for col in add_regressors_df.columns:
            if col != "ds":
                m.add_regressor(col)
        # merge regressors into df_prophet
        df_prophet = df_prophet.merge(add_regressors_df, on="ds", how="left")
    m.fit(df_prophet)
    # save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, MODEL_PATH)
    print(f"Saved Prophet model to {MODEL_PATH}")
    # create future dataframe
    future = m.make_future_dataframe(periods=periods)
    if add_regressors_df is not None:
        # naive: forward fill regressors for future
        future = future.merge(add_regressors_df, on="ds", how="left").ffill().bfill()
    forecast = m.predict(future)
    forecast[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(FORECAST_CSV, index=False)
    print(f"Saved forecast to {FORECAST_CSV}")
    return m, forecast

def run():
    df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
    # Prepare base df for Prophet
    df_prophet = df.groupby("date")["revenue_npr"].sum().reset_index().rename(columns={"date":"ds","revenue_npr":"y"})
    # Add regressors: rainfall, temperature, discount, customer_traffic, festival_season
    regressors = df.groupby("date")[["rainfall_mm","temperature","discount_percent","customer_traffic","festival_season"]].mean().reset_index().rename(columns={"date":"ds"})
    m, forecast = train_global_model(df_prophet, add_regressors_df=regressors, periods=90)
    return m, forecast

if __name__ == "__main__":
    run()

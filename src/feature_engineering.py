# src/feature_engineering.py
"""
Add time-based, festival, weather and other derived features.
Also create transaction_id for Apriori if not available.
"""

import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path("data/processed/cleaned_sales_data.csv")
FEATURE_PATH = Path("data/processed/featurized_sales_data.csv")

# Simple Nepali month mapping (approx by Gregorian month)
NEPALI_MONTHS = ["Baisakh","Jestha","Ashadh","Shrawan","Bhadra","Ashwin","Kartik","Mangsir","Poush","Magh","Falgun","Chaitra"]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.day_name()
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekofyear"] = df["date"].dt.isocalendar().week
    df["nepali_month"] = df["date"].dt.month.apply(lambda m: NEPALI_MONTHS[(m-1)%12])
    # salary_day flag - first week of Nepali month often sees spikes (approximation)
    df["is_salary_week"] = df["date"].dt.day.between(1,7).astype(int)
    return df

def encode_holidays(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure boolean flags exist
    festival_cols = ["is_dashain","is_tihar","is_holi","is_eid","is_lhosar","is_holiday","holiday_name"]
    for c in festival_cols:
        if c not in df.columns:
            df[c] = False if c!="holiday_name" else ""
    # Consolidate into festival_season and days_until_{festival} could be added later
    df["festival_season"] = df[["is_dashain","is_tihar","is_holi","is_eid","is_lhosar"]].any(axis=1).astype(int)
    return df

def create_transaction_id(df: pd.DataFrame) -> pd.DataFrame:
    # If you don't have transaction id, create a synthetic one:
    # Group by date + store_id + a random session identifier based on time-of-day footprint.
    # This is heuristic; replace if you have real transaction data.
    df = df.sort_values(["date","store_id"])
    # create session bucket using cumulative count per store per day
    df["txn_count_store_day"] = df.groupby([df["date"].dt.date, "store_id"]).cumcount()
    df["transaction_id"] = df["date"].dt.strftime("%Y%m%d") + "_" + df["store_id"].astype(str) + "_" + (df["txn_count_store_day"] // 5).astype(str)
    df = df.drop(columns=["txn_count_store_day"])
    return df

def run():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    df = add_time_features(df)
    df = encode_holidays(df)
    df = create_transaction_id(df)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURE_PATH, index=False)
    print(f"Featurized data saved to {FEATURE_PATH}")

if __name__ == "__main__":
    run()

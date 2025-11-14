# src/data_preprocessing.py
"""
Load raw CSV, basic cleaning, simple imputation, and save processed CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/sales_with_racks_nepal_electronics.csv")
PROCESSED_PATH = Path("data/processed/cleaned_sales_data.csv")

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # 2. Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # 3. Remove exact duplicates
    df = df.drop_duplicates()

    # 4. Fix negative or impossible values
    numeric_cols = ["quantity_sold","unit_price_npr","discount_percent",
                    "revenue_npr","cost_npr","profit_npr","customer_traffic",
                    "inventory_level","rainfall_mm","temperature","petrol_price"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5. Fill or drop missing for critical columns
    df = df.dropna(subset=["date","product_id","store_id","quantity_sold","unit_price_npr"])

    # 6. Fill numeric NaNs with sensible defaults (median)
    for c in numeric_cols:
        if c in df.columns:
            median = df[c].median()
            df[c] = df[c].fillna(median)

    # 7. Remove rows with zero or negative quantity
    df = df[df["quantity_sold"] > 0]

    return df

def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to {path}")

def run():
    df = load_raw()
    df = basic_cleaning(df)
    save_processed(df)

if __name__ == "__main__":
    run()

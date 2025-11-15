import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import json

# --- Tab imports ---
from forecast_tab import render_forecast_tab
from comparision_tab import render_comparison_tab
from product_tab import render_product_tab
from explorer_tab import render_explorer_tab

# ---------------------------------------------
# Paths
# ---------------------------------------------
DATA_PATH = Path("data/processed/featurized_sales_data.csv")
RULES_PATH = Path("models/apriori_rules.pkl")
RULES_CSV = Path("data/processed/association_rules.csv")

# Forecast paths
FORECAST_CSV = Path("data/processed/prophet_forecast.csv")
XGB_FORECAST_CSV = Path("data/processed/xgboost_forecast.csv")
LGBM_FORECAST_CSV = Path("data/processed/lightgbm_forecast.csv")
ENSEMBLE_FORECAST_CSV = Path("data/processed/ensemble_forecast.csv")

# Metadata paths
PROPHET_METADATA = Path("models/prophet_metadata.json")
XGBOOST_METADATA = Path("models/xgboost_metadata.json")
LIGHTGBM_METADATA = Path("models/lightgbm_metadata.json")
ENSEMBLE_METADATA = Path("models/ensemble_metadata.json")

MODEL_COMPARISON = Path("data/processed/model_comparison.csv")

# Product recommendation paths
RECOMMENDATION_CSV = Path("data/processed/product_recommendations.csv")
PLACEMENT_CSV = Path("data/processed/placement_suggestions.csv")

FORECAST_DAYS = 90

# ---------------------------------------------
# Streamlit Config
# ---------------------------------------------
st.set_page_config(
    page_title="Nepal Retail Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------------------------------------
# Cached Loaders
# ---------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_data
def load_rules():
    if RULES_CSV.exists():
        return pd.read_csv(RULES_CSV)
    if RULES_PATH.exists():
        return joblib.load(RULES_PATH)
    return None

@st.cache_data
def load_product_recommendations():
    if RECOMMENDATION_CSV.exists():
        return pd.read_csv(RECOMMENDATION_CSV)
    return pd.DataFrame()

@st.cache_data
def load_placement_suggestions():
    if PLACEMENT_CSV.exists():
        return pd.read_csv(PLACEMENT_CSV)
    return pd.DataFrame()

@st.cache_data
def load_forecast(model_type="Prophet"):
    path_map = {
        "Prophet": FORECAST_CSV,
        "XGBoost": XGB_FORECAST_CSV,
        "LightGBM": LGBM_FORECAST_CSV,
        "Ensemble": ENSEMBLE_FORECAST_CSV
    }
    path = path_map.get(model_type)
    if path and path.exists():
        return pd.read_csv(path, parse_dates=["ds"])
    return None

@st.cache_data
def load_metadata(model_type="Prophet"):
    path_map = {
        "Prophet": PROPHET_METADATA,
        "XGBoost": XGBOOST_METADATA,
        "LightGBM": LIGHTGBM_METADATA,
        "Ensemble": ENSEMBLE_METADATA
    }
    path = path_map.get(model_type)
    if path and path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_comparison():
    if MODEL_COMPARISON.exists():
        return pd.read_csv(MODEL_COMPARISON)
    return None

# ---------------------------------------------
# Main Dashboard UI
# ---------------------------------------------
def main():
    df = load_data()
    rules = load_rules()

    # NEW: Load product recommendation datasets
    reco_df = load_product_recommendations()
    placement_df = load_placement_suggestions()

    # Sidebar
    st.sidebar.title("Dashboard Filters")

    product_list = ["All Products"] + sorted(df["category"].unique().tolist())
    selected_product = st.sidebar.selectbox("Select Product:", product_list)

    st.sidebar.markdown("---")

    api_key = st.sidebar.text_input("Enter Google AI API Key:", type="password")
    st.sidebar.markdown("---")

    # Filter data
    df_filtered = df if selected_product == "All Products" else df[df["category"] == selected_product]

    # Title
    st.title("📊 Nepal Retail Analytics Dashboard")
    st.write("""
    Explore forecasting, market basket associations, recommendations, and data insights.
    """)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Sales Forecasting",
        "📊 Model Comparison",
        "🛒 Product Associations",
        "📂 Data Explorer & KPIs"
    ])

    # TAB 1
    with tab1:
        render_forecast_tab(
            df_filtered,
            selected_product,
            api_key,
            load_forecast,
            load_metadata,
            FORECAST_DAYS
        )

    # TAB 2
    with tab2:
        render_comparison_tab(load_comparison, load_metadata)

    # TAB 3
    with tab3:
        # FIXED: pass rules + reco_df + placement_df
        render_product_tab(rules, reco_df, placement_df)

    # TAB 4
    with tab4:
        render_explorer_tab(df_filtered, selected_product)

# ---------------------------------------------
if __name__ == "__main__":
    main()

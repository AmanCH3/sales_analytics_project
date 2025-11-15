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

# Product recommendation paths - UPDATED
RECOMMENDATION_CSV = Path("data/processed/product_category_recommendations.csv")
PLACEMENT_CSV = Path("data/processed/rack_aisle_suggestions.csv")
CATEGORY_SUMMARY_CSV = Path("data/processed/category_summary.csv")

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
# Cached Loaders with Debugging
# ---------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_data
def load_rules():
    if RULES_CSV.exists():
        df = pd.read_csv(RULES_CSV)
        st.sidebar.success(f"✅ Loaded {len(df)} association rules")
        return df
    if RULES_PATH.exists():
        df = joblib.load(RULES_PATH)
        st.sidebar.success(f"✅ Loaded {len(df)} association rules")
        return df
    st.sidebar.error("❌ No association rules found")
    return None

@st.cache_data
def load_product_recommendations():
    """Load product-level recommendations with category info"""
    if RECOMMENDATION_CSV.exists():
        df = pd.read_csv(RECOMMENDATION_CSV)
        st.sidebar.success(f"✅ Loaded {len(df)} product recommendations")
        # Debug: Show columns
        print(f"Product Reco Columns: {df.columns.tolist()}")
        return df
    st.sidebar.error(f"❌ File not found: {RECOMMENDATION_CSV}")
    return pd.DataFrame()

@st.cache_data
def load_placement_suggestions():
    """Load placement suggestions"""
    if PLACEMENT_CSV.exists():
        df = pd.read_csv(PLACEMENT_CSV)
        st.sidebar.success(f"✅ Loaded {len(df)} placement suggestions")
        return df
    st.sidebar.error(f"❌ File not found: {PLACEMENT_CSV}")
    return pd.DataFrame()

@st.cache_data
def load_category_summary():
    """Load category-to-category summary"""
    if CATEGORY_SUMMARY_CSV.exists():
        df = pd.read_csv(CATEGORY_SUMMARY_CSV)
        st.sidebar.success(f"✅ Loaded {len(df)} category pairs")
        # Debug: Show columns and sample data
        print(f"Category Summary Columns: {df.columns.tolist()}")
        if len(df) > 0:
            print(f"Lift range: {df['avg_lift'].min():.3f} to {df['avg_lift'].max():.3f}")
            print(f"Sample data:\n{df.head(3)}")
        return df
    st.sidebar.error(f"❌ File not found: {CATEGORY_SUMMARY_CSV}")
    return pd.DataFrame()

@st.cache_data
def build_hierarchical_data(product_reco_df):
    """Build hierarchical dict: category_pair -> product recommendations"""
    if len(product_reco_df) == 0:
        return {}
    
    # Check if required columns exist
    required_cols = ['product', 'recommended_product', 'product_category', 'recommended_category', 'lift', 'confidence', 'support']
    missing_cols = [col for col in required_cols if col not in product_reco_df.columns]
    
    if missing_cols:
        print(f"⚠️ Missing columns in product recommendations: {missing_cols}")
        print(f"Available columns: {product_reco_df.columns.tolist()}")
        return {}
    
    hierarchical = {}
    
    # Group by category pairs
    for (cat_from, cat_to), group in product_reco_df.groupby(['product_category', 'recommended_category']):
        key = f"{cat_from} → {cat_to}"
        hierarchical[key] = group.sort_values('lift', ascending=False).rename(columns={
            'product': 'product_from',
            'recommended_product': 'product_to',
            'product_category': 'category_from',
            'recommended_category': 'category_to'
        })
    
    print(f"✅ Created {len(hierarchical)} category pairs in hierarchical data")
    st.sidebar.info(f"📊 {len(hierarchical)} category pairs available")
    
    return hierarchical

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
    
    # Sidebar
    st.sidebar.title("Dashboard Filters")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Data Status")
    
    # Load all data with status indicators
    rules = load_rules()
    reco_df = load_product_recommendations()
    placement_df = load_placement_suggestions()
    category_summary = load_category_summary()
    
    # Build hierarchical structure
    hierarchical_data = build_hierarchical_data(reco_df)
    
    st.sidebar.markdown("---")

    product_list = ["All Products"] + sorted(df["category"].unique().tolist())
    selected_product = st.sidebar.selectbox("Select Product:", product_list)

    st.sidebar.markdown("---")

    api_key = st.sidebar.text_input("Enter Google AI API Key:", type="password")

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

    # TAB 3 - FIXED: Pass all required parameters
    with tab3:
        # Debug info at top
        if len(category_summary) > 0:
            st.info(f"📊 Loaded: {len(category_summary)} category pairs, {len(reco_df)} product recommendations, {len(placement_df)} placements")
        
        render_product_tab(
            rules_df=rules,
            reco_df=reco_df,
            placement_df=placement_df,
            category_recs=category_summary,
            hierarchical_data=hierarchical_data
        )

    # TAB 4
    with tab4:
        render_explorer_tab(df_filtered, selected_product)

# ---------------------------------------------
if __name__ == "__main__":
    main()
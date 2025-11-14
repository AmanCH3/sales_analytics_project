import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import plotly.express as px

# ---------------------------------------------
# Paths
# ---------------------------------------------
DATA_PATH = Path("data/processed/featurized_sales_data.csv")
FORECAST_CSV = Path("data/processed/prophet_forecast.csv")
RULES_PATH = Path("models/apriori_rules.pkl")
RULES_CSV = Path("data/processed/association_rules.csv")

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
def load_forecast():
    if FORECAST_CSV.exists():
        return pd.read_csv(FORECAST_CSV, parse_dates=["ds"])
    return None

@st.cache_data
def load_rules():
    if RULES_CSV.exists():
        return pd.read_csv(RULES_CSV)
    if RULES_PATH.exists():
        return joblib.load(RULES_PATH)
    return None

# ---------------------------------------------
# Dashboard UI
# ---------------------------------------------
def main():

    # Title
    st.title("📊 Nepal Retail Analytics Dashboard")
    st.write("""
    Welcome to your **Retail Sales Forecasting & Product Placement Dashboard**.  
    This dashboard helps decision-makers quickly understand:
    - 📈 Sales trends  
    - 🔮 Forecasted performance  
    - 🛒 Product associations for better placements  
    - 💡 Actionable insights powered by machine learning  
    """)

    # ---------------------------------------------
    # Load main dataset
    # ---------------------------------------------
    df = load_data()

    st.subheader("📂 Dataset Snapshot")
    st.write("Below is the first few rows of your cleaned and feature-engineered sales dataset.")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------
    # KPIs Section with Insights
    # ---------------------------------------------
    st.subheader("📌 Key Business Metrics (KPIs)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"{df['revenue_npr'].sum():,.0f} NPR")
    col2.metric("Avg Daily Revenue", f"{df.groupby('date')['revenue_npr'].sum().mean():,.0f} NPR")
    col3.metric("Total Transactions", df["transaction_id"].nunique())
    col4.metric("Unique Products", df["product_id"].nunique())

    st.info("""
    **Insight:**  
    These KPIs give you an instant understanding of business scale, customer activity, and product diversity.  
    Useful for evaluating store performance and planning inventory or promotions.
    """)

    st.markdown("---")

    # ---------------------------------------------
    # Forecast Section
    # ---------------------------------------------
    st.subheader("🔮 Sales Forecast (Prophet Model)")

    forecast = load_forecast()
    if forecast is not None:
        fig = px.line(
            forecast,
            x="ds",
            y=["yhat", "yhat_lower", "yhat_upper"],
            labels={"ds": "Date", "value": "Forecasted Sales"},
            title="30-Day Sales Forecast"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success("""
        **Interpretation:**  
        - `yhat` = central forecast  
        - `yhat_lower` and `yhat_upper` = uncertainty range  
        - Use this to plan **inventory**, **staffing**, and **discount strategies**.  
        """)
    else:
        st.warning("⚠ No forecast found. Run `src/forecasting_model.py` to generate forecasts.")

    st.markdown("---")

    # ---------------------------------------------
    # Apriori Rules (Product Placement Insights)
    # ---------------------------------------------
    st.subheader("🛒 Product Placement Recommendations (Apriori Rules)")

    rules = load_rules()

    if rules is not None:
        st.write("""
        Products that frequently appear together in customer baskets help improve **store layout**,  
        **cross-selling**, and **promotion combos**.
        """)

        if isinstance(rules, pd.DataFrame):
            st.dataframe(
                rules.sort_values("confidence", ascending=False).head(30),
                use_container_width=True
            )
        else:
            try:
                st.dataframe(pd.DataFrame(rules).head(30))
            except:
                st.write(rules)

        st.success("""
        **How to use:**  
        - Place products with high *support* close to each other.  
        - Promote high-confidence pairs as bundle offers.  
        - Use high-lift rules for premium cross-sell opportunities.  
        """)
    else:
        st.warning("⚠ No Apriori rules found. Run `src/apriori_model.py` to generate rules.")

    st.markdown("---")

    st.caption("Built with ❤️ using Streamlit, Prophet, and Apriori — optimized for Nepal retail analytics.")

# ---------------------------------------------
if __name__ == "__main__":
    main()

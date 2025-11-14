import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import plotly.express as px

from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini

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

# Define forecast period (used in UI text)
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
    """Loads the main featurized dataset."""
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_data
def load_rules():
    """Loads the pre-calculated apriori rules."""
    if RULES_CSV.exists():
        return pd.read_csv(RULES_CSV)
    if RULES_PATH.exists():
        return joblib.load(RULES_PATH)
    return None

@st.cache_data
def load_forecast(model_type="Prophet"):
    """Loads the specified pre-calculated forecast CSV."""
    if model_type == "Prophet" and FORECAST_CSV.exists():
        return pd.read_csv(FORECAST_CSV, parse_dates=["ds"])
    if model_type == "XGBoost" and XGB_FORECAST_CSV.exists():
        return pd.read_csv(XGB_FORECAST_CSV, parse_dates=["ds"])
    if model_type == "LightGBM" and LGBM_FORECAST_CSV.exists():
        return pd.read_csv(LGBM_FORECAST_CSV, parse_dates=["ds"])
    return None

# ---------------------------------------------
# Dashboard UI
# ---------------------------------------------
def main():

    # --- Main Data Loading ---
    df = load_data()

    # --- Sidebar for Filters ---
    st.sidebar.title("Dashboard Filters")
    st.sidebar.markdown("Use the filters below to drill down into the data.")

    # Product Filter
    product_list = ["All Products"] + sorted(df["category"].unique().tolist())
    selected_product = st.sidebar.selectbox(
        "Select Product:",
        product_list,
        help="Select a specific product to update KPIs and the 'Actuals' line on the forecast chart."
    )
    
    st.sidebar.markdown("---")
    
    # --- NEW: API Key Input ---
    st.sidebar.title("🤖 AI Configuration")
    api_key = st.sidebar.text_input(
        "Enter Google AI API Key:",
        type="password",
        help="Get your key from Google AI Studio."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with ❤️ using Streamlit, Prophet, XGBoost, LightGBM, and Apriori.")


    # --- Filter Data Based on Selection ---
    if selected_product == "All Products":
        df_filtered = df
    else:
        df_filtered = df[df["product_id"] == selected_product].copy()


    # --- Main Page Title ---
    st.title("📊 Nepal Retail Analytics Dashboard")
    st.write("""
    Welcome to your **Retail Sales Forecasting & Product Placement Dashboard**.  
    Use the tabs below to explore sales forecasts, product associations, and raw data.
    """)

    # --- Main Tabbed Interface ---
    tab1, tab2, tab3 = st.tabs(["📈 Sales Forecasting", "🛒 Product Associations", "📂 Data Explorer & KPIs"])

    # --- TAB 1: Sales Forecasting ---
    with tab1:
        st.subheader("🔮 Sales Forecast Comparison")
        st.write(f"Displaying forecasts for: **{selected_product}**")
        
        model_choice = st.selectbox(
            "Choose a forecast model:",
            ("Prophet", "XGBoost", "LightGBM"),
            help="Select the model to view its forecast."
        )

        forecast = load_forecast(model_choice)
        
        if forecast is not None:
            # Aggregate the FILTERED data for actuals
            df_agg = df_filtered.groupby('date')['revenue_npr'].sum().reset_index().rename(columns={"date":"ds", "revenue_npr":"y"})
            
            # Merge forecast (global) with actuals (filtered or global)
            plot_df = forecast.merge(df_agg, on="ds", how="left")
            
            y_columns = ["yhat"]
            if model_choice == "Prophet":
                y_columns.extend(["yhat_lower", "yhat_upper"])
            
            if 'y' in plot_df.columns:
                plot_df['y'] = plot_df['y'].fillna(0) # Fill NaNs for AI analysis
                y_columns.append('y')
            
            # Clean up df for the AI
            plot_df_cleaned = plot_df[y_columns + ['ds']].copy()
            plot_df_cleaned.columns = plot_df_cleaned.columns.str.replace('_', ' ')


            fig = px.line(
                plot_df,
                x="ds",
                y=y_columns,
                labels={"ds": "Date", "value": "Sales (NPR)"},
                title=f"{model_choice} Forecast vs. Actual Sales for {selected_product}"
            )
            
            # Style the lines
            fig.for_each_trace(lambda t: t.update(name="Actuals (Filtered)") if t.name == "y" else t)
            fig.for_each_trace(lambda t: t.update(line=dict(dash='dash', color='rgba(255,255,255,0.5)')) if t.name in ["yhat_lower", "yhat_upper"] else t)
            fig.for_each_trace(lambda t: t.update(name="Forecast") if t.name == "yhat" else t)
            fig.for_each_trace(lambda t: t.update(line=dict(color='red', width=3)) if t.name == "Actuals (Filtered)" else t)

            st.plotly_chart(fig, use_container_width=True)

            # --- NEW: AI Explanation Button ---
            if st.button(f"🤖 Explain this {model_choice} Chart", use_container_width=True):
                if not api_key:
                    st.error("Please enter your Google AI API Key in the sidebar to use this feature.")
                else:
                    try:
                        with st.spinner("🧠 AI is analyzing the chart, please wait..."):
                            # Initialize the LLM
                            llm = GoogleGemini(api_key=api_key)
                            
                            # Create the SmartDataframe
                            sdf = SmartDataframe(plot_df_cleaned, config={"llm": llm})
                            
                            # Create a dynamic prompt
                            prompt = f"""
                            Analyze the provided time-series data. 
                            - 'ds' is the date.
                            - 'yhat' is the {model_choice} model's sales forecast.
                            - 'y' is the actual sales for {selected_product}.
                            - {'yhat lower and yhat upper are the uncertainty interval for the Prophet forecast.' if model_choice == 'Prophet' else ''}

                            Provide a brief, insightful analysis in 3 bullet points:
                            1.  What is the main trend shown in the forecast ('yhat')?
                            2.  How well does the forecast ('yhat') match the actual sales ('y')? Point out any major differences.
                            3.  Identify one key insight or potential anomaly a manager should look at.
                            """
                            
                            response = sdf.chat(prompt)
                            st.info(f"**🤖 AI Analysis:**\n\n{response}")

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
            
            # --- (Existing interpretation boxes) ---
            if selected_product == "All Products":
                st.success(f"""
                **Interpretation ({model_choice}):** - `yhat` = {FORECAST_DAYS}-day future forecast for **All Products**.  
                - `y` = Actual sales for **All Products**.
                - `yhat_lower/upper` (if shown) = Prophet's uncertainty range.
                """)
            else:
                st.info(f"""
                **Note:** You are viewing the **global** forecast for all products (`yhat`) overlaid with the **filtered** actual sales for `{selected_product}` (`Actuals (Filtered)`). 
                This helps you see how one product's sales contribute to the overall trend.
                """)
        else:
            st.warning(f"⚠ No forecast found for {model_choice}. Run `src/{model_choice.lower()}_model.py` to generate forecasts.")

    # --- TAB 2: Product Placement ---
    with tab2:
        st.subheader("🛒 Product Placement Recommendations (Apriori Rules)")
        
        rules = load_rules()

        if rules is not None:
            st.info("ℹ️ These association rules are calculated across **all** transactions and are not affected by the sidebar filter.")
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
            **How to use:** - Place products with high *support* close to each other.  
            - Promote high-confidence pairs as bundle offers.  
            - Use high-lift rules for premium cross-sell opportunities.  
            """)
        else:
            st.warning("⚠ No Apriori rules found. Run `src/apriori_model.py` to generate rules.")

    # --- TAB 3: Data Explorer & KPIs ---
    with tab3:
        st.subheader(f"📌 Key Business Metrics for: {selected_product}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"{df_filtered['revenue_npr'].sum():,.0f} NPR")
        col2.metric("Avg Daily Revenue", f"{df_filtered.groupby('date')['revenue_npr'].sum().mean():,.0f} NPR")
        col3.metric("Total Transactions", df_filtered["transaction_id"].nunique())
        col4.metric("Unique Products", df_filtered["product_id"].nunique())

        st.info("""
        **Insight:** These KPIs reflect the data selected in the sidebar. 
        Use this to evaluate performance for individual products or the entire store.
        """)
        
        st.markdown("---")
        
        st.subheader(f"📂 Dataset Snapshot for: {selected_product}")
        st.write("Below is the first few rows of the sales dataset, filtered by your selection.")
        st.dataframe(df_filtered.head(100), use_container_width=True, height=300)

# ---------------------------------------------
if __name__ == "__main__":
    main()
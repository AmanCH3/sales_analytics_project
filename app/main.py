# app/main_improved.py
import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json

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
ENSEMBLE_FORECAST_CSV = Path("data/processed/ensemble_forecast.csv")

# Metadata paths
PROPHET_METADATA = Path("models/prophet_metadata.json")
XGBOOST_METADATA = Path("models/xgboost_metadata.json")
LIGHTGBM_METADATA = Path("models/lightgbm_metadata.json")
ENSEMBLE_METADATA = Path("models/ensemble_metadata.json")
MODEL_COMPARISON = Path("data/processed/model_comparison.csv")

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
    """Load model metadata"""
    path_map = {
        "Prophet": PROPHET_METADATA,
        "XGBoost": XGBOOST_METADATA,
        "LightGBM": LIGHTGBM_METADATA,
        "Ensemble": ENSEMBLE_METADATA
    }
    
    path = path_map.get(model_type)
    if path and path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_comparison():
    """Load model comparison data"""
    if MODEL_COMPARISON.exists():
        return pd.read_csv(MODEL_COMPARISON)
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
    
    # --- API Key Input ---
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
    Use the tabs below to explore sales forecasts, model comparison, product associations, and raw data.
    """)

    # --- Main Tabbed Interface ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Sales Forecasting", 
        "📊 Model Comparison",
        "🛒 Product Associations", 
        "📂 Data Explorer & KPIs"
    ])

    # --- TAB 1: Sales Forecasting ---
    with tab1:
        st.subheader("🔮 Sales Forecast Comparison")
        st.write(f"Displaying forecasts for: **{selected_product}**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            model_choice = st.selectbox(
                "Choose a forecast model:",
                ("Ensemble", "Prophet", "XGBoost", "LightGBM"),
                help="Select the model to view its forecast. Ensemble combines all three models."
            )
        
        with col2:
            show_individual = st.checkbox(
                "Show all models",
                value=False,
                help="Compare all models on the same chart"
            )

        forecast = load_forecast(model_choice)
        
        if forecast is not None:
            # Aggregate the FILTERED data for actuals
            df_agg = df_filtered.groupby('date')['revenue_npr'].sum().reset_index().rename(
                columns={"date": "ds", "revenue_npr": "y"}
            )
            
            # Ensure ds column is datetime for proper merging
            df_agg['ds'] = pd.to_datetime(df_agg['ds'])
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            
            if show_individual:
                # Load all forecasts for comparison
                all_forecasts = {}
                for model in ["Prophet", "XGBoost", "LightGBM"]:
                    fc = load_forecast(model)
                    if fc is not None:
                        fc['ds'] = pd.to_datetime(fc['ds'])
                        all_forecasts[model] = fc
                
                # Ensure actuals have proper datetime
                df_agg['ds'] = pd.to_datetime(df_agg['ds'])
                
                # Create comparison chart
                fig = go.Figure()
                
                # Add actual sales - ONLY historical data (not future dates)
                # Filter actuals to only show dates that exist in the data
                actuals_to_plot = df_agg.copy()
                
                fig.add_trace(go.Scatter(
                    x=actuals_to_plot['ds'],
                    y=actuals_to_plot['y'],
                    mode='lines',
                    name='Actual Sales',
                    line=dict(color='red', width=3)
                ))
                
                # Add all model predictions
                colors = {'Prophet': 'blue', 'XGBoost': 'green', 'LightGBM': 'orange'}
                for model_name, fc_df in all_forecasts.items():
                    fig.add_trace(go.Scatter(
                        x=fc_df['ds'],
                        y=fc_df['yhat'],
                        mode='lines',
                        name=f'{model_name} Forecast',
                        line=dict(color=colors[model_name], width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"All Models Comparison for {selected_product}",
                    xaxis_title="Date",
                    yaxis_title="Sales (NPR)",
                    hovermode='x unified',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Single model view
                # Ensure proper datetime types
                forecast['ds'] = pd.to_datetime(forecast['ds'])
                df_agg['ds'] = pd.to_datetime(df_agg['ds'])
                
                # Get the last date of actual data
                last_actual_date = df_agg['ds'].max()
                
                # Split forecast into historical and future
                forecast_historical = forecast[forecast['ds'] <= last_actual_date].copy()
                forecast_future = forecast[forecast['ds'] > last_actual_date].copy()
                
                # Merge only historical forecast with actuals
                plot_df = forecast.copy()
                plot_df = plot_df.merge(df_agg, on="ds", how="left")
                
                # Create figure
                fig = go.Figure()
                
                # Add actual sales (red line - only historical)
                fig.add_trace(go.Scatter(
                    x=df_agg['ds'],
                    y=df_agg['y'],
                    mode='lines',
                    name='Actuals (Filtered)',
                    line=dict(color='red', width=3)
                ))
                
                # Add forecast (blue line)
                fig.add_trace(go.Scatter(
                    x=plot_df['ds'],
                    y=plot_df['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='rgb(0, 145, 255)', width=2)
                ))
                
                # Add confidence intervals if available
                if model_choice in ["Prophet", "Ensemble"]:
                    # Upper bound
                    fig.add_trace(go.Scatter(
                        x=plot_df['ds'],
                        y=plot_df['yhat_upper'],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(color='rgba(255,255,255,0)', width=0),
                        showlegend=False
                    ))
                    
                    # Lower bound with fill
                    fig.add_trace(go.Scatter(
                        x=plot_df['ds'],
                        y=plot_df['yhat_lower'],
                        mode='lines',
                        name='Confidence Interval',
                        line=dict(color='rgba(255,255,255,0)', width=0),
                        fillcolor='rgba(0, 145, 255, 0.2)',
                        fill='tonexty'
                    ))
                
                fig.update_layout(
                    title=f"{model_choice} Forecast vs. Actual Sales for {selected_product}",
                    xaxis_title="Date",
                    yaxis_title="Sales (NPR)",
                    hovermode='x unified',
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                # Prepare data for AI (simplified version)
                y_columns = ["yhat"]
                if 'y' in plot_df.columns:
                    y_columns.append('y')
                
                plot_df_cleaned = plot_df[y_columns + ['ds']].copy()
                plot_df_cleaned.columns = plot_df_cleaned.columns.str.replace('_', ' ')

                # --- AI Explanation Button ---
                if st.button(f"🤖 Explain this {model_choice} Chart", use_container_width=True):
                    if not api_key:
                        st.error("Please enter your Google AI API Key in the sidebar to use this feature.")
                    else:
                        try:
                            with st.spinner("🧠 AI is analyzing the chart, please wait..."):
                                llm = GoogleGemini(api_key=api_key)
                                sdf = SmartDataframe(plot_df_cleaned, config={"llm": llm})
                                
                                prompt = f"""
                                Analyze the provided time-series data. 
                                - 'ds' is the date.
                                - 'yhat' is the {model_choice} model's sales forecast.
                                - 'y' is the actual sales for {selected_product}.
                                
                                Provide a brief, insightful analysis in 3 bullet points:
                                1. What is the main trend shown in the forecast ('yhat')?
                                2. How well does the forecast ('yhat') match the actual sales ('y')? 
                                3. Identify one key insight or potential anomaly a manager should look at.
                                """
                                
                                response = sdf.chat(prompt)
                                st.info(f"**🤖 AI Analysis:**\n\n{response}")

                        except Exception as e:
                            st.error(f"An error occurred during analysis: {e}")
            
            # Display model metadata
            metadata = load_metadata(model_choice)
            if metadata:
                with st.expander(f"📋 {model_choice} Model Details"):
                    col1, col2, col3 = st.columns(3)
                    
                    test_metrics = metadata.get('test_metrics', {})
                    if test_metrics:
                        with col1:
                            st.metric("MAE (Test)", f"{test_metrics.get('MAE', 0):,.0f} NPR")
                        with col2:
                            st.metric("RMSE (Test)", f"{test_metrics.get('RMSE', 0):,.0f} NPR")
                        with col3:
                            st.metric("R² Score", f"{test_metrics.get('R2', 0):.4f}")
                    
                    # Show ensemble weights if applicable
                    if model_choice == "Ensemble":
                        weights = metadata.get('weights', {})
                        if weights:
                            st.write("**Ensemble Weights:**")
                            weight_df = pd.DataFrame([
                                {"Model": k, "Weight": f"{v:.2%}"} 
                                for k, v in weights.items()
                            ])
                            st.dataframe(weight_df, use_container_width=True, hide_index=True)
                    
                    st.caption(f"Last trained: {metadata.get('timestamp', 'Unknown')}")
            
            # Interpretation
            if selected_product == "All Products":
                st.success(f"""
                **Interpretation ({model_choice}):**
                - `yhat` = {FORECAST_DAYS}-day future forecast for **All Products**
                - `y` = Actual sales for **All Products**
                - `yhat_lower/upper` (if shown) = Uncertainty range
                """)
            else:
                st.info(f"""
                **Note:** You are viewing the **global** forecast overlaid with **filtered** actual sales 
                for `{selected_product}`. This helps see how one product contributes to the overall trend.
                """)
        else:
            st.warning(f"⚠ No forecast found for {model_choice}. Run the corresponding model script to generate forecasts.")

    # --- TAB 2: Model Comparison ---
    with tab2:
        st.subheader("📊 Model Performance Comparison")
        
        comparison_df = load_comparison()
        
        if comparison_df is not None:
            st.write("""
            Compare the performance of all forecasting models on the test set. 
            Lower values for MAE, RMSE, and MAPE indicate better performance.
            """)
            
            # Display metrics table
            st.dataframe(
                comparison_df.style.highlight_min(
                    subset=['MAE', 'RMSE', 'MAPE'],
                    color='lightgreen'
                ),
                use_container_width=True,
                hide_index=True
            )
            
            # Best model
            best_model = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']
            st.success(f"🏆 **Best performing model (by MAE):** {best_model}")
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig_mae = px.bar(
                    comparison_df,
                    x='Model',
                    y='MAE',
                    title='Mean Absolute Error (MAE) Comparison',
                    color='MAE',
                    color_continuous_scale='RdYlGn_r'
                )
                fig_mae.update_layout(showlegend=False)
                st.plotly_chart(fig_mae, use_container_width=True)
            
            with col2:
                fig_mape = px.bar(
                    comparison_df,
                    x='Model',
                    y='MAPE',
                    title='Mean Absolute Percentage Error (MAPE) Comparison',
                    color='MAPE',
                    color_continuous_scale='RdYlGn_r'
                )
                fig_mape.update_layout(showlegend=False)
                st.plotly_chart(fig_mape, use_container_width=True)
            
            # Feature importance comparison
            st.markdown("---")
            st.subheader("📈 Feature Importance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                xgb_meta = load_metadata("XGBoost")
                if xgb_meta and 'top_10_features' in xgb_meta:
                    st.write("**XGBoost Top Features:**")
                    feat_df = pd.DataFrame([
                        {"Feature": k, "Importance": v} 
                        for k, v in xgb_meta['top_10_features'].items()
                    ])
                    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                lgbm_meta = load_metadata("LightGBM")
                if lgbm_meta and 'top_10_features' in lgbm_meta:
                    st.write("**LightGBM Top Features:**")
                    feat_df = pd.DataFrame([
                        {"Feature": k, "Importance": v} 
                        for k, v in lgbm_meta['top_10_features'].items()
                    ])
                    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠ No comparison data found. Run `src/ensemble_model.py` to generate model comparison.")
            st.info("Make sure all three models (Prophet, XGBoost, LightGBM) have been trained first.")

    # --- TAB 3: Product Placement ---
    with tab3:
        st.subheader("🛒 Product Placement Recommendations (Apriori Rules)")
        
        rules = load_rules()

        if rules is not None:
            st.info("ℹ️ These association rules are calculated across **all** transactions and are not affected by the sidebar filter.")
            st.write("""
            Products that frequently appear together in customer baskets help improve **store layout**,  
            **cross-selling**, and **promotion combos**.
            """)

            if isinstance(rules, pd.DataFrame):
                # Add filters for rules
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_support = st.slider("Minimum Support", 0.0, 1.0, 0.01, 0.01)
                with col2:
                    min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.2, 0.05)
                with col3:
                    min_lift = st.slider("Minimum Lift", 0.0, 5.0, 1.0, 0.1)
                
                # Filter rules
                filtered_rules = rules[
                    (rules['support'] >= min_support) &
                    (rules['confidence'] >= min_confidence) &
                    (rules['lift'] >= min_lift)
                ].sort_values("confidence", ascending=False)
                
                st.write(f"Showing {len(filtered_rules)} rules (out of {len(rules)} total)")
                
                st.dataframe(
                    filtered_rules.head(50),
                    use_container_width=True
                )
                
                # Visualizations
                if len(filtered_rules) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.scatter(
                            filtered_rules.head(100),
                            x='support',
                            y='confidence',
                            size='lift',
                            hover_data=['antecedents', 'consequents'],
                            title='Support vs Confidence (sized by Lift)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        top_rules = filtered_rules.head(10)
                        fig = px.bar(
                            top_rules,
                            x='confidence',
                            y=top_rules.index.astype(str),
                            orientation='h',
                            title='Top 10 Rules by Confidence'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                try:
                    st.dataframe(pd.DataFrame(rules).head(30))
                except:
                    st.write(rules)

            st.success("""
            **How to use:**
            - Place products with high *support* close to each other
            - Promote high-confidence pairs as bundle offers
            - Use high-lift rules for premium cross-sell opportunities
            """)
        else:
            st.warning("⚠ No Apriori rules found. Run `src/apriori_model.py` to generate rules.")

    # --- TAB 4: Data Explorer & KPIs ---
    with tab4:
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
        
        # Time series of revenue
        st.markdown("---")
        st.subheader("📈 Revenue Trend Over Time")
        
        daily_revenue = df_filtered.groupby('date')['revenue_npr'].sum().reset_index()
        fig = px.line(
            daily_revenue,
            x='date',
            y='revenue_npr',
            title=f'Daily Revenue Trend for {selected_product}'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader(f"📂 Dataset Snapshot for: {selected_product}")
        st.write("Below is the first few rows of the sales dataset, filtered by your selection.")
        st.dataframe(df_filtered.head(100), use_container_width=True, height=300)

# ---------------------------------------------
if __name__ == "__main__":
    main()
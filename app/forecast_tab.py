import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini

def render_forecast_tab(df_filtered, selected_product, api_key, load_forecast, load_metadata, FORECAST_DAYS):
    """
    Renders the Sales Forecasting tab.
    """
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
            
            # Add actual sales
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
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            df_agg['ds'] = pd.to_datetime(df_agg['ds'])
            
            plot_df = forecast.copy()
            plot_df = plot_df.merge(df_agg, on="ds", how="left")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_agg['ds'],
                y=df_agg['y'],
                mode='lines',
                name='Actuals (Filtered)',
                line=dict(color='red', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=plot_df['ds'],
                y=plot_df['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='rgb(0, 145, 255)', width=2)
            ))
            
            if model_choice in ["Prophet", "Ensemble"]:
                fig.add_trace(go.Scatter(
                    x=plot_df['ds'],
                    y=plot_df['yhat_upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='rgba(255,255,255,0)', width=0),
                    showlegend=False
                ))
                
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

            y_columns = ["yhat"]
            if 'y' in plot_df.columns:
                y_columns.append('y')
            
            plot_df_cleaned = plot_df[y_columns + ['ds']].copy()
            plot_df_cleaned.columns = plot_df_cleaned.columns.str.replace('_', ' ')

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
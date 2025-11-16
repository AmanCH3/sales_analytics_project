import streamlit as st
import pandas as pd
import plotly.express as px
import json

def render_comparison_tab(load_comparison, load_metadata):
    """
    Renders the Model Performance Comparison tab.
    """
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
            width="stretch",
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
            st.plotly_chart(fig_mae, width="stretch")
        
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
            st.plotly_chart(fig_mape, width="stretch")
        
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
                st.plotly_chart(fig, width="stretch")
        
        with col2:
            lgbm_meta = load_metadata("LightGBM")
            if lgbm_meta and 'top_10_features' in lgbm_meta:
                st.write("**LightGBM Top Features:**")
                feat_df = pd.DataFrame([
                    {"Feature": k, "Importance": v} 
                    for k, v in lgbm_meta['top_10_features'].items()
                ])
                fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig, width="stretch")
        
    else:
        st.warning("⚠ No comparison data found. Run `src/ensemble_model.py` to generate model comparison.")
        st.info("Make sure all three models (Prophet, XGBoost, LightGBM) have been trained first.")

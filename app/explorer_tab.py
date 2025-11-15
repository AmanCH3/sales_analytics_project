import streamlit as st
import pandas as pd
import plotly.express as px

def render_explorer_tab(df_filtered, selected_product):
    """
    Renders the Data Explorer & KPIs tab.
    """
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
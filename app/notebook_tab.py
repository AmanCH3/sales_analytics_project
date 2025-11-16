import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set plot style
sns.set_theme(style="whitegrid")

def render_notebook_tab(df_full):
    """
    Renders the Notebook Analysis tab with diagrams from the data cleaning
    and apriori analysis notebooks.
    """
    st.subheader("��� Data Analysis Diagrams")
    st.write("Visual analysis from the data cleaning and association rule mining notebooks.")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "��� Revenue Trends",
        "��� Feature Distributions", 
        "��� Correlations",
        "��� Day of Week Analysis",
        "��� Festival Season Impact"
    ])
    
    # ==========================================
    # TAB 1: REVENUE OVER TIME
    # ==========================================
    with tab1:
        st.subheader("Revenue Trend Over Time")
        st.write("Daily total revenue from the entire dataset.")
        
        try:
            # Calculate daily revenue
            df_daily_revenue = df_full.groupby('date')['revenue_npr'].sum().reset_index()
            df_daily_revenue = df_daily_revenue.sort_values('date')
            
            # Create plotly chart
            fig = px.line(
                df_daily_revenue,
                x='date',
                y='revenue_npr',
                title='Total Daily Revenue Over Time',
                labels={'revenue_npr': 'Revenue (NPR)', 'date': 'Date'},
                height=500
            )
            
            fig.update_layout(
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Revenue", f"₨{df_daily_revenue['revenue_npr'].sum():,.0f}")
            with col2:
                st.metric("Avg Daily Revenue", f"₨{df_daily_revenue['revenue_npr'].mean():,.0f}")
            with col3:
                st.metric("Max Daily Revenue", f"₨{df_daily_revenue['revenue_npr'].max():,.0f}")
            with col4:
                st.metric("Min Daily Revenue", f"₨{df_daily_revenue['revenue_npr'].min():,.0f}")
        
        except Exception as e:
            st.error(f"Error creating revenue trend chart: {e}")
    
    # ==========================================
    # TAB 2: FEATURE DISTRIBUTIONS
    # ==========================================
    with tab2:
        st.subheader("Distributions of Key Numeric Features")
        st.write("Histogram distributions for important features in the dataset.")
        
        try:
            numeric_cols = [
                "quantity_sold", 
                "unit_price_npr", 
                "revenue_npr", 
                "customer_traffic", 
                "temperature", 
                "rainfall_mm"
            ]
            
            # Filter to columns that exist
            available_cols = [col for col in numeric_cols if col in df_full.columns]
            
            if available_cols:
                # Let user select which columns to display
                selected_cols = st.multiselect(
                    "Select features to display:",
                    available_cols,
                    default=available_cols[:3],
                    key="feature_selection"
                )
                
                if selected_cols:
                    # Create subplots
                    num_cols = len(selected_cols)
                    ncols = 3
                    nrows = (num_cols + ncols - 1) // ncols
                    
                    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
                    if nrows == 1 and ncols > 1:
                        axes = axes.flatten()
                    elif isinstance(axes, np.ndarray):
                        axes = axes.flatten()
                    else:
                        axes = [axes]
                    
                    for idx, col in enumerate(selected_cols):
                        data = df_full[col].dropna()
                        axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                        axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                        axes[idx].set_xlabel(col)
                        axes[idx].set_ylabel('Frequency')
                        axes[idx].grid(True, alpha=0.3)
                    
                    # Hide unused subplots
                    for idx in range(len(selected_cols), len(axes)):
                        axes[idx].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistics table
                    with st.expander("��� Detailed Statistics"):
                        stats_data = df_full[selected_cols].describe().T
                        st.dataframe(
                            stats_data.style.background_gradient(cmap='RdYlGn'),
                            use_container_width=True
                        )
            else:
                st.warning("No numeric columns found in the dataset.")
        
        except Exception as e:
            st.error(f"Error creating distribution charts: {e}")
    
    # ==========================================
    # TAB 3: CORRELATION HEATMAP
    # ==========================================
    with tab3:
        st.subheader("Correlation Heatmap of Numeric Features")
        st.write("Shows relationships between numeric features (1.0 = perfect positive correlation, -1.0 = perfect negative).")
        
        try:
            numeric_cols = [
                "quantity_sold", 
                "unit_price_npr", 
                "revenue_npr", 
                "customer_traffic", 
                "temperature", 
                "rainfall_mm",
                "discount_percent",
                "profit_npr",
                "inventory_level"
            ]
            
            available_cols = [col for col in numeric_cols if col in df_full.columns]
            
            if available_cols:
                df_corr = df_full[available_cols].corr()
                
                # Create correlation heatmap
                fig = plt.figure(figsize=(12, 8))
                sns.heatmap(
                    df_corr, 
                    annot=True, 
                    fmt=".2f", 
                    cmap="coolwarm", 
                    linewidths=0.5,
                    center=0,
                    cbar_kws={'label': 'Correlation Coefficient'}
                )
                plt.title('Correlation Heatmap of Numeric Features', fontweight='bold', fontsize=14)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Find strongest correlations
                with st.expander("��� Strongest Correlations"):
                    # Get upper triangle of correlation matrix
                    corr_pairs = []
                    for i in range(len(df_corr.columns)):
                        for j in range(i + 1, len(df_corr.columns)):
                            corr_pairs.append({
                                'Feature 1': df_corr.columns[i],
                                'Feature 2': df_corr.columns[j],
                                'Correlation': df_corr.iloc[i, j]
                            })
                    
                    corr_df = pd.DataFrame(corr_pairs)
                    corr_df = corr_df.reindex(corr_df['Correlation'].abs().values.argsort())[::-1]
                    
                    st.dataframe(
                        corr_df.style.background_gradient(
                            subset=['Correlation'],
                            cmap='RdYlGn',
                            vmin=-1,
                            vmax=1
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
        
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {e}")
    
    # ==========================================
    # TAB 4: DAY OF WEEK ANALYSIS
    # ==========================================
    with tab4:
        st.subheader("Revenue Analysis by Day of Week")
        st.write("How daily revenue varies across different days of the week.")
        
        try:
            if 'day_of_week' in df_full.columns:
                # Aggregate revenue by day of week
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                
                daily_stats = df_full.groupby(['date', 'day_of_week'])['revenue_npr'].sum().reset_index()
                
                # Create box plot
                fig = px.box(
                    daily_stats,
                    x='day_of_week',
                    y='revenue_npr',
                    title='Daily Revenue Distribution by Day of Week',
                    category_orders={'day_of_week': day_order},
                    labels={'revenue_npr': 'Daily Revenue (NPR)', 'day_of_week': 'Day of Week'},
                    height=500
                )
                
                fig.update_layout(
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                with st.expander("��� Day-wise Statistics"):
                    day_stats = daily_stats.groupby('day_of_week')['revenue_npr'].agg([
                        ('Count', 'count'),
                        ('Mean', 'mean'),
                        ('Median', 'median'),
                        ('Std Dev', 'std'),
                        ('Max', 'max'),
                        ('Min', 'min')
                    ]).reindex(day_order)
                    
                    st.dataframe(
                        day_stats.style.format("{:,.0f}"),
                        use_container_width=True
                    )
            else:
                st.info("Day of week data not available. Run data preprocessing first.")
        
        except Exception as e:
            st.error(f"Error creating day of week analysis: {e}")
    
    # ==========================================
    # TAB 5: FESTIVAL SEASON IMPACT
    # ==========================================
    with tab5:
        st.subheader("Revenue Impact: Festival vs. Non-Festival Days")
        st.write("Comparison of revenue during festival seasons vs. regular days.")
        
        try:
            if 'festival_season' in df_full.columns:
                # Aggregate by festival season
                festival_stats = df_full.groupby(['date', 'festival_season'])['revenue_npr'].sum().reset_index()
                festival_stats['season_label'] = festival_stats['festival_season'].apply(
                    lambda x: 'Festival Season' if x == 1 else 'Regular Days'
                )
                
                # Box plot
                fig = px.box(
                    festival_stats,
                    x='season_label',
                    y='revenue_npr',
                    title='Daily Revenue: Festival vs. Regular Days',
                    labels={'revenue_npr': 'Daily Revenue (NPR)', 'season_label': 'Season Type'},
                    color='season_label',
                    color_discrete_map={'Festival Season': '#FF6B6B', 'Regular Days': '#4ECDC4'},
                    height=500
                )
                
                fig.update_layout(
                    hovermode='x unified',
                    template='plotly_white',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2 = st.columns(2)
                
                festival_revenue = festival_stats[festival_stats['season_label'] == 'Festival Season']['revenue_npr']
                regular_revenue = festival_stats[festival_stats['season_label'] == 'Regular Days']['revenue_npr']
                
                with col1:
                    st.metric(
                        "Festival Season",
                        f"₨{festival_revenue.mean():,.0f}",
                        delta=f"n={len(festival_revenue)} days"
                    )
                
                with col2:
                    st.metric(
                        "Regular Days",
                        f"₨{regular_revenue.mean():,.0f}",
                        delta=f"n={len(regular_revenue)} days"
                    )
                
                # Lift calculation
                festival_mean = festival_revenue.mean()
                regular_mean = regular_revenue.mean()
                if regular_mean > 0:
                    lift = ((festival_mean - regular_mean) / regular_mean) * 100
                    st.info(f"�� **Festival Impact:** Festivals drive **{lift:+.1f}%** change in daily revenue")
                
                # Detailed breakdown
                with st.expander("��� Festival Types"):
                    festival_cols = ['is_dashain', 'is_tihar', 'is_holi', 'is_eid', 'is_lhosar']
                    available_festivals = [col for col in festival_cols if col in df_full.columns]
                    
                    if available_festivals:
                        festival_impact = []
                        for festival_col in available_festivals:
                            festival_name = festival_col.replace('is_', '').title()
                            festival_days = df_full[df_full[festival_col] == 1].groupby('date')['revenue_npr'].sum()
                            
                            if len(festival_days) > 0:
                                festival_impact.append({
                                    'Festival': festival_name,
                                    'Days': len(festival_days),
                                    'Avg Revenue': festival_days.mean(),
                                    'Total Revenue': festival_days.sum()
                                })
                        
                        if festival_impact:
                            festival_df = pd.DataFrame(festival_impact)
                            st.dataframe(
                                festival_df.style.format({
                                    'Avg Revenue': '₨{:,.0f}',
                                    'Total Revenue': '₨{:,.0f}'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
            else:
                st.info("Festival season data not available. Run data preprocessing first.")
        
        except Exception as e:
            st.error(f"Error creating festival analysis: {e}")

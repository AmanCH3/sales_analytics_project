import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np

def render_product_tab(rules_df, reco_df=None, placement_df=None, category_recs=None, hierarchical_data=None):
    """
    Enhanced Product Tab with Category-First Recommendations
    """

    st.header("🛒 Smart Product Recommendations")
    
    # Debug: Show what data we have
    with st.expander("🔍 Debug Info - Click to expand", expanded=False):
        st.write("**Data Status:**")
        st.write(f"- Association Rules: {len(rules_df) if rules_df is not None else 0} rows")
        st.write(f"- Product Recommendations: {len(reco_df) if reco_df is not None else 0} rows")
        st.write(f"- Category Summary: {len(category_recs) if category_recs is not None else 0} rows")
        st.write(f"- Placement Suggestions: {len(placement_df) if placement_df is not None else 0} rows")
        st.write(f"- Hierarchical Data: {len(hierarchical_data) if hierarchical_data else 0} category pairs")
        
        if category_recs is not None and len(category_recs) > 0:
            st.write(f"**Category Summary Columns:** {category_recs.columns.tolist()}")
            st.write(f"**Lift Range:** {category_recs['avg_lift'].min():.3f} to {category_recs['avg_lift'].max():.3f}")
    
    # Tab structure
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️ Category Recommendations", 
        "🎯 Product Recommendations",
        "🕸️ Association Network",
        "🛍️ Store Placement"
    ])
    
    # ==========================================
    # TAB 1: CATEGORY RECOMMENDATIONS (PRIMARY)
    # ==========================================
    with tab1:
        st.subheader("📊 Category-Level Recommendations")
        st.info("💡 **Start here!** See which product categories are frequently bought together.")
        
        if category_recs is not None and len(category_recs) > 0:
            
            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Category Pairs", len(category_recs))
            with col2:
                avg_lift = category_recs['avg_lift'].mean()
                st.metric("Avg Lift", f"{avg_lift:.2f}")
            with col3:
                strong_pairs = len(category_recs[category_recs['avg_lift'] > 2.0])
                st.metric("Strong Pairs (>2.0)", strong_pairs)
            with col4:
                total_products = category_recs['num_product_pairs'].sum()
                st.metric("Total Product Pairs", int(total_products))
            
            st.markdown("---")
            
            # Interactive Category Explorer
            st.subheader("🔍 Category Association Explorer")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.write("**Filters**")
                
                # Get actual lift range from data
                min_lift_data = float(category_recs['avg_lift'].min())
                max_lift_data = float(category_recs['avg_lift'].max())
                
                min_category_lift = st.slider(
                    "Minimum Lift", 
                    0.0, 
                    max(5.0, max_lift_data), 
                    min_lift_data,  # Start at minimum value in data
                    0.1, 
                    key="cat_lift",
                    help=f"Data range: {min_lift_data:.2f} to {max_lift_data:.2f}"
                )
                
                # Get available strength levels
                available_strengths = category_recs['strength'].unique().tolist()
                strength_filter = st.multiselect(
                    "Strength Level",
                    available_strengths,
                    default=available_strengths,  # Select all by default
                    key="strength_filter"
                )
                
                # Category selector
                all_categories = sorted(set(category_recs['category_from'].unique()) | 
                                      set(category_recs['category_to'].unique()))
                selected_category = st.selectbox("Focus on Category", ["All"] + all_categories)
            
            with col1:
                # Filter data
                filtered_cats = category_recs[
                    (category_recs['avg_lift'] >= min_category_lift) &
                    (category_recs['strength'].isin(strength_filter))
                ].copy()
                
                if selected_category != "All":
                    filtered_cats = filtered_cats[
                        (filtered_cats['category_from'] == selected_category) |
                        (filtered_cats['category_to'] == selected_category)
                    ]
                
                # Visualization: Category Network
                if len(filtered_cats) > 0:
                    st.write(f"**Category Association Network** ({len(filtered_cats)} pairs)")
                    cat_network_fig = create_category_network(filtered_cats)
                    st.plotly_chart(cat_network_fig, use_container_width=True)
                else:
                    st.warning(f"No categories match the current filters. Try lowering the minimum lift below {min_category_lift:.2f}")
            
            st.markdown("---")
            
            # Category Recommendation Table with Drill-Down
            st.subheader("📋 Detailed Category Associations")
            
            if len(filtered_cats) > 0:
                # Display with expandable rows
                for idx, row in filtered_cats.head(20).iterrows():
                    with st.expander(
                        f"🏷️ **{row['category_from']}** → **{row['category_to']}** "
                        f"(Lift: {row['avg_lift']:.2f}, {row['strength']})",
                        expanded=False
                    ):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Average Lift", f"{row['avg_lift']:.2f}")
                            # Only show if column exists
                            if 'max_lift' in row:
                                st.metric("Max Lift", f"{row['max_lift']:.2f}")
                        
                        with col2:
                            # Only show if columns exist
                            if 'avg_confidence' in row:
                                st.metric("Average Confidence", f"{row['avg_confidence']:.2%}")
                            if 'max_confidence' in row:
                                st.metric("Max Confidence", f"{row['max_confidence']:.2%}")
                        
                        with col3:
                            st.metric("Product Pairs", int(row['num_product_pairs']))
                            st.metric("Strength", row['strength'])
                        
                        # Show top products for this category pair
                        if hierarchical_data:
                            key = f"{row['category_from']} → {row['category_to']}"
                            if key in hierarchical_data:
                                st.write("**🎯 Top Product Recommendations:**")
                                product_df = hierarchical_data[key].head(10)
                                
                                # Format for display
                                display_cols = ['product_from', 'product_to', 'lift', 'confidence']
                                display_df = product_df[display_cols].copy()
                                display_df.columns = ['Buy This', 'Recommend This', 'Lift', 'Confidence']
                                
                                st.dataframe(
                                    display_df.style.background_gradient(
                                        subset=['Lift', 'Confidence'],
                                        cmap='YlGn'
                                    ),
                                    use_container_width=True,
                                    height=300
                                )
                            else:
                                st.info(f"No product details available for this category pair")
            
            # Bar charts
            st.markdown("---")
            st.subheader("📊 Top Category Associations")
            
            if len(filtered_cats) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**By Lift (Strength)**")
                    top_by_lift = create_category_bar_chart(filtered_cats.head(10), 'avg_lift', 'Average Lift')
                    st.plotly_chart(top_by_lift, use_container_width=True)
                
                with col2:
                    st.write("**By Number of Product Pairs**")
                    top_by_count = create_category_bar_chart(filtered_cats.head(10), 'num_product_pairs', 'Product Pairs')
                    st.plotly_chart(top_by_count, use_container_width=True)
        
        else:
            st.warning("⚠️ No category recommendations available.")
            st.info("💡 **How to fix this:**\n1. Make sure your source data has a 'category' column\n2. Run: `python src/apriori_model.py`\n3. Check that `data/processed/category_summary.csv` exists and has data")
    
    # ==========================================
    # TAB 2: PRODUCT RECOMMENDATIONS
    # ==========================================
    with tab2:
        st.subheader("🎯 Product-Level Recommendations")
        st.info("💡 Specific product-to-product recommendations.")
        
        if hierarchical_data and len(hierarchical_data) > 0:
            # Smart Product Search
            st.subheader("🔍 Find Recommendations for a Specific Product")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                # Get all unique products
                all_products = set()
                for df in hierarchical_data.values():
                    all_products.update(df['product_from'].unique())
                    all_products.update(df['product_to'].unique())
                
                if len(all_products) > 0:
                    search_product = st.selectbox(
                        "Select Product",
                        sorted(list(all_products)),
                        key="product_search"
                    )
                    
                    min_prod_lift = st.slider("Min Lift", 0.0, 5.0, 1.0, 0.1, key="prod_lift")
                else:
                    st.error("No products found in hierarchical data")
                    search_product = None
            
            with col1:
                if search_product:
                    st.write(f"**Recommendations for: {search_product}**")
                    
                    # Find all recommendations
                    recommendations = []
                    for key, df in hierarchical_data.items():
                        matching = df[
                            (df['product_from'] == search_product) &
                            (df['lift'] >= min_prod_lift)
                        ]
                        if len(matching) > 0:
                            recommendations.append(matching)
                    
                    if recommendations:
                        combined_recs = pd.concat(recommendations).sort_values('lift', ascending=False)
                        
                        st.write(f"**Found {len(combined_recs)} recommendations across {combined_recs['category_to'].nunique()} categories**")
                        
                        for category in combined_recs['category_to'].unique():
                            cat_recs = combined_recs[combined_recs['category_to'] == category].head(5)
                            
                            with st.expander(f"📦 {category} ({len(cat_recs)} products)", expanded=True):
                                display_df = cat_recs[['product_to', 'lift', 'confidence', 'support']].copy()
                                display_df.columns = ['Product', 'Lift', 'Confidence', 'Support']
                                
                                st.dataframe(
                                    display_df.style.background_gradient(
                                        subset=['Lift', 'Confidence'],
                                        cmap='Greens'
                                    ),
                                    use_container_width=True
                                )
                    else:
                        st.info(f"No recommendations found for '{search_product}' with lift >= {min_prod_lift:.2f}")
        
        elif reco_df is not None and len(reco_df) > 0:
            # Fallback: Show all recommendations
            st.write("**All Product Recommendations**")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                search_text = st.text_input("Search products", key="prod_search")
                min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05, key="prod_conf")
            
            with col1:
                filtered = reco_df[reco_df['confidence'] >= min_conf].copy()
                
                if search_text:
                    filtered = filtered[
                        filtered['product'].str.contains(search_text, case=False, na=False) |
                        filtered['recommended_product'].str.contains(search_text, case=False, na=False)
                    ]
                
                if len(filtered) > 0:
                    st.dataframe(
                        filtered.style.background_gradient(
                            subset=['confidence', 'lift'],
                            cmap='Greens'
                        ),
                        use_container_width=True,
                        height=500
                    )
                else:
                    st.info("No products match your search criteria")
        else:
            st.warning("No product recommendation data available.")
    
    # ==========================================
    # TAB 3: ASSOCIATION NETWORK
    # ==========================================
    with tab3:
        st.subheader("🕸️ Product Association Network")
        
        if rules_df is not None and len(rules_df) > 0:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.write("**Network Controls**")
                
                # Get actual ranges
                min_lift_rules = float(rules_df['lift'].min())
                max_lift_rules = float(rules_df['lift'].max())
                
                min_lift = st.slider(
                    "Minimum Lift", 
                    min_lift_rules, 
                    max(5.0, max_lift_rules), 
                    min(1.2, max_lift_rules), 
                    0.1, 
                    key="net_lift"
                )
                min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.3, 0.05, key="net_conf")
                top_n = st.slider("Top N Rules", 10, min(100, len(rules_df)), 30, 10, key="net_top")
            
            with col1:
                filtered_rules = rules_df[
                    (rules_df['lift'] >= min_lift) & 
                    (rules_df['confidence'] >= min_confidence)
                ].nlargest(min(top_n, len(rules_df)), 'lift')
                
                if len(filtered_rules) > 0:
                    fig = create_product_network(filtered_rules)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rules in Network", len(filtered_rules))
                    with col2:
                        st.metric("Avg Lift", f"{filtered_rules['lift'].mean():.2f}")
                    with col3:
                        st.metric("Avg Confidence", f"{filtered_rules['confidence'].mean():.2%}")
                else:
                    st.warning("No rules match current filters.")
        else:
            st.warning("No association rules available. Run `python src/apriori_model.py`")
    
    # ==========================================
    # TAB 4: STORE PLACEMENT
    # ==========================================
    with tab4:
        st.subheader("🛍️ Store Shelf Placement Analysis")
        
        if placement_df is not None and len(placement_df) > 0:
            
            # KPIs
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_priority = len(placement_df[placement_df['priority'] == 'High'])
                st.metric("High Priority", high_priority, help="Lift > 2.0")
            
            with col2:
                medium_priority = len(placement_df[placement_df['priority'] == 'Medium'])
                st.metric("Medium Priority", medium_priority, help="Lift 1.5-2.0")
            
            with col3:
                cross_aisle = len(placement_df[~placement_df['same_aisle']])
                st.metric("Cross-Aisle Pairs", cross_aisle)
            
            st.markdown("---")
            
            # Views
            view1, view2, view3 = st.tabs(["🎯 Priority Actions", "📊 Placement Table", "📍 Heatmap"])
            
            with view1:
                st.write("**Top Priority Placement Changes**")
                
                priority_filter = st.multiselect(
                    "Priority Level",
                    ['High', 'Medium', 'Low'],
                    default=['High', 'Medium'],
                    key="placement_priority"
                )
                
                filtered_placement = placement_df[
                    placement_df['priority'].isin(priority_filter)
                ].head(15)
                
                for idx, row in filtered_placement.iterrows():
                    priority_color = "🔴" if row['priority'] == 'High' else "🟡" if row['priority'] == 'Medium' else "🟢"
                    
                    with st.expander(
                        f"{priority_color} {row['product_1']} ↔ {row['product_2']} (Lift: {row['lift']:.2f})",
                        expanded=(row['priority'] == 'High')
                    ):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Product 1**")
                            st.write(f"📦 {row['product_1']}")
                            st.write(f"🏢 Aisle: {row['p1_aisle']}")
                            st.write(f"📍 Rack: {row['p1_rack']}")
                        
                        with col2:
                            st.write("**Product 2**")
                            st.write(f"📦 {row['product_2']}")
                            st.write(f"🏢 Aisle: {row['p2_aisle']}")
                            st.write(f"📍 Rack: {row['p2_rack']}")
                        
                        with col3:
                            st.write("**Metrics**")
                            st.metric("Lift", f"{row['lift']:.2f}")
                            st.metric("Confidence", f"{row['confidence']:.2%}")
                            st.write(f"**Priority:** {row['priority']}")
                        
                        # Suggestion
                        if row['same_rack']:
                            st.success(row['suggestion'])
                        elif row['same_aisle']:
                            st.warning(row['suggestion'])
                        else:
                            st.error(row['suggestion'])
            
            with view2:
                st.write("**Complete Placement Suggestions**")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_placement_lift = st.number_input("Min Lift", 0.0, 10.0, 1.0, 0.1, key="place_lift")
                with col2:
                    sort_by = st.selectbox("Sort by", ["lift", "confidence", "priority"], key="place_sort")
                
                filtered = placement_df[placement_df['lift'] >= min_placement_lift].copy()
                
                if sort_by == "priority":
                    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
                    filtered['priority_num'] = filtered['priority'].map(priority_order)
                    filtered = filtered.sort_values(['priority_num', 'lift'], ascending=[True, False])
                    filtered = filtered.drop('priority_num', axis=1)
                else:
                    filtered = filtered.sort_values(sort_by, ascending=False)
                
                st.dataframe(
                    filtered.style.background_gradient(
                        subset=['lift', 'confidence'],
                        cmap='YlOrRd'
                    ),
                    use_container_width=True,
                    height=400
                )
            
            with view3:
                st.write("**Aisle Association Heatmap**")
                try:
                    heatmap_fig = create_aisle_heatmap(placement_df)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not create heatmap: {e}")
        
        else:
            st.info("No placement data available.")


# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def create_category_network(category_df):
    """Create network graph for category associations"""
    G = nx.DiGraph()
    
    for _, row in category_df.iterrows():
        weight = row.get('avg_lift', 1.0)
        confidence = row.get('avg_confidence', 0.5)
        G.add_edge(row['category_from'], row['category_to'], weight=weight, confidence=confidence)
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = list(G.nodes())
    node_size = [30 + G.degree(node) * 10 for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hovertext=[f"{node}<br>Connections: {G.degree(node)}" for node in G.nodes()],
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_size,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Connections"),
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
    )
    
    return fig


def create_category_bar_chart(df, metric, title):
    """Create bar chart for category metrics"""
    if len(df) == 0:
        return go.Figure()
    
    df = df.head(10).copy()
    df['label'] = df['category_from'] + ' → ' + df['category_to']
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['label'],
            x=df[metric],
            orientation='h',
            marker=dict(
                color=df[metric],
                colorscale='Blues',
                showscale=False
            ),
            text=df[metric].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else str(int(x))),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title='Category Pair',
        height=400,
        margin=dict(l=200)
    )
    
    return fig


def create_product_network(rules_df):
    """Create network graph for product associations"""
    G = nx.DiGraph()
    
    for _, row in rules_df.iterrows():
        antecedents = list(row['antecedents']) if isinstance(row['antecedents'], (set, list)) else [str(row['antecedents'])]
        consequents = list(row['consequents']) if isinstance(row['consequents'], (set, list)) else [str(row['consequents'])]
        
        for ant in antecedents:
            for cons in consequents:
                G.add_edge(str(ant), str(cons), weight=row['lift'], confidence=row['confidence'])
    
    if len(G.nodes()) == 0:
        return go.Figure()
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_size = [20 + G.degree(node) * 5 for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(n).split()[0][:15] for n in G.nodes()],
        hovertext=[f"{node}<br>Connections: {G.degree(node)}" for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=node_size,
            color=node_size,
            colorbar=dict(thickness=15, title='Connections'),
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )
    
    return fig


def create_aisle_heatmap(placement_df):
    """Create heatmap for aisle associations"""
    aisle_pairs = placement_df.groupby(['p1_aisle', 'p2_aisle'])['lift'].mean().reset_index()
    heatmap_data = aisle_pairs.pivot(index='p1_aisle', columns='p2_aisle', values='lift').fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        text=heatmap_data.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Avg Lift")
    ))
    
    fig.update_layout(
        title='Aisle Association Strength',
        xaxis_title='Aisle 2',
        yaxis_title='Aisle 1',
        height=500
    )
    
    return fig
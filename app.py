
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="Real Estate Analytics Dashboard")

# Load data
@st.cache_data
def load_data():
    excel_file = 'Project 1_Sales Review_10012025.xlsx'
    dfs = {
        'collection': pd.read_excel(excel_file, sheet_name='Collection Analysis'),
        'sales': pd.read_excel(excel_file, sheet_name='Sales Analysis'),
        'monthly': pd.read_excel(excel_file, sheet_name='Monthly Data'),
        'mis': pd.read_excel(excel_file, sheet_name='Monthly MIS Check'),
        'summary': pd.read_excel(excel_file, sheet_name='Sales Summary')
    }
    return dfs

# Initialize
st.title("Real Estate Analytics Dashboard")
dfs = load_data()

# Sidebar filters
st.sidebar.header("Filters")
sale_type = st.sidebar.multiselect("Sale Type", ["Old", "New"], default=["Old", "New"])

# Main layout using columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Collection Overview")
    # Collection metrics
    collection_data = dfs['collection']
    total_collection = collection_data.select_dtypes(include=[np.number]).sum().sum()
    st.metric("Total Collection (₹ Cr)", f"{total_collection:.2f}")
    
    # Collection Trend
    fig_collection = px.line(collection_data, 
                           title="Collection Trend",
                           template="plotly_white")
    st.plotly_chart(fig_collection, use_container_width=True)

with col2:
    st.subheader("Sales Performance")
    # Sales metrics
    sales_data = dfs['sales']
    total_sales = sales_data.select_dtypes(include=[np.number]).sum().sum()
    st.metric("Total Sales (₹ Cr)", f"{total_sales:.2f}")
    
    # Sales by Type
    fig_sales = px.bar(sales_data,
                      title="Sales by Type",
                      template="plotly_white")
    st.plotly_chart(fig_sales, use_container_width=True)

# Monthly Analysis
st.header("Monthly Performance")
monthly_data = dfs['monthly']
col3, col4, col5 = st.columns(3)

with col3:
    # Monthly Collections
    fig_monthly_collection = px.area(monthly_data,
                                   title="Monthly Collections",
                                   template="plotly_white")
    st.plotly_chart(fig_monthly_collection, use_container_width=True)

with col4:
    # Monthly Sales
    fig_monthly_sales = px.bar(monthly_data,
                              title="Monthly Sales",
                              template="plotly_white")
    st.plotly_chart(fig_monthly_sales, use_container_width=True)

with col5:
    # Collection Efficiency
    fig_efficiency = go.Figure()
    fig_efficiency.add_trace(go.Indicator(
        mode="gauge+number",
        title={'text': "Collection Efficiency"},
        value=85,
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig_efficiency, use_container_width=True)

# Detailed Analysis Section
st.header("Detailed Analysis")
tabs = st.tabs(["BHK-wise Analysis", "Tower Analysis", "Payment Plans"])

with tabs[0]:
    # BHK-wise Analysis
    st.subheader("BHK Distribution")
    # Add BHK distribution visualization here

with tabs[1]:
    # Tower Analysis
    st.subheader("Tower-wise Performance")
    # Add tower-wise analysis here

with tabs[2]:
    # Payment Plans Analysis
    st.subheader("Payment Plan Distribution")
    # Add payment plan analysis here

# Download Section
st.sidebar.markdown("---")
st.sidebar.header("Download Reports")
if st.sidebar.button("Download Full Report"):
    # Add download functionality here
    pass

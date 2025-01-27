import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import re
import io
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stSelectbox { margin-bottom: 1rem; }
    .plot-container {
        margin-bottom: 2rem;
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .custom-metric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced normalization functions
def clean_numeric(value):
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        clean_str = re.sub(r'[^\d.-]', '', value)
        try:
            return float(clean_str)
        except ValueError:
            return 0
    return float(value)

def normalize_bhk(bhk):
    if pd.isna(bhk):
        return "Not Specified"
    
    bhk = str(bhk).upper().strip()
    bhk = re.sub(r'\s+', ' ', bhk)
    
    numeric_part = re.search(r'\d+', bhk)
    if numeric_part:
        number = numeric_part.group()
        return f"{number}-BHK"
    
    if bhk == 'SHOP' or 'SHOP' in bhk:
        return 'SHOP'
    
    return "Not Specified"

def normalize_tower(tower):
    if pd.isna(tower):
        return "Not Specified"
    
    tower = str(tower).upper().strip()
    tower = re.sub(r'\s+', ' ', tower)
    
    if 'RETAIL' in tower:
        return 'RETAIL'
    
    if tower.startswith('CA'):
        match = re.search(r'(\d+)', tower)
        if match:
            number = match.group(1).zfill(2)
            return f"CA {number}"
    
    return tower

def normalize_status(status):
    if pd.isna(status):
        return "Not Specified"
    
    status = str(status).upper().strip()
    status_map = {
        'SOLD': 'SOLD',
        'AVAILABLE': 'AVAILABLE',
        'CANCEL': 'CANCELLED',
        'CANCELLED': 'CANCELLED',
        'BLOCKED': 'BLOCKED',
        'NA': 'NOT SPECIFIED',
        '': 'NOT SPECIFIED'
    }
    return status_map.get(status, status)

def normalize_sale_type(sale_type):
    if pd.isna(sale_type):
        return "Not Specified"
    
    sale_type = str(sale_type).upper().strip()
    sale_type = re.sub(r'\s+', ' ', sale_type)
    
    if 'OLD' in sale_type:
        return 'OLD SALE'
    elif 'NEW' in sale_type:
        return 'NEW SALE'
    elif 'CANCEL' in sale_type:
        return 'CANCELLED'
    elif 'TRANSFER' in sale_type:
        return 'TRANSFER'
    
    return sale_type

def normalize_payment_plan(plan):
    if pd.isna(plan):
        return "Not Specified"
    
    plan = str(plan).upper().strip()
    plan = re.sub(r'\s+', ' ', plan)
    
    # Add common payment plan variations
    plan_map = {
        'CONSTRUCTION LINKED': 'CONSTRUCTION LINKED PLAN',
        'CLP': 'CONSTRUCTION LINKED PLAN',
        'DOWN PAYMENT': 'DOWN PAYMENT PLAN',
        'DP': 'DOWN PAYMENT PLAN',
        'FLEXI': 'FLEXI PAYMENT PLAN',
        'FLEXIBLE': 'FLEXI PAYMENT PLAN'
    }
    
    return plan_map.get(plan, plan)

def normalize_customer_name(name):
    if pd.isna(name):
        return "Not Specified"
    
    name = str(name).strip()
    # Standardize prefixes
    prefixes = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'M/s']
    name_parts = name.split()
    if name_parts and name_parts[0].lower().replace('.', '') in [p.lower().replace('.', '') for p in prefixes]:
        prefix = name_parts[0].replace('.', '') + '.'
        rest_of_name = ' '.join(name_parts[1:])
        name = f"{prefix} {rest_of_name}"
    
    return name.title()

def normalize_all_columns(df):
    """Apply normalization to all relevant columns"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Map columns to their normalization functions
    normalization_map = {
        'BHK': normalize_bhk,
        'Tower': normalize_tower,
        'Current Status': normalize_status,
        'Old sale / New sale': normalize_sale_type,
        'Payment Plan': normalize_payment_plan,
        'Customer Name': normalize_customer_name,
        'Name': normalize_customer_name,
        'Cancellation / Transfer': normalize_sale_type
    }
    
    # Numeric columns for cleaning
    numeric_columns = [
        'Total Consideration', 'Required Collection', 'Current collection',
        'Area', 'BSP', 'Collection', 'Sale Consideration'
    ]
    
    # Apply normalizations
    for col in df.columns:
        if col in normalization_map:
            df[col] = df[col].apply(normalization_map[col])
        elif col in numeric_columns:
            df[col] = df[col].apply(clean_numeric)
        elif 'date' in col.lower() or 'month' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    return df

# Session state initialization
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# File upload section
st.title("Real Estate Analytics Dashboard")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        with st.spinner('Processing data...'):
            # Read all sheets
            excel_file = pd.ExcelFile(uploaded_file)
            
            required_sheets = ['Collection Analysis', 'Sales Analysis', 'Monthly Data', 'Sales Summary']
            
            # Check if required sheets exist
            missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
            if missing_sheets:
                st.error(f"Missing required sheets: {', '.join(missing_sheets)}")
                st.stop()
            
            # Read and normalize all sheets
            collection_df = pd.read_excel(excel_file, 'Collection Analysis', skiprows=3)
            collection_df = normalize_all_columns(collection_df)
            
            sales_df = pd.read_excel(excel_file, 'Sales Analysis', skiprows=3)
            sales_df = normalize_all_columns(sales_df)
            
            monthly_df = pd.read_excel(excel_file, 'Monthly Data', skiprows=2)
            monthly_df = normalize_all_columns(monthly_df)
            
            summary_df = pd.read_excel(excel_file, 'Sales Summary', skiprows=2)
            summary_df = normalize_all_columns(summary_df)
            
            st.session_state.data_loaded = True
            st.session_state.collection_df = collection_df
            st.session_state.sales_df = sales_df
            st.session_state.monthly_df = monthly_df
            st.session_state.summary_df = summary_df
            
            st.success("Data loaded successfully!")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()

if st.session_state.data_loaded:
    # Sidebar filters
    st.sidebar.title("Filters")
    
    collection_df = st.session_state.collection_df
    
    # Get unique values for filters
    towers = sorted([t for t in collection_df['Tower'].unique() if t != "Not Specified"])
    bhk_types = sorted([b for b in collection_df['BHK'].unique() if b != "Not Specified"])
    
    selected_tower = st.sidebar.selectbox("Select Tower", ["All Towers"] + towers)
    selected_bhk = st.sidebar.selectbox("Select BHK Type", ["All BHK"] + bhk_types)
    
    # Filter data
    filtered_df = collection_df.copy()
    if selected_tower != "All Towers":
        filtered_df = filtered_df[filtered_df['Tower'] == selected_tower]
    if selected_bhk != "All BHK":
        filtered_df = filtered_df[filtered_df['BHK'] == selected_bhk]
    
    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    # Key metrics
    with col1:
        st.markdown("""
            <div class="custom-metric">
                <h3>Total Units</h3>
                <h2 style="color: #1f77b4;">{}</h2>
            </div>
        """.format(len(filtered_df)), unsafe_allow_html=True)
    
    with col2:
        total_consideration = filtered_df['Total Consideration'].sum()
        st.markdown("""
            <div class="custom-metric">
                <h3>Total Consideration</h3>
                <h2 style="color: #1f77b4;">₹{:,.0f}</h2>
            </div>
        """.format(total_consideration), unsafe_allow_html=True)
    
    with col3:
        current_collection = filtered_df['Current collection'].sum()
        st.markdown("""
            <div class="custom-metric">
                <h3>Current Collection</h3>
                <h2 style="color: #1f77b4;">₹{:,.0f}</h2>
            </div>
        """.format(current_collection), unsafe_allow_html=True)
    
    with col4:
        total_area = filtered_df['Area'].sum()
        st.markdown("""
            <div class="custom-metric">
                <h3>Total Area (sq ft)</h3>
                <h2 style="color: #1f77b4;">{:,.0f}</h2>
            </div>
        """.format(total_area), unsafe_allow_html=True)

    st.markdown("---")
    
    # Charts section
    st.markdown('<p class="section-title">Sales Analytics</p>', unsafe_allow_html=True)
    
    # Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Unit Distribution by Tower")
        tower_dist = filtered_df['Tower'].value_counts()
        fig_tower = px.bar(
            x=tower_dist.index,
            y=tower_dist.values,
            labels={'x': 'Tower', 'y': 'Number of Units'},
            color=tower_dist.values,
            color_continuous_scale='Viridis'
        )
        fig_tower.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        st.plotly_chart(fig_tower, use_container_width=True)
    
    with col2:
        st.subheader("BHK Distribution")
        bhk_dist = filtered_df['BHK'].value_counts()
        fig_bhk = px.pie(
            values=bhk_dist.values,
            names=bhk_dist.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_bhk.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_bhk, use_container_width=True)

    # Continue with the rest of your visualizations...
    # [Previous visualization code remains the same]

    # Monthly Analysis
    st.markdown('<p class="section-title">Monthly Trends</p>', unsafe_allow_html=True)

    monthly_df = st.session_state.monthly_df
    monthly_filtered = monthly_df.copy()
    if selected_tower != "All Towers":
        monthly_filtered = monthly_filtered[monthly_filtered['Tower'] == selected_tower]
    if selected_bhk != "All BHK":
        monthly_filtered = monthly_filtered[monthly_filtered['BHK'] == selected_bhk]

    monthly_agg = monthly_filtered.groupby(['Month No', 'Old sale / New sale']).size().reset_index(name='Count')
    fig_monthly = px.line(
        monthly_agg,
        x='Month No',
        y='Count',
        color='Old sale / New sale',
        markers=True,
        line_shape='spline'
     )
    fig_monthly.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title="Month Number",
        yaxis_title="Number of Sales"
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Continue with other visualizations and sections...
    # [Rest of your dashboard code remains the same]

else:
    # Show welcome message when no data is loaded
    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>Welcome to the Real Estate Analytics Dashboard</h2>
            <p>Please upload an Excel file to begin analysis.</p>
            <p>The file should contain the following sheets:</p>
            <ul style="list-style-type: none;">
                <li>Collection Analysis</li>
                <li>Sales Analysis</li>
                <li>Monthly Data</li>
                <li>Sales Summary</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # BSP Analysis
    st.markdown('<p class="section-title">Pricing Analytics</p>', unsafe_allow_html=True)
    
    fig_bsp = px.box(
        filtered_df,
        x='Tower',
        y='BSP',
        color='BHK',
        points="all",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_bsp.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title="Tower",
        yaxis_title="BSP (₹/sq ft)"
    )
    st.plotly_chart(fig_bsp, use_container_width=True)

    # Payment Plan Analysis
    st.markdown('<p class="section-title">Payment Plans</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Payment Plan Distribution")
        payment_dist = filtered_df['Payment Plan'].value_counts()
        fig_payment = px.pie(
            values=payment_dist.values,
            names=payment_dist.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_payment.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_payment, use_container_width=True)
    
    with col2:
        st.subheader("Collection Efficiency")
        filtered_df['Collection Percentage'] = (filtered_df['Current collection'] / 
                                              filtered_df['Required Collection'] * 100)
        fig_efficiency = px.histogram(
            filtered_df,
            x='Collection Percentage',
            color='BHK',
            nbins=20,
            opacity=0.7
        )
        fig_efficiency.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Collection Percentage",
            yaxis_title="Number of Units"
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)

    # Detailed Data Table
    st.markdown('<p class="section-title">Detailed Unit Information</p>', unsafe_allow_html=True)
    
    # Add column selector
    available_columns = filtered_df.columns.tolist()
    default_columns = ['Apt No', 'BHK', 'Tower', 'Area', 'Current Status', 
                      'Total Consideration', 'Current collection', 'Customer Name']
    selected_columns = st.multiselect(
        "Select columns to display",
        available_columns,
        default=default_columns
    )

    # Show filtered dataframe with selected columns
    if selected_columns:
        st.dataframe(
            filtered_df[selected_columns].sort_values('Apt No'),
            use_container_width=True,
            hide_index=True
        )

    # Download section
    st.markdown("---")
    st.markdown('<p class="section-title">Export Data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"real_estate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    
    with col2:
        # Export summary statistics
        summary_stats = pd.DataFrame({
            'Metric': ['Total Units', 'Total Area (sq ft)', 'Total Consideration', 'Current Collection',
                      'Average BSP', 'Average Unit Area'],
            'Value': [
                len(filtered_df),
                f"{filtered_df['Area'].sum():,.0f}",
                f"₹{filtered_df['Total Consideration'].sum():,.0f}",
                f"₹{filtered_df['Current collection'].sum():,.0f}",
                f"₹{filtered_df['BSP'].mean():,.2f}",
                f"{filtered_df['Area'].mean():,.0f}"
            ]
        })
        csv_summary = summary_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Summary Statistics",
            data=csv_summary,
            file_name=f"summary_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    # Data Quality Section
    st.markdown("---")
    st.markdown('<p class="section-title">Data Quality Report</p>', unsafe_allow_html=True)
    
    with st.expander("View Data Quality Issues"):
        data_quality_issues = []
        
        # Check for missing values
        for column in filtered_df.columns:
            missing_count = filtered_df[column].isna().sum()
            if missing_count > 0:
                data_quality_issues.append(f"Missing values in {column}: {missing_count} records")
        
        # Check for zero or negative values in numeric columns
        numeric_cols = ['Total Consideration', 'Required Collection', 'Current collection', 'Area', 'BSP']
        for col in numeric_cols:
            if col in filtered_df.columns:
                zero_count = (filtered_df[col] == 0).sum()
                neg_count = (filtered_df[col] < 0).sum()
                if zero_count > 0:
                    data_quality_issues.append(f"Zero values in {col}: {zero_count} records")
                if neg_count > 0:
                    data_quality_issues.append(f"Negative values in {col}: {neg_count} records")
        
        if data_quality_issues:
            for issue in data_quality_issues:
                st.warning(issue)
        else:
            st.success("No major data quality issues found!")

    # Footer
    st.markdown("---")
    st.markdown(f"*Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Add cache clearing button in sidebar
    if st.sidebar.button("Clear Cache and Reset"):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.experimental_rerun()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import re
import logging
import warnings

# Configure logging and suppress warnings
logging.getLogger("streamlit_health_check").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

def safe_string_handling(value):
    """Safely convert any value to string"""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d')
    try:
        return str(value).strip()
    except:
        return ""

def clean_numeric(value):
    """Clean numeric values"""
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        clean_str = re.sub(r'[^\d.-]', '', value)
        try:
            return float(clean_str)
        except ValueError:
            return 0
    if isinstance(value, (int, float)):
        return float(value)
    return 0

def normalize_bhk(bhk):
    """Normalize BHK values"""
    value = safe_string_handling(bhk)
    if not value:
        return "Not Specified"
    
    value = value.upper()
    numeric_part = re.search(r'\d+', value)
    if numeric_part:
        number = numeric_part.group()
        return f"{number}-BHK"
    
    if 'SHOP' in value:
        return 'SHOP'
    
    return "Not Specified"

def normalize_tower(tower):
    """Normalize tower names"""
    value = safe_string_handling(tower)
    if not value:
        return "Not Specified"
    
    value = value.upper()
    if 'RETAIL' in value:
        return 'RETAIL'
    
    if value.startswith('CA'):
        match = re.search(r'(\d+)', value)
        if match:
            number = match.group(1).zfill(2)
            return f"CA {number}"
    
    return value

def normalize_status(status):
    """Normalize status values"""
    value = safe_string_handling(status)
    if not value:
        return "Not Specified"
    
    value = value.upper()
    status_map = {
        'SOLD': 'SOLD',
        'AVAILABLE': 'AVAILABLE',
        'CANCEL': 'CANCELLED',
        'CANCELLED': 'CANCELLED',
        'BLOCKED': 'BLOCKED',
    }
    return status_map.get(value, "Not Specified")

def normalize_sale_type(sale_type):
    """Normalize sale type values"""
    value = safe_string_handling(sale_type)
    if not value:
        return "Not Specified"
    
    value = value.upper()
    if 'OLD' in value:
        return 'OLD SALE'
    elif 'NEW' in value:
        return 'NEW SALE'
    elif 'CANCEL' in value:
        return 'CANCELLED'
    elif 'TRANSFER' in value:
        return 'TRANSFER'
    
    return value

def process_dataframe(df, sheet_name):
    """Process and clean dataframe"""
    try:
        df = df.copy()
        
        # Basic cleaning
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Normalize string columns
        if 'BHK' in df.columns:
            df['BHK'] = df['BHK'].apply(normalize_bhk)
        if 'Tower' in df.columns:
            df['Tower'] = df['Tower'].apply(normalize_tower)
        if 'Current Status' in df.columns:
            df['Current Status'] = df['Current Status'].apply(normalize_status)
        if 'Old sale / New sale' in df.columns:
            df['Old sale / New sale'] = df['Old sale / New sale'].apply(normalize_sale_type)
        
        # Clean numeric columns
        numeric_columns = ['Total Consideration', 'Required Collection', 'Current collection', 
                         'Area', 'BSP', 'Collection', 'Sale Consideration']
        for col in df.columns:
            if any(num_col.lower() in col.lower() for num_col in numeric_columns):
                df[col] = df[col].apply(clean_numeric)
        
        return df
    
    except Exception as e:
        st.error(f"Error processing {sheet_name}: {str(e)}")
        return pd.DataFrame()

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Main dashboard title
st.title("Real Estate Analytics Dashboard")
st.markdown("---")

# File upload section
uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        with st.spinner('Processing data...'):
            # Read Excel file
            excel_file = pd.ExcelFile(uploaded_file)
            required_sheets = ['Collection Analysis', 'Sales Analysis', 'Monthly Data', 'Sales Summary']
            
            # Verify required sheets
            missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
            if missing_sheets:
                st.error(f"Missing required sheets: {', '.join(missing_sheets)}")
                st.stop()
            
            # Process each sheet
            collection_df = process_dataframe(
                pd.read_excel(excel_file, 'Collection Analysis', skiprows=3),
                'Collection Analysis'
            )
            sales_df = process_dataframe(
                pd.read_excel(excel_file, 'Sales Analysis', skiprows=3),
                'Sales Analysis'
            )
            monthly_df = process_dataframe(
                pd.read_excel(excel_file, 'Monthly Data', skiprows=2),
                'Monthly Data'
            )
            summary_df = process_dataframe(
                pd.read_excel(excel_file, 'Sales Summary', skiprows=2),
                'Sales Summary'
            )
            
            # Store in session state
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
    try:
        # Sidebar filters
        st.sidebar.title("Filters")
        
        # Get the dataframe from session state
        collection_df = st.session_state.collection_df
        
        # Get unique values for filters
        towers = sorted([t for t in collection_df['Tower'].unique() 
                       if t != "Not Specified" and pd.notna(t)])
        bhk_types = sorted([b for b in collection_df['BHK'].unique() 
                          if b != "Not Specified" and pd.notna(b)])
        
        selected_tower = st.sidebar.selectbox("Select Tower", ["All Towers"] + towers)
        selected_bhk = st.sidebar.selectbox("Select BHK Type", ["All BHK"] + bhk_types)
        
        # Filter data
        df = collection_df.copy()
        if selected_tower != "All Towers":
            df = df[df['Tower'] == selected_tower]
        if selected_bhk != "All BHK":
            df = df[df['BHK'] == selected_bhk]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Units", len(df))
        
        with col2:
            total_consideration = df['Total Consideration'].sum()
            st.metric("Total Consideration", f"₹{total_consideration:,.0f}")
        
        with col3:
            current_collection = df['Current collection'].sum()
            st.metric("Current Collection", f"₹{current_collection:,.0f}")
        
        with col4:
            total_area = df['Area'].sum()
            st.metric("Total Area (sq ft)", f"{total_area:,.0f}")
        
        # Unit Distribution
        st.markdown("---")
        st.markdown('<p class="section-title">Sales Analytics</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Unit Distribution by Tower")
            tower_dist = df['Tower'].value_counts()
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
            bhk_dist = df['BHK'].value_counts()
            fig_bhk = px.pie(
                values=bhk_dist.values,
                names=bhk_dist.index,
                hole=0.4
            )
            fig_bhk.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig_bhk, use_container_width=True)
        
        # Collection Analysis
        st.markdown('<p class="section-title">Collection Analytics</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Collection vs Required Collection by Tower")
            tower_collection = df.groupby('Tower').agg({
                'Required Collection': 'sum',
                'Current collection': 'sum'
            }).reset_index()
            
            fig_collection = go.Figure()
            fig_collection.add_trace(go.Bar(
                name='Required Collection',
                x=tower_collection['Tower'],
                y=tower_collection['Required Collection'],
                marker_color='#1f77b4'
            ))
            fig_collection.add_trace(go.Bar(
                name='Current Collection',
                x=tower_collection['Tower'],
                y=tower_collection['Current collection'],
                marker_color='#2ca02c'
            ))
            fig_collection.update_layout(
                barmode='group',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig_collection, use_container_width=True)
        
        with col2:
            st.subheader("Collection Efficiency")
            df['Collection Percentage'] = (
                df['Current collection'] / df['Required Collection'] * 100
            ).clip(0, 100)
            
            fig_efficiency = px.histogram(
                df,
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
        
        # BSP Analysis
        st.markdown('<p class="section-title">Pricing Analytics</p>', unsafe_allow_html=True)
        
        if 'BSP' in df.columns:
            fig_bsp = px.box(
                df,
                x='Tower',
                y='BSP',
                color='BHK',
                points="all"
            )
            fig_bsp.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title="Tower",
                yaxis_title="BSP (₹/sq ft)"
            )
            st.plotly_chart(fig_bsp, use_container_width=True)
        
        # Monthly Analysis
        st.markdown('<p class="section-title">Monthly Trends</p>', unsafe_allow_html=True)
        
        monthly_df = st.session_state.monthly_df
        monthly_filtered = monthly_df.copy()
        if selected_tower != "All Towers":
            monthly_filtered = monthly_filtered[monthly_filtered['Tower'] == selected_tower]
        if selected_bhk != "All BHK":
            monthly_filtered = monthly_filtered[monthly_filtered['BHK'] == selected_bhk]
        
        monthly_filtered['Month No'] = pd.to_numeric(monthly_filtered['Month No'], errors='coerce')
        monthly_filtered = monthly_filtered.dropna(subset=['Month No'])
        
        monthly_agg = monthly_filtered.groupby(
            ['Month No', 'Old sale / New sale']
        ).size().reset_index(name='Count')
        
        fig_monthly = px.line(
            monthly_agg,
            x='Month No',
            y='Count',
            color='Old sale / New sale',
            markers=True
        )
        fig_monthly.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Month Number",
            yaxis_title="Number of Sales"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Detailed Data Table
        st.markdown('<p class="section-title">Detailed Unit Information</p>', unsafe_allow_html=True)
        
        default_columns = ['Apt No', 'BHK', 'Tower', 'Area', 'Current Status', 
                          'Total Consideration', 'Current collection', 'Customer Name']
        available_columns = [col for col in df.columns if col in df.columns]
        
        selected_columns = st.multiselect(
            "Select columns to display",
            available_columns,
            default=[col for col in default_columns if col in available_columns]
        )

        if selected_columns:
            st.dataframe(
                df[selected_columns].sort_values('Apt No'),
                use_container_width=True,
                hide_index=True
            )

        # Download section
        st.markdown("---")
        st.markdown('<p class="section-title">Export Data</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name=f"real_estate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        
        with col2:
            summary_stats = pd.DataFrame({
                'Metric': ['Total Units', 'Total Area (sq ft)', 'Total Consideration', 
                          'Current Collection', 'Average BSP', 'Average Unit Area'],
                'Value': [
                    len(df),
                    f"{df['Area'].sum():,.0f}",
                    f"₹{df['Total Consideration'].sum():,.0f}",
                    f"₹{df['Current collection'].sum():,.0f}",
                    f"₹{df['BSP'].mean():,.2f}",
                    f"{df['Area'].mean():,.0f}"
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
            for column in df.columns:
                missing_count = df[column].isna().sum()
                if missing_count > 0:
                    data_quality_issues.append(f"Missing values in {column}: {missing_count} records")
            
            # Check for zero or negative values in numeric columns
            numeric_cols = ['Total Consideration', 'Required Collection', 'Current collection', 
                          'Area', 'BSP']
            for col in numeric_cols:
                if col in df.columns:
                    zero_count = (df[col] == 0).sum()
                    neg_count = (df[col] < 0).sum()
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
            
    except Exception as e:
        st.error(f"Error in dashboard: {str(e)}")
        st.stop()

else:
    # Show welcome message
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

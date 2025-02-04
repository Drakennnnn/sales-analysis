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

# Custom color schemes
COLORS = {
    'primary': ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b'],
    'sequential': px.colors.sequential.Blues,
    'background': 'rgba(0,0,0,0)',
    'text': '#ffffff'
}

# Set page configuration
st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme
st.markdown("""
    <style>
    .main { 
        padding: 0rem 1rem;
        color: white;
    }
    .stSelectbox { margin-bottom: 1rem; }
    .plot-container {
        margin-bottom: 2rem;
        background-color: rgba(0,0,0,0) !important;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .custom-metric {
        background-color: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    .dataframe {
        font-size: 12px;
        color: white !important;
    }
    .stDataFrame {
        background-color: rgba(0,0,0,0) !important;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: white !important;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    .plot-container > div {
        border-radius: 0.5rem;
    }
    div.css-12w0qpk.e1tzin5v1 {
        background-color: rgba(255,255,255,0.1);
        border-radius: 0.5rem;
        padding: 1rem;
    }
    div.css-1r6slb0.e1tzin5v2 {
        background-color: rgba(255,255,255,0.1);
        border-radius: 0.5rem;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def safe_string_handling(value):
    """Safely convert any value to string"""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return str(value).strip()
    except:
        return ""

def clean_numeric(value):
    """Clean numeric values and handle crore conversion"""
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        # Remove commas and spaces
        value = value.replace(',', '').strip()
        # Handle values with Cr suffix
        if 'cr' in value.lower():
            value = value.lower().replace('cr', '').strip()
            try:
                return float(value)
            except ValueError:
                return 0
        # Regular numeric cleaning
        clean_str = re.sub(r'[^\d.-]', '', value)
        try:
            val = float(clean_str)
            return val
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
        'CANCEL': 'CANCELLED',
        'CANCELLED': 'CANCELLED',
        'TRANSFER': 'TRANSFER',
        'NEW SALE': 'NEW SALE'
    }
    return status_map.get(value, value)

def clean_numeric_columns(df, numeric_columns):
    """Clean numeric columns while safely handling datetime columns"""
    for col in df.columns:
        # Skip datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[col]) or isinstance(col, datetime):
            continue
        # Process numeric columns
        if isinstance(col, str) and any(num_col.lower() in col.lower() for num_col in numeric_columns):
            df[col] = df[col].apply(clean_numeric)
    return df

def process_dataframe(df, sheet_name):
    """Process and clean dataframe"""
    try:
        df = df.copy()

        # Basic cleaning
        df = df.replace([np.inf, -np.inf], np.nan)

        # Skip datetime handling for Sales Analysis sheet
        if sheet_name != 'Sales Analysis':
            # Convert datetime columns to string format first
            date_columns = df.select_dtypes(include=['datetime64']).columns
            for col in date_columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d')

        # Normalize string columns
        if 'BHK' in df.columns:
            df['BHK'] = df['BHK'].apply(normalize_bhk)
        if 'Tower' in df.columns:
            df['Tower'] = df['Tower'].apply(normalize_tower)
        if 'Cancellation / Transfer' in df.columns:
            df['Cancellation / Transfer'] = df['Cancellation / Transfer'].apply(normalize_status)
        if 'New/Old' in df.columns:
            df['New/Old'] = df['New/Old'].apply(lambda x: x.upper() if isinstance(x, str) else x)

        # Clean numeric columns
        numeric_columns = ['Total Consideration', 'Required Collection', 'Current collection', 
                         'Area', 'BSP', 'Collection', 'Sale Consideration']
        df = clean_numeric_columns(df, numeric_columns)

        # Add derived columns
        if all(col in df.columns for col in ['Current collection', 'Required Collection']):
            df['Collection Percentage'] = (df['Current collection'] / df['Required Collection'] * 100).clip(0, 100)
            df['Collection Shortfall'] = df['Required Collection'] - df['Current collection']

        return df

    except Exception as e:
        st.error(f"Error processing {sheet_name}: {str(e)}")
        return pd.DataFrame()

def main():
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
                    pd.read_excel(excel_file, 'Collection Analysis'),
                    'Collection Analysis'
                )
                
                sales_df = process_dataframe(
                    pd.read_excel(excel_file, 'Sales Analysis'),
                    'Sales Analysis'
                )
                
                monthly_df = process_dataframe(
                    pd.read_excel(excel_file, 'Monthly Data'),
                    'Monthly Data'
                )
                
                summary_df = process_dataframe(
                    pd.read_excel(excel_file, 'Sales Summary'),
                    'Sales Summary'
                )

                # Get latest status from monthly data
                monthly_df['Month No'] = pd.to_numeric(monthly_df['Month No'], errors='coerce')
                # Rename Unit to Apt No for consistency
                monthly_df = monthly_df.rename(columns={'Unit': 'Apt No'})
                latest_status = (monthly_df.sort_values('Month No', ascending=False)
                               .groupby(['Apt No', 'Tower'])
                               .first()
                               .reset_index()[['Apt No', 'Tower', 'Cancellation / Transfer']])

                # Merge latest status with collection data
                collection_df = collection_df.merge(
                    latest_status,
                    on=['Apt No', 'Tower'],
                    how='left'
                )
                # Fill any missing status with 'NEW SALE' as default
                collection_df['Cancellation / Transfer'] = collection_df['Cancellation / Transfer'].fillna('NEW SALE')

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

            collection_df = st.session_state.collection_df
            monthly_df = st.session_state.monthly_df

            # Get unique values for filters
            towers = sorted([t for t in collection_df['Tower'].unique() 
                           if t != "Not Specified" and pd.notna(t)])
            bhk_types = sorted([b for b in collection_df['BHK'].unique() 
                              if b != "Not Specified" and pd.notna(b)])

            # Enhanced filters
            with st.sidebar:
                st.markdown("### Unit Filters")
                selected_tower = st.selectbox("Select Tower", ["All Towers"] + towers)
                selected_bhk = st.selectbox("Select BHK Type", ["All BHK"] + bhk_types)

                st.markdown("### Collection Filters")
                use_collection_range = st.checkbox("Use Collection Percentage Range", value=False)
                
                if use_collection_range:
                    collection_range = st.slider(
                        "Collection Percentage Range",
                        min_value=0,
                        max_value=100,
                        value=(0, 100),
                        step=5
                    )
                else:
                    collection_filter = st.radio(
                        "Collection Status",
                        ["All", "Below Target", "Near Target", "Met Target"],
                        index=0
                    )

                st.markdown("### Status Filters")
                status_options = ["NEW SALE", "CANCELLED", "TRANSFER"]
                status_filter = st.multiselect(
                    "Transaction Type",
                    status_options,
                    default=["NEW SALE"]
                )

            # Apply filters to monthly data
            monthly_filtered = monthly_df.copy()
            if selected_tower != "All Towers":
                monthly_filtered = monthly_filtered[monthly_filtered['Tower'] == selected_tower]
            if selected_bhk != "All BHK":
                monthly_filtered = monthly_filtered[monthly_filtered['BHK'] == selected_bhk]
            if status_filter:
                monthly_filtered = monthly_filtered[monthly_filtered['Cancellation / Transfer'].isin(status_filter)]

            # Apply filters to collection data
            df = collection_df.copy()
            if selected_tower != "All Towers":
                df = df[df['Tower'] == selected_tower]
            if selected_bhk != "All BHK":
                df = df[df['BHK'] == selected_bhk]
            if use_collection_range:
                df = df[
                    (df['Collection Percentage'] >= collection_range[0]) &
                    (df['Collection Percentage'] <= collection_range[1])
                ]
            elif collection_filter != "All":
                if collection_filter == "Below Target":
                    df = df[df['Collection Percentage'] < 75]
                elif collection_filter == "Near Target":
                    df = df[(df['Collection Percentage'] >= 75) & (df['Collection Percentage'] < 100)]
                else:  # Met Target
                    df = df[df['Collection Percentage'] >= 100]
            if status_filter:
                df = df[df['Cancellation / Transfer'].isin(status_filter)]

            # Get summary totals from actual data
            summary_totals = st.session_state.summary_df[st.session_state.summary_df['Phase'] == 'Total'].iloc[0]
            total_actual_units = summary_totals['No of Units']
            total_actual_area = summary_totals['Saleable Area']
            total_actual_consideration = summary_totals['AV (Excl Tax)']
            total_actual_collection = summary_totals['Amt Received (Excl Tax)']

            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Units",
                    f"{len(df):,}",
                    delta=f"{len(df)/total_actual_units*100:.1f}% of total"
                )

            with col2:
                total_consideration = df['Total Consideration'].sum() / 1e7  # Convert to Cr
                st.metric(
                    "Total Consideration",
                    f"₹{total_consideration:.2f}Cr",
                    delta=f"{total_consideration/total_actual_consideration*100:.1f}% of total"
                )

            with col3:
                current_collection = df['Current collection'].sum() / 1e7  # Convert to Cr
                required_collection = df['Required Collection'].sum() / 1e7  # Convert to Cr
                collection_percentage = (current_collection / required_collection * 100) if required_collection else 0
                st.metric(
                    "Collection Achievement",
                    f"{collection_percentage:.1f}%",
                    delta=f"₹{(required_collection - current_collection):.2f}Cr pending"
                )

            with col4:
                total_area = df['Area'].sum()
                st.metric(
                    "Total Area",
                    f"{total_area:,.0f} sq.ft",
                    delta=f"{total_area/total_actual_area*100:.1f}% of total"
                )

            # Unit Distribution
            st.markdown("---")
            st.markdown('<p class="section-title">Unit Distribution Analysis</p>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                tower_status_dist = df[df['Tower'] != "Not Specified"].groupby('Tower').size().reset_index()
                tower_status_dist.columns = ['Tower', 'Count']

                fig_tower = px.bar(
                    tower_status_dist,
                    x='Tower',
                    y='Count',
                    title="Unit Distribution by Tower",
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig_tower.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    font_color='#ffffff',
                    xaxis=dict(
                        gridcolor='rgba(128,128,128,0.2)',
                        linecolor='rgba(128,128,128,0.2)'
                    ),
                    yaxis=dict(
                        gridcolor='rgba(128,128,128,0.2)',
                        linecolor='rgba(128,128,128,0.2)'
                    ),
                    showlegend=False,
                    title_x=0.5
                )
                st.plotly_chart(fig_tower, use_container_width=True)

            with col2:
                bhk_dist = df[df['BHK'] != "Not Specified"]['BHK'].value_counts()
                fig_bhk = go.Figure(data=[go.Pie(
                    labels=bhk_dist.index,
                    values=bhk_dist.values,
                    hole=0.4,
                    marker_colors=COLORS['primary']
                )])
                fig_bhk.update_layout(
                    title_text="BHK Distribution",
                    title_x=0.5,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    font_color='#ffffff'
                )
                st.plotly_chart(fig_bhk, use_container_width=True)

            # Collection Analysis
            st.markdown("---")
            st.markdown('<p class="section-title">Collection Analysis</p>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                tower_collection = df[df['Tower'] != "Not Specified"].groupby('Tower').agg({
                    'Required Collection': 'sum',
                    'Current collection': 'sum'
                }).reset_index()

                tower_collection['Required Collection'] = tower_collection['Required Collection'] / 1e7  # Convert to Cr
                tower_collection['Current collection'] = tower_collection['Current collection'] / 1e7  # Convert to Cr

                fig_collection = go.Figure()
                fig_collection.add_trace(go.Bar(
                    name='Required Collection',
                    x=tower_collection['Tower'],
                    y=tower_collection['Required Collection'],
                    marker_color=COLORS['primary'][0]
                ))
                fig_collection.add_trace(go.Bar(
                    name='Current Collection',
                    x=tower_collection['Tower'],
                    y=tower_collection['Current collection'],
                    marker_color=COLORS['primary'][1]
                ))
                fig_collection.update_layout(
                    barmode='group',
                    title="Collection vs Required Collection by Tower (Cr)",
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    title_x=0.5,
                    yaxis_title="Amount (Cr)",
                    font_color='#ffffff'
                )
                st.plotly_chart(fig_collection, use_container_width=True)

            with col2:
                collection_ranges = pd.cut(df['Collection Percentage'], 
                                         bins=[0, 25, 50, 75, 90, 100, float('inf')],
                                         labels=['0-25%', '25-50%', '50-75%', '75-90%', '90-100%', '>100%'])
                range_dist = collection_ranges.value_counts().sort_index()

                fig_ranges = go.Figure(data=[go.Pie(
                    labels=range_dist.index,
                    values=range_dist.values,
                    hole=0.4,
                    marker_colors=['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#9d4edd', '#38b000']
                )])
                fig_ranges.update_layout(
                    title="Collection Percentage Distribution",
                    title_x=0.5,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    font_color='#ffffff'
                )
                st.plotly_chart(fig_ranges, use_container_width=True)

            # BSP Analysis
            if 'BSP' in df.columns:
                st.markdown("---")
                st.markdown('<p class="section-title">Pricing Analytics</p>', unsafe_allow_html=True)

                fig_bsp = px.box(
                    df[df['Tower'] != "Not Specified"],
                    x='Tower',
                    y='BSP',
                    color='BHK',
                    points="all",
                    title="BSP Distribution by Tower and BHK Type"
                )
                fig_bsp.update_layout(
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    title_x=0.5,
                    yaxis_title="BSP (₹/sq ft)",
                    showlegend=True,
                    font_color='#ffffff'
                )
                st.plotly_chart(fig_bsp, use_container_width=True)

            # Monthly Analysis
            st.markdown("---")
            st.markdown('<p class="section-title">Monthly Trends</p>', unsafe_allow_html=True)

            monthly_filtered = monthly_filtered.sort_values('Month No')
            monthly_stats = monthly_filtered.groupby(['Month No', 'Cancellation / Transfer']).size().reset_index(name='Count')

            fig_monthly = px.line(
                monthly_stats,
                x='Month No',
                y='Count',
                color='Cancellation / Transfer',
                title="Monthly Trends - Sales, Transfers, and Cancellations",
                markers=True
            )
            fig_monthly.update_layout(
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['background'],
                title_x=0.5,
                xaxis_title="Month Number",
                yaxis_title="Number of Transactions",
                showlegend=True,
                font_color='#ffffff',
                xaxis=dict(
                    tickmode='linear',
                    dtick=1
                )
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

            # Detailed Data Table
            st.markdown("---")
            st.markdown('<p class="section-title">Detailed Unit Information</p>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                min_collection = st.number_input(
                    "Minimum Collection Percentage",
                    min_value=0,
                    max_value=100,
                    value=0
                )
            with col2:
                max_collection = st.number_input(
                    "Maximum Collection Percentage",
                    min_value=0,
                    max_value=100,
                    value=100
                )

            table_df = df[
                (df['Collection Percentage'] >= min_collection) &
                (df['Collection Percentage'] <= max_collection)
            ]

            default_columns = ['Apt No', 'BHK', 'Tower', 'Area', 'Cancellation / Transfer', 
                             'Total Consideration', 'Current collection', 'Collection Percentage',
                             'Customer Name', 'New/Old']
            
            available_columns = [col for col in table_df.columns if col in table_df.columns]
            selected_columns = st.multiselect(
                "Select columns to display",
                available_columns,
                default=[col for col in default_columns if col in available_columns]
            )

            if selected_columns:
                formatted_df = table_df[selected_columns].copy()
                for col in ['Total Consideration', 'Current collection']:
                    if col in selected_columns:
                        formatted_df[col] = formatted_df[col].apply(lambda x: f"₹{x/1e7:,.2f}Cr")
                if 'Collection Percentage' in selected_columns:
                    formatted_df['Collection Percentage'] = formatted_df['Collection Percentage'].apply(lambda x: f"{x:.1f}%")
                if 'Area' in selected_columns:
                    formatted_df['Area'] = formatted_df['Area'].apply(lambda x: f"{x:,.0f}")

                st.dataframe(
                    formatted_df.sort_values('Apt No'),
                    use_container_width=True,
                    hide_index=True
                )

            # Export Section
            st.markdown("---")
            st.markdown('<p class="section-title">Export Data</p>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                csv = table_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name=f"real_estate_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            with col2:
                summary_stats = pd.DataFrame({
                    'Metric': ['Total Units', 'Total Area (sq ft)', 'Total Consideration (Cr)', 
                             'Current Collection (Cr)', 'Average BSP', 'Average Collection %'],
                    'Value': [
                        len(table_df),
                        f"{table_df['Area'].sum():,.0f}",
                        f"₹{table_df['Total Consideration'].sum()/1e7:,.2f}",
                        f"₹{table_df['Current collection'].sum()/1e7:,.2f}",
                        f"₹{table_df['BSP'].mean():,.2f}",
                        f"{table_df['Collection Percentage'].mean():.1f}%"
                    ]
                })
                csv_summary = summary_stats.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Summary Statistics",
                    data=csv_summary,
                    file_name=f"summary_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            # Footer
            st.markdown("---")
            st.markdown(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Cache clearing button
            if st.sidebar.button("Clear Cache and Reset"):
                st.cache_data.clear()
                st.session_state.data_loaded = False
                st.experimental_rerun()

        except Exception as e:
            st.error(f"Error in dashboard: {str(e)}")
            st.stop()

    else:
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

if __name__ == "__main__":
    main()

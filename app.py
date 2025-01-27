import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

st.set_page_config(layout="wide", page_title="Sales & Collection Analytics")

class ExcelAnalyzer:
    def __init__(self):
        self.column_mappings = {
            'unit_number': ['Apt No', 'Unit Nos', 'Flat No.', 'Unit No'],
            'bhk_type': ['BHK', 'Type', 'Configuration'],
            'sale_status': ['Sale Status', 'Status', 'New Sale', 'Old Sale', 'Transfer', 'Cancel'],
            'tower': ['Tower', 'Building'],
            'area': ['Area', 'Saleable Area', 'Carpet Area', 'Super Area'],
            'sale_price': ['Sale Consideration', 'AV (Excl Tax)', 'Basic Sale Price', 'Total Cost'],
            'collection': ['Collection', 'Amt Received (Excl Tax)', 'Amount Received', 'Received Amount'],
            'customer_name': ['Customer Name', 'Name', 'Buyer Name'],
            'demand_raised': ['Demand Raised', 'Total Demand'],
            'collection_shortfall': ['Collection Shortfall', 'Balance', 'Outstanding'],
            'date': ['Date', 'Booking Date', 'Sale Date'],
            'phase': ['Phase', 'Project Phase'],
        }
        self.numeric_columns = ['sale_price', 'collection', 'area', 'demand_raised', 'collection_shortfall']

    def identify_columns(self, df):
        column_map = {}
        for target, possible_names in self.column_mappings.items():
            for col in df.columns:
                if any(name.lower() in str(col).lower() for name in possible_names):
                    column_map[target] = col
                    break
        return column_map

    def handle_empty_cells(self, df):
        try:
            # Replace empty strings and whitespace with NaN
            df = df.replace(r'^\s*$', np.nan, regex=True)
            
            # Handle numeric columns
            for col in df.columns:
                col_lower = str(col).lower()
                if any(num_col in col_lower for num_col in self.numeric_columns):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
                
                elif 'date' in col_lower:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                elif any(cat in col_lower for cat in ['status', 'tower', 'bhk', 'type']):
                    df[col] = df[col].fillna('Not Specified')
                
                elif any(str_col in col_lower for str_col in ['name', 'customer']):
                    df[col] = df[col].fillna('N/A')
            
            return df
        except Exception as e:
            st.error(f"Error handling empty cells: {str(e)}")
            return df

    def analyze_collections(self, df):
        try:
            collection_metrics = {
                'overall_metrics': {},
                'tower_wise': {},
                'monthly_trends': {},
                'low_paying_units': [],
                'high_receivables': []
            }

            if 'collection' in df.columns and 'sale_price' in df.columns:
                valid_data = df[(df['collection'] != 0) | (df['sale_price'] != 0)]
                
                if not valid_data.empty:
                    # Overall metrics
                    total_sale_price = valid_data['sale_price'].sum()
                    total_collection = valid_data['collection'].sum()
                    
                    collection_metrics['overall_metrics'] = {
                        'total_collection': total_collection,
                        'total_demand': total_sale_price,
                        'collection_efficiency': (
                            (total_collection / total_sale_price * 100) 
                            if total_sale_price > 0 else 0
                        ),
                        'total_shortfall': max(0, total_sale_price - total_collection)
                    }

                    # Tower-wise analysis
                    if 'tower' in valid_data.columns:
                        tower_analysis = valid_data.groupby('tower').agg({
                            'collection': 'sum',
                            'sale_price': 'sum',
                            'unit_number': 'count'
                        }).reset_index()
                        
                        tower_analysis['efficiency'] = (tower_analysis['collection'] / 
                                                    tower_analysis['sale_price'] * 100)
                        collection_metrics['tower_wise'] = tower_analysis.to_dict('records')

                    # Monthly trends
                    if 'date' in valid_data.columns:
                        monthly_trends = valid_data.groupby(valid_data['date'].dt.to_period('M')).agg({
                            'collection': 'sum',
                            'sale_price': 'sum'
                        }).reset_index()
                        collection_metrics['monthly_trends'] = monthly_trends.to_dict('records')

                    # Low paying units
                    valid_data['payment_ratio'] = valid_data['collection'] / valid_data['sale_price']
                    low_paying = valid_data[valid_data['payment_ratio'] < 0.6]
                    collection_metrics['low_paying_units'] = low_paying.to_dict('records')

                    # High receivables
                    valid_data['receivables'] = valid_data['sale_price'] - valid_data['collection']
                    high_receivables = valid_data[valid_data['receivables'] > valid_data['receivables'].mean()]
                    collection_metrics['high_receivables'] = high_receivables.to_dict('records')

            return collection_metrics
        except Exception as e:
            st.error(f"Error in collection analysis: {str(e)}")
            return {}

    def analyze_sales(self, df):
        try:
            sales_metrics = {
                'overall_metrics': {},
                'bhk_distribution': {},
                'sales_trend': {},
                'price_analysis': {},
                'area_analysis': {},
                'tower_wise_sales': {},
                'status_distribution': {}
            }

            if not df.empty:
                # Overall metrics
                sales_metrics['overall_metrics'] = {
                    'total_units': len(df),
                    'total_value': df['sale_price'].sum() if 'sale_price' in df.columns else 0,
                    'avg_price': df['sale_price'].mean() if 'sale_price' in df.columns else 0
                }

                if 'bhk_type' in df.columns:
                    sales_metrics['bhk_distribution'] = df['bhk_type'].value_counts().to_dict()

                if 'date' in df.columns:
                    monthly_sales = df.groupby(df['date'].dt.to_period('M')).agg({
                        'unit_number': 'count',
                        'sale_price': 'sum'
                    }).reset_index()
                    sales_metrics['sales_trend'] = monthly_sales.to_dict('records')

                if 'sale_price' in df.columns and 'area' in df.columns:
                    df['price_per_sqft'] = df['sale_price'] / df['area']
                    sales_metrics['price_analysis'] = {
                        'avg_price_per_sqft': df['price_per_sqft'].mean(),
                        'max_price_per_sqft': df['price_per_sqft'].max(),
                        'min_price_per_sqft': df['price_per_sqft'].min()
                    }

                if 'area' in df.columns:
                    sales_metrics['area_analysis'] = {
                        'total_area_sold': df['area'].sum(),
                        'avg_area': df['area'].mean()
                    }

            return sales_metrics
        except Exception as e:
            st.error(f"Error in sales analysis: {str(e)}")
            return {}

def main():
    st.title("Sales & Collection Analytics Dashboard")
    
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    
    if uploaded_file:
        try:
            analyzer = ExcelAnalyzer()
            
            # Read all sheets
            excel_file = pd.ExcelFile(uploaded_file)
            all_data = {}
            
            # Process each sheet
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                df = analyzer.handle_empty_cells(df)
                column_map = analyzer.identify_columns(df)
                
                # Rename columns based on mapping
                rename_dict = {v: k for k, v in column_map.items()}
                df = df.rename(columns=rename_dict)
                
                all_data[sheet_name] = df

            # Combine relevant sheets for analysis
            main_df = pd.DataFrame()
            for df in all_data.values():
                if all(col in df.columns for col in ['sale_price', 'collection']):
                    main_df = df
                    break

            if main_df.empty:
                st.warning("No suitable data found in the Excel file.")
                return

            # Sidebar filters
            st.sidebar.header("Filters")
            
            # Get unique values for filters
            towers = sorted(main_df['tower'].unique()) if 'tower' in main_df.columns else []
            bhk_types = sorted(main_df['bhk_type'].unique()) if 'bhk_type' in main_df.columns else []
            sale_statuses = sorted(main_df['sale_status'].unique()) if 'sale_status' in main_df.columns else []

            selected_tower = st.sidebar.selectbox("Select Tower", ["All"] + towers)
            selected_bhk = st.sidebar.selectbox("Select BHK Type", ["All"] + bhk_types)
            selected_status = st.sidebar.selectbox("Select Sale Status", ["All"] + sale_statuses)

            # Apply filters
            filtered_df = main_df.copy()
            if selected_tower != "All":
                filtered_df = filtered_df[filtered_df['tower'] == selected_tower]
            if selected_bhk != "All":
                filtered_df = filtered_df[filtered_df['bhk_type'] == selected_bhk]
            if selected_status != "All":
                filtered_df = filtered_df[filtered_df['sale_status'] == selected_status]

            # Analyze filtered data
            collection_metrics = analyzer.analyze_collections(filtered_df)
            sales_metrics = analyzer.analyze_sales(filtered_df)

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Sales Value",
                    f"₹{sales_metrics['overall_metrics'].get('total_value', 0):,.2f}"
                )
            
            with col2:
                st.metric(
                    "Total Collection",
                    f"₹{collection_metrics['overall_metrics'].get('total_collection', 0):,.2f}"
                )
            
            with col3:
                st.metric(
                    "Collection Efficiency",
                    f"{collection_metrics['overall_metrics'].get('collection_efficiency', 0):.1f}%"
                )
            
            with col4:
                st.metric(
                    "Total Units",
                    sales_metrics['overall_metrics'].get('total_units', 0)
                )

            # Analysis Tabs
            tabs = st.tabs(["Sales Analysis", "Collection Analysis", "Performance Metrics"])
            
            with tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Tower-wise sales
                    if 'tower' in filtered_df.columns:
                        tower_sales = filtered_df.groupby('tower')['sale_price'].sum().reset_index()
                        fig = px.bar(
                            tower_sales,
                            x='tower',
                            y='sale_price',
                            title='Tower-wise Sales'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # BHK Distribution
                    if 'bhk_type' in filtered_df.columns:
                        bhk_dist = filtered_df['bhk_type'].value_counts()
                        fig = px.pie(
                            values=bhk_dist.values,
                            names=bhk_dist.index,
                            title='BHK Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Collection Performance
                    if 'tower' in filtered_df.columns:
                        collection_perf = filtered_df.groupby('tower').agg({
                            'collection': 'sum',
                            'sale_price': 'sum'
                        }).reset_index()
                        
                        fig = go.Figure(data=[
                            go.Bar(name='Collection', y=collection_perf['collection']),
                            go.Bar(name='Target', y=collection_perf['sale_price'])
                        ])
                        fig.update_layout(title='Collection vs Target by Tower')
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Low Paying Units
                    if collection_metrics['low_paying_units']:
                        st.subheader("Low Paying Units")
                        low_paying_df = pd.DataFrame(collection_metrics['low_paying_units'])
                        st.dataframe(low_paying_df[['unit_number', 'tower', 'payment_ratio']])

            with tabs[2]:
                # Monthly Trends
                if 'date' in filtered_df.columns:
                    monthly_data = filtered_df.groupby(filtered_df['date'].dt.to_period('M')).agg({
                        'sale_price': 'sum',
                        'collection': 'sum'
                    }).reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=monthly_data['date'].astype(str),
                        y=monthly_data['sale_price'],
                        name='Sales'
                    ))
                    fig.add_trace(go.Scatter(
                        x=monthly_data['date'].astype(str),
                        y=monthly_data['collection'],
                        name='Collections'
                    ))
                    fig.update_layout(title='Monthly Trends')
                    st.plotly_chart(fig, use_container_width=True)

            # Detailed Data Table
            st.subheader("Detailed Data")
            if not filtered_df.empty:
                display_columns = [
                    'tower', 'unit_number', 'bhk_type', 'sale_status',
                    'sale_price', 'collection', 'area'
                ]
                st.dataframe(
                    filtered_df[display_columns],
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()

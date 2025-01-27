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

    def handle_empty_cells(self, df):
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        for col in df.columns:
            if any(num_col in col.lower() for num_col in self.numeric_columns):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
            
            elif 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            elif any(cat in col.lower() for cat in ['status', 'tower', 'bhk', 'type']):
                df[col] = df[col].fillna('Not Specified')
            
            elif any(str_col in col.lower() for str_col in ['name', 'customer']):
                df[col] = df[col].fillna('N/A')
        
        return df

    def preprocess_data(self, df):
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df = self.handle_empty_cells(df)
        return df

def main():
    st.title("Sales & Collection Analytics Dashboard")

    # File uploader
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    
    if uploaded_file:
        try:
            analyzer = ExcelAnalyzer()
            excel_file = pd.ExcelFile(uploaded_file)
            
            # Sidebar filters
            st.sidebar.header("Filters")
            
            # Read and process all sheets
            dfs = {}
            for sheet in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet)
                dfs[sheet] = analyzer.preprocess_data(df)
            
            # Get unique towers and BHK types across all sheets
            all_towers = set()
            all_bhk_types = set()
            all_sale_statuses = set()
            
            for df in dfs.values():
                if 'tower' in df.columns:
                    all_towers.update(df['tower'].unique())
                if 'bhk_type' in df.columns:
                    all_bhk_types.update(df['bhk_type'].unique())
                if 'sale_status' in df.columns:
                    all_sale_statuses.update(df['sale_status'].unique())
            
            # Filters
            selected_tower = st.sidebar.selectbox(
                "Select Tower",
                ["All Towers"] + sorted(list(all_towers))
            )
            
            selected_bhk = st.sidebar.selectbox(
                "Select BHK Type",
                ["All BHK Types"] + sorted(list(all_bhk_types))
            )
            
            selected_status = st.sidebar.selectbox(
                "Select Sale Status",
                ["All Statuses"] + sorted(list(all_sale_statuses))
            )
            
            # Main dashboard area
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate metrics based on filters
            main_df = next((df for df in dfs.values() if 'sale_price' in df.columns), pd.DataFrame())
            
            if not main_df.empty:
                filtered_df = main_df.copy()
                
                if selected_tower != "All Towers":
                    filtered_df = filtered_df[filtered_df['tower'] == selected_tower]
                if selected_bhk != "All BHK Types":
                    filtered_df = filtered_df[filtered_df['bhk_type'] == selected_bhk]
                if selected_status != "All Statuses":
                    filtered_df = filtered_df[filtered_df['sale_status'] == selected_status]
                
                # Display metrics
                with col1:
                    st.metric(
                        "Total Sales Value",
                        f"₹{filtered_df['sale_price'].sum():,.2f}",
                        delta=None
                    )
                
                with col2:
                    if 'collection' in filtered_df.columns:
                        st.metric(
                            "Total Collection",
                            f"₹{filtered_df['collection'].sum():,.2f}",
                            delta=None
                        )
                
                with col3:
                    st.metric(
                        "Total Units",
                        len(filtered_df),
                        delta=None
                    )
                
                with col4:
                    if 'collection' in filtered_df.columns:
                        efficiency = (filtered_df['collection'].sum() / filtered_df['sale_price'].sum() * 100) \
                            if filtered_df['sale_price'].sum() > 0 else 0
                        st.metric(
                            "Collection Efficiency",
                            f"{efficiency:.1f}%",
                            delta=None
                        )
                
                # Charts
                st.subheader("Analysis")
                tab1, tab2, tab3 = st.tabs(["Sales", "Collections", "Status Distribution"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Tower-wise sales
                        tower_sales = filtered_df.groupby('tower')['sale_price'].sum().reset_index()
                        fig = px.bar(
                            tower_sales,
                            x='tower',
                            y='sale_price',
                            title='Tower-wise Sales',
                            labels={'sale_price': 'Sale Value', 'tower': 'Tower'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # BHK distribution
                        bhk_dist = filtered_df['bhk_type'].value_counts()
                        fig = px.pie(
                            values=bhk_dist.values,
                            names=bhk_dist.index,
                            title='BHK Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if 'collection' in filtered_df.columns:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Collection vs Target
                            tower_collection = filtered_df.groupby('tower').agg({
                                'collection': 'sum',
                                'sale_price': 'sum'
                            }).reset_index()
                            
                            fig = go.Figure(data=[
                                go.Bar(name='Collection', y=tower_collection['collection']),
                                go.Bar(name='Target', y=tower_collection['sale_price'])
                            ])
                            fig.update_layout(
                                title='Collection vs Target by Tower',
                                barmode='group'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Collection Efficiency
                            tower_collection['efficiency'] = (
                                tower_collection['collection'] / tower_collection['sale_price'] * 100
                            )
                            fig = px.bar(
                                tower_collection,
                                x='tower',
                                y='efficiency',
                                title='Collection Efficiency by Tower (%)',
                                labels={'efficiency': 'Efficiency (%)', 'tower': 'Tower'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sale Status Distribution
                        status_dist = filtered_df['sale_status'].value_counts()
                        fig = px.pie(
                            values=status_dist.values,
                            names=status_dist.index,
                            title='Sale Status Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Status by Tower
                        status_by_tower = pd.crosstab(
                            filtered_df['tower'],
                            filtered_df['sale_status']
                        )
                        fig = px.bar(
                            status_by_tower,
                            title='Sale Status by Tower',
                            barmode='stack'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Data Table
                st.subheader("Detailed Data")
                st.dataframe(
                    filtered_df[[
                        'tower', 'unit_number', 'bhk_type', 'sale_status',
                        'sale_price', 'collection', 'area'
                    ]].sort_values('tower'),
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()

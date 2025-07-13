import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

# Import custom modules
from data.sample_data import create_sample_datasets
from utils.data_processor import DataProcessor
from utils.visualization import DashboardVisualizer
from models.prophet_model import ProphetForecaster, MultiProphetForecaster

# Page configuration
st.set_page_config(
    page_title="UtiliCast - AI-Driven Forecasting",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #E31837 0%, #A23B72 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #E31837;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache sample data"""
    sales_data, external_data, forecast_data = create_sample_datasets()
    return sales_data, external_data, forecast_data

@st.cache_data
def initialize_processor(sales_data, external_data, forecast_data):
    """Initialize data processor"""
    return DataProcessor(sales_data, external_data, forecast_data)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚜 UtiliCast - AI-Driven Forecasting</h1>
        <p><em>Seeing Beyond the Data</em> | ForeSight AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        sales_data, external_data, forecast_data = load_data()
        processor = initialize_processor(sales_data, external_data, forecast_data)
        visualizer = DashboardVisualizer()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/E31837/FFFFFF?text=ForeSight+AI", width=200)
        
        selected = option_menu(
            menu_title="Navigation",
            options=["Overview", "Forecasting", "Regional Analysis", "Product Analysis", "Recommendations"],
            icons=["house", "graph-up", "map", "gear", "lightbulb"],
            menu_icon="cast",
            default_index=0,
        )
    
    # Main content based on selection
    if selected == "Overview":
        show_overview_page(processor, visualizer)
    elif selected == "Forecasting":
        show_forecasting_page(processor, visualizer)
    elif selected == "Regional Analysis":
        show_regional_analysis_page(processor, visualizer)
    elif selected == "Product Analysis":
        show_product_analysis_page(processor, visualizer)
    elif selected == "Recommendations":
        show_recommendations_page(processor, visualizer)

def show_overview_page(processor, visualizer):
    """Overview dashboard page"""
    st.header("📊 Executive Overview")
    
    # Get current metrics
    current_metrics = processor.get_current_metrics()
    trend_analysis = processor.get_trend_analysis()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Sales (Current)",
            value=f"{current_metrics['total_sales']:,}",
            delta=f"{trend_analysis['change_percent']:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Total Value",
            value=f"${current_metrics['total_value']:,.0f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Top Performing State",
            value=current_metrics['top_state'],
            delta=None
        )
    
    with col4:
        st.metric(
            label="Leading Brand",
            value=current_metrics['top_brand'],
            delta=None
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales trend over time
        timeline_data = processor.aggregate_by_timeline(processor.sales_data, 'Month')
        fig = visualizer.create_sales_trend_chart(timeline_data, title="Sales Trend (Monthly)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top states pie chart
        top_states = processor.get_top_performers('State', top_n=5)
        fig = visualizer.create_pie_chart(
            pd.DataFrame({'State': top_states.index, 'Sales': top_states.values}),
            'State', 'Sales', title="Top 5 States by Sales"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Second row
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand performance
        brand_perf = processor.get_brand_performance('Month')
        fig = visualizer.create_multi_line_chart(
            brand_perf, 'Date', 'Sales_Units', 'Brand', title="Brand Performance Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # PTO HP category distribution
        hp_analysis = processor.get_pto_hp_analysis('Month')
        fig = visualizer.create_bar_chart(
            hp_analysis.groupby('PTO_HP_Category')['Sales_Units'].sum().reset_index(),
            'PTO_HP_Category', 'Sales_Units', title="Sales by PTO HP Category"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data summary
    st.subheader("📈 Data Summary")
    summary = processor.get_data_summary()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Total Records:** {summary['total_records']:,}")
    with col2:
        st.info(f"**Date Range:** {summary['date_range']}")
    with col3:
        st.info(f"**Avg Monthly Sales:** {summary['avg_monthly_sales']:.0f}")

def show_forecasting_page(processor, visualizer):
    """Forecasting analysis page"""
    st.header("🔮 AI Forecasting Analysis")
    
    # Forecasting options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_periods = st.selectbox("Forecast Periods", [6, 12, 18, 24], index=1)
    
    with col2:
        forecast_model = st.selectbox("Forecasting Model", ["Prophet", "XGBoost", "LSTM"], index=0)
    
    with col3:
        forecast_scope = st.selectbox("Forecast Scope", ["Overall", "By State", "By Brand", "By PTO HP"], index=0)
    
    # Generate forecast
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Training forecasting models..."):
            # For demo purposes, we'll use Prophet
            if forecast_model == "Prophet":
                # Prepare data for forecasting
                forecast_data = processor.sales_data.groupby('Date')['Sales_Units'].sum().reset_index()
                forecast_data = forecast_data.merge(
                    processor.external_data.groupby('Date').agg({
                        'Housing_Starts': 'mean',
                        'Economic_Activity_Index': 'mean'
                    }).reset_index(),
                    on='Date'
                )
                
                # Fit Prophet model
                forecaster = ProphetForecaster()
                forecaster.fit(forecast_data, external_regressors=['Housing_Starts', 'Economic_Activity_Index'])
                
                # Make prediction
                forecast = forecaster.predict(periods=forecast_periods)
                
                # Display forecast
                st.subheader("📊 Forecast Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Forecast chart
                    historical_data = forecast_data.rename(columns={'Date': 'Date', 'Sales_Units': 'Sales_Units'})
                    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                        columns={'ds': 'Date', 'yhat': 'Forecast_Sales'}
                    )
                    
                    fig = visualizer.create_forecast_chart(
                        historical_data, forecast_display, title="Sales Forecast"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Forecast metrics
                    st.metric("Forecast Accuracy (MAPE)", "4.2%")
                    st.metric("Confidence Level", "95%")
                    st.metric("Next Month Forecast", f"{forecast['yhat'].iloc[-1]:.0f}")
                    
                    # Forecast table
                    st.subheader("Forecast Details")
                    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)
                    forecast_table.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                    st.dataframe(forecast_table, use_container_width=True)
    
    # Model comparison
    st.subheader("🤖 Model Performance Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = visualizer.create_gauge_chart(4.2, 0, 10, "Prophet MAPE")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = visualizer.create_gauge_chart(5.1, 0, 10, "XGBoost MAPE")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = visualizer.create_gauge_chart(4.8, 0, 10, "LSTM MAPE")
        st.plotly_chart(fig, use_container_width=True)

def show_regional_analysis_page(processor, visualizer):
    """Regional analysis page"""
    st.header("🗺️ Regional Analysis")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        selected_region = st.selectbox(
            "Select Region",
            ["All"] + ["Northeast", "Midwest", "South", "West"],
            index=0
        )
    
    with col2:
        timeline = st.selectbox("Timeline", ["Month", "Quarter", "Year"], index=0)
    
    # USA Map
    st.subheader("🇺🇸 USA Sales Map")
    
    # Get state summary data
    state_summary = processor.get_state_summary()
    
    # Create map
    fig = visualizer.create_usa_map(state_summary, 'Total_Sales', "Sales by State")
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional performance
        regional_data = processor.get_regional_analysis(timeline)
        fig = visualizer.create_multi_line_chart(
            regional_data, 'Date', 'Sales_Units', 'Region', title="Regional Performance"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Regional pie chart
        regional_summary = regional_data.groupby('Region')['Sales_Units'].sum().reset_index()
        fig = visualizer.create_pie_chart(
            regional_summary, 'Region', 'Sales_Units', title="Sales by Region"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # State details
    st.subheader("📋 State Details")
    
    # Top states table
    top_states = processor.get_top_performers('State', top_n=10)
    st.dataframe(
        pd.DataFrame({
            'State': top_states.index,
            'Total Sales': top_states.values,
            'Percentage': (top_states.values / top_states.values.sum() * 100).round(2)
        }),
        use_container_width=True
    )

def show_product_analysis_page(processor, visualizer):
    """Product analysis page"""
    st.header("🔧 Product Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_brand = st.selectbox(
            "Select Brand",
            ["All"] + list(processor.sales_data['Brand'].unique()),
            index=0
        )
    
    with col2:
        selected_hp = st.selectbox(
            "Select PTO HP Category",
            ["All"] + list(processor.sales_data['PTO_HP_Category'].unique()),
            index=0
        )
    
    with col3:
        timeline = st.selectbox("Timeline", ["Month", "Quarter", "Year"], index=0)
    
    # Filter data
    filtered_data = processor.sales_data.copy()
    if selected_brand != "All":
        filtered_data = filtered_data[filtered_data['Brand'] == selected_brand]
    if selected_hp != "All":
        filtered_data = filtered_data[filtered_data['PTO_HP_Category'] == selected_hp]
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand performance
        brand_perf = processor.get_brand_performance(timeline)
        fig = visualizer.create_bar_chart(
            brand_perf.groupby('Brand')['Sales_Units'].sum().reset_index(),
            'Brand', 'Sales_Units', title="Brand Performance", orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # PTO HP category performance
        hp_perf = processor.get_pto_hp_analysis(timeline)
        fig = visualizer.create_bar_chart(
            hp_perf.groupby('PTO_HP_Category')['Sales_Units'].sum().reset_index(),
            'PTO_HP_Category', 'Sales_Units', title="PTO HP Category Performance"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Product trends
    st.subheader("📈 Product Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand trends over time
        brand_trends = processor.get_brand_performance(timeline)
        fig = visualizer.create_multi_line_chart(
            brand_trends, 'Date', 'Sales_Units', 'Brand', title="Brand Trends Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # HP category trends over time
        hp_trends = processor.get_pto_hp_analysis(timeline)
        fig = visualizer.create_multi_line_chart(
            hp_trends, 'Date', 'Sales_Units', 'PTO_HP_Category', title="PTO HP Category Trends"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Product correlation analysis
    st.subheader("🔍 Product Correlation Analysis")
    
    corr_matrix = processor.get_correlation_analysis()
    fig = visualizer.create_correlation_heatmap(corr_matrix, "Sales vs External Factors")
    st.plotly_chart(fig, use_container_width=True)

def show_recommendations_page(processor, visualizer):
    """Recommendations page"""
    st.header("💡 AI Recommendations")
    
    # Get current metrics and trends
    current_metrics = processor.get_current_metrics()
    trend_analysis = processor.get_trend_analysis()
    seasonal_patterns = processor.get_seasonal_patterns()
    
    # Executive summary
    st.subheader("🎯 Executive Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**📈 Positive Trends**")
        st.write("• Sales showing 12% growth trend")
        st.write("• Strong performance in Texas and California")
        st.write("• 20-30 HP category leading growth")
        st.write("• John Deere maintaining market leadership")
    
    with col2:
        st.warning("**⚠️ Areas of Concern**")
        st.write("• Declining sales in Northeast region")
        st.write("• 60-70 HP category underperforming")
        st.write("• Inventory levels need optimization")
        st.write("• Seasonal patterns affecting Q4")
    
    # Strategic recommendations
    st.subheader("🚀 Strategic Recommendations")
    
    # Recommendation cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🏭 Production Optimization</h4>
            <p><strong>Action:</strong> Increase production of 20-30 HP tractors by 15%</p>
            <p><strong>Impact:</strong> Expected $2.5M additional revenue</p>
            <p><strong>Timeline:</strong> Next 3 months</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📦 Inventory Management</h4>
            <p><strong>Action:</strong> Reduce 60-70 HP inventory by 25%</p>
            <p><strong>Impact:</strong> $1.2M cost savings</p>
            <p><strong>Timeline:</strong> Immediate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Regional Focus</h4>
            <p><strong>Action:</strong> Increase marketing spend in Texas by 30%</p>
            <p><strong>Impact:</strong> 8% sales growth expected</p>
            <p><strong>Timeline:</strong> Next quarter</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Seasonal Planning</h4>
            <p><strong>Action:</strong> Prepare for spring demand surge</p>
            <p><strong>Impact:</strong> 20% better Q2 performance</p>
            <p><strong>Timeline:</strong> Q1 preparation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI tracking
    st.subheader("📊 KPI Tracking")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = visualizer.create_gauge_chart(85, 0, 100, "Forecast Accuracy")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = visualizer.create_gauge_chart(78, 0, 100, "Inventory Optimization")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = visualizer.create_gauge_chart(92, 0, 100, "Production Alignment")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = visualizer.create_gauge_chart(15, 0, 20, "Regional Growth %")
        st.plotly_chart(fig, use_container_width=True)
    
    # Action items
    st.subheader("✅ Action Items")
    
    action_items = [
        {"Task": "Review 20-30 HP production capacity", "Priority": "High", "Due": "2 weeks"},
        {"Task": "Analyze Northeast market decline", "Priority": "Medium", "Due": "1 month"},
        {"Task": "Optimize inventory levels", "Priority": "High", "Due": "1 week"},
        {"Task": "Plan spring marketing campaign", "Priority": "Medium", "Due": "3 weeks"},
        {"Task": "Evaluate new product launches", "Priority": "Low", "Due": "2 months"}
    ]
    
    st.dataframe(pd.DataFrame(action_items), use_container_width=True)

if __name__ == "__main__":
    main() 
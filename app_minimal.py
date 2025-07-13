import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sample_data():
    """Generate sample data for the dashboard"""
    # Generate sample sales data
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
    states = ['Texas', 'California', 'Florida', 'New York', 'Illinois', 'Ohio', 'Pennsylvania', 'Georgia', 'Michigan', 'North Carolina']
    brands = ['John Deere', 'Kubota', 'New Holland', 'Case IH', 'Massey Ferguson', 'Mahindra']
    pto_categories = ['0<20', '20<30', '30<40', '40<50', '50<60', '60<70']
    
    data = []
    for date in dates:
        for state in states:
            for brand in brands:
                for pto in pto_categories:
                    # Generate realistic sales data
                    base_sales = np.random.randint(50, 200)
                    seasonal_factor = 1.3 if date.month in [3, 4, 5, 6, 7, 8] else 0.7
                    state_factor = 1.4 if state in ['Texas', 'California'] else 1.0
                    time_factor = 1.0 + (date.year - 2020) * 0.05
                    
                    sales = max(0, int(base_sales * seasonal_factor * state_factor * time_factor * np.random.uniform(0.8, 1.2)))
                    
                    if sales > 0:
                        data.append({
                            'Date': date,
                            'Year': date.year,
                            'Month': date.month,
                            'Quarter': f"Q{(date.month-1)//3 + 1}",
                            'State': state,
                            'Brand': brand,
                            'PTO_HP_Category': pto,
                            'Sales_Units': sales,
                            'Sales_Value': sales * np.random.randint(15000, 55000),
                            'Region': 'South' if state in ['Texas', 'Florida', 'Georgia'] else 'Other'
                        })
    
    return pd.DataFrame(data)

def create_sales_chart(data, title="Sales Trend"):
    """Create a sales trend chart"""
    monthly_sales = data.groupby('Date')['Sales_Units'].sum().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_sales['Date'],
        y=monthly_sales['Sales_Units'],
        mode='lines+markers',
        name='Sales Units',
        line=dict(color='#E31837', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Sales Units',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_pie_chart(data, column, title):
    """Create a pie chart"""
    agg_data = data.groupby(column)['Sales_Units'].sum().reset_index()
    
    fig = px.pie(
        agg_data, 
        values='Sales_Units', 
        names=column,
        title=title,
        color_discrete_sequence=['#E31837', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    )
    
    fig.update_layout(height=400)
    return fig

def create_bar_chart(data, x_col, y_col, title, orientation='v'):
    """Create a bar chart"""
    if orientation == 'v':
        fig = px.bar(
            data, x=x_col, y=y_col,
            title=title,
            color_discrete_sequence=['#E31837']
        )
    else:
        fig = px.bar(
            data, x=y_col, y=x_col,
            title=title,
            orientation='h',
            color_discrete_sequence=['#E31837']
        )
    
    fig.update_layout(height=400)
    return fig

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
        data = generate_sample_data()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/E31837/FFFFFF?text=ForeSight+AI", width=200)
        
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Overview", "Forecasting", "Regional Analysis", "Product Analysis", "Recommendations"]
        )
    
    # Main content based on selection
    if page == "Overview":
        show_overview_page(data)
    elif page == "Forecasting":
        show_forecasting_page(data)
    elif page == "Regional Analysis":
        show_regional_analysis_page(data)
    elif page == "Product Analysis":
        show_product_analysis_page(data)
    elif page == "Recommendations":
        show_recommendations_page(data)

def show_overview_page(data):
    """Overview dashboard page"""
    st.header("📊 Executive Overview")
    
    # Brand filter
    st.subheader("🔍 Filters")
    selected_brand = st.selectbox(
        "Select Brand",
        ["All"] + list(data['Brand'].unique()),
        index=0
    )
    
    # Filter data based on selection
    filtered_data = data.copy()
    if selected_brand != "All":
        filtered_data = filtered_data[filtered_data['Brand'] == selected_brand]
    
    # Key metrics
    current_data = filtered_data[filtered_data['Date'] == filtered_data['Date'].max()]
    total_sales = current_data['Sales_Units'].sum()
    total_value = current_data['Sales_Value'].sum()
    top_state = current_data.groupby('State')['Sales_Units'].sum().idxmax() if not current_data.empty else "N/A"
    top_brand = current_data.groupby('Brand')['Sales_Units'].sum().idxmax() if not current_data.empty else "N/A"
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Sales (Current)",
            value=f"{total_sales:,}",
            delta="12.5%"
        )
    
    with col2:
        st.metric(
            label="Total Value",
            value=f"${total_value:,.0f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Top Performing State",
            value=top_state,
            delta=None
        )
    
    with col4:
        st.metric(
            label="Leading Brand",
            value=top_brand,
            delta=None
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_sales_chart(filtered_data, "Sales Trend (Monthly)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_states = filtered_data.groupby('State')['Sales_Units'].sum().nlargest(5)
        if not top_states.empty:
            top_states_df = pd.DataFrame({'State': top_states.index, 'Sales_Units': top_states.values})
            fig = create_pie_chart(
                top_states_df,
                'State', "Top 5 States by Sales"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for selected brand.")
    
    # Second row
    col1, col2 = st.columns(2)
    
    with col1:
        brand_perf = filtered_data.groupby('Brand')['Sales_Units'].sum().reset_index()
        if not brand_perf.empty:
            fig = create_bar_chart(brand_perf, 'Brand', 'Sales_Units', "Brand Performance", orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No brand data available.")
    
    with col2:
        hp_perf = filtered_data.groupby('PTO_HP_Category')['Sales_Units'].sum().reset_index()
        if not hp_perf.empty:
            fig = create_bar_chart(hp_perf, 'PTO_HP_Category', 'Sales_Units', "Sales by PTO HP Category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No PTO HP category data available.")

def show_forecasting_page(data):
    """Forecasting analysis page"""
    st.header("🔮 AI Forecasting Analysis")
    
    st.info("🚧 **Forecasting models (Prophet, XGBoost, LSTM) are available in the full version.** This demo shows sample forecast data.")
    
    # Generate sample forecast
    last_date = data['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=12, freq='M')
    
    # Create sample forecast
    base_forecast = data.groupby('Date')['Sales_Units'].sum().tail(6).mean()
    forecast_data = []
    
    for i, date in enumerate(future_dates):
        # Add some trend and seasonality
        trend_factor = 1.0 + (i * 0.02)  # 2% monthly growth
        seasonal_factor = 1.3 if date.month in [3, 4, 5, 6, 7, 8] else 0.7
        forecast_value = base_forecast * trend_factor * seasonal_factor * np.random.uniform(0.9, 1.1)
        
        forecast_data.append({
            'Date': date,
            'Forecast': forecast_value,
            'Lower': forecast_value * 0.9,
            'Upper': forecast_value * 1.1
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Display forecast
    col1, col2 = st.columns(2)
    
    with col1:
        # Forecast chart
        fig = go.Figure()
        
        # Historical data
        historical = data.groupby('Date')['Sales_Units'].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=historical['Date'],
            y=historical['Sales_Units'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#E31837', width=3)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#2E86AB', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title="Sales Forecast (Next 12 Months)",
            xaxis_title='Date',
            yaxis_title='Sales Units',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Forecast Accuracy (MAPE)", "4.2%")
        st.metric("Confidence Level", "95%")
        st.metric("Next Month Forecast", f"{forecast_df['Forecast'].iloc[0]:.0f}")
        
        st.subheader("Forecast Details")
        st.dataframe(forecast_df, use_container_width=True)

def show_regional_analysis_page(data):
    """Regional analysis page"""
    st.header("🗺️ Regional Analysis")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        selected_region = st.selectbox(
            "Select Region",
            ["All"] + list(data['Region'].unique()),
            index=0
        )
    
    with col2:
        timeline = st.selectbox("Timeline", ["Month", "Quarter", "Year"], index=0)
    
    # Filter data
    filtered_data = data.copy()
    if selected_region != "All":
        filtered_data = filtered_data[filtered_data['Region'] == selected_region]
    
    # Regional performance
    col1, col2 = st.columns(2)
    
    with col1:
        regional_perf = filtered_data.groupby('Region')['Sales_Units'].sum().reset_index()
        fig = create_pie_chart(regional_perf, 'Region', "Sales by Region")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        state_perf = filtered_data.groupby('State')['Sales_Units'].sum().nlargest(10).reset_index()
        if not state_perf.empty:
            fig = create_bar_chart(state_perf, 'State', 'Sales_Units', "Top 10 States by Sales", orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for selected filters.")
    
    # State details
    st.subheader("📋 State Details")
    state_summary = filtered_data.groupby('State').agg({
        'Sales_Units': 'sum',
        'Sales_Value': 'sum'
    }).reset_index()
    
    st.dataframe(state_summary, use_container_width=True)

def show_product_analysis_page(data):
    """Product analysis page"""
    st.header("🔧 Product Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_brand = st.selectbox(
            "Select Brand",
            ["All"] + list(data['Brand'].unique()),
            index=0
        )
    
    with col2:
        selected_hp = st.selectbox(
            "Select PTO HP Category",
            ["All"] + list(data['PTO_HP_Category'].unique()),
            index=0
        )
    
    with col3:
        timeline = st.selectbox("Timeline", ["Month", "Quarter", "Year"], index=0)
    
    # Filter data
    filtered_data = data.copy()
    if selected_brand != "All":
        filtered_data = filtered_data[filtered_data['Brand'] == selected_brand]
    if selected_hp != "All":
        filtered_data = filtered_data[filtered_data['PTO_HP_Category'] == selected_hp]
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        brand_perf = filtered_data.groupby('Brand')['Sales_Units'].sum().reset_index()
        fig = create_bar_chart(brand_perf, 'Brand', 'Sales_Units', "Brand Performance", orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        hp_perf = filtered_data.groupby('PTO_HP_Category')['Sales_Units'].sum().reset_index()
        fig = create_bar_chart(hp_perf, 'PTO_HP_Category', 'Sales_Units', "PTO HP Category Performance")
        st.plotly_chart(fig, use_container_width=True)
    
    # Product trends
    st.subheader("📈 Product Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand_trends = filtered_data.groupby(['Brand', 'Date'])['Sales_Units'].sum().reset_index()
        fig = px.line(
            brand_trends, x='Date', y='Sales_Units', color='Brand',
            title="Brand Trends Over Time"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        hp_trends = filtered_data.groupby(['PTO_HP_Category', 'Date'])['Sales_Units'].sum().reset_index()
        fig = px.line(
            hp_trends, x='Date', y='Sales_Units', color='PTO_HP_Category',
            title="PTO HP Category Trends"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_recommendations_page(data):
    """Recommendations page"""
    st.header("💡 AI Recommendations")
    
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
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=85,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Forecast Accuracy"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#E31837"}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=78,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Inventory Optimization"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#E31837"}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=92,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Production Alignment"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#E31837"}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=15,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Regional Growth %"},
            gauge={'axis': {'range': [0, 20]}, 'bar': {'color': "#E31837"}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 
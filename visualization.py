import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from folium import plugins
import streamlit as st

class DashboardVisualizer:
    def __init__(self):
        self.colors = {
            'primary': '#E31837',  # UtiliCast red
            'secondary': '#2E86AB',
            'accent': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#6C5B7B',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
        
        self.color_palette = [
            '#E31837', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
            '#6C5B7B', '#8B4513', '#228B22', '#FF6347', '#4169E1'
        ]
    
    def create_sales_trend_chart(self, data, timeline='Month', title="Sales Trend Over Time"):
        """Create line chart for sales trends"""
        fig = go.Figure()
        
        # Add sales line
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Sales_Units'],
            mode='lines+markers',
            name='Sales Units',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=6)
        ))
        
        # Add sales value on secondary y-axis
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Sales_Value'],
            mode='lines+markers',
            name='Sales Value ($)',
            yaxis='y2',
            line=dict(color=self.colors['secondary'], width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Sales Units',
            yaxis2=dict(
                title='Sales Value ($)',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_bar_chart(self, data, x_col, y_col, title="Bar Chart", 
                        color_col=None, orientation='v'):
        """Create bar chart"""
        if orientation == 'v':
            fig = px.bar(
                data, x=x_col, y=y_col, color=color_col,
                title=title,
                color_discrete_sequence=self.color_palette,
                template='plotly_white'
            )
        else:
            fig = px.bar(
                data, x=y_col, y=x_col, color=color_col,
                title=title,
                color_discrete_sequence=self.color_palette,
                template='plotly_white',
                orientation='h'
            )
        
        fig.update_layout(
            height=400,
            showlegend=True if color_col else False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ) if color_col else None
        )
        
        return fig
    
    def create_pie_chart(self, data, names_col, values_col, title="Pie Chart"):
        """Create pie chart"""
        fig = px.pie(
            data, values=values_col, names=names_col,
            title=title,
            color_discrete_sequence=self.color_palette,
            template='plotly_white'
        )
        
        fig.update_layout(
            height=400,
            showlegend=True
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def create_scatter_plot(self, data, x_col, y_col, color_col=None, 
                           size_col=None, title="Scatter Plot"):
        """Create scatter plot"""
        fig = px.scatter(
            data, x=x_col, y=y_col, color=color_col, size=size_col,
            title=title,
            color_discrete_sequence=self.color_palette,
            template='plotly_white',
            hover_data=[color_col] if color_col else None
        )
        
        fig.update_layout(
            height=400,
            showlegend=True if color_col else False
        )
        
        return fig
    
    def create_correlation_heatmap(self, corr_matrix, title="Correlation Heatmap"):
        """Create correlation heatmap"""
        fig = px.imshow(
            corr_matrix,
            title=title,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            template='plotly_white'
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
    
    def create_multi_line_chart(self, data, x_col, y_col, color_col, title="Multi-Line Chart"):
        """Create multi-line chart for comparing different categories"""
        fig = px.line(
            data, x=x_col, y=y_col, color=color_col,
            title=title,
            color_discrete_sequence=self.color_palette,
            template='plotly_white'
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_area_chart(self, data, x_col, y_col, color_col=None, title="Area Chart"):
        """Create area chart"""
        fig = px.area(
            data, x=x_col, y=y_col, color=color_col,
            title=title,
            color_discrete_sequence=self.color_palette,
            template='plotly_white'
        )
        
        fig.update_layout(
            height=400,
            showlegend=True if color_col else False
        )
        
        return fig
    
    def create_gauge_chart(self, value, min_val, max_val, title="Gauge Chart"):
        """Create gauge chart for KPIs"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': (max_val + min_val) / 2},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': self.colors['primary']},
                'steps': [
                    {'range': [min_val, (max_val + min_val) * 0.5], 'color': "lightgray"},
                    {'range': [(max_val + min_val) * 0.5, max_val], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            template='plotly_white'
        )
        
        return fig
    
    def create_metric_card(self, title, value, change=None, change_type="neutral"):
        """Create metric card for dashboard"""
        colors = {
            "positive": "#28a745",
            "negative": "#dc3545",
            "neutral": "#6c757d"
        }
        
        change_color = colors.get(change_type, colors["neutral"])
        
        html = f"""
        <div style="
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            border-left: 4px solid {self.colors['primary']};
        ">
            <h3 style="margin: 0; color: #6c757d; font-size: 14px;">{title}</h3>
            <h2 style="margin: 10px 0; color: #343a40; font-size: 24px; font-weight: bold;">{value}</h2>
        """
        
        if change is not None:
            arrow = "↗️" if change_type == "positive" else "↘️" if change_type == "negative" else "→"
            html += f'<p style="margin: 0; color: {change_color}; font-size: 12px;">{arrow} {change}</p>'
        
        html += "</div>"
        
        return html
    
    def create_forecast_chart(self, historical_data, forecast_data, title="Sales Forecast"):
        """Create forecast chart with historical and predicted data"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Sales_Units'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=6)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Forecast_Sales'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color=self.colors['secondary'], width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        # Add confidence interval if available
        if 'Forecast_Lower' in forecast_data.columns and 'Forecast_Upper' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast_Upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast_Lower'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(46, 134, 171, 0.2)',
                name='Confidence Interval',
                line=dict(width=0),
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Sales Units',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_seasonal_decomposition(self, data, title="Seasonal Decomposition"):
        """Create seasonal decomposition chart"""
        # This is a simplified version - in practice, you'd use statsmodels
        fig = sp.make_subplots(
            rows=3, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal'),
            vertical_spacing=0.1
        )
        
        # Original data
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Sales_Units'], 
                      mode='lines', name='Original', line=dict(color=self.colors['primary'])),
            row=1, col=1
        )
        
        # Simple trend (moving average)
        trend = data['Sales_Units'].rolling(window=12, center=True).mean()
        fig.add_trace(
            go.Scatter(x=data['Date'], y=trend, 
                      mode='lines', name='Trend', line=dict(color=self.colors['secondary'])),
            row=2, col=1
        )
        
        # Simple seasonal (residuals)
        seasonal = data['Sales_Units'] - trend
        fig.add_trace(
            go.Scatter(x=data['Date'], y=seasonal, 
                      mode='lines', name='Seasonal', line=dict(color=self.colors['accent'])),
            row=3, col=1
        )
        
        fig.update_layout(
            title=title,
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_dashboard_summary(self, metrics):
        """Create dashboard summary with key metrics"""
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                label="Total Sales",
                value=f"{metrics['total_sales']:,}",
                delta=f"{metrics.get('sales_change', 0):.1f}%"
            )
        
        with cols[1]:
            st.metric(
                label="Total Value",
                value=f"${metrics['total_value']:,.0f}",
                delta=f"{metrics.get('value_change', 0):.1f}%"
            )
        
        with cols[2]:
            st.metric(
                label="Top State",
                value=metrics['top_state'],
                delta=None
            )
        
        with cols[3]:
            st.metric(
                label="Top Brand",
                value=metrics['top_brand'],
                delta=None
            )
    
    def create_interactive_filters(self):
        """Create interactive filters for the dashboard"""
        st.sidebar.header("Filters")
        
        # Timeline filter
        timeline = st.sidebar.selectbox(
            "Timeline",
            ["Month", "Quarter", "Year"],
            index=0
        )
        
        # Date range filter
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(datetime(2023, 1, 1), datetime(2024, 12, 31)),
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2024, 12, 31)
        )
        
        # State filter
        states = st.sidebar.multiselect(
            "States",
            ["All"] + ["Texas", "California", "Florida", "New York", "Illinois", "Ohio"],
            default=["All"]
        )
        
        # Brand filter
        brands = st.sidebar.multiselect(
            "Brands",
            ["All"] + ["John Deere", "Kubota", "New Holland", "Case IH", "Massey Ferguson"],
            default=["All"]
        )
        
        # PTO HP Category filter
        pto_categories = st.sidebar.multiselect(
            "PTO HP Categories",
            ["All"] + ["0<20", "20<30", "30<40", "40<50", "50<60", "60<70"],
            default=["All"]
        )
        
        return {
            'timeline': timeline,
            'date_range': date_range,
            'states': states,
            'brands': brands,
            'pto_categories': pto_categories
        }
    
    def create_usa_map(self, state_data, metric_col='Total_Sales', title="USA Sales Map"):
        """Create interactive USA map"""
        # Create a simple choropleth map using plotly
        fig = px.choropleth(
            state_data,
            locations='State',
            locationmode='USA-states',
            color=metric_col,
            scope='usa',
            title=title,
            color_continuous_scale='Reds',
            template='plotly_white'
        )
        
        fig.update_layout(
            height=500,
            geo=dict(
                showlakes=True,
                lakecolor='rgb(255, 255, 255)'
            )
        )
        
        return fig
    
    def create_comparison_chart(self, data1, data2, labels, title="Comparison Chart"):
        """Create comparison chart between two datasets"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=data1.index,
            y=data1.values,
            name=labels[0],
            marker_color=self.colors['primary']
        ))
        
        fig.add_trace(go.Bar(
            x=data2.index,
            y=data2.values,
            name=labels[1],
            marker_color=self.colors['secondary']
        ))
        
        fig.update_layout(
            title=title,
            barmode='group',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, sales_data, external_data, forecast_data):
        self.sales_data = sales_data.copy()
        self.external_data = external_data.copy()
        self.forecast_data = forecast_data.copy()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess all datasets"""
        # Convert date columns
        for df in [self.sales_data, self.external_data, self.forecast_data]:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
        
        # Add additional time-based features
        self._add_time_features()
        
        # Merge external data with sales data
        self._merge_external_data()
    
    def _add_time_features(self):
        """Add time-based features to sales data"""
        self.sales_data['Month_Name'] = self.sales_data['Date'].dt.strftime('%B')
        self.sales_data['Year_Month'] = self.sales_data['Date'].dt.strftime('%Y-%m')
        self.sales_data['Season'] = self.sales_data['Date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
    
    def _merge_external_data(self):
        """Merge external indicators with sales data"""
        # Aggregate external data by state and date
        external_agg = self.external_data.groupby(['State', 'Date']).agg({
            'Housing_Starts': 'mean',
            'Precipitation_mm': 'mean',
            'Economic_Activity_Index': 'mean',
            'Interest_Rate_Percent': 'mean',
            'Fuel_Price_USD': 'mean'
        }).reset_index()
        
        # Merge with sales data
        self.sales_data = self.sales_data.merge(
            external_agg, on=['State', 'Date'], how='left'
        )
    
    def filter_data(self, start_date=None, end_date=None, states=None, 
                   brands=None, pto_hp_categories=None, regions=None):
        """Filter data based on various criteria"""
        filtered_data = self.sales_data.copy()
        
        if start_date:
            filtered_data = filtered_data[filtered_data['Date'] >= start_date]
        
        if end_date:
            filtered_data = filtered_data[filtered_data['Date'] <= end_date]
        
        if states:
            if isinstance(states, str):
                states = [states]
            filtered_data = filtered_data[filtered_data['State'].isin(states)]
        
        if brands:
            if isinstance(brands, str):
                brands = [brands]
            filtered_data = filtered_data[filtered_data['Brand'].isin(brands)]
        
        if pto_hp_categories:
            if isinstance(pto_hp_categories, str):
                pto_hp_categories = [pto_hp_categories]
            filtered_data = filtered_data[filtered_data['PTO_HP_Category'].isin(pto_hp_categories)]
        
        if regions:
            if isinstance(regions, str):
                regions = [regions]
            filtered_data = filtered_data[filtered_data['Region'].isin(regions)]
        
        return filtered_data
    
    def aggregate_by_timeline(self, data, timeline='Month'):
        """Aggregate data by different timeline periods"""
        if timeline == 'Month':
            group_cols = ['Year', 'Month', 'Month_Name']
        elif timeline == 'Quarter':
            group_cols = ['Year', 'Quarter']
        elif timeline == 'Year':
            group_cols = ['Year']
        else:
            group_cols = ['Date']
        
        agg_data = data.groupby(group_cols).agg({
            'Sales_Units': 'sum',
            'Sales_Value': 'sum',
            'Housing_Starts': 'mean',
            'Precipitation_mm': 'mean',
            'Economic_Activity_Index': 'mean',
            'Interest_Rate_Percent': 'mean',
            'Fuel_Price_USD': 'mean'
        }).reset_index()
        
        # Create a proper date column for sorting
        if timeline == 'Month':
            agg_data['Date'] = pd.to_datetime(agg_data['Year'].astype(str) + '-' + 
                                            agg_data['Month'].astype(str).str.zfill(2) + '-01')
        elif timeline == 'Quarter':
            agg_data['Date'] = pd.to_datetime(agg_data['Year'].astype(str) + '-' + 
                                            (agg_data['Quarter'].str[1].astype(int) * 3).astype(str) + '-01')
        elif timeline == 'Year':
            agg_data['Date'] = pd.to_datetime(agg_data['Year'].astype(str) + '-01-01')
        
        return agg_data.sort_values('Date')
    
    def get_state_summary(self, state=None):
        """Get summary statistics for states"""
        if state:
            data = self.sales_data[self.sales_data['State'] == state]
        else:
            data = self.sales_data
        
        state_summary = data.groupby('State').agg({
            'Sales_Units': ['sum', 'mean', 'count'],
            'Sales_Value': ['sum', 'mean'],
            'Housing_Starts': 'mean',
            'Economic_Activity_Index': 'mean'
        }).reset_index()
        
        # Flatten column names
        state_summary.columns = ['State', 'Total_Sales', 'Avg_Sales', 'Sales_Count', 
                               'Total_Value', 'Avg_Value', 'Avg_Housing_Starts', 'Avg_Economic_Index']
        
        return state_summary.sort_values('Total_Sales', ascending=False)
    
    def get_brand_performance(self, timeline='Month'):
        """Get brand performance over time"""
        brand_perf = self.sales_data.groupby(['Brand', 'Date']).agg({
            'Sales_Units': 'sum',
            'Sales_Value': 'sum'
        }).reset_index()
        
        return self.aggregate_by_timeline(brand_perf, timeline)
    
    def get_pto_hp_analysis(self, timeline='Month'):
        """Get PTO HP category analysis over time"""
        hp_analysis = self.sales_data.groupby(['PTO_HP_Category', 'Date']).agg({
            'Sales_Units': 'sum',
            'Sales_Value': 'sum'
        }).reset_index()
        
        return self.aggregate_by_timeline(hp_analysis, timeline)
    
    def get_regional_analysis(self, timeline='Month'):
        """Get regional analysis over time"""
        regional_analysis = self.sales_data.groupby(['Region', 'Date']).agg({
            'Sales_Units': 'sum',
            'Sales_Value': 'sum',
            'Housing_Starts': 'mean',
            'Economic_Activity_Index': 'mean'
        }).reset_index()
        
        return self.aggregate_by_timeline(regional_analysis, timeline)
    
    def get_correlation_analysis(self):
        """Get correlation between sales and external factors"""
        correlation_data = self.sales_data.groupby(['State', 'Date']).agg({
            'Sales_Units': 'sum',
            'Housing_Starts': 'mean',
            'Precipitation_mm': 'mean',
            'Economic_Activity_Index': 'mean',
            'Interest_Rate_Percent': 'mean',
            'Fuel_Price_USD': 'mean'
        }).reset_index()
        
        # Calculate correlations
        corr_matrix = correlation_data[['Sales_Units', 'Housing_Starts', 'Precipitation_mm', 
                                      'Economic_Activity_Index', 'Interest_Rate_Percent', 'Fuel_Price_USD']].corr()
        
        return corr_matrix
    
    def get_forecast_data(self, periods=12):
        """Get forecast data for the specified periods"""
        return self.forecast_data.copy()
    
    def get_current_metrics(self):
        """Get current period metrics"""
        current_date = self.sales_data['Date'].max()
        current_data = self.sales_data[self.sales_data['Date'] == current_date]
        
        if current_data.empty:
            # If no current data, get the most recent available
            current_data = self.sales_data[self.sales_data['Date'] == self.sales_data['Date'].max()]
        
        metrics = {
            'current_date': current_data['Date'].iloc[0],
            'total_sales': current_data['Sales_Units'].sum(),
            'total_value': current_data['Sales_Value'].sum(),
            'avg_housing_starts': current_data['Housing_Starts'].mean(),
            'avg_economic_index': current_data['Economic_Activity_Index'].mean(),
            'top_state': current_data.groupby('State')['Sales_Units'].sum().idxmax(),
            'top_brand': current_data.groupby('Brand')['Sales_Units'].sum().idxmax(),
            'top_hp_category': current_data.groupby('PTO_HP_Category')['Sales_Units'].sum().idxmax()
        }
        
        return metrics
    
    def get_trend_analysis(self, metric='Sales_Units', periods=6):
        """Get trend analysis for the specified metric"""
        # Get recent data
        recent_data = self.sales_data.groupby('Date')[metric].sum().tail(periods)
        
        if len(recent_data) < 2:
            return {'trend': 'stable', 'change_percent': 0, 'direction': 'neutral'}
        
        # Calculate trend
        first_value = recent_data.iloc[0]
        last_value = recent_data.iloc[-1]
        
        if first_value == 0:
            change_percent = 0
        else:
            change_percent = ((last_value - first_value) / first_value) * 100
        
        if change_percent > 5:
            trend = 'increasing'
            direction = 'up'
        elif change_percent < -5:
            trend = 'decreasing'
            direction = 'down'
        else:
            trend = 'stable'
            direction = 'neutral'
        
        return {
            'trend': trend,
            'change_percent': round(change_percent, 2),
            'direction': direction,
            'recent_data': recent_data.tolist()
        }
    
    def get_seasonal_patterns(self):
        """Get seasonal patterns in sales"""
        seasonal_data = self.sales_data.groupby(['Season', 'Year']).agg({
            'Sales_Units': 'sum',
            'Sales_Value': 'sum'
        }).reset_index()
        
        # Calculate seasonal averages
        seasonal_avg = seasonal_data.groupby('Season').agg({
            'Sales_Units': 'mean',
            'Sales_Value': 'mean'
        }).reset_index()
        
        return seasonal_avg.sort_values('Sales_Units', ascending=False)
    
    def get_top_performers(self, category='State', metric='Sales_Units', top_n=10):
        """Get top performers in a category"""
        if category == 'State':
            data = self.sales_data.groupby('State')[metric].sum().sort_values(ascending=False)
        elif category == 'Brand':
            data = self.sales_data.groupby('Brand')[metric].sum().sort_values(ascending=False)
        elif category == 'PTO_HP_Category':
            data = self.sales_data.groupby('PTO_HP_Category')[metric].sum().sort_values(ascending=False)
        elif category == 'Region':
            data = self.sales_data.groupby('Region')[metric].sum().sort_values(ascending=False)
        else:
            return pd.Series()
        
        return data.head(top_n)
    
    def get_data_summary(self):
        """Get overall data summary"""
        summary = {
            'total_records': len(self.sales_data),
            'date_range': f"{self.sales_data['Date'].min().strftime('%Y-%m-%d')} to {self.sales_data['Date'].max().strftime('%Y-%m-%d')}",
            'total_states': self.sales_data['State'].nunique(),
            'total_brands': self.sales_data['Brand'].nunique(),
            'total_hp_categories': self.sales_data['PTO_HP_Category'].nunique(),
            'total_sales': self.sales_data['Sales_Units'].sum(),
            'total_value': self.sales_data['Sales_Value'].sum(),
            'avg_monthly_sales': self.sales_data.groupby('Date')['Sales_Units'].sum().mean()
        }
        
        return summary 
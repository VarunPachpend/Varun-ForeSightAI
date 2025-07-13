import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class SampleDataGenerator:
    def __init__(self):
        self.states = [
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
            'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
            'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
            'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
            'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
            'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
            'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia',
            'Wisconsin', 'Wyoming'
        ]
        
        self.brands = ['John Deere', 'Kubota', 'New Holland', 'Case IH', 'Massey Ferguson', 
                      'Mahindra', 'Kioti', 'Yanmar', 'LS Tractor', 'Tym']
        
        self.pto_hp_categories = ['0<20', '20<30', '30<40', '40<50', '50<60', '60<70']
        
        self.product_names = {
            '0<20': ['Compact Utility Tractor', 'Sub-Compact Tractor', 'Garden Tractor'],
            '20<30': ['Small Utility Tractor', 'Compact Tractor', 'Residential Tractor'],
            '30<40': ['Mid-Size Utility Tractor', 'Compact Utility Tractor', 'Small Farm Tractor'],
            '40<50': ['Utility Tractor', 'Mid-Size Tractor', 'Small Farm Tractor'],
            '50<60': ['Large Utility Tractor', 'Mid-Size Farm Tractor', 'Commercial Tractor'],
            '60<70': ['Large Farm Tractor', 'Commercial Utility Tractor', 'Heavy Duty Tractor']
        }

    def generate_sales_data(self, start_date='2020-01-01', end_date='2024-12-31'):
        """Generate comprehensive tractor sales data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        data = []
        
        for date in date_range:
            for state in self.states:
                for pto_hp in self.pto_hp_categories:
                    for brand in self.brands:
                        # Base sales with seasonal patterns
                        base_sales = self._get_base_sales(pto_hp, state, date)
                        
                        # Add some randomness
                        sales = max(0, int(base_sales * random.uniform(0.8, 1.2)))
                        
                        if sales > 0:
                            product_name = random.choice(self.product_names[pto_hp])
                            
                            data.append({
                                'Date': date,
                                'Year': date.year,
                                'Month': date.month,
                                'Quarter': f"Q{(date.month-1)//3 + 1}",
                                'State': state,
                                'Brand': brand,
                                'PTO_HP_Category': pto_hp,
                                'Product_Name': product_name,
                                'Sales_Units': sales,
                                'Sales_Value': sales * self._get_unit_price(pto_hp),
                                'Region': self._get_region(state)
                            })
        
        return pd.DataFrame(data)

    def generate_external_data(self, start_date='2020-01-01', end_date='2024-12-31'):
        """Generate external indicator data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        external_data = []
        
        for date in date_range:
            for state in self.states:
                # Housing starts (seasonal pattern with growth trend)
                housing_starts = self._generate_housing_starts(date, state)
                
                # Precipitation (seasonal pattern)
                precipitation = self._generate_precipitation(date, state)
                
                # Economic activity (GDP proxy)
                economic_activity = self._generate_economic_activity(date, state)
                
                # Interest rates (national trend)
                interest_rate = self._generate_interest_rate(date)
                
                # Fuel prices (with volatility)
                fuel_price = self._generate_fuel_price(date)
                
                external_data.append({
                    'Date': date,
                    'State': state,
                    'Housing_Starts': housing_starts,
                    'Precipitation_mm': precipitation,
                    'Economic_Activity_Index': economic_activity,
                    'Interest_Rate_Percent': interest_rate,
                    'Fuel_Price_USD': fuel_price,
                    'Region': self._get_region(state)
                })
        
        return pd.DataFrame(external_data)

    def _get_base_sales(self, pto_hp, state, date):
        """Calculate base sales considering multiple factors"""
        # Base sales by PTO HP category
        base_sales_by_hp = {
            '0<20': 150, '20<30': 200, '30<40': 180, 
            '40<50': 160, '50<60': 120, '60<70': 80
        }
        
        base = base_sales_by_hp[pto_hp]
        
        # Seasonal adjustment (spring/summer peak)
        seasonal_factor = 1.0
        if date.month in [3, 4, 5, 6, 7, 8]:  # Spring/Summer
            seasonal_factor = 1.3
        elif date.month in [12, 1, 2]:  # Winter
            seasonal_factor = 0.7
        
        # State-specific adjustments
        state_factors = {
            'Texas': 1.4, 'California': 1.3, 'Florida': 1.2,
            'New York': 1.1, 'Illinois': 1.1, 'Ohio': 1.0,
            'Alaska': 0.6, 'Hawaii': 0.8, 'Rhode Island': 0.8
        }
        state_factor = state_factors.get(state, 1.0)
        
        # Growth trend over time
        time_factor = 1.0 + (date.year - 2020) * 0.05
        
        return base * seasonal_factor * state_factor * time_factor

    def _get_unit_price(self, pto_hp):
        """Get average unit price by PTO HP category"""
        prices = {
            '0<20': 15000, '20<30': 22000, '30<40': 28000,
            '40<50': 35000, '50<60': 45000, '60<70': 55000
        }
        return prices[pto_hp]

    def _get_region(self, state):
        """Map states to regions"""
        regions = {
            'Northeast': ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 
                         'Connecticut', 'New York', 'New Jersey', 'Pennsylvania'],
            'Midwest': ['Ohio', 'Indiana', 'Illinois', 'Michigan', 'Wisconsin', 'Minnesota', 
                       'Iowa', 'Missouri', 'North Dakota', 'South Dakota', 'Nebraska', 'Kansas'],
            'South': ['Delaware', 'Maryland', 'Virginia', 'West Virginia', 'Kentucky', 'Tennessee', 
                     'North Carolina', 'South Carolina', 'Georgia', 'Florida', 'Alabama', 
                     'Mississippi', 'Arkansas', 'Louisiana', 'Oklahoma', 'Texas'],
            'West': ['Montana', 'Idaho', 'Wyoming', 'Colorado', 'New Mexico', 'Arizona', 
                    'Utah', 'Nevada', 'Washington', 'Oregon', 'California', 'Alaska', 'Hawaii']
        }
        
        for region, states in regions.items():
            if state in states:
                return region
        return 'Other'

    def _generate_housing_starts(self, date, state):
        """Generate housing starts data"""
        # Base housing starts by state
        base_starts = {
            'Texas': 8000, 'California': 7000, 'Florida': 6000,
            'New York': 3000, 'Illinois': 2500, 'Ohio': 2000
        }
        base = base_starts.get(state, 1000)
        
        # Seasonal pattern
        seasonal_factor = 1.0
        if date.month in [3, 4, 5, 6]:  # Spring peak
            seasonal_factor = 1.2
        elif date.month in [12, 1, 2]:  # Winter low
            seasonal_factor = 0.8
        
        # Growth trend
        growth_factor = 1.0 + (date.year - 2020) * 0.03
        
        return max(0, int(base * seasonal_factor * growth_factor * random.uniform(0.9, 1.1)))

    def _generate_precipitation(self, date, state):
        """Generate precipitation data"""
        # Base precipitation by region
        base_precip = {
            'Northeast': 100, 'Midwest': 80, 'South': 120, 'West': 60
        }
        region = self._get_region(state)
        base = base_precip.get(region, 80)
        
        # Seasonal pattern
        seasonal_factor = 1.0
        if date.month in [3, 4, 5]:  # Spring rains
            seasonal_factor = 1.3
        elif date.month in [7, 8]:  # Summer dry
            seasonal_factor = 0.7
        
        return max(0, base * seasonal_factor * random.uniform(0.8, 1.2))

    def _generate_economic_activity(self, date, state):
        """Generate economic activity index"""
        # Base economic activity by state
        base_activity = {
            'California': 120, 'Texas': 115, 'New York': 110,
            'Florida': 105, 'Illinois': 100, 'Ohio': 95
        }
        base = base_activity.get(state, 90)
        
        # Gradual growth trend
        growth_factor = 1.0 + (date.year - 2020) * 0.02
        
        return base * growth_factor * random.uniform(0.95, 1.05)

    def _generate_interest_rate(self, date):
        """Generate interest rate data"""
        # Simulate interest rate trends
        base_rate = 3.0
        if date.year >= 2022:
            base_rate += (date.year - 2022) * 1.5
        
        return base_rate + random.uniform(-0.2, 0.2)

    def _generate_fuel_price(self, date):
        """Generate fuel price data"""
        # Base fuel price with volatility
        base_price = 3.0
        
        # Add some volatility
        if date.year >= 2022:
            base_price += 1.0
        
        return base_price + random.uniform(-0.5, 0.5)

    def get_forecast_data(self, periods=12):
        """Generate future forecast data"""
        last_date = datetime(2024, 12, 31)
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=periods, freq='M')
        
        forecast_data = []
        
        for date in future_dates:
            for state in self.states:
                for pto_hp in self.pto_hp_categories:
                    # Generate forecast with trend continuation
                    base_sales = self._get_base_sales(pto_hp, state, date)
                    forecast_sales = int(base_sales * random.uniform(0.9, 1.1))
                    
                    if forecast_sales > 0:
                        product_name = random.choice(self.product_names[pto_hp])
                        
                        forecast_data.append({
                            'Date': date,
                            'Year': date.year,
                            'Month': date.month,
                            'Quarter': f"Q{(date.month-1)//3 + 1}",
                            'State': state,
                            'PTO_HP_Category': pto_hp,
                            'Product_Name': product_name,
                            'Forecast_Sales': forecast_sales,
                            'Forecast_Value': forecast_sales * self._get_unit_price(pto_hp),
                            'Region': self._get_region(state)
                        })
        
        return pd.DataFrame(forecast_data)

# Create sample data instances
def create_sample_datasets():
    """Create and return sample datasets for the dashboard"""
    generator = SampleDataGenerator()
    
    # Generate main sales data
    sales_data = generator.generate_sales_data()
    
    # Generate external data
    external_data = generator.generate_external_data()
    
    # Generate forecast data
    forecast_data = generator.get_forecast_data()
    
    return sales_data, external_data, forecast_data

if __name__ == "__main__":
    # Test data generation
    sales_df, external_df, forecast_df = create_sample_datasets()
    
    print(f"Sales data shape: {sales_df.shape}")
    print(f"External data shape: {external_df.shape}")
    print(f"Forecast data shape: {forecast_df.shape}")
    
    print("\nSample sales data:")
    print(sales_df.head())
    
    print("\nSample external data:")
    print(external_df.head()) 
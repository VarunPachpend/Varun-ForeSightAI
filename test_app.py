#!/usr/bin/env python3
"""
Test script for UtiliCast project
This script tests the basic functionality of the project components
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_generation():
    """Test data generation functionality"""
    print("🧪 Testing data generation...")
    
    try:
        from data.sample_data import create_sample_datasets
        
        # Generate sample data
        sales_data, external_data, forecast_data = create_sample_datasets()
        
        print(f"✅ Sales data shape: {sales_data.shape}")
        print(f"✅ External data shape: {external_data.shape}")
        print(f"✅ Forecast data shape: {forecast_data.shape}")
        
        # Check data quality
        print(f"✅ Sales data columns: {list(sales_data.columns)}")
        print(f"✅ Date range: {sales_data['Date'].min()} to {sales_data['Date'].max()}")
        print(f"✅ States covered: {sales_data['State'].nunique()}")
        print(f"✅ Brands covered: {sales_data['Brand'].nunique()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_data_processing():
    """Test data processing functionality"""
    print("\n🧪 Testing data processing...")
    
    try:
        from data.sample_data import create_sample_datasets
        from utils.data_processor import DataProcessor
        
        # Generate and process data
        sales_data, external_data, forecast_data = create_sample_datasets()
        processor = DataProcessor(sales_data, external_data, forecast_data)
        
        # Test various processing functions
        current_metrics = processor.get_current_metrics()
        trend_analysis = processor.get_trend_analysis()
        state_summary = processor.get_state_summary()
        
        print(f"✅ Current metrics: {len(current_metrics)} metrics calculated")
        print(f"✅ Trend analysis: {trend_analysis['trend']} trend detected")
        print(f"✅ State summary: {len(state_summary)} states analyzed")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality"""
    print("\n🧪 Testing visualization...")
    
    try:
        from data.sample_data import create_sample_datasets
        from utils.data_processor import DataProcessor
        from utils.visualization import DashboardVisualizer
        
        # Generate data and create visualizer
        sales_data, external_data, forecast_data = create_sample_datasets()
        processor = DataProcessor(sales_data, external_data, forecast_data)
        visualizer = DashboardVisualizer()
        
        # Test chart creation
        timeline_data = processor.aggregate_by_timeline(processor.sales_data, 'Month')
        fig = visualizer.create_sales_trend_chart(timeline_data)
        
        print(f"✅ Chart created successfully: {type(fig)}")
        print(f"✅ Color palette: {len(visualizer.color_palette)} colors")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

def test_prophet_model():
    """Test Prophet model functionality"""
    print("\n🧪 Testing Prophet model...")
    
    try:
        from data.sample_data import create_sample_datasets
        from models.prophet_model import ProphetForecaster
        
        # Generate data
        sales_data, external_data, forecast_data = create_sample_datasets()
        
        # Prepare data for Prophet
        test_data = sales_data.groupby('Date')['Sales_Units'].sum().reset_index()
        test_data = test_data.merge(
            external_data.groupby('Date').agg({
                'Housing_Starts': 'mean',
                'Economic_Activity_Index': 'mean'
            }).reset_index(),
            on='Date'
        )
        
        # Test Prophet forecaster
        forecaster = ProphetForecaster()
        forecaster.fit(test_data, external_regressors=['Housing_Starts', 'Economic_Activity_Index'])
        
        print("✅ Prophet model fitted successfully")
        
        # Test prediction
        forecast = forecaster.predict(periods=6)
        print(f"✅ Forecast generated: {len(forecast)} periods")
        
        return True
        
    except Exception as e:
        print(f"❌ Prophet model failed: {e}")
        return False

def test_xgboost_model():
    """Test XGBoost model functionality"""
    print("\n🧪 Testing XGBoost model...")
    
    try:
        from data.sample_data import create_sample_datasets
        from models.xgboost_model import XGBoostForecaster
        
        # Generate data
        sales_data, external_data, forecast_data = create_sample_datasets()
        
        # Prepare data for XGBoost
        test_data = sales_data.groupby('Date')['Sales_Units'].sum().reset_index()
        test_data = test_data.merge(
            external_data.groupby('Date').agg({
                'Housing_Starts': 'mean',
                'Economic_Activity_Index': 'mean'
            }).reset_index(),
            on='Date'
        )
        
        # Test XGBoost forecaster
        forecaster = XGBoostForecaster()
        metrics = forecaster.fit(test_data)
        
        print(f"✅ XGBoost model fitted successfully")
        print(f"✅ Test MAPE: {metrics['test_mape']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ XGBoost model failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚜 UtiliCast Project Test Suite")
    print("=" * 50)
    
    tests = [
        test_data_generation,
        test_data_processing,
        test_visualization,
        test_prophet_model,
        test_xgboost_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to run.")
        print("\nTo run the dashboard:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the app: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProphetForecaster:
    def __init__(self):
        self.model = None
        self.fitted = False
        self.forecast_periods = 12
        
    def prepare_data(self, data, target_col='Sales_Units', date_col='Date'):
        """Prepare data for Prophet model"""
        # Prophet requires columns named 'ds' (date) and 'y' (target)
        prophet_data = data.copy()
        prophet_data['ds'] = pd.to_datetime(prophet_data[date_col])
        prophet_data['y'] = prophet_data[target_col]
        
        # Aggregate by date if multiple records per date
        if len(prophet_data) > len(prophet_data['ds'].unique()):
            prophet_data = prophet_data.groupby('ds').agg({
                'y': 'sum',
                'Housing_Starts': 'mean',
                'Precipitation_mm': 'mean',
                'Economic_Activity_Index': 'mean',
                'Interest_Rate_Percent': 'mean',
                'Fuel_Price_USD': 'mean'
            }).reset_index()
        
        return prophet_data[['ds', 'y'] + [col for col in prophet_data.columns 
                                          if col not in ['ds', 'y'] and col in 
                                          ['Housing_Starts', 'Precipitation_mm', 
                                           'Economic_Activity_Index', 'Interest_Rate_Percent', 
                                           'Fuel_Price_USD']]]
    
    def fit(self, data, target_col='Sales_Units', date_col='Date', 
            external_regressors=None, seasonality_mode='multiplicative'):
        """Fit Prophet model with optional external regressors"""
        # Prepare data
        prophet_data = self.prepare_data(data, target_col, date_col)
        
        # Initialize Prophet model
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        # Add external regressors if provided
        if external_regressors:
            for regressor in external_regressors:
                if regressor in prophet_data.columns:
                    self.model.add_regressor(regressor)
        
        # Fit the model
        self.model.fit(prophet_data)
        self.fitted = True
        
        return self.model
    
    def predict(self, periods=12, external_data=None):
        """Make predictions with Prophet model"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='M')
        
        # Add external regressors to future data if provided
        if external_data is not None and hasattr(self.model, 'extra_regressors'):
            for regressor in self.model.extra_regressors:
                if regressor in external_data.columns:
                    # For future periods, use the last known value or trend
                    last_value = external_data[regressor].iloc[-1]
                    future[regressor] = last_value
        
        # Make prediction
        forecast = self.model.predict(future)
        
        return forecast
    
    def get_forecast_components(self, forecast):
        """Extract forecast components (trend, seasonal, etc.)"""
        components = ['trend', 'yearly', 'multiplicative_terms', 'additive_terms']
        available_components = [col for col in forecast.columns if col in components]
        
        return forecast[['ds'] + available_components]
    
    def evaluate_model(self, data, target_col='Sales_Units', date_col='Date'):
        """Evaluate model performance"""
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Prepare test data
        test_data = self.prepare_data(data, target_col, date_col)
        
        # Make predictions on test data
        future = self.model.make_future_dataframe(periods=0, freq='M')
        forecast = self.model.predict(future)
        
        # Calculate metrics
        actual = test_data['y'].values
        predicted = forecast['yhat'].values[:len(actual)]
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Calculate MAE
        mae = np.mean(np.abs(actual - predicted))
        
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae,
            'actual': actual,
            'predicted': predicted
        }
    
    def plot_forecast(self, forecast, data=None):
        """Plot forecast results"""
        if data is not None:
            return self.model.plot(forecast, data)
        else:
            return self.model.plot(forecast)
    
    def plot_components(self, forecast):
        """Plot forecast components"""
        return self.model.plot_components(forecast)
    
    def get_feature_importance(self):
        """Get feature importance for external regressors"""
        if not self.fitted or not hasattr(self.model, 'extra_regressors'):
            return None
        
        # Prophet doesn't provide direct feature importance
        # This is a simplified approach
        importance = {}
        for regressor in self.model.extra_regressors:
            importance[regressor] = 1.0  # Placeholder
        
        return importance

class MultiProphetForecaster:
    """Forecaster for multiple categories (states, brands, PTO HP categories)"""
    
    def __init__(self):
        self.models = {}
        self.fitted = False
    
    def fit_multiple(self, data, group_col, target_col='Sales_Units', 
                    date_col='Date', external_regressors=None):
        """Fit Prophet models for multiple groups"""
        self.models = {}
        
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]
            
            if len(group_data) > 10:  # Minimum data points required
                forecaster = ProphetForecaster()
                try:
                    forecaster.fit(group_data, target_col, date_col, external_regressors)
                    self.models[group] = forecaster
                except Exception as e:
                    print(f"Error fitting model for {group}: {e}")
        
        self.fitted = len(self.models) > 0
        return self.models
    
    def predict_multiple(self, periods=12, external_data=None):
        """Make predictions for all groups"""
        if not self.fitted:
            raise ValueError("Models must be fitted before making predictions")
        
        all_forecasts = {}
        
        for group, model in self.models.items():
            try:
                forecast = model.predict(periods, external_data)
                all_forecasts[group] = forecast
            except Exception as e:
                print(f"Error predicting for {group}: {e}")
        
        return all_forecasts
    
    def get_ensemble_forecast(self, periods=12, external_data=None, 
                            aggregation_method='mean'):
        """Get ensemble forecast across all groups"""
        individual_forecasts = self.predict_multiple(periods, external_data)
        
        if not individual_forecasts:
            return None
        
        # Combine forecasts
        all_predictions = []
        for group, forecast in individual_forecasts.items():
            forecast_copy = forecast.copy()
            forecast_copy['group'] = group
            all_predictions.append(forecast_copy)
        
        combined_forecast = pd.concat(all_predictions, ignore_index=True)
        
        # Aggregate by date
        if aggregation_method == 'mean':
            ensemble = combined_forecast.groupby('ds').agg({
                'yhat': 'mean',
                'yhat_lower': 'mean',
                'yhat_upper': 'mean'
            }).reset_index()
        elif aggregation_method == 'median':
            ensemble = combined_forecast.groupby('ds').agg({
                'yhat': 'median',
                'yhat_lower': 'median',
                'yhat_upper': 'median'
            }).reset_index()
        
        return ensemble
    
    def evaluate_multiple(self, data, group_col, target_col='Sales_Units', 
                         date_col='Date'):
        """Evaluate all models"""
        results = {}
        
        for group, model in self.models.items():
            group_data = data[data[group_col] == group]
            if len(group_data) > 0:
                try:
                    metrics = model.evaluate_model(group_data, target_col, date_col)
                    results[group] = metrics
                except Exception as e:
                    print(f"Error evaluating model for {group}: {e}")
        
        return results
    
    def get_overall_performance(self, evaluation_results):
        """Get overall performance metrics"""
        if not evaluation_results:
            return None
        
        all_mape = [result['MAPE'] for result in evaluation_results.values()]
        all_rmse = [result['RMSE'] for result in evaluation_results.values()]
        all_mae = [result['MAE'] for result in evaluation_results.values()]
        
        return {
            'Overall_MAPE': np.mean(all_mape),
            'Overall_RMSE': np.mean(all_rmse),
            'Overall_MAE': np.mean(all_mae),
            'Best_Group': min(evaluation_results.items(), key=lambda x: x[1]['MAPE'])[0],
            'Worst_Group': max(evaluation_results.items(), key=lambda x: x[1]['MAPE'])[0]
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    from data.sample_data import SampleDataGenerator
    
    generator = SampleDataGenerator()
    sales_data, external_data, forecast_data = generator.create_sample_datasets()
    
    # Test single Prophet model
    forecaster = ProphetForecaster()
    
    # Prepare data with external regressors
    test_data = sales_data.groupby('Date')['Sales_Units'].sum().reset_index()
    test_data = test_data.merge(
        external_data.groupby('Date').agg({
            'Housing_Starts': 'mean',
            'Economic_Activity_Index': 'mean'
        }).reset_index(),
        on='Date'
    )
    
    # Fit model
    forecaster.fit(test_data, external_regressors=['Housing_Starts', 'Economic_Activity_Index'])
    
    # Make prediction
    forecast = forecaster.predict(periods=6)
    
    print("Prophet Forecast Results:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # Test multi-group forecaster
    multi_forecaster = MultiProphetForecaster()
    multi_forecaster.fit_multiple(sales_data, 'State', external_regressors=['Housing_Starts'])
    
    ensemble_forecast = multi_forecaster.get_ensemble_forecast(periods=6)
    print("\nEnsemble Forecast Results:")
    print(ensemble_forecast.tail()) 
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class XGBoostForecaster:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names = []
        
    def prepare_features(self, data, target_col='Sales_Units', date_col='Date'):
        """Prepare features for XGBoost model"""
        df = data.copy()
        df['Date'] = pd.to_datetime(df[date_col])
        
        # Time-based features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['week_of_year'] = df['Date'].dt.isocalendar().week
        
        # Cyclical encoding for seasonal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Lag features
        df = df.sort_values('Date')
        for lag in [1, 2, 3, 6, 12]:
            df[f'sales_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling features
        for window in [3, 6, 12]:
            df[f'sales_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'sales_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        
        # External features (if available)
        external_features = ['Housing_Starts', 'Precipitation_mm', 'Economic_Activity_Index', 
                           'Interest_Rate_Percent', 'Fuel_Price_USD']
        
        for feature in external_features:
            if feature in df.columns:
                # Lag external features
                for lag in [1, 2, 3]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
                
                # Rolling features for external data
                for window in [3, 6]:
                    df[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window=window).mean()
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Select features
        feature_cols = [col for col in df.columns if col not in [date_col, target_col, 'Date']]
        self.feature_names = feature_cols
        
        return df[feature_cols], df[target_col]
    
    def fit(self, data, target_col='Sales_Units', date_col='Date', 
            external_regressors=None, test_size=0.2):
        """Fit XGBoost model"""
        # Prepare features
        X, y = self.prepare_features(data, target_col, date_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=10
        )
        
        # Fit model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        self.fitted = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mape': np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100,
            'test_mape': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        }
        
        return metrics
    
    def predict(self, data, periods=12, external_data=None):
        """Make predictions with XGBoost model"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features for prediction
        X_pred, _ = self.prepare_features(data, date_col='Date')
        
        # Scale features
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make predictions
        predictions = self.model.predict(X_pred_scaled)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': data['Date'],
            'yhat': predictions,
            'yhat_lower': predictions * 0.9,  # Simple confidence interval
            'yhat_upper': predictions * 1.1
        })
        
        return forecast_df
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.fitted:
            return None
        
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        importance_df = self.get_feature_importance()
        if importance_df is None:
            return None
        
        top_features = importance_df.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='#E31837'
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=400,
            template='plotly_white'
        )
        
        return fig

class MultiXGBoostForecaster:
    """XGBoost forecaster for multiple categories"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.fitted = False
    
    def fit_multiple(self, data, group_col, target_col='Sales_Units', 
                    date_col='Date', external_regressors=None):
        """Fit XGBoost models for multiple groups"""
        self.models = {}
        self.scalers = {}
        
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]
            
            if len(group_data) > 20:  # Minimum data points required
                try:
                    forecaster = XGBoostForecaster()
                    metrics = forecaster.fit(group_data, target_col, date_col, external_regressors)
                    self.models[group] = forecaster
                    self.scalers[group] = forecaster.scaler
                except Exception as e:
                    print(f"Error fitting XGBoost model for {group}: {e}")
        
        self.fitted = len(self.models) > 0
        return self.models
    
    def predict_multiple(self, data, group_col, periods=12):
        """Make predictions for all groups"""
        if not self.fitted:
            raise ValueError("Models must be fitted before making predictions")
        
        all_forecasts = {}
        
        for group, model in self.models.items():
            try:
                group_data = data[data[group_col] == group]
                if len(group_data) > 0:
                    forecast = model.predict(group_data, periods)
                    forecast['group'] = group
                    all_forecasts[group] = forecast
            except Exception as e:
                print(f"Error predicting for {group}: {e}")
        
        return all_forecasts
    
    def get_ensemble_forecast(self, data, group_col, periods=12, 
                            aggregation_method='mean'):
        """Get ensemble forecast across all groups"""
        individual_forecasts = self.predict_multiple(data, group_col, periods)
        
        if not individual_forecasts:
            return None
        
        # Combine forecasts
        all_predictions = []
        for group, forecast in individual_forecasts.items():
            all_predictions.append(forecast)
        
        combined_forecast = pd.concat(all_predictions, ignore_index=True)
        
        # Aggregate by date
        if aggregation_method == 'mean':
            ensemble = combined_forecast.groupby('Date').agg({
                'yhat': 'mean',
                'yhat_lower': 'mean',
                'yhat_upper': 'mean'
            }).reset_index()
        elif aggregation_method == 'median':
            ensemble = combined_forecast.groupby('Date').agg({
                'yhat': 'median',
                'yhat_lower': 'median',
                'yhat_upper': 'median'
            }).reset_index()
        
        return ensemble
    
    def get_overall_feature_importance(self):
        """Get overall feature importance across all models"""
        all_importance = []
        
        for group, model in self.models.items():
            importance_df = model.get_feature_importance()
            if importance_df is not None:
                importance_df['group'] = group
                all_importance.append(importance_df)
        
        if all_importance:
            combined_importance = pd.concat(all_importance, ignore_index=True)
            overall_importance = combined_importance.groupby('feature')['importance'].mean().reset_index()
            return overall_importance.sort_values('importance', ascending=False)
        
        return None

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    from data.sample_data import SampleDataGenerator
    
    generator = SampleDataGenerator()
    sales_data, external_data, forecast_data = generator.create_sample_datasets()
    
    # Test single XGBoost model
    forecaster = XGBoostForecaster()
    
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
    metrics = forecaster.fit(test_data)
    print("XGBoost Training Metrics:")
    print(metrics)
    
    # Make prediction
    forecast = forecaster.predict(test_data, periods=6)
    print("\nXGBoost Forecast Results:")
    print(forecast.tail())
    
    # Feature importance
    importance = forecaster.get_feature_importance()
    print("\nTop 10 Feature Importance:")
    print(importance.head(10))
    
    # Test multi-group forecaster
    multi_forecaster = MultiXGBoostForecaster()
    multi_forecaster.fit_multiple(sales_data, 'State')
    
    ensemble_forecast = multi_forecaster.get_ensemble_forecast(sales_data, 'State', periods=6)
    print("\nEnsemble XGBoost Forecast Results:")
    print(ensemble_forecast.tail()) 
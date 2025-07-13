import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class LSTMForecaster:
    def __init__(self, sequence_length=12, n_features=1):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.fitted = False
        
    def create_sequences(self, data, target_col='Sales_Units'):
        """Create sequences for LSTM model"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[target_col].values[i-self.sequence_length:i])
            y.append(data[target_col].values[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, data, target_col='Sales_Units', date_col='Date'):
        """Prepare data for LSTM model"""
        df = data.copy()
        df['Date'] = pd.to_datetime(df[date_col])
        df = df.sort_values('Date')
        
        # Reshape data for scaling
        values = df[target_col].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = self.create_sequences(pd.DataFrame({target_col: scaled_values.flatten()}))
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        
        return X, y, df
    
    def build_model(self, units=50, dropout=0.2):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(dropout),
            LSTM(units=units, return_sequences=False),
            Dropout(dropout),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def fit(self, data, target_col='Sales_Units', date_col='Date', 
            epochs=100, batch_size=32, validation_split=0.2):
        """Fit LSTM model"""
        # Prepare data
        X, y, df = self.prepare_data(data, target_col, date_col)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build and fit model
        self.model = self.build_model()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        self.fitted = True
        
        # Calculate metrics
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Inverse transform predictions
        y_pred_train = self.scaler.inverse_transform(y_pred_train)
        y_pred_test = self.scaler.inverse_transform(y_pred_test)
        y_train_orig = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_orig = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        metrics = {
            'train_mae': mean_absolute_error(y_train_orig, y_pred_train),
            'test_mae': mean_absolute_error(y_test_orig, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train_orig, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_test)),
            'train_mape': np.mean(np.abs((y_train_orig - y_pred_train) / y_train_orig)) * 100,
            'test_mape': np.mean(np.abs((y_test_orig - y_pred_test) / y_test_orig)) * 100,
            'history': history
        }
        
        return metrics
    
    def predict(self, data, periods=12):
        """Make predictions with LSTM model"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X, y, df = self.prepare_data(data, date_col='Date')
        
        # Get the last sequence for prediction
        last_sequence = X[-1:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(periods):
            # Predict next value
            next_pred = self.model.predict(current_sequence)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        # Create future dates
        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=periods, freq='M')
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'yhat': predictions.flatten(),
            'yhat_lower': predictions.flatten() * 0.9,  # Simple confidence interval
            'yhat_upper': predictions.flatten() * 1.1
        })
        
        return forecast_df
    
    def predict_sequence(self, data, target_col='Sales_Units', date_col='Date'):
        """Predict on the entire sequence"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X, y, df = self.prepare_data(data, target_col, date_col)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(predictions)
        actual = self.scaler.inverse_transform(y.reshape(-1, 1))
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Date': df['Date'].iloc[self.sequence_length:],
            'actual': actual.flatten(),
            'predicted': predictions.flatten()
        })
        
        return results_df

class MultiLSTMForecaster:
    """LSTM forecaster for multiple categories"""
    
    def __init__(self, sequence_length=12):
        self.models = {}
        self.scalers = {}
        self.sequence_length = sequence_length
        self.fitted = False
    
    def fit_multiple(self, data, group_col, target_col='Sales_Units', 
                    date_col='Date', epochs=50, batch_size=32):
        """Fit LSTM models for multiple groups"""
        self.models = {}
        self.scalers = {}
        
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]
            
            if len(group_data) > 30:  # Minimum data points required
                try:
                    forecaster = LSTMForecaster(sequence_length=self.sequence_length)
                    metrics = forecaster.fit(group_data, target_col, date_col, 
                                           epochs=epochs, batch_size=batch_size)
                    self.models[group] = forecaster
                    self.scalers[group] = forecaster.scaler
                except Exception as e:
                    print(f"Error fitting LSTM model for {group}: {e}")
        
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
    
    def evaluate_multiple(self, data, group_col, target_col='Sales_Units', 
                         date_col='Date'):
        """Evaluate all models"""
        results = {}
        
        for group, model in self.models.items():
            group_data = data[data[group_col] == group]
            if len(group_data) > 0:
                try:
                    predictions = model.predict_sequence(group_data, target_col, date_col)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(predictions['actual'], predictions['predicted'])
                    rmse = np.sqrt(mean_squared_error(predictions['actual'], predictions['predicted']))
                    mape = np.mean(np.abs((predictions['actual'] - predictions['predicted']) / predictions['actual'])) * 100
                    
                    results[group] = {
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape,
                        'predictions': predictions
                    }
                except Exception as e:
                    print(f"Error evaluating model for {group}: {e}")
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    from data.sample_data import SampleDataGenerator
    
    generator = SampleDataGenerator()
    sales_data, external_data, forecast_data = generator.create_sample_datasets()
    
    # Test single LSTM model
    forecaster = LSTMForecaster(sequence_length=12)
    
    # Prepare data
    test_data = sales_data.groupby('Date')['Sales_Units'].sum().reset_index()
    
    # Fit model
    metrics = forecaster.fit(test_data, epochs=50)
    print("LSTM Training Metrics:")
    print(metrics)
    
    # Make prediction
    forecast = forecaster.predict(test_data, periods=6)
    print("\nLSTM Forecast Results:")
    print(forecast)
    
    # Test multi-group forecaster
    multi_forecaster = MultiLSTMForecaster(sequence_length=12)
    multi_forecaster.fit_multiple(sales_data, 'State', epochs=30)
    
    ensemble_forecast = multi_forecaster.get_ensemble_forecast(sales_data, 'State', periods=6)
    print("\nEnsemble LSTM Forecast Results:")
    print(ensemble_forecast.tail()) 
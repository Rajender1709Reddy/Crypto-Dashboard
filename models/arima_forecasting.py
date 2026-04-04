import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_project_root

def evaluate_forecast(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def run_arima(train_series, test_series, order=(5,1,0)):
    print(f"Training ARIMA model with order {order}...")
    # Add index freq to avoid warnings if possible
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    predictions = fitted_model.forecast(steps=len(test_series))
    predictions.index = test_series.index
    
    metrics = evaluate_forecast(test_series, predictions)
    return predictions, metrics

def run_sarima(train_series, test_series, order=(1,1,1), seasonal_order=(1,1,1,7)):
    print(f"Training SARIMA model with order {order} and seasonal_order {seasonal_order}...")
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit(disp=False)
    predictions = fitted_model.forecast(steps=len(test_series))
    predictions.index = test_series.index
    
    metrics = evaluate_forecast(test_series, predictions)
    return predictions, metrics

if __name__ == "__main__":
    data_path = os.path.join(get_project_root(), "data", "processed_data.csv")
    if not os.path.exists(data_path):
        print(f"Processed data not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    
    # Train-test split (80-20)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    print("--- ARIMA Evaluation ---")
    arima_preds, arima_metrics = run_arima(train['Close'], test['Close'])
    print("ARIMA Metrics:", arima_metrics)
    
    print("\n--- SARIMA Evaluation ---")
    sarima_preds, sarima_metrics = run_sarima(train['Close'], test['Close'])
    print("SARIMA Metrics:", sarima_metrics)

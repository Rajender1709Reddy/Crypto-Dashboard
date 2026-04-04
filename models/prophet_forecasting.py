import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_project_root

def run_prophet(df):
    print("Training Facebook Prophet model...")
    # Prophet requires 'ds' and 'y' columns
    prophet_df = df.reset_index()[['Date', 'Close']]
    prophet_df.columns = ['ds', 'y']
    
    # Drop rows with NaN if any exist
    prophet_df.dropna(inplace=True)
    
    train_size = int(len(prophet_df) * 0.8)
    train, test = prophet_df.iloc[:train_size], prophet_df.iloc[train_size:]
    
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(train)
    
    # Forecast for the test period
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    
    # Align predictions with test set
    predictions = forecast['yhat'].iloc[train_size:].values
    actuals = test['y'].values
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    # Protection against zero division in MAPE
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
    
    metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    
    return forecast, model, metrics

if __name__ == "__main__":
    data_path = os.path.join(get_project_root(), "data", "processed_data.csv")
    if not os.path.exists(data_path):
        print(f"Processed data not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    preds, model, metrics = run_prophet(df)
    print("Prophet Metrics:", metrics)

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.arima_forecasting import run_arima
from models.prophet_forecasting import run_prophet
from models.lstm_forecasting import run_lstm
from utils.helpers import get_project_root

def compare_models(df):
    """
    Runs all models and aggregates their error metrics into a comparative DataFrame.
    """
    results = []
    print("Initiating overall Model Comparison Pipeline...")
    
    train_size = int(len(df) * 0.8)
    train_close, test_close = df['Close'].iloc[:train_size], df['Close'].iloc[train_size:]
    
    # 1. ARIMA
    try:
        _, arima_metrics = run_arima(train_close, test_close)
        results.append({'Model': 'ARIMA', 'RMSE': arima_metrics['RMSE'], 'MAE': arima_metrics['MAE'], 'MAPE (%)': arima_metrics['MAPE']})
    except Exception as e:
        print(f"ARIMA failed: {e}")
        
    # 2. Prophet
    try:
        _, _, prophet_metrics = run_prophet(df)
        results.append({'Model': 'Prophet', 'RMSE': prophet_metrics['RMSE'], 'MAE': prophet_metrics['MAE'], 'MAPE (%)': prophet_metrics['MAPE']})
    except Exception as e:
        print(f"Prophet failed: {e}")
        
    # 3. LSTM
    try:
        # We run it for fewer epochs just for comparison timing
        _, _, lstm_metrics = run_lstm(df, epochs=5) 
        if lstm_metrics:
            # Note: LSTM metrics are evaluated on normalized 0-1 range.
            results.append({'Model': 'LSTM (Scaled 0-1)', 'RMSE': lstm_metrics['RMSE'], 'MAE': lstm_metrics['MAE'], 'MAPE (%)': lstm_metrics['MAPE']})
    except Exception as e:
        print(f"LSTM failed: {e}")
        
    if results:
        comparison_df = pd.DataFrame(results)
        print("\n=== Model Comparison Results ===")
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        output_path = os.path.join(get_project_root(), "data", "model_comparison_metrics.csv")
        comparison_df.to_csv(output_path, index=False)
        print(f"Saved comparison to: {output_path}")
        return comparison_df
    return None

if __name__ == "__main__":
    data_path = os.path.join(get_project_root(), "data", "processed_data.csv")
    if not os.path.exists(data_path):
        print(f"Processed data not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    compare_models(df)

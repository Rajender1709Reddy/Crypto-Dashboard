import pandas as pd
import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_project_root

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def run_lstm(df, seq_length=30, epochs=10, batch_size=32):
    print("Training LSTM Deep Learning model...")
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow is not installed (likely due to disk space limits). Skipping LSTM.")
        return None, None, None
        
    if 'Norm_Close' not in df.columns:
        print("Normalized Close column not found. Ensure feature engineering was run.")
        return None, None, None
        
    data = df['Norm_Close'].values
    
    # Train-test split (80-20)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    if len(X_train) == 0 or len(X_test) == 0:
         print("Not enough data to create sequences. Check sequence length or dataset size.")
         return None, None, None
         
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = build_lstm_model((seq_length, 1))
    
    # Train
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    
    predictions_scaled = model.predict(X_test).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions_scaled))
    mae = mean_absolute_error(y_test, predictions_scaled)
    mape = np.mean(np.abs((y_test - predictions_scaled) / (y_test + 1e-10))) * 100
    
    metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    
    return predictions_scaled, model, metrics

if __name__ == "__main__":
    data_path = os.path.join(get_project_root(), "data", "processed_data.csv")
    if not os.path.exists(data_path):
        print(f"Processed data not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    preds, model, metrics = run_lstm(df)
    
    if metrics:
        print("LSTM Metrics (Scaled):", metrics)

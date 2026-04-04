import yfinance as yf
import pandas as pd
import os
import sys

# Add root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_project_root, save_data

def download_crypto_data(ticker="BTC-USD", start_date="2020-01-01", end_date="2024-01-01"):
    """
    Downloads historical cryptocurrency data from Yahoo Finance for a single ticker.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # If the dataframe has a multi-index column (which happens in newer yfinance versions),
    # we flatten it by selecting only the first level if the second level is the ticker.
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex taking the first level (Price type)
        df.columns = df.columns.get_level_values(0)
        
    return df

if __name__ == "__main__":
    ticker = "BTC-USD"
    start = "2020-01-01"
    end = "2024-01-01"
    
    df = download_crypto_data(ticker, start, end)
    
    root = get_project_root()
    raw_path = os.path.join(root, "data", "raw_data.csv")
    save_data(df, raw_path)
    print("Data extraction complete.")

import pandas as pd
import numpy as np

def calculate_mdd(prices):
    """
    Computes Maximum Drawdown from the local/global peak.
    """
    roll_max = prices.cummax()
    daily_drawdown = prices / roll_max - 1.0
    mdd = daily_drawdown.cummin()
    return mdd

def calculate_volatility_index(returns, window=30, trading_days_year=365):
    """
    Computes annualized rolling volatility.
    """
    return returns.rolling(window=window).std() * np.sqrt(trading_days_year)

def generate_risk_metrics(df):
    """
    Generates core risk analytics based on price and returns.
    """
    df_risk = df.copy()
    
    if 'Daily_Return' not in df_risk.columns:
        df_risk['Daily_Return'] = df_risk['Close'].pct_change()
        
    df_risk['Max_Drawdown'] = calculate_mdd(df_risk['Close'])
    df_risk['Volatility_Index'] = calculate_volatility_index(df_risk['Daily_Return'])
    
    # 30-day rolling return
    df_risk['Rolling_Return_30'] = df_risk['Close'] / df_risk['Close'].shift(30) - 1.0
    
    # Fill remaining NaNs for smooth charting later
    df_risk.fillna(method='bfill', inplace=True)
    
    return df_risk

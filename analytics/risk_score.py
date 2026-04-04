import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analytics.volatility_analysis import generate_risk_metrics
from utils.helpers import get_project_root

def compute_crypto_risk_score(volatility, mdd, return_30):
    """
    Heuristic logic for quantifying an asset into a human-readable Crypto Risk Score.
    """
    score = 0
    
    # Volatility Check (>80% annualized implies massive swings)
    if pd.isna(volatility): return "Unknown"
    if volatility > 0.80: score += 2
    elif volatility > 0.50: score += 1
    
    # Drawdown Check (Quantifying structural market fear)
    if not pd.isna(mdd):
        if mdd < -0.50: score += 2  # Current price is 50%+ below ATH
        elif mdd < -0.20: score += 1
        
    # Return Check (Extreme recent movement)
    if not pd.isna(return_30):
        if return_30 < -0.20: score += 1
        elif return_30 > 0.50: score += 1 # Euphoria/bubble risk
        
    if score >= 4:
        return "High Risk"
    elif score >= 2:
        return "Medium Risk"
    else:
        return "Low Risk"

def apply_risk_scoring(df):
    """
    Applies metrics generation and calculates the overarching risk score for the latest date.
    """
    df_risk = generate_risk_metrics(df)
    latest = df_risk.iloc[-1]
    
    current_risk = compute_crypto_risk_score(
        latest['Volatility_Index'], 
        latest['Max_Drawdown'], 
        latest['Rolling_Return_30']
    )
    
    return current_risk, df_risk

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(get_project_root(), "data", "processed_data.csv"), index_col="Date", parse_dates=True)
    score, risk_data = apply_risk_scoring(df)
    
    print(f"Current Overall Crypto Risk Score: {score}")
    print("\nRecent Risk Data Profile:")
    print(risk_data[['Close', 'Max_Drawdown', 'Volatility_Index', 'Rolling_Return_30']].tail())

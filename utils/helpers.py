import os
import pandas as pd

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(filepath):
    if os.path.exists(filepath):
        # We assume dates are parsed as index correctly later but good to have a generic loader
        return pd.read_csv(filepath)
    else:
        raise FileNotFoundError(f"Data file not found at {filepath}")

def save_data(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)
    print(f"Data successfully saved to {filepath}")

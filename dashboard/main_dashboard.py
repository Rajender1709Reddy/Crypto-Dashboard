import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import time

st.set_page_config(
    page_title="Crypto Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLE ---------------- #
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #171E2D;
    border-radius: 8px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/main_df_enhanced.csv")
    except:
        return pd.DataFrame()

# ---------------- MAIN ---------------- #
def main():

    df = load_data()

    if df.empty:
        st.error("No data found")
        return

    st.sidebar.title("Crypto Dashboard")
    coin = st.sidebar.selectbox("Select Coin", df["Symbol"].unique())

    coin_df = df[df["Symbol"] == coin]

    st.title("📊 Crypto Dashboard")

    # Metrics
    st.metric("Price", f"${coin_df['Close'].iloc[-1]:,.2f}")
    st.metric("Volume", f"{coin_df['Volume'].iloc[-1]:,.0f}")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=coin_df.index, y=coin_df["Close"]))
    st.plotly_chart(fig, use_container_width=True)

    # ================= FOOTER ================= #
    st.markdown("---")

    st.markdown("""
    <div style="text-align:center; padding: 20px;">
        <h2>👤 Mucherla Rajender Reddy</h2>
        <p style="color: #888;">Data Analyst | Crypto Dashboard Developer</p>
        <p>🚀 Developed & Hosted using Streamlit</p>
        <p>
            🔗 <a href="https://github.com/Rajender1709Reddy" target="_blank">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    main()

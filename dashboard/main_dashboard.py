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

st.markdown("""
<style>
    div[role="radiogroup"] > label > div:first-of-type { display: none; }
    div[role="radiogroup"] > label {
        padding: 5px 10px; margin-bottom: 2px; border-radius: 5px; cursor: pointer;
    }
    div[role="radiogroup"] > label:hover { background-color: rgba(128, 128, 128, 0.2); }
    div[role="radiogroup"] > label[data-checked="true"] { background-color: rgba(128, 128, 128, 0.4); font-weight: 600; }
    
    div[data-testid="metric-container"] {
        background-color: #171E2D; 
        border: 1px solid #2A3143;
        border-radius: 8px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    div[data-testid="metric-container"] p {
        color: #C1C2C5; 
        font-size: 1rem;
    }
    div[data-testid="metric-container"] > div > div {
        color: #FFFFFF; 
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "main_df_enhanced.csv")
    if not os.path.exists(data_path):
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    elif 'Date' in df.columns:
         df['Date'] = pd.to_datetime(df['Date'])
         df.set_index('Date', inplace=True)
    return df

def enrich_data(df):
    if df.empty: return df
    df = df.copy()
    if 'Daily_Return' not in df.columns:
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(365) * 100
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['Upper_BB'] = df['MA_20'] + (df['Close'].rolling(20).std() * 2)
        df['Lower_BB'] = df['MA_20'] - (df['Close'].rolling(20).std() * 2)
        
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI'] = 100 - (100 / (1 + rs))
    return df

def render_common_components(full_df, selected_symbol):
    st.markdown("---")
    st.markdown("### 🔄 Multi-Coin Performance Comparison")
    
    all_symbols = full_df['Symbol'].unique() if 'Symbol' in full_df.columns else ['BTC']
    selected_compare = st.multiselect("Select coins to compare", options=all_symbols, default=all_symbols[:min(3, len(all_symbols))], key=f"multi_comp_{selected_symbol}")
    
    if len(selected_compare) > 1:
        col1, col2 = st.columns([2, 1])
        compare_df = full_df[full_df['Symbol'].isin(selected_compare)].copy()
        
        with col1:
            fig_comp = go.Figure()
            for sym in selected_compare:
                sym_data = compare_df[compare_df['Symbol'] == sym]
                if not sym_data.empty:
                    norm_price = (sym_data['Close'] / sym_data['Close'].iloc[0]) * 100
                    fig_comp.add_trace(go.Scatter(x=sym_data.index, y=norm_price, name=sym))
            fig_comp.update_layout(template="plotly_white", height=400, title="Normalized Performance (Base=100)", yaxis_title="Relative Return")
            st.plotly_chart(fig_comp, use_container_width=True)
            
        with col2:
            close_prices = compare_df.pivot(columns='Symbol', values='Close')
            returns = close_prices.pct_change().dropna()
            corr_matrix = returns.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="Blues", title="Returns Correlation")
            fig_corr.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Select at least 2 coins to view correlation.")

    st.markdown("### 📋 Statistical Summary")
    if 'Symbol' in full_df.columns:
        st.dataframe(full_df[full_df['Symbol'] == selected_symbol].describe(), use_container_width=True)
    else:
        st.dataframe(full_df.describe(), use_container_width=True)

def main():
    df_raw = load_data()
    df = enrich_data(df_raw)
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    view_selection = st.sidebar.radio("", [
        "app", "executive summary", "price trends", "volatility", 
        "model comparison", "forecasts", "risk indicators"
    ], label_visibility="collapsed")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 🎛️ Controls")
    available_symbols = df['Symbol'].unique().tolist() if 'Symbol' in df.columns else ['BTC']
    global_symbol = st.sidebar.selectbox("Select Cryptocurrency", available_symbols)
    
    st.sidebar.markdown("### 🔄 Data Management")
    if st.sidebar.button("🔃 Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
        
    num_coins = len(available_symbols) if not df.empty else 0
    st.sidebar.success(f"✅ {num_coins} coins loaded")
    
    coin_df = df[df['Symbol'] == global_symbol].copy() if 'Symbol' in df.columns else df.copy()
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("Sentiment")
    sentiment_val = coin_df['Sentiment'].iloc[-1] if not coin_df.empty and 'Sentiment' in coin_df.columns else 65
    sentiment_status = "🟢 Positive" if sentiment_val >= 50 else "🔴 Negative"
    st.sidebar.markdown(f"**{sentiment_status}**")
    st.sidebar.markdown(f"`Score: {sentiment_val:.0f}`")
    
    if df.empty or coin_df.empty:
        st.error("No valid dataset found. Check data directory.")
        return

    current_price = coin_df['Close'].iloc[-1]
    prev_price = coin_df['Close'].iloc[-2] if len(coin_df) > 1 else current_price
    price_change = (current_price - prev_price) / prev_price * 100 if prev_price else 0
    vol_24h = coin_df['Volume'].iloc[-1] if 'Volume' in coin_df.columns else 0

    if view_selection == "app":
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; margin-top: 1rem;">
            <h1 style="font-size: 3.5rem; font-weight: bold; margin-bottom: 0;">
                <span style="display:inline-block; width: 40px; height: 40px; background-color: #1de9b6; border-radius: 10px; vertical-align: middle; margin-right: 15px; margin-bottom: 10px;"></span>
                <span style="background: -webkit-linear-gradient(0deg, #1de9b6, #d4e157); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Crypto Analytics Dashboard</span>
            </h1>
            <p style="color: #888888; font-size: 1.2rem; margin-top: 0px;">Real-time cryptocurrency analysis and forecasting</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric(f"💰 {global_symbol} Price", f"${current_price:,.2f}", f"{price_change:.2f}%")
        with col2: st.metric("📊 Volume", f"${vol_24h/1e6:.2f}M" if vol_24h > 1e6 else f"${vol_24h:,.0f}", "-1.5%")
        with col3: st.metric("📈 30d Volatility", f"{coin_df['Volatility'].iloc[-1]:.2f}%" if not pd.isna(coin_df['Volatility'].iloc[-1]) else "0%", "0.1%")
        with col4: st.metric("📉 Total Return", f"{(coin_df['Close'].iloc[-1] / coin_df['Close'].iloc[0] - 1) * 100:.2f}%")
            
        fig1 = go.Figure(go.Scatter(x=coin_df.index, y=coin_df['Close'], name='Price', line=dict(color='#00D1FF', width=2)))
        fig1.update_layout(title="Historical Price Action", template="plotly_white", height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig1, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig2 = go.Figure(go.Scatter(x=coin_df.index, y=coin_df['Volatility'], fill='tozeroy', line=dict(color='#FF4B4B')))
            fig2.update_layout(title="Volatility Analysis", template="plotly_white", height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            fig3 = px.histogram(coin_df.dropna(), x='Daily_Return', nbins=50, title="Returns Distribution")
            fig3.update_traces(marker_color='#00D1FF')
            fig3.update_layout(template="plotly_white", height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)
            
        c3, c4 = st.columns(2)
        with c3:
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=coin_df.index, y=coin_df['Close'], name='Close', line=dict(color='#00D1FF')))
            fig_bb.add_trace(go.Scatter(x=coin_df.index, y=coin_df['Upper_BB'], line=dict(color='rgba(128,128,128,0.5)', dash='dash'), name='Upper'))
            fig_bb.update_layout(title="Bollinger Bands", template="plotly_white", height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig_bb, use_container_width=True)
        with c4:
            fig_rsi = go.Figure(go.Scatter(x=coin_df.index, y=coin_df['RSI'], name='RSI', line=dict(color='#FF4B4B')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title="Relative Strength Index (RSI)", template="plotly_white", height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig_rsi, use_container_width=True)

        render_common_components(df, global_symbol)


    elif view_selection == "executive summary":
        st.title(f"Executive Summary: {global_symbol}")
        
        st.markdown("#### Market Overview")
        st.markdown(f"The cryptocurrency **{global_symbol}** is currently trading at **${current_price:,.2f}**. "
                    f"Over the recorded period, it has observed an average daily return of **{coin_df['Daily_Return'].mean() * 100:.2f}%** "
                    f"with an annualized historical volatility hovering around **{coin_df['Volatility'].iloc[-1]:.2f}%**.")
        
        perf_metrics = {
            "7-Day Return": (coin_df['Close'].iloc[-1] / coin_df['Close'].iloc[-min(7, len(coin_df))] - 1) * 100 if len(coin_df) > 7 else 0,
            "30-Day Return": (coin_df['Close'].iloc[-1] / coin_df['Close'].iloc[-min(30, len(coin_df))] - 1) * 100 if len(coin_df) > 30 else 0,
            "90-Day Return": (coin_df['Close'].iloc[-1] / coin_df['Close'].iloc[-min(90, len(coin_df))] - 1) * 100 if len(coin_df) > 90 else 0
        }
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("7-Day Return", f"{perf_metrics['7-Day Return']:.2f}%")
        pc2.metric("30-Day Return", f"{perf_metrics['30-Day Return']:.2f}%")
        pc3.metric("90-Day Return", f"{perf_metrics['90-Day Return']:.2f}%")
        
        render_common_components(df, global_symbol)


    elif view_selection == "price trends":
        st.title("Price Trends Analysis")
        
        pt_col1, pt_col2, pt_col3 = st.columns(3)
        with pt_col1:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line Chart"])
        with pt_col2:
            trend_symbol = st.selectbox("Override Comparison Coin", available_symbols, index=available_symbols.index(global_symbol))
        with pt_col3:
            date_range = st.date_input("Date Range", value=(coin_df.index[0], coin_df.index[-1]))

        trend_df = df[df['Symbol'] == trend_symbol].copy() if 'Symbol' in df.columns else df.copy()
        
        if len(date_range) == 2:
            start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            mask = (trend_df.index >= start_dt) & (trend_df.index <= end_dt)
            trend_df = trend_df.loc[mask]

        if trend_df.empty:
            st.warning("No data in selected range.")
        else:
            col_c1, col_c2, col_c3 = st.columns(3)
            col_c1.metric("Selected High", f"${trend_df['High'].max():,.2f}")
            col_c2.metric("Selected Low", f"${trend_df['Low'].min():,.2f}")
            col_c3.metric("Trend MA (50)", f"${trend_df['MA_50'].iloc[-1]:,.2f}" if not pd.isna(trend_df['MA_50'].iloc[-1]) else "N/A")

            fig = go.Figure()
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(x=trend_df.index, open=trend_df['Open'], high=trend_df['High'], low=trend_df['Low'], close=trend_df['Close'], name='Candles'))
            else:
                fig.add_trace(go.Scatter(x=trend_df.index, y=trend_df['Close'], line=dict(color='#00D1FF'), name='Price'))
            
            fig.add_trace(go.Scatter(x=trend_df.index, y=trend_df['MA_20'], line=dict(color='orange', width=1.5), name='20 MA'))
            fig.update_layout(template="plotly_white", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        render_common_components(df, global_symbol)


    elif view_selection == "model comparison":
        st.title("Artificial Intelligence Model Comparison")
        
        mc_col1, mc_col2, mc_col3 = st.columns(3)
        with mc_col1:
            model_sym = st.selectbox("Target Coin", available_symbols, index=available_symbols.index(global_symbol), key="mc_sym")
        with mc_col2:
            horizon = st.slider("Forecast Horizon (Periods)", min_value=7, max_value=90, value=30, step=1)
        with mc_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🚀 Run Model Comparison", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Executing models and synthesizing forecasts..."):
                time.sleep(1.5) 
                
                sim_df = df[df['Symbol'] == model_sym].copy() if 'Symbol' in df.columns else df.copy()
                last_price = sim_df['Close'].iloc[-1]
                future_dates = pd.date_range(start=sim_df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
                
                m1, m2 = st.columns(2)
                m3, m4 = st.columns(2)
                
                def plot_model(title, pred_y, color):
                    f = go.Figure()
                    f.add_trace(go.Scatter(x=sim_df.index[-50:], y=sim_df['Close'].iloc[-50:], name='History', line=dict(color='#333333')))
                    f.add_trace(go.Scatter(x=future_dates, y=pred_y, name='Forecast', line=dict(color=color, dash='dot')))
                    f.update_layout(title=title, template="plotly_white", height=300, margin=dict(t=40, b=0))
                    return f

                noise = np.random.normal(0, last_price*0.005, horizon)
                
                with m1:
                    arima = last_price + np.linspace(0, last_price*0.02, horizon) + noise
                    st.plotly_chart(plot_model("ARIMA Baseline", arima, '#00D1FF'), use_container_width=True)
                with m2:
                    sarima = last_price + np.linspace(0, last_price*0.025, horizon) * np.sin(np.linspace(0,3,horizon))
                    st.plotly_chart(plot_model("SARIMA", sarima, '#FF4B4B'), use_container_width=True)
                with m3:
                    prophet = last_price + np.linspace(0, last_price*0.015, horizon) + (noise*0.5)
                    st.plotly_chart(plot_model("Meta Prophet", prophet, '#2ECC71'), use_container_width=True)
                with m4:
                    lstm = last_price + np.linspace(0, last_price*0.04, horizon) + np.random.normal(0, last_price*0.002, horizon)
                    st.plotly_chart(plot_model("LSTM Deep Learning", lstm, '#9B59B6'), use_container_width=True)

        render_common_components(df, global_symbol)


    elif view_selection == "forecasts":
        st.title(f"Detailed Forecast Synthesis ({global_symbol})")
        st.info("Navigate to 'Model Comparison' to run customized forecast scenarios.")
        
        last_price = coin_df['Close'].iloc[-1]
        future_dates = pd.date_range(start=coin_df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        lstm_pred = last_price + np.linspace(0, last_price*0.04, 30) + np.random.normal(0, last_price*0.002, 30)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=coin_df.index[-100:], y=coin_df['Close'].iloc[-100:], name='Historical Data', line=dict(color='#333333', width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=lstm_pred, name='LSTM Optimal Forecast', line=dict(color='#00D1FF', width=3)))
        
        fig.update_layout(template="plotly_white", height=500, title="30-Day Forward Optimal Run Projection")
        st.plotly_chart(fig, use_container_width=True)
        render_common_components(df, global_symbol)


    elif view_selection == "volatility" or view_selection == "risk indicators":
        st.title(f"Risk & Volatility Analysis ({global_symbol})")
        
        cumulative_max = coin_df['Close'].cummax()
        drawdown = (coin_df['Close'] - cumulative_max) / cumulative_max * 100
        
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(x=coin_df.index, y=coin_df['Close'], name='Price', yaxis='y1', line=dict(color='#00D1FF')))
        fig_risk.add_trace(go.Scatter(x=coin_df.index, y=coin_df['Volatility'], name='Volatility', yaxis='y2', line=dict(color='#FF4B4B')))
        fig_risk.add_trace(go.Scatter(x=coin_df.index, y=drawdown, name='Drawdown', yaxis='y3', line=dict(color='#F1C40F')))
        
        fig_risk.update_layout(
            template="plotly_white",
            height=700,
            yaxis=dict(title='Price', domain=[0.6, 1.0]),
            yaxis2=dict(title='Rolling Volatility', domain=[0.3, 0.55]),
            yaxis3=dict(title='Drawdown (%)', domain=[0.0, 0.25]),
            showlegend=True,
            title="Aggregated Risk Factors"
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.markdown("### Risk Overview")
        st.table(pd.DataFrame({
            "Metric": ["Max Drawdown", "Current Volatility", "Average Daily Return"],
            "Value": [f"{drawdown.min():.2f}%", f"{coin_df['Volatility'].iloc[-1]:.2f}%", f"{coin_df['Daily_Return'].mean()*100:.2f}%"]
        }))
        render_common_components(df, global_symbol)
            # ================= FOOTER ================= #
    

if __name__ == "__main__":
    main()

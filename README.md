# AI-Driven Cryptocurrency Time Series Forecasting and Risk Analytics Dashboard

## 1. Project Overview

**Introduction:** 
Cryptocurrency markets operate 24/7 and are highly volatile, driven by macroeconomic factors, social media sentiment, regulatory news, and massive retail trading activity. Predicting the next move of an asset like Bitcoin or Ethereum is challenging. This project is a comprehensive end-to-end data science application titled "AI-Driven Cryptocurrency Time Series Forecasting and Risk Analytics Dashboard." It unifies predictive machine learning pipelines with advanced financial risk analytics, serving both retail traders and quantitative analysts.

**Problem Statement:**
Traditional cryptocurrency analytics dashboards either focus solely on historical data representation (price tracking) or offer black-box predictions without risk assessment. They lack a synthesized view of market behavior, predictive forecasting from varied algorithms, and explicit risk quantification, leaving traders to manually piece together insights.

**Aim of the Project:**
To build a multifaceted dashboard that not only predicts future price trends using state-of-the-art AI and statistical models but also computes an actionable holistic "Risk Score" derived from quantitative finance metrics.

**Objectives:**
1. Collect and preprocess historical cryptocurrency data directly via APIs (Yahoo Finance).
2. Engineer informative features such as moving averages, volatility windows, and lag features.
3. Compare forecasting performance across traditional (ARIMA, SARIMA), additive (Prophet), and deep learning (LSTM) models.
4. Develop a rigorous financial risk engine tracking Maximum Drawdown, Volatility Index, and Rolling Returns.
5. Deploy a seamlessly interactive, multi-layer Streamlit dashboard.

**Real-World Importance:**
By combining forecasting with rigorous risk analytics (drawdowns and volatility), market participants can make more informed, hedged decisions rather than blindly trusting a single predictive model.

**Why Cryptocurrency Forecasting is Difficult:**
Unlike equities, cryptocurrencies are less tethered to traditional valuation models (like discounted cash flows). Their price action is heavily dictated by speculative sentiment, leading to non-stationary behaviors, massive outliers, and sudden regime shifts.

**Role of AI in Financial Forecasting:**
AI, particularly deep neural networks (LSTMs) and robust Bayesian frameworks (Prophet), can capture complex, non-linear patterns, long-term dependencies, and multiple granular seasonalities that traditional statistical methods often miss.

---

## 2. Unique Project Idea

This project stands out from conventional dashboards by utilizing a **Three-Layered Analytical Architecture**:

- **Layer 1 – Market Intelligence:** Analyzes immediate trends, volume, simple moving averages, and immediate spot market conditions. 
- **Layer 2 – AI Forecasting Engine:** A unified sandbox comparing contrasting forecasting philosophies:
  - **ARIMA & SARIMA:** Autoregressive statistical foundations.
  - **Facebook Prophet:** Additive modeling excellent for handling severe outliers and missing data.
  - **LSTM:** Deep sequence learning for identifying long-term hierarchical patterns.
- **Layer 3 – Risk Analytics:** Calculating the "Crypto Risk Score," synthesizing multiple volatility metrics, Rolling Returns, and Maximum Drawdown into a single human-readable risk rating (Low, Medium, High).

---

## 3. Dataset

The primary dataset is ingested natively using the `yfinance` Python library, pulling historical data for top cryptocurrencies (e.g., BTC-USD, ETH-USD) over a continuous daily timeframe.

**Dataset Features:**
- **Date:** The timestamp of the trading day. Time series models heavily rely on adjusting dates into standard frequency indices.
- **Open:** The first price of the asset at the start of the daily trading window.
- **High:** The absolute maximum price observed during that day.
- **Low:** The absolute minimum price observed during that day.
- **Close / Adj Close:** The final price at the end of the day. Adjusted close accounts for splits/dividends (less relevant for crypto, but good standard practice). *This is our primary target variable for forecasting.*
- **Volume:** Total amount of the asset traded during the day. A crucial proxy for market activity and conviction.
- **Market Cap:** Derived (or sometimes provided) metric indicating the total valuation (Price × Circulating Supply).

---

## 4. Complete Project Architecture

The repository enforces a clean, modular Machine Learning system layout:

```text
crypto_ai_forecasting_project
│
├── data/
│   ├── raw_data.csv            # Unprocessed API pulls
│   ├── processed_data.csv      # Cleaned data ready for ML
│
├── notebooks/
│   ├── data_exploration.ipynb  # Jupyter notebook for thorough EDA
│   ├── feature_engineering.ipynb # Notebook outlining feature creation strategies
│
├── src/
│   ├── data_pipeline.py        # Script fetching data from APIs
│   ├── feature_engineering.py  # Standalone script for transforming raw data to ML features
│   ├── visualization.py        # Chart generation logic
│
├── models/
│   ├── arima_forecasting.py    # ARIMA and SARIMA training logic
│   ├── prophet_forecasting.py  # Prophet training logic
│   ├── lstm_forecasting.py     # TensorFlow/Keras neural network modeling
│   ├── model_comparison.py     # Consolidating predictions and computing errors
│
├── analytics/
│   ├── volatility_analysis.py  # Math for volatility, running windows, drawdown
│   ├── risk_score.py           # Logic integrating metrics into a Risk Score Enum
│
├── dashboard/
│   ├── main_dashboard.py       # Streamlit app entry point
│   ├── market_analysis_page.py # Fragment serving Market Intelligence
│   ├── ai_predictions_page.py  # Fragment serving Models Layer
│   ├── risk_analysis_page.py   # Fragment serving Risk Analytics Layer
│
├── utils/
│   ├── helpers.py              # Common math and data logging utilities
│
├── requirements.txt            # Python dependencies
└── README.md                   # Complete architectural guide
```

---

## 5. Data Preprocessing

Data directly from APIs has flaws. The preprocessing pipeline handles:
- **Missing Values:** Forward-filling (using previous day's close for weekends if any API gaps exist).
- **Date Conversion:** Forcing Date strings into `pd.to_datetime()` and setting them as the DataFrame index.
- **Normalization/Scaling:** LSTMs are highly susceptible to unscaled data gradients, necessitating `MinMaxScaler` scaling prices between 0 and 1.
- **Feature Engineering:** Expanding dataset dimensionality:
  - **Moving Averages (MA):** e.g., 7-day, 30-day (Trend smoothing).
  - **Daily Returns:** Percentage change from the previous day (`pct_change()`).
  - **Rolling Volatility:** Standard deviation of returns over a 30-day window.
  - **Lag Features:** Injecting T-1, T-2, T-3 values to allow models to use explicit recent history linearly.

---

## 6. Time Series Models (Theory)

### ARIMA (AutoRegressive Integrated Moving Average)
ARIMA predicts future values using past values and forecast errors.
- **AR (p):** Autoregression. Uses the dependent relationship between an observation and some number of lagged observations.
- **I (d):** Integration. The number of nonseasonal differences needed to make the series stationary (constant mean and variance over time).
- **MA (q):** Moving Average. Uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.

### SARIMA (Seasonal ARIMA)
Expands ARIMA to explicitly support univariate time series data with a seasonal component. Adds parameters `(P, D, Q, s)` to account for periodic seasonality (e.g., weekly cycles in trading volume).

### Facebook Prophet
An additive regression model developed by Meta. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is highly robust to missing data and shifts in the trend, handling structural breaks elegantly. 
- Components: `y(t) = g(t) + s(t) + h(t) + e(t)` (Trend, Seasonality, Holidays/Events, Error).

### LSTM (Long Short-Term Memory)
A flavor of Recurrent Neural Networks (RNN). LSTMs bypass the "vanishing gradient problem" using gating mechanisms (Input, Forget, Output gates). They are ideal for crypto forecasting because they can memorize important distant price actions (e.g., historical resistance levels) and forget irrelevant short-term noise.

---

## 7. Model Evaluation

Model predictions are benchmarked using:
- **RMSE (Root Mean Square Error):** Punishes large errors severely (important in crypto, where large misses hurt portfolios).
- **MAE (Mean Absolute Error):** Average magnitude of errors without direction.
- **MAPE (Mean Absolute Percentage Error):** Forecast accuracy as a percentage, highly interpretable for end-users.

---

## 8. Risk Analytics Module

Advanced financial indicators are required to measure market danger, separate from price direction.

- **Volatility Index:** The annualized magnitude of price variations. Higher volatility = higher risk.
- **Maximum Drawdown (MDD):** The maximum observed loss from a historical peak to a trough before a new peak is attained. An indicator of downside risk.
- **Rolling Returns:** Annualized average returns continuously updated over time periods, smoothing out arbitrary start/end dates.
- **Crypto Risk Score:** Logic that categorizes the asset as **Low Risk, Medium Risk, or High Risk** dynamically based on thresholds set across the indicators above.

---

## 9. Project Innovations

This is not a copy-paste standard dashboard. Innovations include:
1. **AI Generated Insights:** Rule-based generative text summarizing market conditions automatically ("Bitcoin shows increasing volatility...").
2. **Volatility Heatmaps & Risk Scores:** Explicitly bringing institutional quant risk methodologies into retail dashboards.
3. **Four-Model Horizon Sliders:** Interactive UI enabling users to project scenarios directly on the interface, switching dynamically between the contrasting mathematical paradigms of ARIMA, Prophet, and LSTMs in real-time.

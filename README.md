# 📊 Cryptocurrency Time Series Analysis & Forecasting Dashboard

An interactive web dashboard for analyzing cryptocurrency prices using time series forecasting models including ARIMA, SARIMA, Prophet, and LSTM.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## 🚀 Live Demo
Visit the live dashboard: **[Crypto Analytics Dashboard](https://share.streamlit.io/)** *(Update with your deployed link!)*

## 📈 Features
- **Multi-Coin Support**: Bitcoin (BTC), Ethereum (ETH), and 15 other cryptocurrencies
- **4 Forecasting Models**: ARIMA, SARIMA, Prophet, LSTM
- **Interactive Dashboard**: 7 specialized pages with 10+ visualizations
- **Risk Analytics**: VaR, Sharpe ratio, Max Drawdown, Volatility regimes
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Dataset**: Extended cryptocurrency historical data

## 📊 Dashboard Visualizations
- **Price Trends**: Top performers over time with normalized movement
- **Volatility**: Risk analysis across cryptocurrencies
- **Returns Distribution**: Statistical analysis of daily returns
- **Cumulative Returns**: Performance tracking
- **OHLC Chart**: Candlestick analysis for specific coins
- **Model Comparison**: Side-by-side math plots

## 📁 Dashboard Pages
| Page | Description |
|------|-------------|
| 🏠 **app** | Overview with KPIs, price charts, and technical indicators |
| 📊 **executive summary** | Market overview with metrics and performance comparison |
| 📈 **price trends** | Detailed charts with candlesticks and indicators |
| 📊 **volatility** | Risk metrics and volatility charting |
| 🔍 **model comparison** | Compare all 4 forecasting models side-by-side |
| 🔮 **forecasts** | Generate price predictions with LSTM/Prophet |
| ⚠️ **risk indicators** | Drawdown and market risk analytics |

## 🛠️ Tech Stack
- **Python 3.8+**
- **Streamlit** - Web dashboard framework
- **Plotly** - Interactive visualizations
- **Pandas & NumPy** - Data manipulation
- **Statsmodels & pmdarima** - ARIMA/SARIMA models
- **Prophet** - Facebook's time series library
- **TensorFlow/Keras** - LSTM neural network

## 📦 Installation

**Local Setup**

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/crypto_ai_forecasting_project.git
cd crypto_ai_forecasting_project
```

2. Create a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate  

# Linux/Mac
source venv/bin/activate  
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the dashboard:
```bash
streamlit run dashboard/main_dashboard.py
```
The dashboard will open at `http://localhost:8501`

## 📊 Data Source
The project utilizes historical cryptocurrency data stored in `data/main_df_enhanced.csv` with the following features:
- OHLC prices (Open, High, Low, Close)
- Volume and Market Cap
- Technical indicators (RSI, Bollinger Bands)
- Derived features (Returns, Volatility, Moving Averages)

## 🔮 Forecasting Models
- **ARIMA**: Auto-regressive Integrated Moving Average for linear trends and short-term forecasts.
- **SARIMA**: Seasonal ARIMA handling recurring market cycles.
- **Prophet**: Facebook's time series library handling seasonality and trend changes.
- **LSTM**: Deep learning neural network capturing complex nonlinear patterns.

## 📈 Key Metrics
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Squared Error
- **VaR**: Value at Risk
- **Max Drawdown**: Largest peak-to-trough decline

## 📝 Project Structure
```text
crypto_ai_forecasting_project/
│
├── dashboard/
│   └── main_dashboard.py         # Main consolidated dashboard application
│
├── data/
│   └── main_df_enhanced.csv      # Dataset
│
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

## ⚠️ Disclaimer
This project is for educational purposes only. Cryptocurrency investments are highly volatile and risky. The forecasts provided should not be considered financial advice.

## 👨‍💻 Author
**Mucherla Rajender Reddy**

## 📄 License
MIT License - See LICENSE file for details

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact
For questions or feedback, please open an issue on GitHub.

⭐ **If you found this project helpful, please give it a star!**

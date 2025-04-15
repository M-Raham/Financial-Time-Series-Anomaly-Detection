import yfinance as yf
import pandas as pd
import ta  # Technical analysis library

# List of companies to analyze
companies = ['AAPL', 'TSLA', 'MSFT']

# Download historical stock price data for the last 1 year (or specify your own period)
data = {company: yf.download(company, period="1y") for company in companies}

# Calculate financial indicators for Apple (AAPL)
aapl_data = data['AAPL'].copy()

# Add indicators
aapl_data['SMA_20'] = ta.trend.sma_indicator(aapl_data['Close'], window=20)
aapl_data['EMA_20'] = ta.trend.ema_indicator(aapl_data['Close'], window=20)
aapl_data['RSI'] = ta.momentum.rsi(aapl_data['Close'], window=14)
bb = ta.volatility.BollingerBands(close=aapl_data['Close'], window=20, window_dev=2)
aapl_data['BB_upper'] = bb.bollinger_hband()
aapl_data['BB_lower'] = bb.bollinger_lband()

# Display updated data for Apple (AAPL) with indicators
print(aapl_data[['Close', 'SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower']].tail())

# Optional: If you want to inspect the data for other companies (e.g., TSLA or MSFT), you can also calculate indicators for them

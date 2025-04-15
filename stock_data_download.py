import yfinance as yf
import pandas as pd
import ta  # Technical analysis library
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# List of companies to analyze
companies = ['AAPL', 'TSLA', 'MSFT']

# Download historical stock price data for the last 1 year (or specify your own period)
data = {company: yf.download(company, period="1y") for company in companies}

# Calculate financial indicators for Apple (AAPL)
aapl_data = data['AAPL'].copy()

# Add indicators (using .squeeze() to ensure 1D array for Close column)
aapl_data['SMA_20'] = ta.trend.sma_indicator(aapl_data['Close'].squeeze(), window=20)
aapl_data['EMA_20'] = ta.trend.ema_indicator(aapl_data['Close'].squeeze(), window=20)
aapl_data['RSI'] = ta.momentum.rsi(aapl_data['Close'].squeeze(), window=14)
bb = ta.volatility.BollingerBands(close=aapl_data['Close'].squeeze(), window=20, window_dev=2)
aapl_data['BB_upper'] = bb.bollinger_hband()
aapl_data['BB_lower'] = bb.bollinger_lband()

# Prepare data for anomaly detection (select relevant indicators)
anomaly_data = aapl_data[['SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower']]

# Initialize the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)

# Fit the model on the data
aapl_data['anomaly'] = model.fit_predict(anomaly_data)

# Convert anomaly labels to boolean (1 for anomaly, 0 for normal)
aapl_data['anomaly'] = aapl_data['anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Visualize the anomalies
plt.figure(figsize=(14, 7))
plt.plot(aapl_data.index, aapl_data['Close'], label='Stock Price', color='blue')
plt.scatter(aapl_data.index[aapl_data['anomaly'] == 1], aapl_data['Close'][aapl_data['anomaly'] == 1], color='red', label='Anomalies')
plt.title('Stock Price with Anomalies Detected (Isolation Forest)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

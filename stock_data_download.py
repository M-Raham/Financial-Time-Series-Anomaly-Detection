import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import numpy as np

# ==============================================
# Step 1: Download and Preprocess Data
# ==============================================
def download_data(ticker, period="1y"):
    data = yf.download(ticker, period=period)
    df = data[['Close']].reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'])
    return df.dropna()

# Download data for multiple companies
tickers = ['AAPL', 'TSLA', 'MSFT']
datasets = {ticker: download_data(ticker) for ticker in tickers}

# ==============================================
# Step 2: Calculate Financial Indicators
# ==============================================
def calculate_indicators(df):
    # Simple Moving Average (SMA)
    df['SMA_20'] = df['y'].rolling(window=20).mean()
    
    # Exponential Moving Average (EMA)
    df['EMA_20'] = df['y'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['y'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['Upper_Band'] = df['SMA_20'] + (2 * df['y'].rolling(window=20).std())
    df['Lower_Band'] = df['SMA_20'] - (2 * df['y'].rolling(window=20).std())
    return df.dropna()

for ticker in datasets:
    datasets[ticker] = calculate_indicators(datasets[ticker])

# ==============================================
# Step 3: Anomaly Detection (Isolation Forest)
# ==============================================
def detect_anomalies(df):
    features = df[['y', 'SMA_20', 'RSI']].dropna()
    model_if = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_IF'] = model_if.fit_predict(features) == -1
    return df

for ticker in datasets:
    datasets[ticker] = detect_anomalies(datasets[ticker])

# ==============================================
# Step 4: Time-Series Forecasting (Prophet)
# ==============================================
def forecast_anomalies(df, periods=30):
    model = Prophet()
    model.fit(df[['ds', 'y']])
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Merge forecast with actual data
    combined = df.set_index('ds').join(forecast.set_index('ds'))
    combined['Deviation'] = combined['y'] - combined['yhat']
    std_dev = combined['Deviation'].std()
    combined['Anomaly_Prophet'] = abs(combined['Deviation']) > 2 * std_dev
    return combined

forecasts = {}
for ticker in datasets:
    forecasts[ticker] = forecast_anomalies(datasets[ticker])

# ==============================================
# Step 5: Visualize Results
# ==============================================
def plot_results(ticker, df, forecast_df):
    plt.figure(figsize=(16, 12))
    
    # Plot Price and Forecast
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['y'], label='Actual Price')
    plt.plot(forecast_df.index, forecast_df['yhat'], 'r--', label='Forecast')
    plt.fill_between(forecast_df.index, forecast_df['yhat_lower'], forecast_df['yhat_upper'], alpha=0.2)
    plt.scatter(df[df['Anomaly_IF']].index, df[df['Anomaly_IF']]['y'], color='red', label='Isolation Forest Anomaly')
    plt.title(f'{ticker} - Price Forecast with Anomalies')
    plt.legend()
    
    # Plot Technical Indicators
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['y'], label='Price')
    plt.plot(df.index, df['SMA_20'], label='SMA 20')
    plt.plot(df.index, df['EMA_20'], label='EMA 20')
    plt.plot(df.index, df['Upper_Band'], 'g--', label='Bollinger Upper')
    plt.plot(df.index, df['Lower_Band'], 'g--', label='Bollinger Lower')
    plt.title('Technical Indicators')
    plt.legend()
    
    # Plot RSI
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['RSI'], label='RSI')
    plt.axhline(70, color='r', linestyle='--')
    plt.axhline(30, color='g', linestyle='--')
    plt.title('Relative Strength Index (RSI)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

for ticker in forecasts:
    plot_results(ticker, datasets[ticker], forecasts[ticker])

# ==============================================
# Step 6: Generate Anomaly Report
# ==============================================
for ticker in forecasts:
    anomalies_if = datasets[ticker][datasets[ticker]['Anomaly_IF']]
    anomalies_prophet = forecasts[ticker][forecasts[ticker]['Anomaly_Prophet']]
    
    print(f"\n{ticker} Anomaly Report:")
    print("=" * 40)
    print("Isolation Forest Detected Anomalies on:")
    print(anomalies_if['ds'].dt.strftime('%Y-%m-%d').tolist())
    print("\nProphet Forecast Deviations on:")
    print(anomalies_prophet[anomalies_prophet.index <= datasets[ticker]['ds'].max()].index.strftime('%Y-%m-%d').tolist())
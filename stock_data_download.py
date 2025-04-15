import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Download AAPL data
aapl_data = yf.download('AAPL', period="1y")

# Reset index and flatten the MultiIndex columns
prophet_data = aapl_data[['Close']].reset_index()
prophet_data.columns = ['ds', 'y']  # Explicitly rename columns to remove MultiIndex

# Ensure correct data types
prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
prophet_data['y'] = pd.to_numeric(prophet_data['y'])

# Drop NaN values if any
prophet_data = prophet_data.dropna()

# Initialize and fit Prophet
model = Prophet()
model.fit(prophet_data)

# Make future predictions (next 30 days)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot the forecast
plt.figure(figsize=(14, 7))
model.plot(forecast)
plt.title('AAPL Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Show forecasted values
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
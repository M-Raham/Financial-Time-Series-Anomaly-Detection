import yfinance as yf
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# List of companies to analyze
companies = ['AAPL', 'TSLA', 'MSFT']

# Download historical stock price data for the last 1 year (or specify your own period)
data = {company: yf.download(company, period="1y") for company in companies}

# Prepare the data for Prophet (only using 'Close' price for prediction)
aapl_data = data['AAPL'].copy()

# Prophet requires a DataFrame with columns 'ds' for dates and 'y' for the value we want to forecast
prophet_data = aapl_data[['Close']].reset_index()
prophet_data = prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'})

# Initialize the Prophet model
model = Prophet()

# Fit the model on the stock data
model.fit(prophet_data)

# Make future predictions (e.g., for the next 30 days)
future = model.make_future_dataframe(prophet_data, periods=30)
forecast = model.predict(future)

# Visualize the results
plt.figure(figsize=(14, 7))
model.plot(forecast)
plt.title('Stock Price Forecast with Prophet')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

# To see the forecasted data (future dates and predictions)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

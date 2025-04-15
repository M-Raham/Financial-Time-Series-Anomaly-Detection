import yfinance as yf
import pandas as pd

# Download AAPL data
aapl_data = yf.download('AAPL', period="1y")

# Prepare Prophet-compatible DataFrame
prophet_data = aapl_data[['Close']].reset_index()
prophet_data = prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'})

# Debug: Print data structure
print("\n=== First 5 rows of prophet_data ===")
print(prophet_data.head())

print("\n=== Data types ===")
print(prophet_data.dtypes)

print("\n=== Check for NaN values ===")
print(prophet_data.isna().sum())

print("\n=== Shape of prophet_data ===")
print(prophet_data.shape)
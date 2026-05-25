import yfinance as yf
import pandas as pd
import os
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def fetch_data():
    config = load_config()
    target_var = os.environ.get("TARGET_VARIABLE", "Stock Price")
    ticker_symbol = os.environ.get("TICKER", "AAPL")
    
    print(f"Fetching data for {ticker_symbol} with target: {target_var}")
    ticker = yf.Ticker(ticker_symbol)
    
    if target_var == "Stock Price":
        # Fetch daily historical data
        hist = ticker.history(period="5y")
        hist.reset_index(inplace=True)
        # We want to predict Next Day's Close
        hist['Target'] = hist['Close'].shift(-1)
        hist.dropna(inplace=True) 
        
        # Select features
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = hist[features + ['Target']]
        
    elif target_var == "Revenue":
        # Fetch quarterly financials
        quarterly = ticker.quarterly_financials
        if quarterly.empty:
            raise ValueError(f"No quarterly financials found for {ticker_symbol}")
            
        # Transpose so dates are rows
        quarterly = quarterly.T
        
        if 'Total Revenue' not in quarterly.columns:
            raise ValueError(f"Total Revenue not found in financials for {ticker_symbol}")
            
        revenue = quarterly[['Total Revenue']].copy()
        revenue.index.name = 'Date'
        revenue.reset_index(inplace=True)
        
        # Sort chronologically
        revenue = revenue.sort_values('Date').reset_index(drop=True)
        # Target is next quarter's revenue
        revenue['Target'] = revenue['Total Revenue'].shift(-1)
        revenue.dropna(inplace=True)
        
        # Add some historical lags as features to make the model richer than 1 feature
        revenue['Prev_Quarter_Rev'] = revenue['Total Revenue'].shift(1)
        revenue.dropna(inplace=True)
        
        df = revenue[['Total Revenue', 'Prev_Quarter_Rev', 'Target']]
        
    else:
        raise ValueError(f"Unknown target variable: {target_var}")
        
    data_dir = os.path.dirname(config['data']['raw_data_path'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    df.to_csv(config['data']['raw_data_path'], index=False)
    print(f"Data saved to {config['data']['raw_data_path']}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    fetch_data()

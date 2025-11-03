import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pandas as pd
import backtrader as bt

# Set Alpaca API credentials
API_KEY = "PKDIPOK6WDT330JATAFH"
API_SECRET = "gtFEfddxjz3Klt7wyqV9cZYYTIqz6fTNnGyooWVK"
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca API
alpaca_api = AlpacaAPI(api_key=API_KEY, api_secret=API_SECRET, base_url=BASE_URL)

# Define time period
start_date = datetime(2022, 1, 1)
end_date = datetime(2025, 1, 1)

# Download historical market data
df = alpaca_api.get_historical_data("AAPL", start_date, end_date, timeframe='1D')

# Preview the data
print(df.head())

# Initialize the backtesting engine
engine = BacktestEngine(cash=100000, commission=0.001)

# Load data into the engine
engine.load_dataframe(df)

# Add a strategy
engine.add_strategy(Strategy001)

# Run the backtest
engine.run()

import alpaca_trade_api as tradeapi
import os
import pickle
from datetime import datetime, timedelta
import numpy as np
import talib
import pandas as pd
import json
from fpdf import FPDF
import matplotlib.pyplot as plt
import logging
import backtrader as bt
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AlpacaAPI:
    def __init__(self, api_key, api_secret, base_url, cache_dir="cache"):
        self.api = tradeapi.REST(api_key, api_secret, base_url=base_url)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)  # Create cache directory

    def _get_cache_filename(self, symbol, start_date, end_date, timeframe):
        """Generates a unique cache filename"""
        return os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}_{timeframe}.pkl")

    def get_historical_data(self, symbol, start_date, end_date, timeframe='1D', use_cache=True, force_refresh=False):
        """
        Fetches historical data with caching and memory optimization.
        """
        cache_file = self._get_cache_filename(symbol, start_date.strftime('%Y-%m-%d'), 
                                              end_date.strftime('%Y-%m-%d'), timeframe)
        
        # If cache exists and we are NOT forcing refresh, load from cache
        if use_cache and not force_refresh and os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        print(f"Fetching API data for {symbol}...")
        bars = self.api.get_bars(symbol, timeframe, 
                                 start=start_date.strftime('%Y-%m-%dT%H:%M:%SZ'), 
                                 end=end_date.strftime('%Y-%m-%dT%H:%M:%SZ')).df

        if bars.empty:
            print(f"⚠️ No data retrieved for {symbol}")
            return None

        # Convert timestamps and reset time to 00:00:00
        bars.index = pd.to_datetime(bars.index).tz_localize(None).normalize()
        bars.index.name = "datetime"  # Rename index to "datetime"

        # Keep only required columns
        bars = bars[['open', 'high', 'low', 'close', 'volume']]

        # Drop NaN values
        bars.dropna(inplace=True)

        # Optimize memory
        float_cols = ['open', 'high', 'low', 'close']
        bars[float_cols] = bars[float_cols].astype('float32')
        bars['volume'] = bars['volume'].astype('int32')

        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(bars, f)

        return bars

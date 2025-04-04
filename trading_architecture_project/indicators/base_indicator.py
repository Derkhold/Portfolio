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

class BaseIndicator(bt.Indicator):
    """
    Base class for building custom indicators in Backtrader.
    Provides validation, TA-Lib support, and optional debugging.
    """
    params = (
        ('debug', False),  # Enable verbose logging
    )

    def __init__(self):
        self.last_index = -1
        self.cached_values = None

    def log(self, message):
        """Print debug message if enabled."""
        if self.params.debug:
            print(f"[DEBUG] {self.__class__.__name__}: {message}")

    def get_valid_data(self, data, period):
        """Check if enough data is available."""
        values = np.array(data.get(size=period))
        if len(values) < period:
            self.log(f"Insufficient data: {len(values)} / {period}")
            return None
        return values

    def handle_talib_output(self, output, default=0):
        """Extract last valid value from TA-Lib output."""
        if output is None or np.all(np.isnan(output)):
            return default
        return output[~np.isnan(output)][-1]

    def apply_talib(self, talib_function, *args):
        """Run TA-Lib function and clean the result."""
        output = talib_function(*args)
        return self.handle_talib_output(output)

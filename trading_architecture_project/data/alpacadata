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


class AlpacaData(bt.feeds.PandasData):
    """
    Custom Backtrader data feed for Alpaca-formatted OHLCV data.
    Ensures proper column mapping and datetime index compliance.
    """

    params = (
        ('datetime', None),   # Use index as datetime
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', -1)  # Required by Backtrader
    )

    def __init__(self):
        """Check that input data has a DatetimeIndex."""
        if not isinstance(self.dataname.index, pd.DatetimeIndex):
            raise ValueError("Input data must have a DatetimeIndex.")

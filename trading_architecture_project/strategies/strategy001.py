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

class Strategy001(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ema20 = self.get_indicator('ema20')
        self.ema50 = self.get_indicator('ema50')
        self.ema100 = self.get_indicator('ema100')
        self.rsi = self.get_indicator('rsi14')
        self.ha_close = bt.indicators.HeikinAshi(self.data).ha_close

    def check_signal(self):
        # If any required indicator is missing, do not proceed
        if any(ind is None for ind in [self.ema20, self.ema50, self.ema100, self.rsi]):
            return None

        # Buy condition: short-term EMA above mid-term EMA, Heikin Ashi close above short-term EMA, RSI above 50
        if self.ema20[0] > self.ema50[0] and self.ha_close[0] > self.ema20[0] and self.rsi[0] > 50:
            self.log("Buy signal detected")
            return "BUY"

        # Sell condition: mid-term EMA above long-term EMA, Heikin Ashi close below short-term EMA, RSI below 50
        if self.ema50[0] > self.ema100[0] and self.ha_close[0] < self.ema20[0] and self.rsi[0] < 50:
            self.log("Sell signal detected")
            return "SELL"

        # No signal detected
        return None

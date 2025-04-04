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

class Strategy002(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rsi = self.get_indicator('rsi14')
        self.bbands = self.get_indicator('bbands')
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low

        # Temporary native indicators (non-custom but non-critical)
        self.stoch = bt.indicators.Stochastic(self.datas[0], period=14, period_dfast=3)
        self.sar = bt.indicators.ParabolicSAR(self.datas[0])

    def is_hammer(self):
        open_ = self.data_open[0]
        close = self.data_close[0]
        high = self.data_high[0]
        low = self.data_low[0]
        body = abs(close - open_)
        lower_shadow = min(open_, close) - low
        upper_shadow = high - max(open_, close)

        return (
            body < (high - low) * 0.3 and
            lower_shadow > body * 2 and
            upper_shadow < body
        )

    def fisher_transform_rsi(self, rsi_val):
        if rsi_val is None or np.isnan(rsi_val):
            return float('nan')

        x = (rsi_val / 100.0) * 2 - 1
        x = np.clip(x, -0.999, 0.999)  # avoid division by zero
        return 0.5 * np.log((1 + x) / (1 - x))

    def check_signal(self):
        if len(self) < 50 or self.rsi is None or self.bbands is None:
            return None

        rsi_val = self.rsi[0]
        fisher_val = self.fisher_transform_rsi(rsi_val)
        stoch_k = self.stoch.percK[0]
        close = self.data_close[0]
        bb_lower = self.bbands.bb_lower[0]
        sar_val = self.sar[0]
        is_hammer = self.is_hammer()

        # Buy signal
        if (
            rsi_val < 30 and
            stoch_k < 20 and
            close < bb_lower and
            is_hammer
        ):
            self.log("Buy signal detected (RSI < 30, Stoch < 20, below BB, Hammer pattern)")
            return "BUY"

        # Sell signal
        if (
            sar_val > close and
            fisher_val > 0.3
        ):
            self.log("Sell signal detected (SAR > close, Fisher RSI > 0.3)")
            return "SELL"

        return None

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

class Strategy003(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Custom indicators already integrated
        self.rsi = self.get_indicator('rsi14')
        self.ema50 = self.get_indicator('ema50')
        self.ema100 = self.get_indicator('ema100')

        # Additional short EMAs and SMA
        self.ema5 = bt.indicators.ExponentialMovingAverage(self.datas[0], period=5)
        self.ema10 = bt.indicators.ExponentialMovingAverage(self.datas[0], period=10)
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=40)

        # Remaining native indicators
        self.stoch_fast = bt.indicators.StochasticFast(self.datas[0], period=14, period_dfast=3)
        self.sar = bt.indicators.ParabolicSAR(self.datas[0])

        # History to compute MFI manually
        self.mfi_period = 14
        self.typical_prices = []
        self.raw_money_flows = []

    def fisher_transform_rsi(self, rsi_val):
        if rsi_val is None or np.isnan(rsi_val):
            return float('nan')
        x = (rsi_val / 100.0) * 2 - 1
        x = np.clip(x, -0.999, 0.999)
        return 0.5 * np.log((1 + x) / (1 - x))

    def compute_mfi(self):
        tp = (self.datas[0].high[0] + self.datas[0].low[0] + self.datas[0].close[0]) / 3
        mf = tp * self.datas[0].volume[0]

        self.typical_prices.append(tp)
        self.raw_money_flows.append(mf)

        if len(self.typical_prices) > self.mfi_period:
            self.typical_prices.pop(0)
            self.raw_money_flows.pop(0)

        if len(self.typical_prices) < self.mfi_period:
            return None

        positive_flows = sum(
            mf for i, mf in enumerate(self.raw_money_flows[1:], start=1)
            if self.typical_prices[i] > self.typical_prices[i - 1]
        )
        negative_flows = sum(
            mf for i, mf in enumerate(self.raw_money_flows[1:], start=1)
            if self.typical_prices[i] < self.typical_prices[i - 1]
        )

        if negative_flows == 0:
            return 100.0

        money_flow_ratio = positive_flows / negative_flows
        mfi = 100 - (100 / (1 + money_flow_ratio))
        return mfi

    def check_signal(self):
        if len(self) < 50 or self.rsi is None or self.ema50 is None or self.ema100 is None:
            return None

        rsi_val = self.rsi[0]
        fisher_val = self.fisher_transform_rsi(rsi_val)
        mfi_val = self.compute_mfi()
        close = self.datas[0].close[0]
        sma_val = self.sma[0]
        ema5_val = self.ema5[0]
        ema10_val = self.ema10[0]
        ema50_val = self.ema50[0]
        ema100_val = self.ema100[0]
        stoch_k = self.stoch_fast.percK[0]
        stoch_d = self.stoch_fast.percD[0]
        sar_val = self.sar[0]

        # Buy signal
        if (
            rsi_val > 0 and rsi_val < 28 and
            close < sma_val and
            fisher_val < -0.94 and
            mfi_val is not None and mfi_val < 16.0 and
            ((ema50_val > ema100_val) or (ema5_val > ema10_val)) and
            stoch_d > stoch_k and stoch_d > 0
        ):
            self.log("Buy signal detected (with manual MFI)")
            return "BUY"

        # Sell signal
        if sar_val > close and fisher_val > 0.3:
            self.log("Sell signal detected (SAR > close, Fisher RSI > 0.3)")
            return "SELL"

        return None

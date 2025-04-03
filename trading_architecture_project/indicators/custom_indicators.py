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

class CustomSMA(BaseIndicator):
    """Simple Moving Average (SMA)."""
    lines = ('sma',)
    params = (('period', 20), ('debug', False))

    def __init__(self):
        super().__init__()
        self.addminperiod(self.params.period)

    def next(self):
        if self.last_index == self.data.datetime[0]:
            return

        self.last_index = self.data.datetime[0]
        close_data = self.get_valid_data(self.data.close, self.params.period)
        if close_data is None:
            return

        self.lines.sma[0] = self.apply_talib(talib.SMA, close_data, self.params.period)
        self.log(f"SMA: {self.lines.sma[0]}")

class CustomEMA(BaseIndicator):
    """Exponential Moving Average (EMA)."""
    lines = ('ema',)
    params = (('period', 20), ('debug', False))

    def __init__(self):
        super().__init__()
        self.addminperiod(self.params.period)

    def next(self):
        if self.last_index == self.data.datetime[0]:
            return

        self.last_index = self.data.datetime[0]
        close_data = self.get_valid_data(self.data.close, self.params.period)
        if close_data is None:
            return

        self.lines.ema[0] = self.apply_talib(talib.EMA, close_data, self.params.period)
        self.log(f"EMA: {self.lines.ema[0]}")


class CustomRSI(BaseIndicator):
    """Relative Strength Index (RSI)."""
    lines = ('rsi',)
    params = (('period', 14), ('debug', False))

    def __init__(self):
        super().__init__()
        self.addminperiod(self.params.period + 1)

    def next(self):
        close_prices = np.array(self.data.close.get(size=self.params.period + 1), dtype=np.float64)
        if len(close_prices) < self.params.period + 1:
            self.lines.rsi[0] = float('nan')
            return

        self.lines.rsi[0] = talib.RSI(close_prices, timeperiod=self.params.period)[-1]
        self.log(f"RSI: {self.lines.rsi[0]}")


class CustomVWAP(BaseIndicator):
    """Volume Weighted Average Price (VWAP)."""
    lines = ('vwap',)
    params = (('window', 20), ('debug', False))

    def __init__(self):
        super().__init__()
        self.addminperiod(self.params.window)

    def next(self):
        if len(self.data) < self.params.window:
            return

        prices = np.array(self.data.close.get(size=self.params.window), dtype=np.float64)
        volumes = np.array(self.data.volume.get(size=self.params.window), dtype=np.float64)

        if len(prices) < self.params.window or np.sum(volumes) == 0:
            return

        self.lines.vwap[0] = np.sum(prices * volumes) / np.sum(volumes)
        self.log(f"VWAP: {self.lines.vwap[0]}")

class CustomMACD(BaseIndicator):
    """MACD (Moving Average Convergence Divergence)."""
    lines = ('macd', 'macd_signal', 'macd_hist')
    params = (('fast', 12), ('slow', 26), ('signal', 9), ('debug', False))

    def __init__(self):
        super().__init__()
        self.addminperiod(self.params.slow + 10)

    def next(self):
        if self.last_index == self.data.datetime[0]:
            return
        self.last_index = self.data.datetime[0]

        close_data = self.get_valid_data(self.data.close, self.params.slow + 10)
        if close_data is None or len(close_data) < self.params.slow:
            return

        close_data = np.asarray(close_data, dtype=np.float64)
        macd, macd_signal, macd_hist = talib.MACD(
            close_data,
            fastperiod=self.params.fast,
            slowperiod=self.params.slow,
            signalperiod=self.params.signal
        )

        if np.isnan(macd[-1]) or np.isnan(macd_signal[-1]) or np.isnan(macd_hist[-1]):
            return

        self.lines.macd[0] = self.handle_talib_output(macd[-1])
        self.lines.macd_signal[0] = self.handle_talib_output(macd_signal[-1])
        self.lines.macd_hist[0] = self.handle_talib_output(macd_hist[-1])


class CustomBollingerBands(BaseIndicator):
    """Bollinger Bands."""
    lines = ('bb_upper', 'bb_middle', 'bb_lower')
    params = (('period', 20), ('nbdev', 2), ('debug', False))

    def __init__(self):
        super().__init__()
        self.addminperiod(self.params.period)

    def next(self):
        close_data = self.get_valid_data(self.data.close, self.params.period)
        if close_data is None:
            return

        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close_data,
            timeperiod=self.params.period,
            nbdevup=self.params.nbdev,
            nbdevdn=self.params.nbdev
        )

        self.lines.bb_upper[0] = self.handle_talib_output(bb_upper)
        self.lines.bb_middle[0] = self.handle_talib_output(bb_middle)
        self.lines.bb_lower[0] = self.handle_talib_output(bb_lower)


class CustomIchimoku(BaseIndicator):
    """Ichimoku Kinko Hyo."""
    lines = ('tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span')
    params = (('tenkan', 9), ('kijun', 26), ('senkou', 52), ('debug', False))

    def __init__(self):
        super().__init__()
        self.addminperiod(self.params.senkou)

    def next(self):
        high_prices = self.get_valid_data(self.data.high, self.params.senkou)
        low_prices = self.get_valid_data(self.data.low, self.params.senkou)
        if high_prices is None or low_prices is None:
            return

        self.lines.tenkan_sen[0] = (np.max(high_prices[-self.params.tenkan:]) + np.min(low_prices[-self.params.tenkan:])) / 2
        self.lines.kijun_sen[0] = (np.max(high_prices[-self.params.kijun:]) + np.min(low_prices[-self.params.kijun:])) / 2
        self.lines.senkou_span_a[0] = (self.lines.tenkan_sen[0] + self.lines.kijun_sen[0]) / 2
        self.lines.senkou_span_b[0] = (np.max(high_prices[-self.params.senkou:]) + np.min(low_prices[-self.params.senkou:])) / 2
        self.lines.chikou_span[0] = self.data.close[-self.params.kijun]

class CustomADX(BaseIndicator):
    """Average Directional Index (ADX)."""
    lines = ('adx',)
    params = (('period', 14), ('debug', False))

    def __init__(self):
        self.addminperiod(self.params.period * 2)
        self.last_index = None

    def log(self, message):
        if self.params.debug:
            print(f"[DEBUG] CustomADX: {message}")

    def next(self):
        if self.last_index == self.data.datetime[0]:
            return
        self.last_index = self.data.datetime[0]

        high = np.array(self.data.high.get(size=self.params.period * 2), dtype=np.float64)
        low = np.array(self.data.low.get(size=self.params.period * 2), dtype=np.float64)
        close = np.array(self.data.close.get(size=self.params.period * 2), dtype=np.float64)

        if len(high) < self.params.period:
            return

        adx = talib.ADX(high, low, close, timeperiod=self.params.period)
        if not np.isnan(adx[-1]):
            self.lines.adx[0] = adx[-1]


class CustomCCI(BaseIndicator):
    """Commodity Channel Index (CCI)."""
    lines = ('cci',)
    params = (('period', 14), ('debug', False))

    def __init__(self):
        super().__init__()
        self.addminperiod(self.params.period)

    def next(self):
        if self.last_index == self.data.datetime[0]:
            return
        self.last_index = self.data.datetime[0]

        high = self.get_valid_data(self.data.high, self.params.period)
        low = self.get_valid_data(self.data.low, self.params.period)
        close = self.get_valid_data(self.data.close, self.params.period)

        if high is None or low is None or close is None:
            return

        self.lines.cci[0] = self.apply_talib(talib.CCI, high, low, close, self.params.period)
        self.log(f"CCI: {self.lines.cci[0]}")


class CustomWilliamsR(BaseIndicator):
    """Williams %R."""
    lines = ('williams_r',)
    params = (('period', 14), ('debug', False))

    def __init__(self):
        super().__init__()
        self.addminperiod(self.params.period)

    def next(self):
        if self.last_index == self.data.datetime[0]:
            return
        self.last_index = self.data.datetime[0]

        high = self.get_valid_data(self.data.high, self.params.period)
        low = self.get_valid_data(self.data.low, self.params.period)
        close = self.get_valid_data(self.data.close, self.params.period)

        if high is None or low is None or close is None:
            return

        self.lines.williams_r[0] = self.apply_talib(talib.WILLR, high, low, close, self.params.period)
        self.log(f"Williams %R: {self.lines.williams_r[0]}")


class CustomVolumeIndicator(bt.Indicator):
    """Composite Volume-Based Indicator (RVOL, VWAP, OBV)."""
    lines = ('rvol', 'vwap', 'obv', 'volume')
    params = (('period', 20), ('debug', False))

    def __init__(self):
        self.addminperiod(self.params.period)
        self.last_index = 0

    def next(self):
        self.last_index = len(self) - 1
        self.lines.volume[0] = self.data.volume[0]

        if self.last_index == self.data.datetime[0]:
            return
        self.last_index = self.data.datetime[0]

        volume = np.array(self.data.volume.get(size=self.params.period), dtype=np.float64)
        close = np.array(self.data.close.get(size=self.params.period), dtype=np.float64)
        high = np.array(self.data.high.get(size=self.params.period), dtype=np.float64)
        low = np.array(self.data.low.get(size=self.params.period), dtype=np.float64)

        if len(volume) < self.params.period:
            return

        avg_volume = np.mean(volume)
        self.lines.rvol[0] = volume[-1] / avg_volume if avg_volume > 0 else 1
        self.lines.vwap[0] = np.sum((high + low + close) / 3 * volume) / np.sum(volume)
        self.lines.obv[0] = talib.OBV(close, volume)[-1]

        if self.params.debug:
            print(f"[DEBUG] RVOL: {self.lines.rvol[0]:.2f} | VWAP: {self.lines.vwap[0]:.2f} | OBV: {self.lines.obv[0]:.2f}")

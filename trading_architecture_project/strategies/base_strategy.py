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

class BaseStrategy(bt.Strategy):
    """
    Modular base strategy that integrates risk, order, position,
    and monitoring logic. Designed to be extended with specific signals.
    """

    params = (
        ('risk_per_trade', 0.02),
        ('min_volume', 100000),
        ('cooldown_days', 1),
        ('indicators', ['sma20', 'ema20', 'ema50', 'ema100', 'rsi14', 'macd', 'bbands']),
        ('max_drawdown_pct', 15),
        ('verbosity', 'normal'),
    )

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.trade_monitor = kwargs.get("trade_monitor", None)
        if self.trade_monitor is None:
            raise ValueError("TradeMonitor is required for tracking trades.")

        self.logger = logging.getLogger("TradingLogger")
        if not self.logger.handlers:
            handler = logging.FileHandler("trading_log.txt", mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.order_manager = OrderManager(self, trade_monitor=self.trade_monitor)
        self.position_manager = PositionManager(self)
        self.risk_manager = RiskManager(self, max_drawdown_pct=self.params.max_drawdown_pct)
        self.last_trade_date = None
        self.indicators = {}
        self.load_indicators()

        if len(self.datas) == 0:
            raise ValueError("No data feed loaded.")
        self.addminperiod(50)

    def log(self, txt, dt=None, log_to_console=True, level="info"):
        if self.params.verbosity == 'silent' and level != 'error':
            return
        dt = dt or self.datas[0].datetime.date(0)
        message = f'{dt.isoformat()}, {txt}'
        if log_to_console:
            print(message)
        self.logger.info(message)

    def load_indicators(self):
        available = {
            'sma20': CustomSMA(self.datas[0], period=20),
            'ema20': CustomEMA(self.datas[0], period=20),
            'ema50': CustomEMA(self.datas[0], period=50),
            'ema100': CustomEMA(self.datas[0], period=100),
            'rsi14': CustomRSI(self.datas[0], period=14),
            'macd': CustomMACD(self.datas[0]),
            'bbands': CustomBollingerBands(self.datas[0]),
            'atr': bt.ind.ATR(self.datas[0], period=14),
            'vwap': CustomVWAP(self.datas[0])
        }
        for name in self.params.indicators:
            if name in available:
                self.indicators[name] = available[name]
                self.log(f"Indicator loaded: {name}")

    def get_indicator(self, name):
        """Returns an indicator if it has been loaded."""
        return self.indicators.get(name, None)

    def notify_order(self, order):
        self.order_manager.notify_order(order)

    def notify_trade(self, trade):
        if trade.isclosed:
            entry_date = trade.open_datetime().date()
            exit_date = trade.close_datetime().date()
            size = abs(trade.size)
            direction = "long" if trade.size > 0 else "short"
            entry_price = trade.price
            exit_price = entry_price + trade.pnl / size if size != 0 else 0
            commission = abs(trade.commission)
            pnl = trade.pnl
            pnl_net = trade.pnlcomm
            risk = self.risk_manager.trade_risk_history[-1] if self.risk_manager.trade_risk_history else None

            self.trade_monitor.record_trade(
                entry_date, exit_date, entry_price, exit_price,
                size, direction, pnl, pnl_net, commission, risk_at_entry=risk
            )

            self.log(f'Trade closed | Net PnL: {pnl_net:.2f}')
            self.position_manager.update_on_exit()

    def should_trade(self):
        if self.datas[0].volume[0] < self.params.min_volume:
            self.log("Volume too low for trading.")
            return False
        return True

    def validate_entry_conditions(self):
        if not self.should_trade():
            return False
        if self.position_manager.is_position_open():
            self.log("Position already open.")
            return False
        if self.last_trade_date and (self.datas[0].datetime.date(0) - self.last_trade_date).days < self.params.cooldown_days:
            self.log("Cooldown active.")
            return False
        if self.risk_manager.is_locked() or self.risk_manager.is_below_min_cash():
            return False
        if self.risk_manager.stop_strategy_if_drawdown_exceeds():
            return False
        return True

    def execute_trade(self, signal):
        price = self.data.close[0]
        atr = self.indicators.get('atr', None)
        stop_loss_distance = atr[0] * 2 if atr else price * 0.02

        size = self.risk_manager.calculate_position_size(
            price=price,
            stop_loss_distance=stop_loss_distance,
            risk_per_trade=self.params.risk_per_trade
        )

        if size <= 0:
            self.log("Invalid position size.")
            return

        if signal == "BUY":
            sl = price - stop_loss_distance
            tp = price + stop_loss_distance * 2
        elif signal == "SELL":
            sl = price + stop_loss_distance
            tp = price - stop_loss_distance * 2
        else:
            return

        self.position_manager.mark_pending_entry(signal.lower())
        self.order_manager.place_entry(signal.lower(), size=size, sl=sl, tp=tp)

    def next(self):
        self.order_manager.expire_old_orders()
        self.order_manager.check_exit_timeout()
        self.position_manager.sync_with_broker()

        signal = self.check_signal() if hasattr(self, "check_signal") else None
        if not signal:
            return

        if self.validate_entry_conditions():
            self.execute_trade(signal)

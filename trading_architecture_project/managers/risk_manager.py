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

class RiskManager:
    """
    Manages risk exposure, drawdowns, and capital protection logic.
    Can halt trading when thresholds are exceeded.
    """

    def __init__(self, strategy, initial_cash=None, max_drawdown_pct=20.0, min_cash_threshold=0):
        self.strategy = strategy
        self.initial_cash = initial_cash or strategy.broker.get_cash()
        self.max_drawdown_pct = max_drawdown_pct
        self.min_cash_threshold = min_cash_threshold
        self.locked_until = None
        self.locked_reason = ""
        self.trade_risk_history = []

    def calculate_position_size(self, price, stop_loss_distance, risk_per_trade=0.02):
        """Computes optimal position size given risk parameters."""
        cash = self.strategy.broker.get_cash()
        capital_at_risk = cash * risk_per_trade

        if stop_loss_distance <= 0:
            self.strategy.log("Invalid stop-loss distance.")
            return 0

        raw_size = capital_at_risk / stop_loss_distance
        max_size = int(cash / price)
        size = int(min(raw_size, max_size))

        if size < 1:
            self.strategy.log("Position size below minimum (1).")
            return 0

        self.register_risk_taken(capital_at_risk)
        self.strategy.log(f"Position: Size={size}, Risk={risk_per_trade*100:.1f}%, SL={stop_loss_distance:.2f}, Cash={cash:.2f}")
        return size

    def register_risk_taken(self, amount):
        """Records the monetary risk taken for a trade."""
        self.trade_risk_history.append(amount)

    def avg_risk_per_trade(self):
        """Returns average monetary risk taken per trade."""
        if not self.trade_risk_history:
            return 0
        return round(sum(self.trade_risk_history) / len(self.trade_risk_history), 2)

    def check_current_drawdown(self):
        """Calculates current drawdown percentage."""
        current_cash = self.strategy.broker.get_value()
        drawdown = 100 * (1 - (current_cash / self.initial_cash))
        return round(drawdown, 2)

    def log_current_drawdown(self):
        """Logs the current drawdown value."""
        dd = self.check_current_drawdown()
        self.strategy.log(f"Current drawdown: {dd:.2f}%")

    def stop_strategy_if_drawdown_exceeds(self, lock_days=3):
        """
        Stops the strategy if drawdown exceeds the allowed maximum.
        Locks further trading for a cooldown period.
        """
        dd = self.check_current_drawdown()
        if dd >= self.max_drawdown_pct:
            self.locked_until = datetime.now() + timedelta(days=lock_days)
            self.locked_reason = (
                f"Drawdown {dd:.2f}% exceeds limit ({self.max_drawdown_pct}%). "
                f"Trading locked for {lock_days} days."
            )
            self.strategy.log(self.locked_reason)
            return True
        return False

    def is_locked(self):
        """Returns True if the strategy is temporarily blocked due to a drawdown lock."""
        if self.locked_until and datetime.now() < self.locked_until:
            self.strategy.log(f"Strategy locked until {self.locked_until.date()} | Reason: {self.locked_reason}")
            return True
        return False

    def is_below_min_cash(self):
        """Returns True if current capital is below the minimum allowed threshold."""
        capital = self.strategy.broker.get_value()
        if self.min_cash_threshold > 0 and capital < self.min_cash_threshold:
            self.strategy.log(f"Capital {capital:.2f} below minimum allowed ({self.min_cash_threshold:.2f})")
            return True
        return False

    def get_drawdown_stats(self):
        """Returns a summary of drawdown and capital protection status."""
        return {
            'initial_cash': round(self.initial_cash, 2),
            'current_cash': round(self.strategy.broker.get_value(), 2),
            'drawdown_pct': self.check_current_drawdown(),
            'max_drawdown_pct': self.max_drawdown_pct,
            'locked_until': self.locked_until,
            'locked_reason': self.locked_reason
        }

    def reset(self, new_initial_cash=None):
        """Resets the initial capital reference point."""
        self.initial_cash = new_initial_cash or self.strategy.broker.get_value()
        self.strategy.log(f"New capital reference set to {self.initial_cash:.2f}")

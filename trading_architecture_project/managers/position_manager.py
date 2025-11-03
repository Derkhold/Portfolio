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

class PositionManager:
    """
    Tracks position state, direction, duration, and entry metadata.
    Synchronizes with Backtrader's internal position object.
    """

    def __init__(self, strategy):
        self.strategy = strategy
        self.position_open_date = None
        self.position_size = 0
        self.position_direction = None
        self.position_status = "flat"  # Options: flat, long, short, pending
        self.entry_details = {}

    def update_on_entry(self, sl=None, tp=None):
        """Records new position and metadata."""
        self.position_open_date = self.strategy.data.datetime.datetime(0)
        self.position_size = abs(self.strategy.position.size)
        self.position_direction = (
            "long" if self.strategy.position.size > 0
            else "short" if self.strategy.position.size < 0
            else None
        )
        self.position_status = self.position_direction or "flat"
        self.entry_details = {
            'entry_price': self.strategy.data.close[0],
            'sl': sl,
            'tp': tp,
            'atr': self.strategy.indicators['atr'][0] if 'atr' in self.strategy.indicators else None,
            'opened_at': self.position_open_date
        }
        self.strategy.log(f"Position opened | Direction: {self.position_direction}, Size: {self.position_size}, Price: {self.entry_details['entry_price']:.2f}")

    def update_on_exit(self):
        """Resets all position-related state on exit."""
        if self.is_position_open():
            duration = self.get_position_duration()
            self.strategy.log(f"Position closed after {duration} days.")
        else:
            self.strategy.log("Position closed.")
        self.position_open_date = None
        self.position_size = 0
        self.position_direction = None
        self.position_status = "flat"
        self.entry_details = {}

    def is_position_open(self):
        """Returns True if a position is currently held."""
        return self.strategy.position and self.strategy.position.size != 0

    def get_position_size(self):
        return abs(self.strategy.position.size)

    def get_position_direction(self):
        return self.position_direction

    def get_position_status(self):
        return self.position_status

    def get_position_duration(self):
        if self.position_open_date is None:
            return 0
        now = self.strategy.data.datetime.datetime(0)
        return (now - self.position_open_date).days

    def get_entry_details(self):
        return self.entry_details or {}

    def mark_pending_entry(self, direction):
        """Called when an entry order is submitted but not yet filled."""
        self.position_status = "pending"
        self.position_direction = direction
        self.strategy.log(f"Pending entry detected: {direction.upper()}")

    def cancel_pending_entry(self):
        """Called when a pending entry order is cancelled."""
        self.position_status = "flat"
        self.position_direction = None
        self.strategy.log("Pending entry cancelled. Back to FLAT.")

    def close_position(self, force_close_after_days=None):
        """Closes the position immediately or after a defined time."""
        if not self.is_position_open():
            return
        if force_close_after_days is not None:
            duration = self.get_position_duration()
            if duration < force_close_after_days:
                return
        self.strategy.log("Forced position close triggered.")
        self.strategy.close()

    def auto_check(self, max_days=5):
        """Checks and closes positions that exceed the max duration."""
        self.close_position(force_close_after_days=max_days)

    def sync_with_broker(self):
        """
        Synchronizes internal state with Backtrader's position object.
        Should be called on each 'next()' to detect position transitions.
        """
        size = self.strategy.position.size
        if size == 0 and self.position_size != 0:
            self.update_on_exit()
        elif size != 0 and self.position_size == 0:
            self.update_on_entry()

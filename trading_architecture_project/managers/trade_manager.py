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

class TradeMonitor:
    """
    Monitors and records closed trades and order events.
    Provides statistics and export functions for reporting.
    """

    def __init__(self):
        self.trades = []          # List of closed trades
        self.order_events = []    # List of all order events (submitted, executed, cancelled)

    def record_trade(self, entry_date, exit_date, entry_price, exit_price,
                     size, direction, pnl, pnl_net, commission, risk_at_entry=None):
        """Stores information about a completed trade."""
        duration = (exit_date - entry_date).days
        self.trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "direction": direction,
            "pnl": pnl,
            "pnl_net": pnl_net,
            "commission": commission,
            "duration_days": duration,
            "risk_at_entry": risk_at_entry,
            "return_risk_ratio": (pnl_net / risk_at_entry) if risk_at_entry else None
        })

    def record_order_event(self, order_dict):
        """Records any order event (submitted, cancelled, executed)."""
        self.order_events.append(order_dict)

    def get_statistics(self):
        """Returns a summary dictionary of trade performance metrics."""
        if not self.trades:
            return {}

        df = pd.DataFrame(self.trades)

        stats = {
            "total_trades": len(df),
            "win_rate": round(100 * (df['pnl_net'] > 0).mean(), 2),
            "avg_pnl": round(df['pnl'].mean(), 2),
            "avg_pnl_net": round(df['pnl_net'].mean(), 2),
            "avg_commission": round(df['commission'].mean(), 4),
            "avg_duration": round(df['duration_days'].mean(), 2),
            "best_trade": df['pnl_net'].max(),
            "worst_trade": df['pnl_net'].min(),
            "avg_r_multiple": round(df['return_risk_ratio'].mean(), 2) if df['return_risk_ratio'].notna().any() else None
        }

        return stats

    def export_order_history(self, filename="order_history.csv"):
        """Exports all recorded order events to a CSV file."""
        if not self.order_events:
            print("No order events to export.")
            return
        df = pd.DataFrame(self.order_events)
        df.to_csv(filename, index=False)
        print(f"Order history exported to {filename}")

    def export_history(self, filename="trade_history.csv"):
        """Exports all closed trades to a CSV file."""
        if not self.trades:
            print("No trades to export.")
            return
        df = pd.DataFrame(self.trades)
        df.to_csv(filename, index=False)
        print(f"Trade history exported to {filename}")

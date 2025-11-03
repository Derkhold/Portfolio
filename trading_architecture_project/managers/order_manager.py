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

class OrderManager:
    """
    Manages order submission, execution, cancellation, SL/TP attachment, and logging.
    Works in coordination with the Backtrader strategy.
    """

    def __init__(self, strategy, trade_monitor=None, order_expiry_minutes=30, sltp_timeout_days=3):
        self.strategy = strategy
        self.trade_monitor = trade_monitor
        self.active_orders = {}
        self.pending_exit = {}
        self.current_order = None
        self.order_expiry_minutes = order_expiry_minutes
        self.sltp_timeout_days = sltp_timeout_days
        self.last_entry_execution_time = None
        self.order_log = []

    def place_entry(self, order_type, size, price=None, sl=None, tp=None):
        """Submits an entry order with optional SL/TP."""
        if self.current_order or size <= 0:
            self.strategy.log("Order blocked: active or invalid size.")
            return

        if order_type.lower() == "buy":
            self.current_order = self.strategy.buy(size=size)
        elif order_type.lower() == "sell":
            self.current_order = self.strategy.sell(size=size)
        else:
            raise ValueError("Invalid order_type: must be 'buy' or 'sell'.")

        self.pending_exit[self.current_order.ref] = {'type': order_type, 'sl': sl, 'tp': tp}
        self.track_order(self.current_order, kind='entry')
        self.strategy.log(f"{order_type.upper()} order submitted | SL: {sl}, TP: {tp}, SIZE: {size}")

    def track_order(self, order, kind="entry"):
        """Adds order to active registry and logs submission."""
        created_time = self.strategy.data.datetime.datetime(0)
        self.active_orders[order.ref] = {
            'type': kind,
            'status': 'submitted',
            'created': created_time,
            'bt_order': order
        }
        self.log_order_event(order, kind, status="submitted", created=created_time)

    def notify_order(self, order):
        """Updates order status on execution or rejection, and attaches SL/TP."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        order_ref = order.ref
        kind = self.active_orders.get(order_ref, {}).get('type', 'unknown')
        exec_time = self.strategy.data.datetime.datetime(0)

        if order.status == order.Completed:
            self.log_order_event(order, kind, status="executed", executed=exec_time)
            if kind == "entry":
                self.last_entry_execution_time = exec_time

            if order_ref in self.pending_exit:
                sl = self.pending_exit[order_ref]['sl']
                tp = self.pending_exit[order_ref]['tp']
                parent_order = order

                if order.isbuy():
                    sl_order = self.strategy.sell(exectype=bt.Order.Stop, price=sl, parent=parent_order)
                    tp_order = self.strategy.sell(exectype=bt.Order.Limit, price=tp, parent=parent_order)
                else:
                    sl_order = self.strategy.buy(exectype=bt.Order.Stop, price=sl, parent=parent_order)
                    tp_order = self.strategy.buy(exectype=bt.Order.Limit, price=tp, parent=parent_order)

                self.track_order(sl_order, kind="sl")
                self.track_order(tp_order, kind="tp")
                self.pending_exit.pop(order_ref)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log_order_event(order, kind, status="cancelled", executed=exec_time)

        self.active_orders.pop(order_ref, None)
        if self.current_order and self.current_order.ref == order_ref:
            self.current_order = None

    def cancel_all_open_orders(self):
        """Cancels all submitted/accepted orders and clears tracking."""
        for order_ref, info in list(self.active_orders.items()):
            order = info.get('bt_order')
            if order and order.status in [order.Submitted, order.Accepted]:
                self.strategy.cancel(order)
                self.log_order_event(order, info['type'], status="cancelled (force)")

            self.active_orders.pop(order_ref, None)

        self.pending_exit.clear()
        self.current_order = None

    def expire_old_orders(self):
        """Cancels orders that have not been executed within the expiry window."""
        now = self.strategy.data.datetime.datetime(0)
        expired_refs = []

        for order_ref, info in self.active_orders.items():
            created = info.get('created')
            if info.get('status') == 'submitted' and (now - created) > timedelta(minutes=self.order_expiry_minutes):
                order = info.get('bt_order')
                if order.status in [order.Submitted, order.Accepted]:
                    self.strategy.cancel(order)
                    self.log_order_event(order, info['type'], status="expired")
                    expired_refs.append(order_ref)

        for ref in expired_refs:
            self.active_orders.pop(ref, None)

    def check_exit_timeout(self):
        """Forcefully closes a position if SL/TP have not triggered after timeout."""
        if not self.last_entry_execution_time:
            return

        now = self.strategy.data.datetime.datetime(0)
        duration = (now - self.last_entry_execution_time).days

        if duration >= self.sltp_timeout_days and self.strategy.position:
            self.strategy.log(f"SL/TP timeout reached after {duration} days. Closing position.")
            self.strategy.close()
            self.last_entry_execution_time = None

    def log_order_event(self, order, kind, status, created=None, executed=None):
        """Stores order event details in log and optionally forwards to TradeMonitor."""
        log_entry = {
            "datetime": self.strategy.data.datetime.datetime(0),
            "ref": order.ref,
            "type": kind,
            "status": status,
            "size": order.size,
            "price": order.created.price if order.created else None,
            "created_at": created,
            "executed_at": executed
        }
        self.order_log.append(log_entry)

        if self.trade_monitor:
            self.trade_monitor.record_order_event(log_entry)

    def get_order_history(self):
        """Returns the full internal order log."""
        return self.order_log

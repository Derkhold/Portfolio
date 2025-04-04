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

class BacktestEngine:
    def __init__(self, cash=10000, commission=0.001,
                 risk_per_trade=0.02, max_drawdown_pct=20, cooldown_days=1):
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.set_cash(cash)
        self.cerebro.broker.setcommission(commission=commission)

        self.risk_per_trade = risk_per_trade
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_days = cooldown_days

        self.trade_monitor = TradeMonitor()

    def add_strategy(self, strategy):
        """Attach the strategy to the engine with proper parameters."""
        self.cerebro.addstrategy(
            strategy,
            risk_per_trade=self.risk_per_trade,
            max_drawdown_pct=self.max_drawdown_pct,
            cooldown_days=self.cooldown_days,
            trade_monitor=self.trade_monitor
        )

    def load_dataframe(self, df):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().tz_localize(None)
        df = df.astype({col: 'float64' for col in ['open', 'high', 'low', 'close', 'volume']})
        data = bt.feeds.PandasData(dataname=df)
        self.cerebro.adddata(data)

    def add_analyzers(self):
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    
    def run(self):
        logging.info("Starting backtest...")
        initial_cash = self.cerebro.broker.getvalue()
        self.add_analyzers()
        results = self.cerebro.run()
        final_cash = self.cerebro.broker.getvalue()
        performance = ((final_cash - initial_cash) / initial_cash) * 100

        self.display_results(results, initial_cash, final_cash, performance)
        self.generate_pdf_report(results, initial_cash, final_cash, performance)

        self.trade_monitor.export_order_history("order_history.csv")
        self.trade_monitor.export_history("trade_history.csv")

        self.cerebro.plot()

    def display_results(self, results, initial_cash, final_cash, performance):
        strat = results[0]
        print("\nBacktest Summary")
        print(f"Initial Capital: {initial_cash:.2f} EUR")
        print(f"Final Capital: {final_cash:.2f} EUR")
        print(f"Performance: {performance:.2f}%")

        drawdown = strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
        trades = strat.analyzers.trades.get_analysis()

        print(f"Max Drawdown: {drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}" if sharpe else "Sharpe Ratio not available")

        total = trades.get('total', {}).get('total', 0)
        won = trades.get('won', {}).get('total', 0)
        lost = trades.get('lost', {}).get('total', 0)

        print(f"Total Trades: {total} | Winners: {won} | Losers: {lost}")

        self.save_results(results, initial_cash, final_cash, performance)

    def generate_pdf_report(self, results, initial_cash, final_cash, performance, filename="backtest_report.pdf"):
        strat = results[0]
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Backtest Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, f"Initial Capital: {initial_cash:.2f} EUR", ln=True)
        pdf.cell(200, 10, f"Final Capital: {final_cash:.2f} EUR", ln=True)
        pdf.cell(200, 10, f"Performance: {performance:.2f}%", ln=True)
        pdf.ln(10)

        drawdown = strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
        trades = strat.analyzers.trades.get_analysis()

        pdf.cell(200, 10, f"Max Drawdown: {drawdown:.2f}%", ln=True)
        pdf.cell(200, 10, f"Sharpe Ratio: {sharpe:.2f}" if sharpe else "Sharpe Ratio not available", ln=True)

        total = trades.get('total', {}).get('total', 0)
        won = trades.get('won', {}).get('total', 0)
        lost = trades.get('lost', {}).get('total', 0)

        pdf.cell(200, 10, f"Trades: {total} | Winners: {won} | Losers: {lost}", ln=True)

        # Performance chart
        plt.savefig("backtest_plot.png", dpi=150)
        pdf.image("backtest_plot.png", x=10, w=190)
        pdf.output(filename)
        print(f"PDF report generated: {filename}")

    def save_results(self, results, initial_cash, final_cash, performance, filename="backtest_results.json"):
        strat = results[0]
        trades = strat.analyzers.trades.get_analysis()
        data = {
            "initial_capital": initial_cash,
            "final_capital": final_cash,
            "performance": performance,
            "max_drawdown": strat.analyzers.drawdown.get_analysis()["max"]["drawdown"],
            "sharpe_ratio": strat.analyzers.sharpe.get_analysis().get("sharperatio", None),
            "total_trades": trades.get('total', {}).get('total', 0),
            "winning_trades": trades.get('won', {}).get('total', 0),
            "losing_trades": trades.get('lost', {}).get('total', 0)
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Results saved to {filename}")

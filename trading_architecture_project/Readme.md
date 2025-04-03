# Modular Algorithmic Trading Architecture

This project presents a modular, extensible framework for designing, testing, and analyzing algorithmic trading strategies in Python.

Rather than focusing on a single strategy or dataset, the goal is to build a **flexible system** that can be adapted to different styles, market environments, and research directions. Each component is designed to be independent and reusable, following a clear **separation of concerns** to ensure maintainability, scalability, and transparency.

---

## Project Overview

The architecture is structured into the following core components:

- **Data Collection**  
  Import and preprocessing of historical market data, including local caching and API integration (e.g., Alpaca).

- **Indicators**  
  Computation of technical indicators such as SMA, EMA, RSI, ATR, MACD, and VWAP using TA-Lib and custom implementations.

- **Managers**
  - `OrderManager`: Handles order placement logic
  - `PositionManager`: Tracks and updates position states
  - `RiskManager`: Applies risk constraints, stop-loss logic, and position sizing
  - `TradeMonitor`: Logs trade executions and performance statistics

- **Strategies**  
  Definition of rule-based strategies that plug into the system and use the above components for execution.

- **Backtest Engine**  
  A custom runner built on top of Backtrader to execute strategies under realistic conditions, with full metrics tracking.

- **Results & Reporting**  
  Export of results in CSV, JSON, and PDF formats. Includes equity curves, trade logs, and summary statistics.

- **Conclusion & Extensions**  
  A final assessment of the framework, with a list of open directions for improvement and future work.

---

## Objective

The purpose of this framework is not to optimize a single strategy, but to develop a **reusable research and execution platform** that can:

- Be extended to multiple asset classes and timeframes  
- Connect to live trading APIs (e.g., Alpaca) for real-time execution  
- Support robust risk evaluation and performance diagnostics  
- Adapt to various market regimes (trend-following, mean-reverting, volatile, etc.)  
- Serve as a foundation for future experimentation and model integration

---

## Project Structure

The project is organized into modular components, each fulfilling a specific role in the trading architecture. Below is the full file structure:

```text
trading_architecture_project/
│
├── data/                          # Historical data ingestion and API access
│   └── alpaca_api.py              # Retrieves and formats price data from Alpaca
│
├── indicators/                    # Technical indicator computation
│   ├── base_indicator.py          # Base class ensuring consistency across indicators
│   └── custom_indicators.py       # Implements SMA, EMA, RSI, MACD, ATR, etc.
│
├── managers/                      # Core trading components
│   ├── order_manager.py           # Handles order creation and execution logic
│   ├── position_manager.py        # Maintains and updates position states
│   ├── risk_manager.py            # Applies risk filters, stop-loss, and position sizing
│   └── trade_monitor.py           # Logs trades, PnL, durations, and metrics
│
├── strategies/                    # Custom strategies built on shared logic
│   ├── base_strategy.py           # Base strategy class with signal and execution interface
│   ├── strategy001.py             # Trend-following strategy
│   ├── strategy002.py             # Mean-reversion strategy
│   └── strategy003.py             # Momentum exhaustion strategy
│
├── engine/                        # Backtesting engine logic
│   └── backtest_engine.py         # Wraps full strategy execution using Backtrader
│
├── reports/                       # Exported results and performance files
│   ├── trade_history.csv          # Log of all executed trades
│   ├── order_history.csv          # Log of orders (timestamps, prices, quantities)
│   ├── backtest_results.json      # Summary of key performance metrics
│   └── backtest_report.pdf        # PDF report of backtest performance
│
├── run_backtest.py                # Main script to launch a full backtest session
└── README.md                      # Project overview and module documentation (this file)

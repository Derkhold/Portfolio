# Backtesting Framework

## BacktestEngine: A Custom Wrapper Around Backtrader

The `BacktestEngine` class is a custom-built wrapper around the Backtrader framework. It manages the full backtesting workflow—from strategy deployment to performance reporting—while remaining fully compatible with the modular architecture.

It automates the following processes:

- Capital and commission configuration  
- Strategy integration with consistent risk parameters  
- Data ingestion via Pandas DataFrames  
- Analyzer setup for key performance metrics (drawdown, Sharpe ratio, returns, trade stats)  
- Performance reporting through:
  - Console summaries  
  - PDF reports with embedded performance charts  
  - JSON and CSV exports for traceability and further analysis

This engine is designed to ensure transparency, reproducibility, and formal reporting, making it suitable for academic validation and iterative strategy research.

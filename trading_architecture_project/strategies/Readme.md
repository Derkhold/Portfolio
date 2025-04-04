# Strategies: Modular Strategy Architecture

## BaseStrategy: A Modular Template for Strategy Logic

The `BaseStrategy` class serves as the central template for developing algorithmic trading strategies within the Backtrader framework. It integrates the different components defined previously to provide a structured and reusable environment for implementing and testing trading logic.

The class coordinates:

- `OrderManager`: For submitting orders, managing SL/TP, and handling expirations  
- `PositionManager`: For tracking open positions, direction, and duration  
- `RiskManager`: For enforcing capital protection and dynamic position sizing  
- `TradeMonitor`: For logging trades and generating performance reports  
- Custom indicators: Loaded dynamically based on strategy parameters

This base class is designed to be extended through subclassing. Custom strategies can override the `check_signal()` method to define their own entry and exit rules.

---

### Core Features

- Dynamic loading of selected technical indicators  
- Stop-loss and take-profit logic integrated into order management  
- Trade cooldown control to prevent overtrading  
- Volume filters and risk-based constraints  
- Unified execution logic consistent across all strategies

> This strategy class serves as the backbone of the framework and enables consistent execution across multiple strategies with minimal duplication.

## Custom Strategies

### Overview

This section presents a set of algorithmic trading strategies designed for testing within the modular architecture. Each strategy combines technical indicators and logical conditions tailored to specific market behaviors, such as trending markets, reversals, or momentum shifts.

All strategies are fully compatible with the architecture's risk management, execution, and reporting components, allowing consistent and scalable testing.

Each strategy is designed to address a specific market regime and is evaluated independently.

---

## Strategy Summary

| Strategy     | Style                      | Entry Logic                          | Exit Logic                          |
|--------------|----------------------------|---------------------------------------|--------------------------------------|
| Strategy001  | Trend Following            | EMA alignment, Heikin-Ashi, RSI       | EMA reversal, RSI confirmation       |
| Strategy002  | Oversold Reversal + Pattern | RSI < 30, Stochastic < 20, Hammer     | Parabolic SAR, Fisher RSI            |
| Strategy003  | Momentum Exhaustion        | RSI, MFI, Fisher RSI (low extremes)   | SAR + Fisher RSI confirmation        |

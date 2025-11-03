# Managers: Modular Trading Control

## Overview

The "Manager" classes offer a modular and structured approach to handling the different components of the trading lifecycle. Instead of embedding all logic within a single strategy class, responsibilities are delegated to specialized modules.

Each manager focuses on a specific task:

- **OrderManager**: Handles order submission, tracking, and execution lifecycle.
- **PositionManager**: Manages position state, size, direction, and trade duration.
- **RiskManager**: Enforces trade-level and portfolio-level risk constraints, including drawdown and capital protection.
- **TradeMonitor**: Records trade metrics and order events, and supports export to CSV and JSON for performance evaluation.

This design improves code readability, maintainability, and testing flexibility, while aligning with clean architecture principles.

### OrderManager: Centralized Order Lifecycle Control

The `OrderManager` class is responsible for managing the entire lifecycle of trading orders.

Key responsibilities include:

- Submitting buy and sell orders  
- Attaching stop-loss (SL) and take-profit (TP) levels to parent orders  
- Tracking all active orders through a centralized registry  
- Handling execution callbacks, rejections, and time-based expirations  
- Logging order events and optionally forwarding them to the `TradeMonitor`

This component acts as the main interface between strategy logic and the Backtrader order execution engine. It ensures reliability, consistency, and traceability throughout the trade process.

### PositionManager: Position Tracking and Synchronization

The `PositionManager` is a stateful utility that monitors the current position held by the strategy. It wraps Backtrader’s native position object to provide higher-level attributes such as position status, direction, duration, and entry metadata.

#### Key Responsibilities

- Maintain position status (open/closed), direction (long/short), and duration  
- Record key information at entry, including price, stop-loss, take-profit, and ATR  
- Automatically detect when positions are opened or closed  
- Support forced exits based on duration thresholds  
- Handle the "pending" entry state when an order is submitted but not yet executed

This module enhances clarity by separating position state management from core strategy logic, allowing for better monitoring and control.

### RiskManager: Capital Preservation and Risk Control

The `RiskManager` class provides capital protection mechanisms at the strategy level. Its goal is to maintain discipline and preserve capital during adverse market conditions or underperformance.

#### Core Responsibilities

- Compute position size based on stop-loss distance and a defined risk percentage  
- Monitor drawdowns in relation to the initial capital  
- Halt trading if the drawdown exceeds a predefined threshold  
- Enforce a minimum capital floor to prevent overexposure  
- Record risk committed per trade and track average risk exposure over time

By isolating capital control logic from signal generation, this module ensures that strategic decisions remain focused, while risk exposure remains within acceptable limits.


### TradeMonitor: Trade and Order History Logging

The `TradeMonitor` class serves as the central logging component for both executed trades and order-related events. It functions as the primary audit trail and reporting mechanism of the trading architecture.

#### Responsibilities

- Record all completed trades along with metadata such as PnL, size, duration, and R-multiple  
- Log all order events including submissions, executions, and cancellations  
- Generate summary statistics for overall trading performance  
- Export trade and order histories to CSV for external analysis and review

This component enables transparency, reproducibility, and structured post-trade analysis, making it essential for both backtesting validation and live trading diagnostics.


# Indicators:

## BaseIndicator – Foundation for Custom Technical Indicators

The `BaseIndicator` class provides a reusable and optimized foundation for implementing custom technical indicators within the Backtrader framework. It simplifies the development of trading signals by handling key tasks such as:

- Validating input data for required lookback periods  
- Wrapping TA-Lib functions for indicator computation  
- Managing debug output for easier testing and verification

This class is intended to be inherited by specific indicators (e.g., RSI, MACD, Bollinger Bands) and promotes a clean, modular, and testable architecture for strategy development.

---

### Core Functionalities

- Avoids redundant computations by caching results  
- Validates historical data length before applying calculations  
- Cleans and extracts values from TA-Lib outputs  
- Supports optional debug logging to assist with indicator development

> This class forms the backbone of all TA-Lib-based custom indicators implemented in this framework.


## Custom TA-Lib Indicators for Algorithmic Trading

This section presents a collection of custom technical indicators implemented using the Backtrader and TA-Lib libraries. These indicators are essential components used for generating trading signals within the system.

Each class inherits from the shared `BaseIndicator`, and includes logic for data validation, memory optimization, and optional debugging to ensure both performance and reliability.

---

### Included Indicators

- Simple Moving Average (SMA)  
- Exponential Moving Average (EMA)  
- Relative Strength Index (RSI)  
- Volume-Weighted Average Price (VWAP)  
- Moving Average Convergence Divergence (MACD)  
- Bollinger Bands  
- Ichimoku Kinko Hyo  
- Average Directional Index (ADX)  
- Commodity Channel Index (CCI)  
- Williams %R  
- Advanced Volume Metrics (RVOL, OBV)

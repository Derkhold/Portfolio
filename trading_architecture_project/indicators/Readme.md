# III – Indicators: BaseIndicator Class

## BaseIndicator – Foundation for Custom Technical Indicators

The `BaseIndicator` class provides a reusable and optimized foundation for implementing custom technical indicators within the Backtrader framework. It simplifies the development of trading signals by handling key tasks such as:

- Validating input data for required lookback periods  
- Wrapping TA-Lib functions for indicator computation  
- Managing debug output for easier testing and verification

This class is intended to be inherited by specific indicators (e.g., RSI, MACD, Bollinger Bands) and promotes a clean, modular, and testable architecture for strategy development.

---

## Core Functionalities

- Avoids redundant computations by caching results  
- Validates historical data length before applying calculations  
- Cleans and extracts values from TA-Lib outputs  
- Supports optional debug logging to assist with indicator development

> This class forms the backbone of all TA-Lib-based custom indicators implemented in this framework.

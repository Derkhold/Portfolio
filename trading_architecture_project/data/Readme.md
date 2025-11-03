## AlpacaAPI - Data Collection 

The `AlpacaAPI` class is responsible for fetching historical OHLCV data (Open, High, Low, Close, Volume) from the Alpaca Market Data API. It is designed to support efficient and reliable access to price data while minimizing latency, API usage, and memory footprint.

This is achieved through a combination of local caching, timestamp normalization, and column type optimization, ensuring consistent and clean data inputs for all downstream processes in the trading pipeline.

---

### Key Features

- Efficient retrieval of historical bar data using Alpaca’s REST API  
- Automatic caching to limit redundant API calls  
- Normalization of timestamps and timezone handling  
- Memory optimization through proper data type conversion  
- Graceful handling of empty or missing datasets  

> The AlpacaAPI serves as the foundational data layer of the entire trading architecture.

## AlpacaData – Custom Data Feed for Backtrader

The `AlpacaData` class is a custom implementation of a data feed compatible with the Backtrader backtesting engine. It is designed to load and validate historical OHLCV data retrieved from the Alpaca API and make it compliant with Backtrader’s internal format.

This wrapper ensures:
- That the input data uses a valid `DatetimeIndex`
- That all essential fields (Open, High, Low, Close, Volume) are correctly mapped
- That the `openinterest` column is present, as required by Backtrader, even if unused

---

### Key Functionalities

- Full compatibility with Backtrader's `PandasData` interface  
- Strict validation of the datetime index  
- No overhead from unnecessary fields  
- Seamless integration with the `AlpacaAPI` class for streamlined data loading

## AlpacaAPI – Historical Market Data Retrieval

The `AlpacaAPI` class is responsible for fetching historical OHLCV data (Open, High, Low, Close, Volume) from the Alpaca Market Data API. It is designed to support efficient and reliable access to price data while minimizing latency, API usage, and memory footprint.

This is achieved through a combination of local caching, timestamp normalization, and column type optimization, ensuring consistent and clean data inputs for all downstream processes in the trading pipeline.

---

## Key Features

- Efficient retrieval of historical bar data using Alpaca’s REST API  
- Automatic caching to limit redundant API calls  
- Normalization of timestamps and timezone handling  
- Memory optimization through proper data type conversion  
- Graceful handling of empty or missing datasets  

> The AlpacaAPI serves as the foundational data layer of the entire trading architecture.

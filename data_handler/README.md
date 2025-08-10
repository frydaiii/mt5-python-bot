# Data Handler Module

The `data_handler` module provides a comprehensive interface for retrieving financial data from MetaTrader 5. It offers functions to get historical data for specific symbols and timeframes, along with utility functions for managing MT5 connections.

## Features

- **Easy MT5 Connection Management**: Initialize and shutdown MT5 connections
- **Flexible Data Retrieval**: Get data by count, date range, or specific periods
- **Multiple Symbol Support**: Retrieve data for multiple symbols simultaneously
- **Symbol Information**: Get detailed symbol specifications and current prices
- **Timeframe Utilities**: Support for all MT5 timeframes with string conversion
- **Error Handling**: Comprehensive error handling and logging
- **Data Validation**: Input validation and data quality checks

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from data_handler import initialize_mt5, shutdown_mt5, get_symbol_data
import MetaTrader5 as mt5

# Initialize MT5 connection
if initialize_mt5():
    try:
        # Get EUR/USD hourly data for the last 100 bars
        df = get_symbol_data("EURUSD", mt5.TIMEFRAME_H1, count=100)
        
        if df is not None:
            print(f"Retrieved {len(df)} bars")
            print(df.head())
        
    finally:
        shutdown_mt5()
```

## Available Functions

### Connection Management

- `initialize_mt5(login=None, password=None, server=None, path=None)` - Initialize MT5 connection
- `shutdown_mt5()` - Close MT5 connection

### Data Retrieval

- `get_symbol_data(symbol, timeframe, count, start_date, end_date)` - Get historical data for a symbol
- `get_multiple_symbols_data(symbols, timeframe, count, start_date, end_date)` - Get data for multiple symbols
- `get_current_price(symbol)` - Get current bid/ask prices

### Symbol Information

- `get_symbol_info(symbol)` - Get detailed symbol information
- `get_available_symbols(group="*")` - Get list of available symbols

### Utilities

- `TIMEFRAMES` - Dictionary of timeframe constants
- `get_timeframe_value(timeframe_str)` - Convert timeframe string to MT5 constant

## Usage Examples

### Basic Data Retrieval

```python
from data_handler import initialize_mt5, shutdown_mt5, get_symbol_data
import MetaTrader5 as mt5

if initialize_mt5():
    try:
        # Get last 50 daily bars for EUR/USD
        df = get_symbol_data("EURUSD", mt5.TIMEFRAME_D1, count=50)
        
        if df is not None:
            print(f"OHLC Data for EURUSD:")
            print(df[['open', 'high', 'low', 'close']].tail())
    finally:
        shutdown_mt5()
```

### Date Range Retrieval

```python
from datetime import datetime, timedelta
from data_handler import initialize_mt5, shutdown_mt5, get_symbol_data
import MetaTrader5 as mt5

if initialize_mt5():
    try:
        # Get data for the last week
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        df = get_symbol_data(
            symbol="GBPUSD",
            timeframe=mt5.TIMEFRAME_H4,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is not None:
            print(f"Data from {start_date.date()} to {end_date.date()}")
            print(f"Number of bars: {len(df)}")
    finally:
        shutdown_mt5()
```

### Multiple Symbols

```python
from data_handler import initialize_mt5, shutdown_mt5, get_multiple_symbols_data
import MetaTrader5 as mt5

if initialize_mt5():
    try:
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        data_dict = get_multiple_symbols_data(
            symbols=symbols,
            timeframe=mt5.TIMEFRAME_H1,
            count=100
        )
        
        for symbol, df in data_dict.items():
            if df is not None:
                print(f"{symbol}: {len(df)} bars")
                print(f"  Latest close: {df['close'].iloc[-1]:.5f}")
    finally:
        shutdown_mt5()
```

### Using Timeframe Strings

```python
from data_handler import initialize_mt5, shutdown_mt5, get_symbol_data, get_timeframe_value

if initialize_mt5():
    try:
        # Use string timeframe instead of MT5 constant
        timeframe = get_timeframe_value("H1")  # 1 hour
        
        df = get_symbol_data("EURUSD", timeframe, count=24)
        
        if df is not None:
            print(f"Last 24 hourly bars for EURUSD")
    finally:
        shutdown_mt5()
```

### Symbol Information

```python
from data_handler import initialize_mt5, shutdown_mt5, get_symbol_info, get_current_price

if initialize_mt5():
    try:
        symbol = "EURUSD"
        
        # Get symbol specifications
        info = get_symbol_info(symbol)
        if info:
            print(f"Symbol: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"Digits: {info['digits']}")
            print(f"Spread: {info['spread']}")
        
        # Get current price
        price = get_current_price(symbol)
        if price:
            print(f"Current Bid: {price['bid']}")
            print(f"Current Ask: {price['ask']}")
    finally:
        shutdown_mt5()
```

## Timeframes

The module supports all MT5 timeframes:

- `M1` - 1 minute
- `M5` - 5 minutes
- `M15` - 15 minutes
- `M30` - 30 minutes
- `H1` - 1 hour
- `H4` - 4 hours
- `D1` - 1 day
- `W1` - 1 week
- `MN1` - 1 month

## Data Format

All historical data is returned as pandas DataFrame with the following columns:

- `time` - Timestamp (used as index)
- `open` - Opening price
- `high` - Highest price
- `low` - Lowest price
- `close` - Closing price
- `tick_volume` - Tick volume
- `spread` - Spread
- `real_volume` - Real volume (if available)

## Error Handling

The module includes comprehensive error handling:

- Connection failures are logged and return `False` or `None`
- Invalid symbols are detected and reported
- Network timeouts are handled gracefully
- All functions include try-catch blocks with detailed logging

## Logging

The module uses Python's built-in logging system. To see detailed logs:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Requirements

- MetaTrader 5 terminal installed and running
- Valid trading account (demo or live)
- Python packages: MetaTrader5, pandas

## Notes

- Ensure MetaTrader 5 is running before initializing the connection
- Some symbols may not be available depending on your broker
- Historical data availability depends on your broker's data provision
- The module automatically handles symbol selection in Market Watch

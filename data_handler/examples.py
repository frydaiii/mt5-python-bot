"""
Example usage of the data_handler module
"""

from data_handler import (
    initialize_mt5, 
    shutdown_mt5, 
    get_symbol_data, 
    get_multiple_symbols_data,
    get_symbol_info,
    get_available_symbols,
    get_current_price,
    TIMEFRAMES,
    get_timeframe_value
)
from datetime import datetime, timedelta
import MetaTrader5 as mt5


def example_basic_usage():
    """Example of basic data retrieval."""
    print("=== Basic Data Retrieval Example ===")
    
    # Initialize MT5
    if not initialize_mt5():
        print("Failed to initialize MT5")
        return
    
    try:
        # Get EUR/USD hourly data for the last 100 bars
        symbol = "EURUSD"
        df = get_symbol_data(symbol, mt5.TIMEFRAME_H1, count=100)
        
        if df is not None:
            print(f"Retrieved {len(df)} bars for {symbol}")
            print("\nFirst 5 rows:")
            print(df.head())
            print("\nLast 5 rows:")
            print(df.tail())
        else:
            print(f"Failed to get data for {symbol}")
            
    finally:
        shutdown_mt5()


def example_date_range_usage():
    """Example of data retrieval with specific date range."""
    print("\n=== Date Range Data Retrieval Example ===")
    
    if not initialize_mt5():
        print("Failed to initialize MT5")
        return
    
    try:
        # Get data for the last 7 days
        symbol = "GBPUSD"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        df = get_symbol_data(
            symbol=symbol,
            timeframe=mt5.TIMEFRAME_H4,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is not None:
            print(f"Retrieved {len(df)} H4 bars for {symbol} from {start_date.date()} to {end_date.date()}")
            print(f"Data range: {df.index[0]} to {df.index[-1]}")
        else:
            print(f"Failed to get data for {symbol}")
            
    finally:
        shutdown_mt5()


def example_multiple_symbols():
    """Example of retrieving data for multiple symbols."""
    print("\n=== Multiple Symbols Data Retrieval Example ===")
    
    if not initialize_mt5():
        print("Failed to initialize MT5")
        return
    
    try:
        # Get data for multiple currency pairs
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        
        data_dict = get_multiple_symbols_data(
            symbols=symbols,
            timeframe=mt5.TIMEFRAME_D1,
            count=50
        )
        
        print(f"Retrieved data for {len(data_dict)} symbols:")
        for symbol, df in data_dict.items():
            if df is not None:
                print(f"  {symbol}: {len(df)} daily bars")
            else:
                print(f"  {symbol}: Failed to retrieve data")
                
    finally:
        shutdown_mt5()


def example_symbol_info():
    """Example of getting symbol information."""
    print("\n=== Symbol Information Example ===")
    
    if not initialize_mt5():
        print("Failed to initialize MT5")
        return
    
    try:
        symbol = "EURUSD"
        info = get_symbol_info(symbol)
        
        if info:
            print(f"Information for {symbol}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"Failed to get information for {symbol}")
            
        # Get current price
        price = get_current_price(symbol)
        if price:
            print(f"\nCurrent price for {symbol}:")
            print(f"  Bid: {price['bid']}")
            print(f"  Ask: {price['ask']}")
            print(f"  Last: {price['last']}")
            print(f"  Time: {price['time']}")
            
    finally:
        shutdown_mt5()


def example_available_symbols():
    """Example of getting available symbols."""
    print("\n=== Available Symbols Example ===")
    
    if not initialize_mt5():
        print("Failed to initialize MT5")
        return
    
    try:
        # Get all Forex symbols
        forex_symbols = get_available_symbols("Forex*")
        print(f"Found {len(forex_symbols)} Forex symbols")
        
        if forex_symbols:
            print("First 10 Forex symbols:")
            for symbol in forex_symbols[:10]:
                print(f"  {symbol}")
                
    finally:
        shutdown_mt5()


def example_using_timeframe_strings():
    """Example of using timeframe strings instead of constants."""
    print("\n=== Using Timeframe Strings Example ===")
    
    if not initialize_mt5():
        print("Failed to initialize MT5")
        return
    
    try:
        symbol = "EURUSD"
        timeframe_str = "H1"  # 1 hour
        
        # Convert string to MT5 constant
        timeframe = get_timeframe_value(timeframe_str)
        
        df = get_symbol_data(symbol, timeframe, count=24)  # Last 24 hours
        
        if df is not None:
            print(f"Retrieved {len(df)} {timeframe_str} bars for {symbol}")
            print("Available timeframes:", list(TIMEFRAMES.keys()))
        else:
            print(f"Failed to get data for {symbol}")
            
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_date_range_usage()
    example_multiple_symbols()
    example_symbol_info()
    example_available_symbols()
    example_using_timeframe_strings()
    
    print("\n=== Examples completed ===")

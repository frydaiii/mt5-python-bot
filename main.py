"""
MT5 Python Trading Bot - Main Application

This script demonstrates how to use the data_handler module to retrieve
financial data from MetaTrader 5 using secure configuration management.
"""

from data_handler import (
    initialize_mt5_with_config,
    shutdown_mt5,
    get_symbol_data,
    get_current_price,
    TIMEFRAMES
)
from trading.utils import rebalance_portfolio, get_portfolio_summary, validate_allocations
from config import config
import MetaTrader5 as mt5


def main():
    """Main application function."""
    print("MT5 Python Trading Bot")
    print("=" * 30)
    
    # Check configuration
    if not config.validate_mt5_config():
        print("❌ Missing required configuration")
        print("Please create a .env file based on .env.example and fill in your MT5 credentials")
        return
    
    # Initialize MT5 connection using configuration
    if not initialize_mt5_with_config():
        print("❌ Failed to initialize MT5 connection")
        print("Please ensure MetaTrader 5 is installed and running")
        print("Also verify your credentials in the .env file")
        return
    
    print("✅ MT5 connection established successfully")
    
    try:
        # Example: Get symbol data for EURUSD
        symbol = "EURUSDm"
        timeframe = TIMEFRAMES['H1']  # 1-hour timeframe
        
        print(f"Fetching data for {symbol} on {timeframe} timeframe...")
        data = get_symbol_data(symbol, timeframe)
        
        if data is not None:
            print(f"📈 Data for {symbol}:\n{data.head()}")
        else:
            print(f"❌ No data found for {symbol} on {timeframe} timeframe")
        
        # Example: Get current price
        current_price = get_current_price(symbol)
        if current_price is not None:
            print(f"💰 Current price of {symbol}: {current_price}")
        else:
            print(f"❌ Failed to retrieve current price for {symbol}")
        
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        
    finally:
        # Clean shutdown
        shutdown_mt5()
        print("\n🔚 MT5 connection closed")


if __name__ == "__main__":
    main()
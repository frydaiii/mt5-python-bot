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
        # Example 1: Get current price for EUR/USD
        symbol = "EURUSD"
        print(f"\n📊 Getting current price for {symbol}...")
        
        current_price = get_current_price(symbol)
        if current_price:
            print(f"   Bid: {current_price['bid']:.5f}")
            print(f"   Ask: {current_price['ask']:.5f}")
            print(f"   Spread: {(current_price['ask'] - current_price['bid']):.5f}")
        else:
            print(f"   ❌ Failed to get current price for {symbol}")
        
        # Example 2: Get historical data
        print(f"\n📈 Getting historical data for {symbol}...")
        
        # Get last 50 H1 bars
        df = get_symbol_data(
            symbol=symbol,
            timeframe=TIMEFRAMES['H1'],
            count=50
        )
        
        if df is not None and len(df) > 0:
            print(f"   ✅ Retrieved {len(df)} hourly bars")
            print(f"   Date range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
            print(f"   Latest close price: {df['close'].iloc[-1]:.5f}")
            print(f"   Highest price: {df['high'].max():.5f}")
            print(f"   Lowest price: {df['low'].min():.5f}")
        else:
            print(f"   ❌ Failed to get historical data for {symbol}")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        
    finally:
        # Clean shutdown
        shutdown_mt5()
        print("\n🔚 MT5 connection closed")


if __name__ == "__main__":
    main()
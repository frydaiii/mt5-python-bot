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
        # Example: Get portfolio summary
        print("\n📊 Getting current portfolio summary...")
        summary = get_portfolio_summary()
        
        if "error" not in summary:
            print(f"Account Balance: ${summary.get('account_balance', 0):,.2f}")
            print(f"Account Equity: ${summary.get('account_equity', 0):,.2f}")
            print(f"Open Positions: {summary.get('number_of_positions', 0)}")
        else:
            print(f"Error getting portfolio summary: {summary['error']}")
        
        # Example: Portfolio rebalancing (dry run)
        print("\n🎯 Example portfolio rebalancing...")
        target_allocations = [
            ("EURUSD", 0.4),   # 40% EUR/USD
            ("GBPUSD", 0.3),   # 30% GBP/USD
            ("USDJPY", 0.2),   # 20% USD/JPY
            ("AUDUSD", 0.1),   # 10% AUD/USD
        ]
        
        # Perform dry run
        results = rebalance_portfolio(target_allocations, dry_run=True)
        
        if "error" not in results:
            print(f"Current weights: {results.get('current_weights', {})}")
            print(f"Target weights: {results.get('target_weights', {})}")
            print(f"Required trades: {results.get('required_trades', {})}")
            
            # Ask user if they want to execute
            if results.get('required_trades'):
                print("\n💡 To execute these trades, set dry_run=False in the rebalance_portfolio call")
        else:
            print(f"Rebalancing error: {results['error']}")

            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        
    finally:
        # Clean shutdown
        shutdown_mt5()
        print("\n🔚 MT5 connection closed")


if __name__ == "__main__":
    main()
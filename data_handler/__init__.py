"""
Data Handler Module for MT5 Trading Bot

This module provides functions to retrieve financial data from MetaTrader 5
for specific symbols and timeframes.
"""

from .mt5_data import (
    initialize_mt5,
    shutdown_mt5,
    get_symbol_data,
    get_multiple_symbols_data,
    get_symbol_info,
    get_available_symbols,
    get_current_price,
    TIMEFRAMES,
    get_timeframe_value,
    initialize_mt5_with_config
)

__version__ = "1.0.0"
__all__ = [
    "initialize_mt5",
    "shutdown_mt5", 
    "get_symbol_data",
    "get_multiple_symbols_data",
    "get_symbol_info",
    "get_available_symbols",
    "get_current_price",
    "TIMEFRAMES",
    "get_timeframe_value",
    "initialize_mt5_with_config"
]

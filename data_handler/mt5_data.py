"""
MT5 Data Handler

This module provides functions to interact with MetaTrader 5 and retrieve
financial data for specific symbols and timeframes.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Tuple
import logging
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

def initialize_mt5(login: Optional[int] = None, 
                   password: Optional[str] = None, 
                   server: Optional[str] = None,
                   path: Optional[str] = None) -> bool:
    """
    Initialize connection to MetaTrader 5.
    
    Args:
        login: Trading account login (if None, will use config values)
        password: Trading account password (if None, will use config values)  
        server: Trading server name (if None, will use config values)
        path: Path to MetaTrader 5 terminal (if None, will use config values)
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        # Use config values if parameters not provided
        credentials = config.get_mt5_credentials()
        
        login = login or credentials.get('login')
        password = password or credentials.get('password')
        server = server or credentials.get('server')
        path = path or credentials.get('path')

        if not mt5.initialize(path=path, login=login, password=password, server=server):
            error_code, error_msg = mt5.last_error()
            logger.error(f"MT5 initialization failed. Error code: {error_code}, Message: {error_msg}")
            return False
                
        logger.info("MT5 initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing MT5: {e}")
        return False


def initialize_mt5_with_config() -> bool:
    """
    Initialize MT5 using configuration from environment variables.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    if not config.validate_mt5_config():
        return False
    
    return initialize_mt5()


def shutdown_mt5() -> None:
    """Shutdown MetaTrader 5 connection."""
    mt5.shutdown()
    logger.info("MT5 connection closed")


def get_symbol_data(symbol: str, 
                   timeframe: int = mt5.TIMEFRAME_H1,
                   count: int = 100,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """
    Get historical data for a specific symbol and timeframe.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'GBPUSD')
        timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1, mt5.TIMEFRAME_D1)
        count: Number of bars to retrieve (used if start_date/end_date not provided)
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data or None if error
    """
    try:
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            error_code, error_msg = mt5.last_error()
            logger.error(f"Symbol {symbol} not found. Error code: {error_code}, Message: {error_msg}")
            return None
            
        # Select the symbol in the Market Watch
        if not mt5.symbol_select(symbol, True):
            error_code, error_msg = mt5.last_error()
            logger.error(f"Failed to select symbol {symbol}. Error code: {error_code}, Message: {error_msg}")
            return None
            
        # Get data based on provided parameters
        if start_date and end_date:
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        elif start_date:
            rates = mt5.copy_rates_from(symbol, timeframe, start_date, count)
        else:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
        if rates is None or len(rates) == 0:
            error_code, error_msg = mt5.last_error()
            logger.warning(f"No data retrieved for {symbol}. Error code: {error_code}, Message: {error_msg}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Set time as index
        df.set_index('time', inplace=True)
        
        # Rename columns for clarity
        df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        
        logger.info(f"Retrieved {len(df)} bars for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {e}")
        return None


def get_multiple_symbols_data(symbols: List[str],
                             timeframe: int = mt5.TIMEFRAME_H1,
                             count: int = 100,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
    """
    Get historical data for multiple symbols.
    
    Args:
        symbols: List of trading symbols
        timeframe: MT5 timeframe constant
        count: Number of bars to retrieve
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with symbol as key and DataFrame as value
    """
    data_dict = {}
    
    for symbol in symbols:
        df = get_symbol_data(symbol, timeframe, count, start_date, end_date)
        if df is not None:
            data_dict[symbol] = df
        else:
            logger.warning(f"Failed to get data for {symbol}")
            
    return data_dict


def get_symbol_info(symbol: str) -> Optional[Dict]:
    """
    Get detailed information about a trading symbol.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dict: Symbol information or None if error
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            error_code, error_msg = mt5.last_error()
            logger.error(f"Symbol {symbol} not found. Error code: {error_code}, Message: {error_msg}")
            return None
            
        # Convert to dictionary for easier handling
        info_dict = {
            'name': symbol_info.name,
            'description': symbol_info.description,
            'currency_base': symbol_info.currency_base,
            'currency_profit': symbol_info.currency_profit,
            'currency_margin': symbol_info.currency_margin,
            'digits': symbol_info.digits,
            'point': symbol_info.point,
            'spread': symbol_info.spread,
            'trade_mode': symbol_info.trade_mode,
            'min_lot': symbol_info.volume_min,
            'max_lot': symbol_info.volume_max,
            'lot_step': symbol_info.volume_step,
            'swap_long': symbol_info.swap_long,
            'swap_short': symbol_info.swap_short,
        }
        
        return info_dict
        
    except Exception as e:
        logger.error(f"Error getting symbol info for {symbol}: {e}")
        return None


def get_available_symbols(group: str = "*") -> List[str]:
    """
    Get list of available trading symbols.
    
    Args:
        group: Symbol group filter (e.g., "*EURUSD*", "Forex*", "*")
        
    Returns:
        List[str]: List of available symbols
    """
    try:
        symbols = mt5.symbols_get(group=group)
        if symbols is None:
            error_code, error_msg = mt5.last_error()
            logger.warning(f"No symbols found. Error code: {error_code}, Message: {error_msg}")
            return []
            
        symbol_names = [symbol.name for symbol in symbols]
        logger.info(f"Found {len(symbol_names)} symbols")
        return symbol_names
        
    except Exception as e:
        logger.error(f"Error getting available symbols: {e}")
        return []


def get_current_price(symbol: str) -> Optional[Dict[str, float]]:
    """
    Get current price information for a symbol.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dict: Current price information (bid, ask, last) or None if error
    """
    try:
        # attempt to enable the display of the GBPUSD in MarketWatch
        selected=mt5.symbol_select(symbol,True)
        if not selected:
            error_code, error_msg = mt5.last_error()
            logger.error(f"Failed to select {symbol}. Error code: {error_code}, Message: {error_msg}")
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            error_code, error_msg = mt5.last_error()
            logger.error(f"Failed to get current price for {symbol}. Error code: {error_code}, Message: {error_msg}")
            return None
            
        price_info = {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'time': datetime.fromtimestamp(tick.time)
        }
        
        return price_info
        
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}")
        return None


# Timeframe constants for easy access
TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
    'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1
}


def get_timeframe_value(timeframe_str: str) -> int:
    """
    Convert timeframe string to MT5 timeframe constant.
    
    Args:
        timeframe_str: Timeframe string (e.g., 'H1', 'D1', 'M15')
        
    Returns:
        int: MT5 timeframe constant
        
    Raises:
        ValueError: If timeframe string is not valid
    """
    if timeframe_str.upper() not in TIMEFRAMES:
        raise ValueError(f"Invalid timeframe: {timeframe_str}. Valid options: {list(TIMEFRAMES.keys())}")
    
    return TIMEFRAMES[timeframe_str.upper()]

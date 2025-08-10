"""
Utility functions for the data handler module.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def validate_symbol_format(symbol: str) -> bool:
    """
    Validate if symbol format is correct.
    
    Args:
        symbol: Trading symbol to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(symbol, str):
        return False
    
    # Basic validation - symbol should be 6-8 characters typically
    if len(symbol) < 3 or len(symbol) > 10:
        return False
        
    # Should contain only alphanumeric characters
    if not symbol.replace('_', '').replace('.', '').isalnum():
        return False
        
    return True


def format_datetime_for_mt5(dt: datetime) -> datetime:
    """
    Format datetime for MT5 compatibility.
    
    Args:
        dt: Datetime object to format
        
    Returns:
        datetime: Formatted datetime
    """
    # MT5 typically works with UTC time
    return dt.replace(microsecond=0)


def get_business_days_count(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of business days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        int: Number of business days
    """
    business_days = 0
    current_date = start_date
    
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
            business_days += 1
        current_date += timedelta(days=1)
        
    return business_days


def calculate_bars_needed(timeframe_minutes: int, 
                         start_date: datetime, 
                         end_date: datetime) -> int:
    """
    Calculate approximate number of bars needed for a given timeframe and date range.
    
    Args:
        timeframe_minutes: Timeframe in minutes
        start_date: Start date
        end_date: End date
        
    Returns:
        int: Estimated number of bars
    """
    total_minutes = (end_date - start_date).total_seconds() / 60
    
    # For forex markets, adjust for weekend gaps
    business_days = get_business_days_count(start_date, end_date)
    total_days = (end_date - start_date).days + 1
    
    # Rough adjustment for market hours (forex trades ~24/5)
    market_ratio = business_days / total_days if total_days > 0 else 1
    adjusted_minutes = total_minutes * market_ratio
    
    return int(adjusted_minutes / timeframe_minutes)


def timeframe_to_minutes(timeframe_constant: int) -> int:
    """
    Convert MT5 timeframe constant to minutes.
    
    Args:
        timeframe_constant: MT5 timeframe constant
        
    Returns:
        int: Timeframe in minutes
    """
    # Import here to avoid circular imports
    import MetaTrader5 as mt5
    
    timeframe_map = {
        mt5.TIMEFRAME_M1: 1,
        mt5.TIMEFRAME_M5: 5,
        mt5.TIMEFRAME_M15: 15,
        mt5.TIMEFRAME_M30: 30,
        mt5.TIMEFRAME_H1: 60,
        mt5.TIMEFRAME_H4: 240,
        mt5.TIMEFRAME_D1: 1440,
        mt5.TIMEFRAME_W1: 10080,
        mt5.TIMEFRAME_MN1: 43200  # Approximate
    }
    
    return timeframe_map.get(timeframe_constant, 60)  # Default to 1 hour


def clean_symbol_name(symbol: str) -> str:
    """
    Clean and standardize symbol name.
    
    Args:
        symbol: Raw symbol name
        
    Returns:
        str: Cleaned symbol name
    """
    # Remove common suffixes and prefixes
    cleaned = symbol.upper()
    
    # Remove common broker suffixes
    suffixes_to_remove = ['.RAW', '.ECN', '.PRO', '.MIC', '.STP']
    for suffix in suffixes_to_remove:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break
    
    return cleaned


def log_data_summary(data, symbol: str, timeframe: str) -> None:
    """
    Log summary information about retrieved data.
    
    Args:
        data: DataFrame or data object
        symbol: Symbol name
        timeframe: Timeframe string
    """
    try:
        if data is not None and hasattr(data, '__len__'):
            logger.info(f"Data summary for {symbol} ({timeframe}): {len(data)} records")
            
            if hasattr(data, 'index') and len(data) > 0:
                logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        else:
            logger.warning(f"No data retrieved for {symbol} ({timeframe})")
    except Exception as e:
        logger.error(f"Error logging data summary: {e}")


def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """
    Validate that date range is logical.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        bool: True if valid, False otherwise
    """
    if start_date >= end_date:
        logger.error("Start date must be before end date")
        return False
        
    # Check if dates are not too far in the future
    now = datetime.now()
    if start_date > now or end_date > now:
        logger.warning("Date range includes future dates")
        
    # Check if dates are not too far in the past (more than 10 years)
    ten_years_ago = now - timedelta(days=3650)
    if start_date < ten_years_ago:
        logger.warning("Start date is more than 10 years ago")
        
    return True

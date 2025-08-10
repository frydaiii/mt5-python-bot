"""
Alpha Factors Implementation for MT5 Trading Bot

This module implements various alpha factors used for quantitative trading strategies.
The alpha factors are based on price, volume, and other market data transformations.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class AlphaOperators:
    """Implementation of alpha factor operators and functions."""
    
    @staticmethod
    def rank(x: pd.Series) -> pd.Series:
        """Cross-sectional rank."""
        return x.rank(pct=True)
    
    @staticmethod
    def delay(x: pd.Series, d: int) -> pd.Series:
        """Value of x d days ago."""
        return x.shift(d)
    
    @staticmethod
    def correlation(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        """Time-serial correlation of x and y for the past d days."""
        return x.rolling(window=d).corr(y)
    
    @staticmethod
    def covariance(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        """Time-serial covariance of x and y for the past d days."""
        return x.rolling(window=d).cov(y)
    
    @staticmethod
    def scale(x: pd.Series, a: float = 1.0) -> pd.Series:
        """Rescaled x such that sum(abs(x)) = a."""
        abs_sum = x.abs().sum()
        if abs_sum == 0:
            return x
        return x * (a / abs_sum)
    
    @staticmethod
    def delta(x: pd.Series, d: int) -> pd.Series:
        """Today's value of x minus the value of x d days ago."""
        return x - x.shift(d)
    
    @staticmethod
    def signedpower(x: pd.Series, a: float) -> pd.Series:
        """x^a with sign preservation."""
        return pd.Series(np.sign(x) * (np.abs(x) ** a), index=x.index)
    
    @staticmethod
    def decay_linear(x: pd.Series, d: int) -> pd.Series:
        """Weighted moving average with linearly decaying weights."""
        weights = np.arange(1, d + 1)
        weights = weights / weights.sum()
        return x.rolling(window=d).apply(lambda vals: np.average(vals, weights=weights))
    
    @staticmethod
    def ts_min(x: pd.Series, d: int) -> pd.Series:
        """Time-series min over the past d days."""
        return x.rolling(window=d).min()
    
    @staticmethod
    def ts_max(x: pd.Series, d: int) -> pd.Series:
        """Time-series max over the past d days."""
        return x.rolling(window=d).max()
    
    @staticmethod
    def ts_argmax(x: pd.Series, d: int) -> pd.Series:
        """Which day ts_max(x, d) occurred on."""
        return x.rolling(window=d).apply(lambda vals: vals.argmax())
    
    @staticmethod
    def ts_argmin(x: pd.Series, d: int) -> pd.Series:
        """Which day ts_min(x, d) occurred on."""
        return x.rolling(window=d).apply(lambda vals: vals.argmin())
    
    @staticmethod
    def ts_rank(x: pd.Series, d: int) -> pd.Series:
        """Time-series rank in the past d days."""
        return x.rolling(window=d).rank(pct=True)
    
    @staticmethod
    def sum(x: pd.Series, d: int) -> pd.Series:
        """Time-series sum over the past d days."""
        return x.rolling(window=d).sum()
    
    @staticmethod
    def product(x: pd.Series, d: int) -> pd.Series:
        """Time-series product over the past d days."""
        return x.rolling(window=d).apply(lambda vals: vals.prod())
    
    @staticmethod
    def stddev(x: pd.Series, d: int) -> pd.Series:
        """Moving time-series standard deviation over the past d days."""
        return x.rolling(window=d).std()
    
    @staticmethod
    def adv(volume: pd.Series, close: pd.Series, d: int) -> pd.Series:
        """Average daily dollar volume for the past d days."""
        dollar_volume = volume * close
        return dollar_volume.rolling(window=d).mean()
    
    @staticmethod
    def indneutralize(x: pd.Series, groups: pd.Series) -> pd.Series:
        """Cross-sectionally neutralize x against groups."""
        # Simple implementation: demean within each group
        result = x.copy()
        for group in groups.unique():
            mask = groups == group
            group_mean = x[mask].mean()
            result[mask] = x[mask] - group_mean
        return result


def alpha07(
    open_prices: pd.Series,
    close_prices: pd.Series, 
    high_prices: pd.Series,
    low_prices: pd.Series,
    volume: pd.Series,
    vwap: Optional[pd.Series] = None
) -> pd.Series:
    """
    Alpha07 Implementation
    
    Formula: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
    
    Parameters:
    -----------
    open_prices : pd.Series
        Daily open prices
    close_prices : pd.Series
        Daily close prices
    high_prices : pd.Series
        Daily high prices
    low_prices : pd.Series
        Daily low prices
    volume : pd.Series
        Daily volume
    vwap : pd.Series, optional
        Volume-weighted average price (if not provided, will be calculated)
    
    Returns:
    --------
    pd.Series
        Alpha07 factor values
    """
    try:
        # Calculate VWAP if not provided
        if vwap is None:
            typical_price = (high_prices + low_prices + close_prices) / 3
            vwap = (typical_price * volume).rolling(window=1).sum() / volume.rolling(window=1).sum()
        
        # Calculate average daily dollar volume for past 20 days (adv20)
        adv20 = AlphaOperators.adv(volume, close_prices, 20)
        
        # Calculate delta(close, 7) - change in close price over 7 days
        delta_close_7 = AlphaOperators.delta(close_prices, 7)
        
        # Calculate abs(delta(close, 7))
        abs_delta_close_7 = pd.Series(np.abs(delta_close_7), index=delta_close_7.index)
        
        # Calculate ts_rank(abs(delta(close, 7)), 60) - time series rank over 60 days
        ts_rank_abs_delta = AlphaOperators.ts_rank(abs_delta_close_7, 60)
        
        # Calculate sign(delta(close, 7))
        sign_delta_close_7 = np.sign(delta_close_7)
        
        # Calculate the main expression: (-1 * ts_rank(...)) * sign(...)
        main_expression = (-1 * ts_rank_abs_delta) * sign_delta_close_7
        
        # Apply conditional logic: (adv20 < volume) ? main_expression : (-1 * 1)
        condition = adv20 < volume
        alpha07_values = np.where(condition, main_expression, -1.0)
        
        # Convert back to pandas Series with original index
        result = pd.Series(alpha07_values, index=close_prices.index)
        
        # Handle any NaN values
        result = result.fillna(0)
        
        logger.info(f"Alpha07 calculated successfully. Non-zero values: {(result != 0).sum()}/{len(result)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating Alpha07: {str(e)}")
        # Return zeros in case of error
        return pd.Series(0, index=close_prices.index)


def calculate_alpha07_from_dataframe(
    df: pd.DataFrame,
    open_col: str = 'open',
    close_col: str = 'close',
    high_col: str = 'high',
    low_col: str = 'low',
    volume_col: str = 'volume',
    vwap_col: Optional[str] = None
) -> pd.Series:
    """
    Convenience function to calculate Alpha07 from a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing OHLCV data
    open_col : str
        Column name for open prices
    close_col : str
        Column name for close prices
    high_col : str
        Column name for high prices
    low_col : str
        Column name for low prices
    volume_col : str
        Column name for volume
    vwap_col : str, optional
        Column name for VWAP (if available)
    
    Returns:
    --------
    pd.Series
        Alpha07 factor values
    """
    vwap = df[vwap_col] if vwap_col and vwap_col in df.columns else None
    
    return alpha07(
        open_prices=df[open_col],
        close_prices=df[close_col],
        high_prices=df[high_col],
        low_prices=df[low_col],
        volume=df[volume_col],
        vwap=vwap
    )


# Example usage and testing functions
def test_alpha07():
    """Test function for Alpha07 implementation."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate synthetic OHLCV data with patterns that will trigger the condition
    close_prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5), index=dates)
    open_prices = pd.Series(close_prices.shift(1) + np.random.randn(100) * 0.2, index=dates)
    high_prices = pd.Series(np.maximum(open_prices, close_prices) + np.abs(np.random.randn(100) * 0.3), index=dates)
    low_prices = pd.Series(np.minimum(open_prices, close_prices) - np.abs(np.random.randn(100) * 0.3), index=dates)
    
    # Create volume pattern where some days have volume > adv20
    # Start with low base volume, then add spikes
    base_volume = np.random.randint(100, 500, 100)  # Lower base volume
    # Add volume spikes on some days (this will make volume > adv20 for those days)
    spike_days = np.random.choice(range(30, 100), 20, replace=False)  # 20 spike days after day 30
    volume_data = base_volume.copy()
    volume_data[spike_days] *= np.random.randint(10, 50, len(spike_days))  # Large volume spikes
    volume = pd.Series(volume_data, index=dates)
    
    # Calculate Alpha07
    alpha_values = alpha07(open_prices, close_prices, high_prices, low_prices, volume)
    
    print(f"Alpha07 Test Results:")
    print(f"Shape: {alpha_values.shape}")
    print(f"Non-NaN values: {alpha_values.notna().sum()}")
    print(f"Min value: {alpha_values.min():.4f}")
    print(f"Max value: {alpha_values.max():.4f}")
    print(f"Mean value: {alpha_values.mean():.4f}")
    print(f"Std deviation: {alpha_values.std():.4f}")
    print(f"Unique values: {alpha_values.nunique()}")
    print(f"Values != -1: {(alpha_values != -1.0).sum()}")
    
    # Show some examples of the calculation working
    if (alpha_values != -1.0).any():
        non_neg_one = alpha_values[alpha_values != -1.0]
        print(f"Non -1.0 values range: {non_neg_one.min():.4f} to {non_neg_one.max():.4f}")
    
    return alpha_values


def example_usage():
    """Example of how to use the alpha07 function with real-like data."""
    print("Example: Using Alpha07 with sample market data")
    
    # Create sample market data
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    np.random.seed(123)
    
    # Simulate price data
    price_base = 100
    returns = np.random.normal(0, 0.02, 90)  # 2% daily volatility
    close_prices = pd.Series(price_base * np.exp(np.cumsum(returns)), index=dates)
    
    # Create OHLC data
    daily_ranges = np.random.uniform(0.005, 0.03, 90)  # 0.5% to 3% daily range
    open_prices = pd.Series(close_prices.shift(1) * (1 + np.random.normal(0, 0.005, 90)), index=dates)
    high_prices = pd.Series(np.maximum(open_prices, close_prices) * (1 + daily_ranges/2), index=dates)
    low_prices = pd.Series(np.minimum(open_prices, close_prices) * (1 - daily_ranges/2), index=dates)
    
    # Create volume with realistic patterns
    avg_volume = 1000000
    volume_volatility = np.random.lognormal(0, 0.5, 90)
    volume = pd.Series(avg_volume * volume_volatility, index=dates).astype(int)
    
    # Calculate Alpha07
    alpha_result = alpha07(open_prices, close_prices, high_prices, low_prices, volume)
    
    print(f"\nAlpha07 Results for 90-day period:")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"Alpha values range: {alpha_result.min():.4f} to {alpha_result.max():.4f}")
    print(f"Mean alpha: {alpha_result.mean():.4f}")
    print(f"Alpha std: {alpha_result.std():.4f}")
    
    # Show some specific dates
    print(f"\nSample alpha values:")
    for i in [20, 40, 60, 80]:
        if i < len(alpha_result):
            print(f"  {dates[i].date()}: {alpha_result.iloc[i]:.4f}")
    
    return {
        'dates': dates,
        'close': close_prices,
        'volume': volume,
        'alpha07': alpha_result
    }

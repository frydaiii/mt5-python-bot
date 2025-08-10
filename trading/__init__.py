"""
Trading module for MT5 Trading Bot

This module contains trading utilities, portfolio management,
alpha, and related trading operations.
"""

# Import commonly used functions and classes for easy access
try:
    from .utils import (
        PortfolioManager,
        PortfolioAllocation,
        PositionInfo,
        rebalance_portfolio,
        get_portfolio_summary,
        validate_allocations
    )
    from .alpha import (
        alpha07,
        calculate_alpha07_from_dataframe,
        AlphaOperators
    )
except ImportError:
    # Handle import errors gracefully if dependencies are missing
    pass

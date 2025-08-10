"""
Test fixtures and mock data for MT5 trading bot tests.

This module contains reusable test data and fixtures that can be imported
by test modules.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


# ==================== SAMPLE DATA ====================

SAMPLE_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", 
    "EURJPY", "GBPJPY", "EURGBP", "AUDCAD", "NZDUSD"
]

SAMPLE_FOREX_PAIRS = {
    "EURUSD": {"base": "EUR", "quote": "USD", "digits": 5, "point": 0.00001, "typical_spread": 2},
    "GBPUSD": {"base": "GBP", "quote": "USD", "digits": 5, "point": 0.00001, "typical_spread": 3},
    "USDJPY": {"base": "USD", "quote": "JPY", "digits": 3, "point": 0.001, "typical_spread": 2},
    "AUDUSD": {"base": "AUD", "quote": "USD", "digits": 5, "point": 0.00001, "typical_spread": 3},
    "USDCAD": {"base": "USD", "quote": "CAD", "digits": 5, "point": 0.00001, "typical_spread": 3},
}

SAMPLE_PRICES = {
    "EURUSD": {"bid": 1.0850, "ask": 1.0852, "last": 1.0851},
    "GBPUSD": {"bid": 1.2650, "ask": 1.2653, "last": 1.2651},
    "USDJPY": {"bid": 149.50, "ask": 149.52, "last": 149.51},
    "AUDUSD": {"bid": 0.6750, "ask": 0.6753, "last": 0.6751},
    "USDCAD": {"bid": 1.3850, "ask": 1.3853, "last": 1.3851},
}

SAMPLE_PORTFOLIO_WEIGHTS = {
    "conservative": {
        "EURUSD": 0.40,
        "GBPUSD": 0.30,
        "USDJPY": 0.20,
        "AUDUSD": 0.10,
    },
    "balanced": {
        "EURUSD": 0.25,
        "GBPUSD": 0.25,
        "USDJPY": 0.20,
        "AUDUSD": 0.15,
        "USDCAD": 0.15,
    },
    "aggressive": {
        "EURUSD": 0.20,
        "GBPUSD": 0.20,
        "USDJPY": 0.20,
        "AUDUSD": 0.20,
        "USDCAD": 0.20,
    }
}


# ==================== DATA GENERATORS ====================

def generate_ohlc_data(
    symbol: str = "EURUSD",
    start_date: Optional[datetime] = None,
    periods: int = 100,
    freq: str = "H",
    base_price: float = 1.0850,
    volatility: float = 0.001,
    trend: float = 0.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic OHLC data for testing.
    
    Args:
        symbol: Trading symbol
        start_date: Start date for data (default: 30 days ago)
        periods: Number of periods to generate
        freq: Frequency ('M1', 'M5', 'H1', 'D1', etc.)
        base_price: Starting price
        volatility: Price volatility (standard deviation)
        trend: Trend factor (positive for uptrend, negative for downtrend)
        seed: Random seed for reproducible data
        
    Returns:
        DataFrame with OHLC data
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    data = []
    current_price = base_price
    
    for i, date in enumerate(dates):
        # Add trend and random walk
        price_change = np.random.normal(trend, volatility)
        current_price += price_change
        
        # Generate OHLC from current price
        volatility_factor = abs(np.random.normal(0, volatility * 0.5))
        
        open_price = current_price + np.random.normal(0, volatility * 0.3)
        high = max(open_price, current_price) + volatility_factor
        low = min(open_price, current_price) - volatility_factor
        close = current_price + np.random.normal(0, volatility * 0.2)
        
        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume
        volume = max(1, int(np.random.lognormal(5, 1)))
        
        data.append({
            "time": date,
            "open": round(open_price, 5),
            "high": round(high, 5),
            "low": round(low, 5), 
            "close": round(close, 5),
            "tick_volume": volume,
            "spread": 2,
            "real_volume": volume * 1000
        })
        
        current_price = close
    
    return pd.DataFrame(data)


def generate_tick_data(
    symbol: str = "EURUSD",
    start_time: Optional[datetime] = None,
    duration_minutes: int = 60,
    base_bid: float = 1.0850,
    spread_points: int = 2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic tick data for testing.
    
    Args:
        symbol: Trading symbol
        start_time: Start time for ticks
        duration_minutes: Duration in minutes
        base_bid: Base bid price
        spread_points: Spread in points
        seed: Random seed
        
    Returns:
        DataFrame with tick data
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(hours=1)
    
    np.random.seed(seed)
    
    # Generate random tick times
    num_ticks = duration_minutes * np.random.randint(10, 50)  # 10-50 ticks per minute
    tick_times = []
    current_time = start_time
    
    for _ in range(num_ticks):
        # Random interval between ticks (0.1 to 10 seconds)
        interval = timedelta(seconds=np.random.exponential(2))
        current_time += interval
        tick_times.append(current_time)
    
    # Generate tick prices
    ticks = []
    current_bid = base_bid
    point_value = SAMPLE_FOREX_PAIRS.get(symbol, {}).get("point", 0.00001)
    spread = spread_points * point_value
    
    for tick_time in tick_times:
        # Random price change
        price_change = np.random.normal(0, point_value * 5)
        current_bid += price_change
        
        ask = current_bid + spread
        last = current_bid + np.random.uniform(0, spread)
        
        ticks.append({
            "time": tick_time,
            "bid": round(current_bid, 5),
            "ask": round(ask, 5),
            "last": round(last, 5),
            "volume": np.random.randint(1, 100),
            "flags": 0
        })
    
    return pd.DataFrame(ticks)


def generate_trading_history(
    num_trades: int = 50,
    start_date: 'Optional[datetime]' = None,
    symbols: Optional[List[str]] = None,
    account_balance: float = 10000.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic trading history for testing.
    
    Args:
        num_trades: Number of trades to generate
        start_date: Start date for trades
        symbols: List of symbols to trade
        account_balance: Starting account balance
        seed: Random seed
        
    Returns:
        DataFrame with trading history
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)
    
    if symbols is None:
        symbols = SAMPLE_SYMBOLS[:5]
    
    np.random.seed(seed)
    
    trades = []
    current_balance = account_balance
    
    for i in range(num_trades):
        symbol = np.random.choice(symbols)
        trade_date = start_date + timedelta(
            days=np.random.randint(0, 90),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        # Trade parameters
        trade_type = np.random.choice([0, 1])  # 0=buy, 1=sell
        volume = round(np.random.uniform(0.01, 1.0), 2)
        entry_price = SAMPLE_PRICES.get(symbol, {"bid": 1.0})["bid"] * (
            1 + np.random.normal(0, 0.01)
        )
        
        # Exit parameters (simulate trade duration and outcome)
        duration_hours = np.random.exponential(24)  # Average 24 hours
        exit_date = trade_date + timedelta(hours=duration_hours)
        
        # Simulate profit/loss (60% win rate)
        is_winner = np.random.random() < 0.6
        if is_winner:
            profit_pips = np.random.uniform(10, 100)
        else:
            profit_pips = -np.random.uniform(5, 150)
        
        point_value = SAMPLE_FOREX_PAIRS.get(symbol, {}).get("point", 0.00001)
        profit = profit_pips * point_value * volume * 100000  # Standard lot size
        
        # Adjust for trade type
        if trade_type == 1:  # Sell position
            profit = -profit
        
        # Commission and swap
        commission = volume * np.random.uniform(3, 7)
        swap = np.random.uniform(-2, 1) * (duration_hours / 24)
        
        net_profit = profit - commission + swap
        current_balance += net_profit
        
        trades.append({
            "ticket": 100000 + i,
            "symbol": symbol,
            "type": trade_type,
            "volume": volume,
            "open_time": trade_date,
            "open_price": round(entry_price, 5),
            "close_time": exit_date,
            "close_price": round(entry_price + (profit_pips * point_value), 5),
            "profit": round(profit, 2),
            "commission": round(-commission, 2),
            "swap": round(swap, 2),
            "net_profit": round(net_profit, 2),
            "balance": round(current_balance, 2),
        })
    
    return pd.DataFrame(trades)


def generate_correlation_matrix(symbols: Optional[List[str]] = None, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic correlation matrix for currency pairs.
    
    Args:
        symbols: List of symbols
        seed: Random seed
        
    Returns:
        Correlation matrix DataFrame
    """
    if symbols is None:
        symbols = SAMPLE_SYMBOLS[:5]
    
    np.random.seed(seed)
    
    # Generate a positive semi-definite correlation matrix
    n = len(symbols)
    A = np.random.randn(n, n)
    corr_matrix = np.dot(A, A.T)
    
    # Normalize to correlation matrix
    diag = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(diag, diag)
    
    # Ensure diagonal is 1.0
    np.fill_diagonal(corr_matrix, 1.0)
    
    return pd.DataFrame(corr_matrix, index=symbols, columns=symbols)


# ==================== TEST SCENARIOS ====================

TEST_SCENARIOS = {
    "normal_market": {
        "volatility": 0.001,
        "trend": 0.0,
        "spread_multiplier": 1.0,
        "description": "Normal market conditions"
    },
    "high_volatility": {
        "volatility": 0.005,
        "trend": 0.0,
        "spread_multiplier": 2.0,
        "description": "High volatility market (news events)"
    },
    "trending_up": {
        "volatility": 0.001,
        "trend": 0.0002,
        "spread_multiplier": 1.0,
        "description": "Strong uptrend"
    },
    "trending_down": {
        "volatility": 0.001,
        "trend": -0.0002,
        "spread_multiplier": 1.0,
        "description": "Strong downtrend"
    },
    "low_liquidity": {
        "volatility": 0.003,
        "trend": 0.0,
        "spread_multiplier": 3.0,
        "description": "Low liquidity conditions"
    },
    "market_crash": {
        "volatility": 0.01,
        "trend": -0.001,
        "spread_multiplier": 5.0,
        "description": "Market crash scenario"
    }
}


# ==================== MOCK RESPONSES ====================

MOCK_MT5_RESPONSES = {
    "account_info": {
        "login": 12345,
        "balance": 10000.0,
        "equity": 10000.0,
        "margin": 0.0,
        "free_margin": 10000.0,
        "margin_level": 0.0,
        "profit": 0.0,
        "currency": "USD",
        "credit": 0.0,
        "margin_so_mode": 0,
        "margin_so_call": 50.0,
        "margin_so_so": 30.0,
        "margin_free_mode": 0,
        "server": "TestServer-Demo",
        "trade_allowed": True,
        "trade_expert": True,
        "limit_orders": 200,
        "margin_call": 50.0,
        "margin_stop_out": 30.0,
    },
    
    "order_send_success": {
        "retcode": 10009,  # TRADE_RETCODE_DONE
        "deal": 123456,
        "order": 123456,
        "volume": 0.1,
        "price": 1.0850,
        "bid": 1.0850,
        "ask": 1.0852,
        "comment": "Test order",
        "request_id": 1,
        "retcode_external": 0,
    },
    
    "order_send_failure": {
        "retcode": 10006,  # TRADE_RETCODE_INVALID
        "deal": 0,
        "order": 0,
        "volume": 0.0,
        "price": 0.0,
        "bid": 0.0,
        "ask": 0.0,
        "comment": "Invalid request",
        "request_id": 1,
        "retcode_external": 0,
    }
}


# ==================== UTILITY FUNCTIONS ====================

def get_test_scenario_data(scenario: str, symbol: str = "EURUSD", **kwargs) -> pd.DataFrame:
    """
    Get test data for a specific market scenario.
    
    Args:
        scenario: Scenario name from TEST_SCENARIOS
        symbol: Trading symbol
        **kwargs: Additional parameters for data generation
        
    Returns:
        DataFrame with scenario-specific data
    """
    if scenario not in TEST_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    scenario_params = TEST_SCENARIOS[scenario].copy()
    scenario_params.update(kwargs)
    
    return generate_ohlc_data(
        symbol=symbol,
        volatility=scenario_params["volatility"],
        trend=scenario_params["trend"],
        **{k: v for k, v in scenario_params.items() 
           if k in ["periods", "freq", "base_price", "seed"]}
    )


def create_test_portfolio(portfolio_type: str = "balanced") -> List[Dict[str, str | float]]:
    """
    Create a test portfolio allocation.
    
    Args:
        portfolio_type: Type of portfolio ("conservative", "balanced", "aggressive")
        
    Returns:
        List of portfolio allocations
    """
    if portfolio_type not in SAMPLE_PORTFOLIO_WEIGHTS:
        raise ValueError(f"Unknown portfolio type: {portfolio_type}")
    
    weights = SAMPLE_PORTFOLIO_WEIGHTS[portfolio_type]
    return [{"symbol": symbol, "weight": weight} for symbol, weight in weights.items()]

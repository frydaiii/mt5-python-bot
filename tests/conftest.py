"""
Pytest configuration and shared fixtures for MT5 trading bot tests.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Disable logging during tests to reduce noise
logging.disable(logging.CRITICAL)


@dataclass
class MockMT5SymbolInfo:
    """Mock MT5 symbol info structure."""
    name: str
    digits: int
    point: float
    spread: int
    trade_calc_mode: int = 0
    trade_mode: int = 4
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01


@dataclass
class MockMT5TickInfo:
    """Mock MT5 tick info structure."""
    time: int
    bid: float
    ask: float
    last: float
    volume: int = 1


@dataclass 
class MockMT5Position:
    """Mock MT5 position structure."""
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    time: int


@dataclass
class MockMT5AccountInfo:
    """Mock MT5 account info structure."""
    login: int = 12345
    balance: float = 10000.0
    equity: float = 10000.0
    margin: float = 0.0
    free_margin: float = 10000.0
    margin_level: float = 0.0
    profit: float = 0.0
    currency: str = "USD"


# ==================== FIXTURES ====================

@pytest.fixture
def sample_symbols():
    """Provide a list of sample trading symbols."""
    return ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]


@pytest.fixture
def sample_prices():
    """Provide sample price data for testing."""
    return {
        "EURUSD": {"bid": 1.0850, "ask": 1.0852},
        "GBPUSD": {"bid": 1.2650, "ask": 1.2652},
        "USDJPY": {"bid": 149.50, "ask": 149.52},
        "AUDUSD": {"bid": 0.6750, "ask": 0.6752},
        "USDCAD": {"bid": 1.3850, "ask": 1.3852},
    }


@pytest.fixture
def sample_ohlc_data():
    """Provide sample OHLC data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
    np.random.seed(42)  # For reproducible tests
    
    data = []
    base_price = 1.0850
    
    for i, date in enumerate(dates):
        open_price = base_price + np.random.normal(0, 0.001)
        high = open_price + abs(np.random.normal(0, 0.002))
        low = open_price - abs(np.random.normal(0, 0.002))
        close = open_price + np.random.normal(0, 0.001)
        volume = np.random.randint(100, 1000)
        
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
        
        base_price = close  # Next candle starts where this one ended
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_symbol_info():
    """Provide mock symbol information."""
    return {
        "EURUSD": MockMT5SymbolInfo(
            name="EURUSD",
            digits=5,
            point=0.00001,
            spread=2
        ),
        "GBPUSD": MockMT5SymbolInfo(
            name="GBPUSD", 
            digits=5,
            point=0.00001,
            spread=3
        ),
        "USDJPY": MockMT5SymbolInfo(
            name="USDJPY",
            digits=3,
            point=0.001,
            spread=2
        ),
    }


@pytest.fixture
def mock_account_info():
    """Provide mock account information."""
    return MockMT5AccountInfo()


@pytest.fixture
def mock_positions():
    """Provide mock position data."""
    return [
        MockMT5Position(
            ticket=123456,
            symbol="EURUSD",
            type=0,  # Buy
            volume=0.1,
            price_open=1.0850,
            price_current=1.0860,
            profit=10.0,
            swap=0.0,
            time=int(datetime.now().timestamp())
        ),
        MockMT5Position(
            ticket=123457,
            symbol="GBPUSD", 
            type=1,  # Sell
            volume=0.2,
            price_open=1.2650,
            price_current=1.2640,
            profit=20.0,
            swap=-1.5,
            time=int(datetime.now().timestamp())
        ),
    ]


@pytest.fixture
def portfolio_allocations():
    """Provide sample portfolio allocations."""
    from trading.utils import PortfolioAllocation
    return [
        PortfolioAllocation("EURUSD", 0.4),
        PortfolioAllocation("GBPUSD", 0.3),
        PortfolioAllocation("USDJPY", 0.2),
        PortfolioAllocation("AUDUSD", 0.1),
    ]


# ==================== MT5 MOCKING FIXTURES ====================

@pytest.fixture
def mock_mt5():
    """Provide a comprehensive mock of the MT5 module."""
    with patch('MetaTrader5') as mock:
        # Setup basic MT5 functions
        mock.initialize.return_value = True
        mock.shutdown.return_value = None
        mock.login.return_value = True
        mock.last_error.return_value = (0, "Success")
        
        # Account info
        mock.account_info.return_value = MockMT5AccountInfo()
        
        # Symbol operations
        mock.symbols_total.return_value = 100
        mock.symbols_get.return_value = ["EURUSD", "GBPUSD", "USDJPY"]
        
        # Position operations
        mock.positions_total.return_value = 0
        mock.positions_get.return_value = []
        
        # Trading operations
        mock.order_send.return_value = Mock(retcode=10009, order=123456)
        mock.order_close.return_value = Mock(retcode=10009)
        
        yield mock


@pytest.fixture
def mock_mt5_with_data(mock_mt5, sample_ohlc_data, mock_symbol_info, sample_prices):
    """Enhanced MT5 mock with realistic data responses."""
    
    def mock_symbol_info_get(symbol):
        return mock_symbol_info.get(symbol)
    
    def mock_symbol_info_tick(symbol):
        if symbol in sample_prices:
            return MockMT5TickInfo(
                time=int(datetime.now().timestamp()),
                bid=sample_prices[symbol]["bid"],
                ask=sample_prices[symbol]["ask"],
                last=sample_prices[symbol]["bid"]
            )
        return None
    
    def mock_copy_rates_from(symbol, timeframe, date_from, count):
        # Return a subset of sample data
        end_idx = min(count, len(sample_ohlc_data))
        return sample_ohlc_data.iloc[:end_idx].to_records(index=False)
    
    def mock_copy_rates_range(symbol, timeframe, date_from, date_to):
        return sample_ohlc_data.to_records(index=False)
    
    # Configure the mock responses
    mock_mt5.symbol_info.side_effect = mock_symbol_info_get
    mock_mt5.symbol_info_tick.side_effect = mock_symbol_info_tick
    mock_mt5.copy_rates_from.side_effect = mock_copy_rates_from
    mock_mt5.copy_rates_range.side_effect = mock_copy_rates_range
    
    return mock_mt5


# ==================== UTILITY FIXTURES ====================

@pytest.fixture
def temp_config():
    """Provide temporary configuration for testing."""
    return {
        "MT5_LOGIN": 12345,
        "MT5_PASSWORD": "test_password",
        "MT5_SERVER": "TestServer-Demo",
        "MT5_PATH": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        "LOG_LEVEL": "DEBUG",
        "DEFAULT_SYMBOLS": ["EURUSD", "GBPUSD", "USDJPY"],
        "DEFAULT_TIMEFRAME": "H1",
        "MAX_RISK_PER_TRADE": 0.02,
        "MAX_PORTFOLIO_RISK": 0.10,
    }


@pytest.fixture
def mock_datetime():
    """Provide a mock datetime for consistent testing."""
    test_datetime = datetime(2024, 1, 15, 12, 0, 0)
    with patch('datetime.datetime') as mock_dt:
        mock_dt.now.return_value = test_datetime
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield test_datetime


# ==================== TEST HELPERS ====================

@pytest.fixture
def assert_dataframe_equal():
    """Provide a helper function to assert DataFrame equality."""
    def _assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs):
        """Assert that two DataFrames are equal with better error messages."""
        try:
            pd.testing.assert_frame_equal(df1, df2, **kwargs)
        except AssertionError as e:
            # Add more context to the error
            print(f"DataFrame 1 shape: {df1.shape}")
            print(f"DataFrame 2 shape: {df2.shape}")
            print(f"DataFrame 1 columns: {list(df1.columns)}")
            print(f"DataFrame 2 columns: {list(df2.columns)}")
            raise e
    
    return _assert_dataframe_equal


@pytest.fixture
def assert_portfolio_allocation():
    """Provide a helper to validate portfolio allocations."""
    def _assert_portfolio_allocation(allocations: List[Any], tolerance: float = 1e-6):
        """Assert that portfolio allocations sum to approximately 1.0."""
        total_weight = sum(alloc.weight for alloc in allocations)
        assert abs(total_weight - 1.0) <= tolerance, \
            f"Portfolio weights sum to {total_weight}, expected ~1.0"
        
        for alloc in allocations:
            assert 0 <= alloc.weight <= 1, \
                f"Invalid weight {alloc.weight} for {alloc.symbol}"
    
    return _assert_portfolio_allocation


# ==================== PYTEST HOOKS ====================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "mt5: mark test as requiring MT5 connection"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Auto-mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Auto-mark integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark tests that use real MT5 connection
        if hasattr(item, 'fixturenames') and 'real_mt5' in item.fixturenames:
            item.add_marker(pytest.mark.mt5)


# Re-enable logging after tests
def pytest_unconfigure(config):
    """Re-enable logging after test session."""
    logging.disable(logging.NOTSET)

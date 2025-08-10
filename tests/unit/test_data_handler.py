"""
Unit tests for data_handler.mt5_data module.

This module contains comprehensive unit tests for MT5 data handling functions.
Tests are organized by functionality and use mocked MT5 responses.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import the modules to test
from data_handler.mt5_data import (
    initialize_mt5,
    get_symbol_info,
    get_current_price,
    # Add other functions as needed
)


class TestMT5Initialization:
    """Test cases for MT5 initialization and connection."""
    
    def test_initialize_mt5_success(self, mock_mt5, temp_config):
        """Test successful MT5 initialization."""
        # Arrange
        mock_mt5.initialize.return_value = True
        
        # Act
        result = initialize_mt5()
        
        # Assert
        assert result is True
        mock_mt5.initialize.assert_called_once()
    
    def test_initialize_mt5_failure(self, mock_mt5):
        """Test MT5 initialization failure."""
        # Arrange
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = (1, "Connection failed")
        
        # Act
        result = initialize_mt5()
        
        # Assert
        assert result is False
        mock_mt5.initialize.assert_called_once()
    
    def test_initialize_mt5_with_custom_credentials(self, mock_mt5):
        """Test MT5 initialization with custom credentials."""
        # Arrange
        mock_mt5.initialize.return_value = True
        login = 12345
        password = "test_pass"
        server = "TestServer"
        path = "C:\\MetaTrader\\terminal.exe"
        
        # Act
        result = initialize_mt5(login=login, password=password, server=server, path=path)
        
        # Assert
        assert result is True
        mock_mt5.initialize.assert_called_once_with(
            path=path, login=login, password=password, server=server
        )
    
    @pytest.mark.parametrize("error_code,error_msg", [
        (1, "Connection failed"),
        (2, "Invalid credentials"), 
        (3, "Server not found"),
    ])
    def test_initialize_mt5_various_errors(self, mock_mt5, error_code, error_msg):
        """Test MT5 initialization with various error codes."""
        # Arrange
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = (error_code, error_msg)
        
        # Act
        result = initialize_mt5()
        
        # Assert
        assert result is False


class TestSymbolOperations:
    """Test cases for symbol-related operations."""
    
    def test_get_symbol_info_success(self, mock_mt5_with_data, mock_symbol_info):
        """Test successful symbol info retrieval."""
        # Act
        result = get_symbol_info("EURUSD")
        
        # Assert
        assert result is not None
        assert result["name"] == "EURUSD"
        assert result["digits"] == 5
        assert result["point"] == 0.00001
    
    def test_get_symbol_info_invalid_symbol(self, mock_mt5):
        """Test symbol info retrieval for invalid symbol."""
        # Arrange
        mock_mt5.symbol_info.return_value = None
        
        # Act
        result = get_symbol_info("INVALID")
        
        # Assert
        assert result is None


class TestPriceData:
    """Test cases for price data retrieval."""
    
    def test_get_current_price_success(self, mock_mt5_with_data, sample_prices):
        """Test successful current price retrieval."""
        # Act
        result = get_current_price("EURUSD")
        
        # Assert
        assert result is not None
        assert "bid" in result
        assert "ask" in result
        assert result["bid"] == sample_prices["EURUSD"]["bid"]
        assert result["ask"] == sample_prices["EURUSD"]["ask"]
    
    def test_get_current_price_invalid_symbol(self, mock_mt5):
        """Test current price retrieval for invalid symbol."""
        # Arrange
        mock_mt5.symbol_info_tick.return_value = None
        
        # Act
        result = get_current_price("INVALID")
        
        # Assert
        assert result is None
    
    @pytest.mark.parametrize("symbol,expected_bid", [
        ("EURUSD", 1.0850),
        ("GBPUSD", 1.2650),
        ("USDJPY", 149.50),
    ])
    def test_get_current_price_multiple_symbols(
        self, mock_mt5_with_data, sample_prices, symbol, expected_bid
    ):
        """Test current price retrieval for multiple symbols."""
        # Act
        result = get_current_price(symbol)
        
        # Assert
        assert result is not None
        assert result["bid"] == expected_bid


class TestErrorHandling:
    """Test cases for error handling and edge cases."""
    
    def test_mt5_connection_lost(self, mock_mt5):
        """Test handling of lost MT5 connection."""
        # Arrange
        mock_mt5.symbol_info_tick.side_effect = Exception("Connection lost")
        
        # Act & Assert
        with pytest.raises(Exception):
            get_current_price("EURUSD")
    
    def test_empty_symbol_string(self, mock_mt5):
        """Test handling of empty symbol string."""
        # Act
        result = get_current_price("")
        
        # Assert
        assert result is None


class TestPerformance:
    """Test cases for performance and optimization."""
    
    def test_concurrent_price_requests(self, mock_mt5_with_data, sample_symbols):
        """Test concurrent price requests for multiple symbols."""
        import concurrent.futures
        
        # Act
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(get_current_price, symbol) 
                for symbol in sample_symbols
            ]
            results = [future.result() for future in futures]
        
        # Assert
        assert len(results) == len(sample_symbols)
        assert all(result is not None for result in results)


# ==================== FIXTURES FOR THIS MODULE ====================

@pytest.fixture
def sample_market_hours():
    """Provide sample market hours data."""
    return {
        "forex": {
            "monday": {"open": "00:00", "close": "24:00"},
            "tuesday": {"open": "00:00", "close": "24:00"},
            "wednesday": {"open": "00:00", "close": "24:00"},
            "thursday": {"open": "00:00", "close": "24:00"},
            "friday": {"open": "00:00", "close": "22:00"},
            "saturday": {"open": None, "close": None},
            "sunday": {"open": "22:00", "close": "24:00"},
        }
    }


# ==================== CUSTOM MARKERS ====================

pytestmark = [
    pytest.mark.unit,
    pytest.mark.data_handler,
]

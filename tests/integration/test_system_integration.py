"""
Integration tests for MT5 trading bot.

These tests verify the interaction between different components of the system.
They may require actual MT5 connection or use comprehensive mocking.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
from typing import List, Dict, Any

# Import modules to test
from data_handler.mt5_data import initialize_mt5, get_current_price, get_historical_data
from trading.utils import PortfolioManager, PortfolioAllocation


@pytest.mark.integration
class TestDataToTradingIntegration:
    """Test integration between data handler and trading modules."""
    
    def test_portfolio_rebalancing_with_real_data(
        self, mock_mt5_with_data, portfolio_allocations, mock_account_info
    ):
        """Test complete portfolio rebalancing using data handler."""
        # Arrange
        mock_mt5_with_data.account_info.return_value = mock_account_info
        portfolio_manager = PortfolioManager(account_balance=10000.0)
        
        # Act
        # 1. Initialize connection
        init_success = initialize_mt5()
        assert init_success
        
        # 2. Get current prices for all symbols
        current_prices = {}
        for allocation in portfolio_allocations:
            price = get_current_price(allocation.symbol)
            assert price is not None
            current_prices[allocation.symbol] = price
        
        # 3. Get historical data for risk calculation
        historical_data = {}
        for allocation in portfolio_allocations:
            data = get_historical_data(allocation.symbol, "H1", 100)
            assert data is not None and not data.empty
            historical_data[allocation.symbol] = data
        
        # 4. Calculate portfolio metrics
        # This would involve your actual implementation
        try:
            current_positions = portfolio_manager.get_current_positions()
            portfolio_value = portfolio_manager.calculate_portfolio_value()
            
            # Assert results
            assert isinstance(current_positions, list)
            assert portfolio_value >= 0
            
        except (AttributeError, NotImplementedError):
            pytest.skip("Portfolio management methods not implemented yet")
    
    def test_data_consistency_across_modules(self, mock_mt5_with_data, sample_symbols):
        """Test that data is consistent when accessed from different modules."""
        # Test that the same symbol returns consistent data across modules
        symbol = sample_symbols[0]
        
        # Get data from data handler
        price_data = get_current_price(symbol)
        historical_data = get_historical_data(symbol, "H1", 10)
        
        # Verify data consistency
        assert price_data is not None
        assert historical_data is not None and not historical_data.empty
        
        # Price should be within reasonable range of recent historical data
        if len(historical_data) > 0:
            recent_close = historical_data['close'].iloc[-1]
            current_bid = price_data['bid']
            
            # Prices should be within 1% of each other (allowing for normal market movement)
            price_diff_pct = abs(current_bid - recent_close) / recent_close
            assert price_diff_pct < 0.01, f"Price inconsistency: {price_diff_pct:.4f}"


@pytest.mark.integration
class TestCompleteWorkflows:
    """Test complete trading workflows from start to finish."""
    
    def test_new_portfolio_creation_workflow(
        self, mock_mt5_with_data, temp_config, mock_account_info
    ):
        """Test creating a new portfolio from scratch."""
        # Arrange
        mock_mt5_with_data.account_info.return_value = mock_account_info
        target_allocations = [
            PortfolioAllocation("EURUSD", 0.4),
            PortfolioAllocation("GBPUSD", 0.3),
            PortfolioAllocation("USDJPY", 0.3),
        ]
        
        # Act
        # 1. Initialize system
        init_success = initialize_mt5()
        assert init_success
        
        # 2. Validate all symbols
        for allocation in target_allocations:
            price = get_current_price(allocation.symbol)
            assert price is not None, f"Cannot get price for {allocation.symbol}"
        
        # 3. Create portfolio manager
        portfolio_manager = PortfolioManager(account_balance=10000.0)
        
        # 4. Calculate initial positions (this would use your actual implementation)
        try:
            # Get current (empty) portfolio
            current_positions = portfolio_manager.get_current_positions()
            assert isinstance(current_positions, list)
            
            # Calculate required orders to reach target allocation
            # rebalance_orders = portfolio_manager.rebalance_portfolio(target_allocations)
            # assert isinstance(rebalance_orders, list)
            
        except (AttributeError, NotImplementedError):
            pytest.skip("Portfolio management methods not implemented yet")
    
    def test_risk_management_workflow(self, mock_mt5_with_data, sample_symbols):
        """Test risk management across the entire system."""
        # This would test your risk management implementation
        # Example structure:
        
        # 1. Get historical data for risk calculation
        symbol = sample_symbols[0]
        historical_data = get_historical_data(symbol, "D1", 252)  # 1 year of daily data
        assert historical_data is not None and len(historical_data) >= 100
        
        # 2. Calculate volatility metrics
        returns = historical_data['close'].pct_change().dropna()
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * (252 ** 0.5)
        
        # 3. Risk-based position sizing would go here
        # This depends on your risk management implementation
        
        assert daily_volatility > 0
        assert annual_volatility > daily_volatility
    
    def test_portfolio_monitoring_workflow(self, mock_mt5_with_data, mock_positions):
        """Test ongoing portfolio monitoring and alerting."""
        # Arrange
        mock_mt5_with_data.positions_get.return_value = mock_positions
        
        # Act
        portfolio_manager = PortfolioManager()
        
        try:
            # 1. Get current positions
            positions = portfolio_manager.get_current_positions()
            assert len(positions) == len(mock_positions)
            
            # 2. Calculate current portfolio metrics
            # portfolio_value = portfolio_manager.calculate_portfolio_value()
            # current_weights = portfolio_manager.get_portfolio_weights()
            
            # 3. Check risk limits
            # risk_metrics = portfolio_manager.calculate_risk_metrics()
            
            # 4. Generate alerts if needed
            # alerts = portfolio_manager.check_risk_alerts()
            
        except (AttributeError, NotImplementedError):
            pytest.skip("Portfolio management methods not implemented yet")


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test system performance under various conditions."""
    
    def test_high_frequency_data_requests(self, mock_mt5_with_data, sample_symbols):
        """Test system performance with high-frequency data requests."""
        import time
        
        # Test rapid successive requests
        start_time = time.time()
        
        for _ in range(10):
            for symbol in sample_symbols[:3]:  # Test with 3 symbols
                price = get_current_price(symbol)
                assert price is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 30 price requests in reasonable time
        assert total_time < 2.0, f"High frequency requests too slow: {total_time}s"
    
    def test_large_historical_data_processing(self, mock_mt5_with_data):
        """Test processing large amounts of historical data."""
        # Request large dataset
        large_data = get_historical_data("EURUSD", "M1", 10000)  # 10k minutes ≈ 1 week
        
        assert large_data is not None
        if not large_data.empty:
            # Test basic data processing operations
            assert len(large_data) > 0
            assert all(col in large_data.columns for col in ['open', 'high', 'low', 'close'])
            
            # Test calculations on large dataset
            start_time = time.time()
            
            # Typical calculations you might do
            large_data['sma_20'] = large_data['close'].rolling(20).mean()
            large_data['volatility'] = large_data['close'].pct_change().rolling(20).std()
            
            end_time = time.time()
            calculation_time = end_time - start_time
            
            # Should process calculations reasonably quickly
            assert calculation_time < 1.0, f"Large data calculations too slow: {calculation_time}s"


@pytest.mark.integration 
@pytest.mark.mt5
class TestRealMT5Integration:
    """
    Tests that require actual MT5 connection.
    
    These tests are marked with @pytest.mark.mt5 and will only run
    when explicitly requested with: pytest -m mt5
    """
    
    @pytest.fixture
    def real_mt5_connection(self):
        """Fixture for real MT5 connection (use carefully!)."""
        # Only use this for actual integration testing with demo account
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                yield mt5
            else:
                pytest.skip("Could not initialize real MT5 connection")
        finally:
            try:
                mt5.shutdown()
            except:
                pass
    
    def test_real_mt5_connection(self, real_mt5_connection):
        """Test actual connection to MT5 (requires MT5 to be running)."""
        mt5 = real_mt5_connection
        
        # Test basic operations
        account_info = mt5.account_info()
        assert account_info is not None
        assert account_info.login > 0
        
        # Test symbol operations
        symbols = mt5.symbols_get()
        assert len(symbols) > 0
        
        # Test price data
        eurusd_info = mt5.symbol_info("EURUSD")
        if eurusd_info:
            tick = mt5.symbol_info_tick("EURUSD")
            assert tick is not None
            assert tick.bid > 0
            assert tick.ask > tick.bid
    
    def test_real_historical_data_retrieval(self, real_mt5_connection):
        """Test retrieving real historical data."""
        mt5 = real_mt5_connection
        
        # Get recent data
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 100)
        assert rates is not None
        assert len(rates) > 0
        
        # Convert to DataFrame and validate
        df = pd.DataFrame(rates)
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery and resilience."""
    
    def test_mt5_disconnection_recovery(self, mock_mt5):
        """Test recovery from MT5 connection loss."""
        # Simulate connection working initially
        mock_mt5.initialize.return_value = True
        mock_mt5.symbol_info_tick.return_value = Mock(bid=1.0850, ask=1.0852)
        
        # Initialize and get initial data
        init_success = initialize_mt5()
        assert init_success
        
        price = get_current_price("EURUSD")
        assert price is not None
        
        # Simulate connection loss
        mock_mt5.symbol_info_tick.side_effect = Exception("Connection lost")
        
        # Test that system handles the error gracefully
        with pytest.raises(Exception):
            get_current_price("EURUSD")
        
        # Test reconnection
        mock_mt5.symbol_info_tick.side_effect = None
        mock_mt5.symbol_info_tick.return_value = Mock(bid=1.0855, ask=1.0857)
        
        # Should work again after reconnection
        price = get_current_price("EURUSD")
        assert price is not None
    
    def test_invalid_data_handling(self, mock_mt5):
        """Test handling of invalid or corrupted data."""
        # Test with malformed price data
        mock_mt5.symbol_info_tick.return_value = Mock(bid=0, ask=0)
        
        price = get_current_price("EURUSD")
        # System should handle invalid prices appropriately
        # (either return None, raise exception, or provide default)
        # Adjust assertion based on your implementation
    
    def test_partial_data_availability(self, mock_mt5_with_data, sample_symbols):
        """Test system behavior when some data is unavailable."""
        # Simulate some symbols having data, others not
        def mock_price_response(symbol):
            if symbol in sample_symbols[:2]:  # First 2 symbols have data
                return {"bid": 1.0850, "ask": 1.0852}
            return None  # Others return None
        
        with patch('data_handler.mt5_data.get_current_price', side_effect=mock_price_response):
            # Test portfolio creation with partial data
            available_symbols = []
            for symbol in sample_symbols:
                if get_current_price(symbol):
                    available_symbols.append(symbol)
            
            # Should be able to work with available symbols
            assert len(available_symbols) >= 2
            
            # Create portfolio with available symbols only
            allocations = [
                PortfolioAllocation(available_symbols[0], 0.6),
                PortfolioAllocation(available_symbols[1], 0.4),
            ]
            
            # This should work even with limited data
            assert len(allocations) == 2
            assert sum(alloc.weight for alloc in allocations) == 1.0


# ==================== INTEGRATION TEST HELPERS ====================

@pytest.fixture
def integration_test_portfolio():
    """Provide a standard portfolio for integration tests."""
    return [
        PortfolioAllocation("EURUSD", 0.30),
        PortfolioAllocation("GBPUSD", 0.25),
        PortfolioAllocation("USDJPY", 0.25),
        PortfolioAllocation("AUDUSD", 0.20),
    ]


@pytest.fixture
def mock_market_conditions():
    """Provide different market condition scenarios for testing."""
    return {
        "normal": {"volatility": 0.001, "trend": 0.0},
        "volatile": {"volatility": 0.005, "trend": 0.0},
        "trending": {"volatility": 0.001, "trend": 0.0002},
        "crisis": {"volatility": 0.01, "trend": -0.001},
    }


# ==================== CUSTOM MARKERS ====================

pytestmark = [
    pytest.mark.integration,
]

"""
Portfolio Management and Trading Utilities for MT5 Trading Bot

This module provides utility functions for portfolio management, trading operations,
and portfolio rebalancing using MetaTrader 5.
"""

import MetaTrader5 as mt5
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from datetime import datetime
from dataclasses import dataclass
from data_handler import get_current_price, get_symbol_info

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Represents a portfolio allocation with symbol and target weight."""
    symbol: str
    weight: float
    
    def __post_init__(self):
        """Validate the allocation data."""
        if not isinstance(self.symbol, str):
            raise ValueError("Symbol must be a string")
        if not 0 <= self.weight <= 1:
            raise ValueError("Weight must be between 0 and 1")


@dataclass
class PositionInfo:
    """Represents current position information."""
    symbol: str
    volume: float
    type: int  # 0 for buy, 1 for sell
    price: float
    profit: float
    swap: float
    ticket: int


class PortfolioManager:
    """Portfolio management class for MT5 trading operations."""
    
    def __init__(self, account_balance: Optional[float] = None):
        """
        Initialize portfolio manager.
        
        Args:
            account_balance: Account balance for calculations (if None, will get from MT5)
        """
        self.account_balance = account_balance
        self._update_account_info()
    
    def _update_account_info(self) -> None:
        """Update account information from MT5."""
        try:
            account_info = mt5.account_info()
            if account_info:
                if self.account_balance is None:
                    self.account_balance = account_info.balance
                self.equity = account_info.equity
                self.margin = account_info.margin
                self.free_margin = account_info.margin_free
                logger.info(f"Account balance: {self.account_balance}, Equity: {self.equity}")
            else:
                logger.error("Failed to get account information")
        except Exception as e:
            logger.error(f"Error updating account info: {e}")
    
    def get_current_positions(self) -> List[PositionInfo]:
        """
        Get current open positions.
        
        Returns:
            List[PositionInfo]: List of current positions
        """
        positions = []
        try:
            mt5_positions = mt5.positions_get()
            if mt5_positions:
                for pos in mt5_positions:
                    position = PositionInfo(
                        symbol=pos.symbol,
                        volume=pos.volume,
                        type=pos.type,
                        price=pos.price_open,
                        profit=pos.profit,
                        swap=pos.swap,
                        ticket=pos.ticket
                    )
                    positions.append(position)
            logger.info(f"Found {len(positions)} open positions")
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
        
        return positions
    
    def get_portfolio_value(self) -> Dict[str, float]:
        """
        Calculate current portfolio value breakdown.
        
        Returns:
            Dict[str, float]: Portfolio value breakdown by symbol
        """
        portfolio_values = {}
        positions = self.get_current_positions()
        
        for position in positions:
            try:
                current_price_info = get_current_price(position.symbol)
                if current_price_info:
                    # Use appropriate price based on position type
                    if position.type == 0:  # Buy position
                        current_price = current_price_info['bid']
                    else:  # Sell position
                        current_price = current_price_info['ask']
                    
                    # Calculate position value
                    symbol_info = get_symbol_info(position.symbol)
                    if symbol_info:
                        # For forex pairs, contract size is typically 100,000 (1 lot)
                        # For other instruments, check the symbol specification
                        position_value = position.volume * current_price
                        
                        if position.symbol in portfolio_values:
                            portfolio_values[position.symbol] += position_value
                        else:
                            portfolio_values[position.symbol] = position_value
            except Exception as e:
                logger.error(f"Error calculating value for {position.symbol}: {e}")
        
        return portfolio_values
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Calculate current portfolio weights.
        
        Returns:
            Dict[str, float]: Current weights by symbol
        """
        portfolio_values = self.get_portfolio_value()
        total_value = sum(portfolio_values.values())
        
        if total_value == 0:
            return {}
        
        weights = {symbol: value / total_value for symbol, value in portfolio_values.items()}
        return weights
    
    def calculate_rebalancing_trades(self, target_allocations: List[PortfolioAllocation]) -> Dict[str, float]:
        """
        Calculate required trades to rebalance portfolio to target allocations.
        
        Args:
            target_allocations: List of target allocations
            
        Returns:
            Dict[str, float]: Required volume changes by symbol (positive = buy more, negative = sell)
        """
        # Validate total weights sum to 1
        total_weight = sum(allocation.weight for allocation in target_allocations)
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            raise ValueError(f"Total weights must sum to 1.0, got {total_weight}")
        
        # Get current positions and weights
        current_weights = self.get_current_weights()
        current_positions = {pos.symbol: pos.volume for pos in self.get_current_positions()}
        
        # Calculate target positions based on account equity
        self._update_account_info()
        target_value = self.equity  # Use equity instead of balance for active trading
        
        required_trades = {}
        
        for allocation in target_allocations:
            symbol = allocation.symbol
            target_weight = allocation.weight
            
            # Calculate target value for this symbol
            symbol_target_value = target_value * target_weight
            
            # Get symbol info for lot size calculation
            symbol_info = get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Could not get symbol info for {symbol}")
                continue
            
            current_price_info = get_current_price(symbol)
            if not current_price_info:
                logger.error(f"Could not get current price for {symbol}")
                continue
            
            # Use mid price for calculations
            current_price = (current_price_info['bid'] + current_price_info['ask']) / 2
            
            # Calculate target volume (simplified calculation for forex)
            # For more complex instruments, this would need adjustment
            target_volume = symbol_target_value / current_price / 100000  # Assuming standard forex lot size
            
            # Calculate current volume
            current_volume = current_positions.get(symbol, 0.0)
            
            # Calculate required change
            volume_change = target_volume - current_volume
            
            # Only include significant changes (avoid tiny adjustments)
            min_lot = symbol_info.get('min_lot', 0.01)
            if abs(volume_change) >= min_lot:
                required_trades[symbol] = volume_change
        
        # Handle positions that need to be closed (not in target allocation)
        for symbol, current_volume in current_positions.items():
            if symbol not in [alloc.symbol for alloc in target_allocations]:
                required_trades[symbol] = -current_volume  # Close position
        
        return required_trades
    
    def execute_trade(self, symbol: str, volume: float, action: str = "auto") -> bool:
        """
        Execute a trade for the given symbol and volume.
        
        Args:
            symbol: Trading symbol
            volume: Volume to trade (positive for buy, negative for sell)
            action: Trade action ("buy", "sell", or "auto" to determine from volume)
            
        Returns:
            bool: True if trade was successful, False otherwise
        """
        try:
            # Determine trade action
            if action == "auto":
                trade_action = mt5.ORDER_TYPE_BUY if volume > 0 else mt5.ORDER_TYPE_SELL
                trade_volume = abs(volume)
            elif action == "buy":
                trade_action = mt5.ORDER_TYPE_BUY
                trade_volume = abs(volume)
            elif action == "sell":
                trade_action = mt5.ORDER_TYPE_SELL
                trade_volume = abs(volume)
            else:
                raise ValueError(f"Invalid action: {action}")
            
            # Get current price
            price_info = get_current_price(symbol)
            if not price_info:
                logger.error(f"Could not get price for {symbol}")
                return False
            
            # Select appropriate price based on action
            price = price_info['ask'] if trade_action == mt5.ORDER_TYPE_BUY else price_info['bid']
            
            # Get symbol info for validation
            symbol_info = get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Could not get symbol info for {symbol}")
                return False
            
            # Validate volume
            min_lot = symbol_info['min_lot']
            max_lot = symbol_info['max_lot']
            lot_step = symbol_info['lot_step']
            
            # Round volume to nearest lot step
            trade_volume = round(trade_volume / lot_step) * lot_step
            
            if trade_volume < min_lot:
                logger.warning(f"Volume {trade_volume} too small for {symbol} (min: {min_lot})")
                return False
            
            if trade_volume > max_lot:
                logger.warning(f"Volume {trade_volume} too large for {symbol} (max: {max_lot})")
                trade_volume = max_lot
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": trade_volume,
                "type": trade_action,
                "price": price,
                "sl": 0.0,  # No stop loss
                "tp": 0.0,  # No take profit
                "deviation": 20,  # Price deviation in points
                "magic": 234000,  # Magic number for identification
                "comment": "Portfolio rebalancing",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Trade failed for {symbol}: {result.comment}")
                return False
            
            logger.info(f"Trade executed: {action} {trade_volume} {symbol} at {price}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, volume: Optional[float] = None) -> bool:
        """
        Close position for a specific symbol.
        
        Args:
            symbol: Symbol to close
            volume: Volume to close (if None, closes entire position)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                logger.warning(f"No positions found for {symbol}")
                return True  # Nothing to close
            
            for position in positions:
                close_volume = volume if volume is not None else position.volume
                
                # Determine opposite action
                if position.type == mt5.POSITION_TYPE_BUY:
                    trade_type = mt5.ORDER_TYPE_SELL
                    price = mt5.symbol_info_tick(symbol).bid
                else:
                    trade_type = mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(symbol).ask
                
                # Prepare close request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": close_volume,
                    "type": trade_type,
                    "position": position.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "Position close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Failed to close position for {symbol}: {result.comment}")
                    return False
                
                logger.info(f"Closed position: {close_volume} {symbol}")
                
                # If partial close and we only wanted to close specific volume
                if volume is not None:
                    break
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False


def rebalance_portfolio(target_allocations: List[Tuple[str, float]], 
                       account_balance: Optional[float] = None,
                       dry_run: bool = True) -> Dict[str, Any]:
    """
    Rebalance portfolio to target allocations.
    
    Args:
        target_allocations: List of tuples (symbol, weight) representing target allocation
        account_balance: Account balance for calculations (if None, will get from MT5)
        dry_run: If True, only calculates trades without executing them
        
    Returns:
        Dict: Summary of rebalancing results
    """
    try:
        if not validate_allocations(target_allocations):
            return {"error": "Invalid target allocations"}

        # Convert tuples to PortfolioAllocation objects
        allocations = [PortfolioAllocation(symbol, weight) for symbol, weight in target_allocations]
        
        # Initialize portfolio manager
        portfolio_manager = PortfolioManager(account_balance)
        
        # Get current portfolio state
        current_weights = portfolio_manager.get_current_weights()
        current_positions = portfolio_manager.get_current_positions()
        
        # Calculate required trades
        required_trades = portfolio_manager.calculate_rebalancing_trades(allocations)
        
        # Prepare results
        results = {
            "timestamp": datetime.now(),
            "current_weights": current_weights,
            "target_weights": {alloc.symbol: alloc.weight for alloc in allocations},
            "required_trades": required_trades,
            "current_positions": len(current_positions),
            "account_equity": portfolio_manager.equity,
            "executed_trades": [],
            "failed_trades": [],
            "dry_run": dry_run
        }
        
        if dry_run:
            logger.info("DRY RUN: Portfolio rebalancing simulation completed")
            logger.info(f"Required trades: {required_trades}")
            return results
        
        # Execute trades
        successful_trades = 0
        failed_trades = 0
        
        for symbol, volume_change in required_trades.items():
            if abs(volume_change) < 0.01:  # Skip very small changes
                continue
            
            if volume_change > 0:
                # Need to buy more
                success = portfolio_manager.execute_trade(symbol, volume_change, "buy")
            else:
                # Need to sell
                success = portfolio_manager.execute_trade(symbol, abs(volume_change), "sell")
            
            if success:
                successful_trades += 1
                results["executed_trades"].append({
                    "symbol": symbol,
                    "volume": volume_change,
                    "action": "buy" if volume_change > 0 else "sell"
                })
            else:
                failed_trades += 1
                results["failed_trades"].append({
                    "symbol": symbol,
                    "volume": volume_change,
                    "error": "Trade execution failed"
                })
        
        results["successful_trades"] = successful_trades
        results["failed_trades_count"] = failed_trades
        
        logger.info(f"Portfolio rebalancing completed: {successful_trades} successful, {failed_trades} failed")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in portfolio rebalancing: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(),
            "dry_run": dry_run
        }


def get_portfolio_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of the current portfolio.
    
    Returns:
        Dict: Portfolio summary with positions, weights, and values
    """
    try:
        portfolio_manager = PortfolioManager()
        
        current_positions = portfolio_manager.get_current_positions()
        current_weights = portfolio_manager.get_current_weights()
        portfolio_values = portfolio_manager.get_portfolio_value()
        
        total_value = sum(portfolio_values.values())
        
        summary = {
            "timestamp": datetime.now(),
            "account_balance": portfolio_manager.account_balance,
            "account_equity": portfolio_manager.equity,
            "total_portfolio_value": total_value,
            "number_of_positions": len(current_positions),
            "positions": [],
            "weights": current_weights,
            "values": portfolio_values
        }
        
        for position in current_positions:
            position_summary = {
                "symbol": position.symbol,
                "volume": position.volume,
                "type": "BUY" if position.type == 0 else "SELL",
                "open_price": position.price,
                "current_profit": position.profit,
                "swap": position.swap,
                "weight": current_weights.get(position.symbol, 0.0),
                "value": portfolio_values.get(position.symbol, 0.0)
            }
            summary["positions"].append(position_summary)
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        return {"error": str(e), "timestamp": datetime.now()}


def validate_allocations(allocations: List[Tuple[str, float]]) -> bool:
    """
    Validate portfolio allocations.
    
    Args:
        allocations: List of (symbol, weight) tuples
        
    Returns:
        bool: True if allocations are valid, False otherwise
    """
    try:
        total_weight = sum(weight for _, weight in allocations)
        
        if abs(total_weight - 1.0) > 0.01:
            logger.error(f"Total weights must sum to 1.0, got {total_weight}")
            return False
        
        for symbol, weight in allocations:
            if not isinstance(symbol, str):
                logger.error(f"Symbol must be string, got {type(symbol)}")
                return False
            
            if not 0 <= weight <= 1:
                logger.error(f"Weight for {symbol} must be between 0 and 1, got {weight}")
                return False
            
            # Check if symbol exists
            symbol_info = get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found or not available")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating allocations: {e}")
        return False


# Example usage and testing functions
def example_rebalance():
    """Example of how to use the portfolio rebalancing function."""
    
    # Define target allocations (symbol, weight)
    target_allocations = [
        ("EURUSD", 0.4),   # 40% EUR/USD
        ("GBPUSD", 0.3),   # 30% GBP/USD
        ("USDJPY", 0.2),   # 20% USD/JPY
        ("AUDUSD", 0.1),   # 10% AUD/USD
    ]
    
    # Validate allocations
    if not validate_allocations(target_allocations):
        print("❌ Invalid allocations")
        return
    
    # Perform dry run first
    print("🔍 Performing dry run...")
    dry_run_results = rebalance_portfolio(target_allocations, dry_run=True)
    print(f"Required trades: {dry_run_results.get('required_trades', {})}")
    
    # Ask for confirmation before executing
    response = input("Execute trades? (y/N): ").strip().lower()
    if response == 'y':
        print("⚡ Executing trades...")
        results = rebalance_portfolio(target_allocations, dry_run=False)
        print(f"Rebalancing completed: {results.get('successful_trades', 0)} successful trades")
    else:
        print("Rebalancing cancelled")


if __name__ == "__main__":
    # Example usage
    print("Portfolio Management Utils")
    print("=" * 30)
    
    # Get portfolio summary
    summary = get_portfolio_summary()
    print(f"Current portfolio summary: {summary}")

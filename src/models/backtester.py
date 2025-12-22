"""
Backtesting module using vectorbt for portfolio simulation.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioBacktester:
    """Backtests trading strategies using vectorbt."""
    
    def __init__(self, config: Dict):
        """
        Initialize PortfolioBacktester.
        
        Args:
            config: Configuration dictionary with strategy parameters
        """
        self.config = config
        self.portfolio = None
        self.signals = None
    
    def create_signals_from_predictions(self, df: pd.DataFrame, 
                                       predictions: np.ndarray,
                                       buy_threshold: float = 0.6,
                                       sell_threshold: float = 0.4) -> pd.DataFrame:
        """
        Convert model predictions to trading signals.
        
        Args:
            df: DataFrame with price data
            predictions: Model predictions (0: Sell, 1: Hold, 2: Buy)
            buy_threshold: Threshold for buy signals (not used with discrete predictions)
            sell_threshold: Threshold for sell signals (not used with discrete predictions)
            
        Returns:
            DataFrame with entry and exit signals
        """
        signals_df = df[['close']].copy()
        
        # Convert predictions to signals
        # 2 = Buy, 0 = Sell, 1 = Hold
        signals_df['entries'] = predictions == 2
        signals_df['exits'] = predictions == 0
        
        logger.info(f"Generated {signals_df['entries'].sum()} entry signals")
        logger.info(f"Generated {signals_df['exits'].sum()} exit signals")
        
        self.signals = signals_df
        return signals_df
    
    def run_backtest(self, price_data: pd.Series, 
                     entries: pd.Series, 
                     exits: pd.Series,
                     initial_capital: float = 100000,
                     commission: float = 0.001) -> vbt.Portfolio:
        """
        Run backtest using vectorbt.
        
        Args:
            price_data: Series of closing prices
            entries: Boolean series of entry signals
            exits: Boolean series of exit signals
            initial_capital: Starting capital
            commission: Commission rate per trade
            
        Returns:
            vectorbt Portfolio object
        """
        logger.info("Running backtest...")
        
        # Create portfolio from signals
        self.portfolio = vbt.Portfolio.from_signals(
            close=price_data,
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=commission,
            freq='1D'
        )
        
        logger.info("Backtest completed")
        return self.portfolio
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate and return performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.portfolio is None:
            logger.error("No portfolio to analyze. Run backtest first.")
            return {}
        
        metrics = {
            'total_return': self.portfolio.total_return(),
            'annual_return': self.portfolio.annualized_return(),
            'sharpe_ratio': self.portfolio.sharpe_ratio(),
            'max_drawdown': self.portfolio.max_drawdown(),
            'win_rate': self.portfolio.trades.win_rate(),
            'total_trades': self.portfolio.trades.count(),
            'final_value': self.portfolio.final_value(),
        }
        
        # Try to get profit factor if available
        try:
            metrics['profit_factor'] = self.portfolio.trades.profit_factor()
        except:
            metrics['profit_factor'] = 0
        
        logger.info("Performance Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        return metrics
    
    def get_portfolio_value(self) -> pd.Series:
        """
        Get portfolio value over time.
        
        Returns:
            Series of portfolio values
        """
        if self.portfolio is None:
            logger.error("No portfolio to analyze. Run backtest first.")
            return pd.Series()
        
        return self.portfolio.value()
    
    def get_returns(self) -> pd.Series:
        """
        Get portfolio returns over time.
        
        Returns:
            Series of returns
        """
        if self.portfolio is None:
            logger.error("No portfolio to analyze. Run backtest first.")
            return pd.Series()
        
        return self.portfolio.returns()
    
    def get_trades_df(self) -> pd.DataFrame:
        """
        Get DataFrame of all trades.
        
        Returns:
            DataFrame with trade information
        """
        if self.portfolio is None:
            logger.error("No portfolio to analyze. Run backtest first.")
            return pd.DataFrame()
        
        return self.portfolio.trades.records_readable
    
    def compare_with_buy_and_hold(self, price_data: pd.Series,
                                  initial_capital: float = 100000) -> Dict:
        """
        Compare strategy performance with buy and hold.
        
        Args:
            price_data: Series of closing prices
            initial_capital: Starting capital
            
        Returns:
            Dictionary with comparison metrics
        """
        if self.portfolio is None:
            logger.error("No portfolio to analyze. Run backtest first.")
            return {}
        
        # Calculate buy and hold returns
        buy_hold_return = (price_data.iloc[-1] / price_data.iloc[0]) - 1
        buy_hold_final = initial_capital * (1 + buy_hold_return)
        
        # Get strategy returns
        strategy_return = self.portfolio.total_return()
        strategy_final = self.portfolio.final_value()
        
        comparison = {
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'strategy_final_value': strategy_final,
            'buy_hold_final_value': buy_hold_final,
            'outperformance': strategy_return - buy_hold_return
        }
        
        logger.info(f"Strategy Return: {strategy_return:.2%}")
        logger.info(f"Buy & Hold Return: {buy_hold_return:.2%}")
        logger.info(f"Outperformance: {comparison['outperformance']:.2%}")
        
        return comparison

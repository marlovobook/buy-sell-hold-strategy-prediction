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


"""
Reinforcement Learning Backtester (Target Weights)
Designed for Continuous Control models (PPO, SAC, TD3) using VectorBT.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import logging
from typing import Dict, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktesterRL")

class PortfolioBacktesterRL:
    """
    Backtester specifically for Reinforcement Learning models that output 
    Target Portfolio Weights (Continuous Control).
    
    Key Difference from Discrete Backtester:
    - Uses 'targetpercent' sizing instead of boolean entry/exit signals.
    - Automatically handles index alignment between RL actions and Price data
      (accounting for the lookback window used during feature engineering).
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Dictionary containing default parameters (commission, initial_capital).
        """
        self.config = config or {}
        self.portfolio = None
        self.aligned_prices = None
        self.aligned_weights = None
        
        # Defaults
        self.default_commission = self.config.get("commission", 0.001)
        self.default_capital = self.config.get("initial_balance", 10000.0)

    def run_backtest(self, 
                     price_data: pd.Series, 
                     predicted_weights: np.ndarray, 
                     lookback_window: int = 0,
                     freq: str = '1D') -> vbt.Portfolio:
        """
        Executes the backtest using Target Weights.

        Args:
            price_data (pd.Series): The full Close price series (Raw data).
            predicted_weights (np.ndarray): The output from model.predict(). 
                                            Values should be between -1.0 and 1.0.
            lookback_window (int): The number of steps the RL environment 'skipped' 
                                   at the start. (e.g., 30). Crucial for alignment.
            freq (str): Data frequency ('1D', '1H', etc.)

        Returns:
            vbt.Portfolio: The calculated vectorbt portfolio object.
        """
        logger.info(f"Preparing Backtest. Raw Prices: {len(price_data)}, Predictions: {len(predicted_weights)}")

        # --- 1. Data Alignment (Critical for RL) ---
        # RL models usually start predicting at index `lookback_window`.
        # We must slice the price data to match the length of the predictions.
        
        # Option A: User passed full price data + lookback
        if len(price_data) > len(predicted_weights):
            if lookback_window == 0:
                # Auto-detect lookback if not provided
                lookback_window = len(price_data) - len(predicted_weights)
                logger.warning(f"Auto-detected lookback window: {lookback_window} steps")
            
            # Slice prices to match predictions (from end)
            # We assume predictions correspond to the *end* of the dataset
            self.aligned_prices = price_data.iloc[-len(predicted_weights):]
        
        # Option B: User already aligned them manually
        elif len(price_data) == len(predicted_weights):
            self.aligned_prices = price_data
            
        else:
            raise ValueError(f"Price data ({len(price_data)}) is shorter than predictions ({len(predicted_weights)}). Check your data splits.")

        # Ensure weights are a Pandas Series with the same index as prices
        # This prevents vectorbt from misaligning dates
        self.aligned_weights = pd.Series(
            data=predicted_weights, 
            index=self.aligned_prices.index, 
            name='TargetWeight'
        )

        # --- 2. Run Simulation ---
        # size_type='targetpercent' tells vectorbt that 0.5 means "50% of portfolio value"
        # rather than "0.5 shares".
        
        logger.info("Running vectorbt simulation...")
        self.portfolio = vbt.Portfolio.from_orders(
            close=self.aligned_prices,
            size=self.aligned_weights,
            size_type='targetpercent',
            init_cash=self.default_capital,
            fees=self.default_commission,
            freq=freq,
            # If allowing shorting, ensure direction logic is handled (VBT handles negative weights as shorts automatically)
        )
        
        logger.info("Backtest successfully completed.")
        return self.portfolio

    def get_performance_metrics(self) -> Dict:
        """
        Extracts key financial metrics from the portfolio.
        """
        if self.portfolio is None:
            logger.error("No portfolio found. Run 'run_backtest' first.")
            return {}

        # Basic VBT stats
        total_return = self.portfolio.total_return()
        sharpe = self.portfolio.sharpe_ratio()
        max_dd = self.portfolio.max_drawdown()
        
        # Advanced stats
        sortino = self.portfolio.sortino_ratio()
        calmar = self.portfolio.calmar_ratio()
        win_rate = self.portfolio.trades.win_rate()
        
        metrics = {
            "Total Return (%)": round(total_return * 100, 2),
            "Annual Return (%)": round(self.portfolio.annualized_return() * 100, 2),
            "Sharpe Ratio": round(sharpe, 4),
            "Sortino Ratio": round(sortino, 4),
            "Max Drawdown (%)": round(max_dd * 100, 2),
            "Calmar Ratio": round(calmar, 4),
            "Win Rate (%)": round(win_rate * 100, 2),
            "Total Trades": self.portfolio.trades.count(),
            "Final Value ($)": round(self.portfolio.final_value(), 2)
        }
        
        return metrics

    def compare_with_benchmark(self, benchmark_prices: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compares RL Strategy vs. Buy & Hold.
        
        Args:
            benchmark_prices: Optional Series. If None, uses the strategy's price data (Buy & Hold the asset itself).
        """
        if self.portfolio is None:
            return pd.DataFrame()

        # If no external benchmark provided, compare against holding the traded asset
        if benchmark_prices is None:
            benchmark_prices = self.aligned_prices

        # Create Buy & Hold portfolio
        bnch_portfolio = vbt.Portfolio.from_holding(
            benchmark_prices, 
            init_cash=self.default_capital,
            freq='1D'
        )
        
        strategy_cum_ret = self.portfolio.cumulative_returns()
        bnch_cum_ret = bnch_portfolio.cumulative_returns()
        
        comparison_df = pd.DataFrame({
            "RL Strategy": strategy_cum_ret,
            "Buy & Hold": bnch_cum_ret
        })
        
        return comparison_df
    
    def compare_with_buy_and_hold_rl(self) -> pd.DataFrame:
        """
        Compares the RL Strategy against a Buy & Hold benchmark 
        using the EXACT same timeframe and price data used in the backtest.
        
        Note: No arguments needed. It uses the self.portfolio data.
        """
        if self.portfolio is None:
            logger.error("Run backtest first.")
            return pd.DataFrame()

        # 1. Extract the exact prices used in the simulation
        # This prevents date mismatch errors if you sliced data externally
        simulation_prices = self.portfolio.close

        # 2. Simulate Buy & Hold on those specific prices
        bnch_portfolio = vbt.Portfolio.from_holding(
            simulation_prices, 
            init_cash=self.default_capital,
            fees=self.default_commission,
            freq='1D'
        )
        
        # 3. Calculate Returns
        strategy_rets = self.portfolio.total_return()
        bnch_rets = bnch_portfolio.total_return()
        
        logger.info(f"Strategy Return: {strategy_rets:.2%}")
        logger.info(f"Buy & Hold Return: {bnch_rets:.2%}")
        logger.info(f"Outperformance: {strategy_rets - bnch_rets:.2%}")
        
        # 4. Create Comparison DataFrame (Cumulative Returns)
        comparison_df = pd.DataFrame({
            "RL Strategy": self.portfolio.cumulative_returns(),
            "Buy & Hold": bnch_portfolio.cumulative_returns()
        })
        
        return comparison_df

    def plot_portfolio(self, show_orders: bool = False):
        """
        Visualizes the equity curve and optionally the orders.
        """
        if self.portfolio is None:
            return None
            
        # Creates a subplot with Value + Drawdown
        fig = self.portfolio.plot(subplots=[
            ('value', dict(title='Portfolio Value')), 
            ('drawdowns', dict(title='Drawdowns'))
        ])
        
        if show_orders:
            # Overlay buy/sell markers on the value plot
            # Note: plotting continuous rebalancing orders can be messy (too many dots)
            # so use this sparingly.
            self.portfolio.trades.plot(fig=fig, plot_close=False)
            
        fig.show()

# --- Example Usage (Commented out) ---
# if __name__ == "__main__":
#     # 1. Assume we have price data (1000 days)
#     prices = pd.Series(np.random.uniform(100, 200, 1000), name='close')
#     
#     # 2. Assume we have 970 predictions from RL (because lookback was 30)
#     # These are target weights: 0.5 = 50% invested, -0.2 = 20% Short
#     predictions = np.random.uniform(-1, 1, 970)
#     
#     # 3. Initialize & Run
#     bt = PortfolioBacktesterRL(config={"initial_balance": 100000, "commission": 0.001})
#     
#     # 4. Run Backtest (It will automatically align the 1000 prices to the 970 predictions)
#     bt.run_backtest(price_data=prices, predicted_weights=predictions, lookback_window=30)
#     
#     # 5. Results
#     print(bt.get_performance_metrics())
#     bt.plot_portfolio()
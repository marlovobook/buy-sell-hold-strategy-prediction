"""
Data collection module using yfinance.
This module handles downloading financial data and calculating technical indicators.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import pandas_ta as ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects and processes financial data from Yahoo Finance."""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str, interval: str = "1d"):
        """
        Initialize DataCollector.
        
        Args:
            symbols: List of stock symbols to download
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1wk, 1mo)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data = {}
    
    def download_data(self) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for all symbols.
        
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        logger.info(f"Downloading data for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                logger.info(f"Downloading {symbol}...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval
                )
                
                if not df.empty:
                    self.data[symbol] = df
                    logger.info(f"Downloaded {len(df)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
        
        return self.data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        data = df.copy()
        
        
        
        # TradingView Technical Indicators using pandas_ta --#
        data.ta.sma(length=20, append=True)
        data.ta.sma(length=50, append=True)
        data.ta.rsi(length=14, append=True)
        data.ta.macd(append=True)
        data.ta.bbands(append=True)
        data.ta.vwap(append=True)
        
        data.ta.roc(length=1, append=True)
        data.ta.atr(length=14, append=True)
        data.ta.tsi(append=True)
        data.ta.adx(append=True)
        data.ta.cci(length=20, append=True)
        #-----------------------------------------------#
        
        # Calculate fibonacci retracement levels (52-week rolling window)
        window = 252  # Approximate trading days in 52 weeks (252 trading days/year)
        data['Rolling_High_52w'] = data['High'].rolling(window=window, min_periods=1).max()
        data['Rolling_Low_52w'] = data['Low'].rolling(window=window, min_periods=1).min()
        diff = data['Rolling_High_52w'] - data['Rolling_Low_52w']
        
        data['Fibo_23.6'] = data['Rolling_High_52w'] - 0.236 * diff
        data['Fibo_38.2'] = data['Rolling_High_52w'] - 0.382 * diff
        data['Fibo_50.0'] = data['Rolling_High_52w'] - 0.5 * diff
        data['Fibo_61.8'] = data['Rolling_High_52w'] - 0.618 * diff
        data['Fibo_78.6'] = data['Rolling_High_52w'] - 0.786 * diff
        data['Fibo_100.0'] = data['Rolling_Low_52w']
        data['Fibo_161.8'] = data['Rolling_High_52w'] - 1.618 * diff
        
        
        #--------------------------
        # # Simple Moving Averages
        # data['SMA_20'] = data['Close'].rolling(window=20).mean()
        # data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # # Relative Strength Index (RSI)
        # delta = data['Close'].diff()
        # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # rs = gain / loss
        # data['RSI'] = 100 - (100 / (1 + rs))
        
        # # MACD
        # exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        # exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        # data['MACD'] = exp1 - exp2
        # data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # # Bollinger Bands
        # data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        # bb_std = data['Close'].rolling(window=20).std()
        # data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        # data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # # Volume indicators
        # data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
        # # Price change percentage
        # data['Price_Change'] = data['Close'].pct_change()
        
        # # Volatility
        # data['Volatility'] = data['Price_Change'].rolling(window=20).std()
        
        
        
        return data
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        Create buy/sell/hold labels based on future price movement.
        
        Args:
            df: DataFrame with price data
            horizon: Number of days to look ahead
            
        Returns:
            DataFrame with labels column
        """
        data = df.copy()
        
        # Calculate future returns
        # Explain how future returns are calculated and what it means
        """
        Future_Return = (Price at day t + horizon - Price at day t) / Price at day t
        This gives the percentage change in price over the next 'horizon' days."""
        data['Future_Return'] = data['Close'].pct_change(horizon).shift(-horizon)
        
        # Create labels: Buy (2), Hold (1), Sell (0)
        # Buy if future return > 2%
        # Sell if future return < -2%
        # Hold otherwise
        data['Label'] = 1  # Default to Hold
        data.loc[data['Future_Return'] > 0.02, 'Label'] = 2  # Buy
        data.loc[data['Future_Return'] < -0.02, 'Label'] = 0  # Sell
        
        return data
    
    def prepare_data(self, symbol: str) -> pd.DataFrame:
        """
        Prepare complete dataset with indicators and labels for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Prepared DataFrame
        """
        if symbol not in self.data:
            logger.error(f"Data not found for {symbol}")
            return pd.DataFrame()
        
        df = self.data[symbol].copy()
        df = self.calculate_technical_indicators(df)
        df = self.create_labels(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # .lower all column names
        df.columns = [col.lower() for col in df.columns]
        
        logger.info(f"Prepared {len(df)} records for {symbol}")
        return df
    
    def save_data(self, filepath: str):
        """
        Save collected data to CSV files.
        
        Args:
            filepath: Base path for saving files
        """
        for symbol, df in self.data.items():
            file_path = f"{filepath}/{symbol}.csv"
            df.to_csv(file_path)
            logger.info(f"Saved data for {symbol} to {file_path}")

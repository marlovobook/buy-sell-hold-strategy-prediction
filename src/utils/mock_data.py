"""
Mock data generator for testing without internet access.
Creates synthetic stock data that resembles real market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_mock_stock_data(symbol: str, start_date: str, end_date: str, 
                             initial_price: float = 100.0) -> pd.DataFrame:
    """
    Generate mock stock data for testing.
    
    Args:
        symbol: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_price: Starting price
        
    Returns:
        DataFrame with OHLCV data
    """
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate date range
    dates = pd.date_range(start=start, end=end, freq='D')
    
    # Generate realistic price movement using random walk with drift
    n_days = len(dates)
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns with slight upward drift
    price_multipliers = np.exp(np.cumsum(returns))
    close_prices = initial_price * price_multipliers
    
    # Generate OHLCV data
    data = {
        'Open': close_prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': close_prices * (1 + np.random.uniform(0.005, 0.03, n_days)),
        'Low': close_prices * (1 + np.random.uniform(-0.03, -0.005, n_days)),
        'Close': close_prices,
        'Volume': np.random.randint(50000000, 150000000, n_days),
        'Dividends': np.zeros(n_days),
        'Stock Splits': np.zeros(n_days)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure High is highest and Low is lowest
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    return df


if __name__ == "__main__":
    # Test the generator
    df = generate_mock_stock_data("AAPL", "2023-01-01", "2024-12-31")
    print(f"Generated {len(df)} days of data")
    print(df.head())
    print(df.tail())

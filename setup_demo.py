"""
Setup script to generate sample data for the Streamlit app demo.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_collector import DataCollector
from src.models.model_trainer import ModelTrainer
from src.models.backtester import PortfolioBacktester
from src.utils.config_loader import load_config
from src.utils.mock_data import generate_mock_stock_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_demo_data():
    """Generate sample data for demo."""
    logger.info("Setting up demo data for Streamlit app...")
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Generate data for all symbols
    data_config = config['data']
    
    for symbol in data_config['symbols'][:2]:  # Use first 2 symbols for demo
        logger.info(f"\nProcessing {symbol}...")
        
        # Generate mock data
        mock_data = generate_mock_stock_data(
            symbol,
            data_config['start_date'],
            data_config['end_date']
        )
        
        # Create collector and prepare data
        collector = DataCollector(
            symbols=[symbol],
            start_date=data_config['start_date'],
            end_date=data_config['end_date'],
            interval=data_config['interval']
        )
        collector.data[symbol] = mock_data
        
        # Prepare data
        prepared_data = collector.prepare_data(symbol)
        
        # Save processed data
        prepared_data.to_csv(f'data/processed/{symbol}_processed.csv')
        logger.info(f"Saved processed data for {symbol}: {len(prepared_data)} records")
    
    # Train models for first symbol
    symbol = data_config['symbols'][0]
    logger.info(f"\nTraining models for {symbol}...")
    
    # Load the prepared data for this symbol
    import pandas as pd
    prepared_data = pd.read_csv(f'data/processed/{symbol}_processed.csv', index_col=0, parse_dates=True)
    
    # Use reduced model configuration for faster demo setup
    model_config = {
        'algorithms': ['random_forest', 'xgboost'],
        'test_size': 0.2,
        'random_forest': {
            'n_estimators': 50,
            'max_depth': 8,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 50,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }
    
    trainer = ModelTrainer(model_config)
    results = trainer.train_all_models(prepared_data)
    
    # Save models
    trainer.save_models('models')
    logger.info("Models saved successfully")
    
    logger.info("\n" + "=" * 50)
    logger.info("Demo data setup complete!")
    logger.info("=" * 50)
    logger.info("\nYou can now run: streamlit run app.py")


if __name__ == "__main__":
    setup_demo_data()

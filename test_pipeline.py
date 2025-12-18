"""
Test script to validate the ML pipeline.
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


def test_pipeline():
    """Test the ML pipeline with reduced data."""
    logger.info("=" * 50)
    logger.info("Testing ML Pipeline")
    logger.info("=" * 50)
    
    try:
        # Load test configuration
        logger.info("\n1. Loading configuration...")
        config = load_config("config/config_test.yaml")
        logger.info("✓ Configuration loaded successfully")
        
        # Data Collection Phase
        logger.info("\n2. Testing Data Collection...")
        data_config = config['data']
        
        # Use mock data for testing (no internet required)
        symbol = data_config['symbols'][0]
        logger.info(f"Generating mock data for {symbol}...")
        mock_data = generate_mock_stock_data(
            symbol,
            data_config['start_date'],
            data_config['end_date']
        )
        
        # Create collector and manually set data
        collector = DataCollector(
            symbols=data_config['symbols'],
            start_date=data_config['start_date'],
            end_date=data_config['end_date'],
            interval=data_config['interval']
        )
        collector.data[symbol] = mock_data
        
        logger.info(f"✓ Mock data generated: {len(mock_data)} records")
        
        # Process data for first symbol
        logger.info(f"\n3. Processing data for {symbol}...")
        prepared_data = collector.prepare_data(symbol)
        
        if prepared_data.empty:
            logger.error("✗ No data to process.")
            return False
        
        logger.info(f"✓ Data processed successfully: {len(prepared_data)} records")
        
        # Model Training Phase
        logger.info("\n4. Testing Model Training...")
        trainer = ModelTrainer(config['models'])
        results = trainer.train_all_models(prepared_data)
        
        logger.info(f"✓ Models trained successfully")
        logger.info(f"  Best model: {trainer.best_model_name}")
        logger.info(f"  Best accuracy: {trainer.results[trainer.best_model_name]['accuracy']:.4f}")
        
        # Backtesting Phase
        logger.info("\n5. Testing Backtesting...")
        backtester = PortfolioBacktester(config['strategy'])
        
        # Get predictions for backtesting
        predictions = trainer.results[trainer.best_model_name]['predictions']
        test_data = prepared_data.iloc[-len(predictions):]
        
        # Create signals
        signals = backtester.create_signals_from_predictions(
            test_data,
            predictions,
            config['strategy']['buy_threshold'],
            config['strategy']['sell_threshold']
        )
        
        # Run backtest
        portfolio = backtester.run_backtest(
            test_data['Close'],
            signals['entries'],
            signals['exits'],
            config['strategy']['initial_capital'],
            config['strategy']['commission']
        )
        
        logger.info("✓ Backtesting completed successfully")
        
        # Get performance metrics
        metrics = backtester.get_performance_metrics()
        
        logger.info("\n" + "=" * 50)
        logger.info("✓ All Tests Passed!")
        logger.info("=" * 50)
        logger.info(f"\nSummary:")
        logger.info(f"  Data Records: {len(prepared_data)}")
        logger.info(f"  Best Model: {trainer.best_model_name}")
        logger.info(f"  Model Accuracy: {trainer.results[trainer.best_model_name]['accuracy']:.4f}")
        logger.info(f"  Portfolio Return: {metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

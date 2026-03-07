"""
Main ML pipeline script following software factory methodology.
This script orchestrates the entire process from data collection to model training.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_collector import DataCollector
from src.models.model_trainer import ModelTrainer
from src.models.backtester import PortfolioBacktester
from src.utils.config_loader import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution."""
    logger.info("=" * 50)
    logger.info("Starting ML Pipeline")
    logger.info("=" * 50)
    
    # Load configuration
    logger.info("\n1. Loading configuration...")
    config = load_config("config/config.yaml")
    
    # Data Collection Phase
    logger.info("\n2. Data Collection Phase")
    data_config = config['data']
    collector = DataCollector(
        symbols=data_config['symbols'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        interval=data_config['interval']
    )
    
    # Download data
    collector.download_data()
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    collector.save_data('data/raw')
    
    # Process data for first symbol (can be extended for multiple symbols)
    symbol = data_config['symbols'][0]
    logger.info(f"\n3. Processing data for {symbol}...")
    prepared_data = collector.prepare_data(symbol)
    
    if prepared_data.empty:
        logger.error("No data to process. Exiting.")
        return
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    prepared_data.to_csv(f'data/processed/{symbol}_processed.csv')
    logger.info(f"Saved processed data: {len(prepared_data)} records")
    
    # Model Training Phase
    logger.info("\n4. Model Training Phase")
    os.makedirs('models', exist_ok=True)
    trainer = ModelTrainer(config['models'])
    results = trainer.train_all_models(prepared_data)
    
    # Save models
    trainer.save_models('models')
    
    # Backtesting Phase
    logger.info("\n5. Backtesting Phase")
    backtester = PortfolioBacktester(config['strategy'])
    
    # Get predictions for backtesting (using test set)
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
    
    # Get performance metrics
    metrics = backtester.get_performance_metrics()
    
    # Compare with buy and hold
    comparison = backtester.compare_with_buy_and_hold(
        test_data['Close'],
        config['strategy']['initial_capital']
    )
    
    logger.info("\n" + "=" * 50)
    logger.info("Pipeline Completed Successfully!")
    logger.info("=" * 50)
    logger.info(f"\nBest Model: {trainer.best_model_name}")
    logger.info(f"Model Accuracy: {trainer.results[trainer.best_model_name]['accuracy']:.4f}")
    logger.info(f"\nStrategy Return: {metrics['total_return']:.2%}")
    logger.info(f"Buy & Hold Return: {comparison['buy_hold_return']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"\nRun Streamlit app to visualize results: streamlit run app.py")


if __name__ == "__main__":
    main()

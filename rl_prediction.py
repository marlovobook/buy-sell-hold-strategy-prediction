import pandas as pd
import numpy as np
import yaml
import logging
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your custom classes
# (Assuming they are in files named model_trainer_rl.py and portfolio_backtester_rl.py)
from model_trainer_rl import ModelTrainerRL, TradingEnvRL
from portfolio_backtester_rl import PortfolioBacktesterRL

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def load_config(path: str):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def get_agent_predictions(model, env: TradingEnvRL) -> np.ndarray:
    """
    The Bridge: Runs the trained Agent through the Test Environment 
    to record the specific actions (target weights) it takes.
    """
    logger.info("Generating predictions from trained agent...")
    
    # 1. Reset Environment
    obs, _ = env.reset()
    done = False
    
    actions = []
    
    # 2. Step-by-Step Inference Loop
    # We must run this loop because RL is state-dependent. 
    # The action at step T depends on the portfolio state at step T-1.
    while not done:
        # predict returns (action, state). We only need action.
        # deterministic=True is standard for backtesting (removes random noise)
        action, _ = model.predict(obs, deterministic=True)
        
        # Store the target weight (the action)
        actions.append(action[0])
        
        # Step the environment forward
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    return np.array(actions)

def main():
    # --- 1. Setup ---
    logger.info("1. Loading Data and Config")
    config = load_config("config/config.yaml")
    
    # Load processed data
    df = pd.read_csv('data/processed/AAPL_processed.csv')
    
    # Ensure date is datetime and set as index (helper for vectorbt plotting later)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # --- 2. Train RL Model ---
    logger.info("2. Initializing Trainer")
    trainer = ModelTrainerRL(config)
    
    # Train models (PPO, A2C, etc.)
    # Note: We split data inside train_all, usually 80% train / 20% test
    results = trainer.train_all(df, test_size=0.2)
    
    # Get the best model (e.g., PPO)
    best_model_name = trainer.best_model_name
    best_model = trainer.best_model
    logger.info(f"Selected Best Model: {best_model_name}")
    
    # --- 3. Prepare Test Data for Backtest ---
    # We must re-create the exact environment used for testing
    # to ensure our predictions match the timeline.
    split_idx = int(len(df) * (1 - 0.2))
    df_test = df.iloc[split_idx:].copy()
    
    # Initialize the Test Environment specifically for prediction
    env_test = TradingEnvRL(
        df_test, 
        initial_balance=config['environment']['initial_balance'],
        commission=config['environment']['commission'],
        lookback_window=config['environment']['lookback_window']
    )
    
    # --- 4. Generate Predictions (The Bridge) ---
    # Run the agent over df_test to get the list of target weights
    predicted_weights = get_agent_predictions(best_model, env_test)
    
    logger.info(f"Generated {len(predicted_weights)} actions.")
    
    # --- 5. Run VectorBT Backtest ---
    logger.info("5. Running VectorBT Backtest")
    
    # Initialize our specialized RL Backtester
    backtester = PortfolioBacktesterRL(config['environment'])
    
    # Run the backtest
    # Note: We pass df_test['close'] because backtester needs raw prices
    # Note: lookback_window=30 tells it to align the data correctly
    portfolio = backtester.run_backtest(
        price_data=df_test['close'],
        predicted_weights=predicted_weights,
        lookback_window=config['environment']['lookback_window']
    )
    
    # --- 6. Results & Visualization ---
    metrics = backtester.get_performance_metrics()
    
    print("\n" + "="*30)
    print(f"FINAL RESULTS ({best_model_name})")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k:<20}: {v}")
    print("="*30 + "\n")
    
    # Plotting
    # This will open a browser window or show in Jupyter
    backtester.plot_portfolio(show_orders=False)

if __name__ == "__main__":
    main()
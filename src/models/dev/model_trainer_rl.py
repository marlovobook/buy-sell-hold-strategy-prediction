"""
Reinforcement Learning model training for stock trading strategies.
Implements PPO, A2C, DDPG, TD3, SAC, DQN and custom reward functions.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Callable, List
import os

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class TradingEnvRL(gym.Env):
    """Custom trading environment for RL agents."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000,
        commission: float = 0.001,
        reward_func: str = "sharpe",
        lookback_window: int = 30,
        exclude_cols: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_func_name = reward_func
        self.lookback_window = lookback_window
        
        # Extract price columns
        self.prices = self.df["close"].values
        self.dates = self.df.index if "date" not in self.df.columns else self.df["date"].values
        
        """
        Yes, RL models can have a datetime column, but it should not be used as a feature for training. Yes, data must be ordered from past to present.
        Why datetime matters for RL:
        1. Temporal ordering is critical – RL agents learn sequential decision-making. If data is shuffled or out of order, the agent learns meaningless patterns (e.g., predicting tomorrow's price using future data).
        2. No data leakage – Using future data as features violates causality and creates unrealistic performance.
        3. Datetime for reference only – Keep it for:
           - Tracking which step corresponds to which real date
           - Analyzing performance across time periods
            Post-training analysis/reporting
        *------*
        Why exclude OHLCV:
            Data leakage & multicollinearity – Close, Open, High, Low are raw price data. If you include them as features AND use close price for portfolio calculations, the model sees redundant information and can overfit to price movements rather than learning trading signals.
            Technical indicators already capture price – The feature columns should be derived indicators (SMA, RSI, MACD, Bollinger Bands, etc.) that represent patterns, not raw prices. These are more meaningful for decision-making.
            Volume – Often causes scale/normalization issues and is less predictive than price-derived features.
        """
        # Default exclusions (OHLCV + date)
        if exclude_cols is None:
            exclude_cols = ["close", "date", "open", "high", "low", "volume"]
        
        # Feature columns (exclude specified columns)
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        self.features = self.df[feature_cols].values if feature_cols else np.zeros((len(self.df), 1))
        
        self.n_features = self.features.shape[1]
        
        # Action space: continuous [-1, 1] where -1=sell all, 0=hold, 1=buy all
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: features + portfolio state (balance, shares, total_value)
        obs_dim = self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.balance = initial_balance
        self.shares = 0
        self.total_asset = initial_balance
        self.portfolio_values = []
        self.returns_history = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares = 0
        self.total_asset = self.initial_balance
        self.portfolio_values = [self.initial_balance]
        self.returns_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (features + portfolio state)."""
        # Get features at current step
        features = self.features[self.current_step]
        
        # Portfolio state: normalized balance, shares, total_value
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.shares,
            self.total_asset / self.initial_balance
        ], dtype=np.float32)
        
        obs = np.concatenate([features, portfolio_state])
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray):
        """Execute one trading step."""
        action_value = float(action[0])
        current_price = self.prices[self.current_step]
        
        # Execute trade based on action
        if action_value > 0:  # Buy
            # Buy amount proportional to action
            max_shares = int(self.balance / (current_price * (1 + self.commission)))
            shares_to_buy = int(max_shares * action_value)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.commission)
                self.balance -= cost
                self.shares += shares_to_buy
                
        elif action_value < 0:  # Sell
            # Sell amount proportional to action
            shares_to_sell = int(self.shares * abs(action_value))
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.commission)
                self.balance += revenue
                self.shares -= shares_to_sell
        
        # Update portfolio value
        prev_total_asset = self.total_asset
        self.total_asset = self.balance + self.shares * current_price
        self.portfolio_values.append(self.total_asset)
        
        # Calculate return
        current_return = (self.total_asset - prev_total_asset) / prev_total_asset
        self.returns_history.append(current_return)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False
        
        obs = self._get_observation() if not terminated else self._get_observation()
        info = {
            "portfolio_value": self.total_asset,
            "return": current_return,
            "balance": self.balance,
            "shares": self.shares,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on configured reward function."""
        if self.reward_func_name == "profit":
            return self._reward_profit()
        elif self.reward_func_name == "sharpe":
            return self._reward_sharpe()
        elif self.reward_func_name == "sortino":
            return self._reward_sortino()
        elif self.reward_func_name == "cvar":
            return self._reward_cvar()
        elif self.reward_func_name == "max_drawdown":
            return self._reward_max_drawdown()
        else:
            return self._reward_profit()
    
    def _reward_profit(self) -> float:
        """Simple profit-based reward."""
        if len(self.returns_history) > 0:
            return self.returns_history[-1] * 100
        return 0.0
    
    def _reward_sharpe(self) -> float:
        """Sharpe ratio reward (risk-adjusted)."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history[-30:])  # Last 30 steps
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-6
        sharpe = mean_return / std_return
        return float(sharpe)
    
    def _reward_sortino(self) -> float:
        """Sortino ratio reward (downside risk-adjusted)."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history[-30:])
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return mean_return * 10
        
        downside_std = np.std(downside_returns) + 1e-6
        sortino = mean_return / downside_std
        return float(sortino)
    
    def _reward_cvar(self, alpha: float = 0.05) -> float:
        """Conditional Value at Risk (CVaR) reward."""
        if len(self.returns_history) < 10:
            return 0.0
        
        returns = np.array(self.returns_history[-30:])
        var = np.quantile(returns, alpha)
        cvar = returns[returns <= var].mean()
        
        # Reward is negative CVaR (we want to minimize losses)
        return float(-cvar * 100)
    
    def _reward_max_drawdown(self) -> float:
        """Maximum drawdown penalty reward."""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        portfolio_array = np.array(self.portfolio_values[-30:])
        running_max = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Penalize large drawdowns, reward small ones
        return float(-max_drawdown * 100)
    
    def render(self):
        """Render environment state."""
        print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, "
              f"Shares: {self.shares}, Total: ${self.total_asset:.2f}")


class ModelTrainerRL:
    """Reinforcement Learning model trainer for stock trading."""
    
    def __init__(self, config: Dict):
        self.config = config or {}
        self.models = {}
        self.results = {}
        self.best_model_name: Optional[str] = None
        self.best_model = None
        self.env_train = None
        self.env_test = None
    
    def prepare_environment(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        reward_func: str = "sharpe",
    ) -> Tuple[TradingEnvRL, TradingEnvRL]:
        """Prepare train and test environments."""
        env_config = self.config.get("environment", {})
        initial_balance = env_config.get("initial_balance", 10000)
        commission = env_config.get("commission", 0.001)
        lookback = env_config.get("lookback_window", 30)
        
        env_train = TradingEnvRL(
            df_train,
            initial_balance=initial_balance,
            commission=commission,
            reward_func=reward_func,
            lookback_window=lookback,
        )
        
        env_test = TradingEnvRL(
            df_test,
            initial_balance=initial_balance,
            commission=commission,
            reward_func=reward_func,
            lookback_window=lookback,
        )
        
        self.env_train = env_train
        self.env_test = env_test
        
        logger.info("Prepared training env with %d steps", len(df_train))
        logger.info("Prepared testing env with %d steps", len(df_test))
        
        return env_train, env_test
    
    def train_ppo(self, env: TradingEnvRL) -> Dict:
        """Train Proximal Policy Optimization agent."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for PPO") from exc
        
        ppo_config = self.config.get("ppo", {})
        timesteps = ppo_config.get("total_timesteps", 10000)
        
        vec_env = DummyVecEnv([lambda: env])
        
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            gamma=ppo_config.get("gamma", 0.99),
            verbose=0,
        )
        
        logger.info("Training PPO for %d timesteps", timesteps)
        model.learn(total_timesteps=timesteps)
        
        return {"model": model, "algorithm": "PPO"}
    
    def train_a2c(self, env: TradingEnvRL) -> Dict:
        """Train Advantage Actor-Critic agent."""
        try:
            from stable_baselines3 import A2C
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for A2C") from exc
        
        a2c_config = self.config.get("a2c", {})
        timesteps = a2c_config.get("total_timesteps", 10000)
        
        vec_env = DummyVecEnv([lambda: env])
        
        model = A2C(
            "MlpPolicy",
            vec_env,
            learning_rate=a2c_config.get("learning_rate", 7e-4),
            n_steps=a2c_config.get("n_steps", 5),
            gamma=a2c_config.get("gamma", 0.99),
            verbose=0,
        )
        
        logger.info("Training A2C for %d timesteps", timesteps)
        model.learn(total_timesteps=timesteps)
        
        return {"model": model, "algorithm": "A2C"}
    
    def train_ddpg(self, env: TradingEnvRL) -> Dict:
        """Train Deep Deterministic Policy Gradient agent."""
        try:
            from stable_baselines3 import DDPG
            from stable_baselines3.common.noise import NormalActionNoise
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for DDPG") from exc
        
        ddpg_config = self.config.get("ddpg", {})
        timesteps = ddpg_config.get("total_timesteps", 10000)
        
        # Action noise for exploration
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        model = DDPG(
            "MlpPolicy",
            env,
            learning_rate=ddpg_config.get("learning_rate", 1e-3),
            buffer_size=ddpg_config.get("buffer_size", 100000),
            batch_size=ddpg_config.get("batch_size", 100),
            gamma=ddpg_config.get("gamma", 0.99),
            action_noise=action_noise,
            verbose=0,
        )
        
        logger.info("Training DDPG for %d timesteps", timesteps)
        model.learn(total_timesteps=timesteps)
        
        return {"model": model, "algorithm": "DDPG"}
    
    def train_td3(self, env: TradingEnvRL) -> Dict:
        """Train Twin Delayed DDPG agent."""
        try:
            from stable_baselines3 import TD3
            from stable_baselines3.common.noise import NormalActionNoise
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for TD3") from exc
        
        td3_config = self.config.get("td3", {})
        timesteps = td3_config.get("total_timesteps", 10000)
        
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=td3_config.get("learning_rate", 1e-3),
            buffer_size=td3_config.get("buffer_size", 100000),
            batch_size=td3_config.get("batch_size", 100),
            gamma=td3_config.get("gamma", 0.99),
            action_noise=action_noise,
            verbose=0,
        )
        
        logger.info("Training TD3 for %d timesteps", timesteps)
        model.learn(total_timesteps=timesteps)
        
        return {"model": model, "algorithm": "TD3"}
    
    def train_sac(self, env: TradingEnvRL) -> Dict:
        """Train Soft Actor-Critic agent."""
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for SAC") from exc
        
        sac_config = self.config.get("sac", {})
        timesteps = sac_config.get("total_timesteps", 10000)
        
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_config.get("learning_rate", 3e-4),
            buffer_size=sac_config.get("buffer_size", 100000),
            batch_size=sac_config.get("batch_size", 256),
            gamma=sac_config.get("gamma", 0.99),
            verbose=0,
        )
        
        logger.info("Training SAC for %d timesteps", timesteps)
        model.learn(total_timesteps=timesteps)
        
        return {"model": model, "algorithm": "SAC"}
    
    def train_dqn(self, env: TradingEnvRL) -> Dict:
        """Train Deep Q-Network agent (requires discrete action space)."""
        try:
            from stable_baselines3 import DQN
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for DQN") from exc
        
        # Note: DQN requires discrete action space
        # We'll modify the environment for DQN
        logger.warning("DQN requires discrete actions. Consider using continuous methods like PPO/SAC.")
        
        dqn_config = self.config.get("dqn", {})
        timesteps = dqn_config.get("total_timesteps", 10000)
        
        # Create discrete wrapper if needed
        from gymnasium.wrappers import TransformAction
        
        # For now, skip DQN or implement discrete wrapper
        logger.info("DQN training skipped - requires discrete action space conversion")
        return {"model": None, "algorithm": "DQN", "skipped": True}
    
    def evaluate_model(self, model, env: TradingEnvRL, n_episodes: int = 1) -> Dict:
        """Evaluate trained RL model."""
        all_rewards = []
        all_portfolio_values = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            all_rewards.append(episode_reward)
            all_portfolio_values.append(env.portfolio_values)
        
        final_value = env.total_asset
        total_return = (final_value - env.initial_balance) / env.initial_balance
        
        # Calculate Sharpe ratio
        if len(env.returns_history) > 1:
            returns = np.array(env.returns_history)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Calculate max drawdown
        portfolio_array = np.array(env.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        metrics = {
            "total_return": float(total_return),
            "final_value": float(final_value),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "avg_reward": float(np.mean(all_rewards)),
        }
        
        logger.info("Evaluation - Return: %.2f%%, Sharpe: %.3f, Max DD: %.2f%%",
                   total_return * 100, sharpe, max_drawdown * 100)
        
        return metrics
    
    def train_all(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        reward_func: str = "sharpe",
    ) -> Dict:
        """Train all configured RL algorithms."""
        # Split data
        split_idx = int(len(df) * (1 - test_size))
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        
        # Prepare environments
        env_train, env_test = self.prepare_environment(df_train, df_test, reward_func)
        
        algorithms = self.config.get("algorithms", ["ppo", "a2c"])
        
        for algo in algorithms:
            try:
                if algo == "ppo":
                    logger.info("Training PPO")
                    result = self.train_ppo(env_train)
                    if result["model"]:
                        metrics = self.evaluate_model(result["model"], env_test)
                        result["metrics"] = metrics
                        self.results["ppo"] = result
                        self.models["ppo"] = result["model"]
                
                elif algo == "a2c":
                    logger.info("Training A2C")
                    result = self.train_a2c(env_train)
                    if result["model"]:
                        metrics = self.evaluate_model(result["model"], env_test)
                        result["metrics"] = metrics
                        self.results["a2c"] = result
                        self.models["a2c"] = result["model"]
                
                elif algo == "ddpg":
                    logger.info("Training DDPG")
                    result = self.train_ddpg(env_train)
                    if result["model"]:
                        metrics = self.evaluate_model(result["model"], env_test)
                        result["metrics"] = metrics
                        self.results["ddpg"] = result
                        self.models["ddpg"] = result["model"]
                
                elif algo == "td3":
                    logger.info("Training TD3")
                    result = self.train_td3(env_train)
                    if result["model"]:
                        metrics = self.evaluate_model(result["model"], env_test)
                        result["metrics"] = metrics
                        self.results["td3"] = result
                        self.models["td3"] = result["model"]
                
                elif algo == "sac":
                    logger.info("Training SAC")
                    result = self.train_sac(env_train)
                    if result["model"]:
                        metrics = self.evaluate_model(result["model"], env_test)
                        result["metrics"] = metrics
                        self.results["sac"] = result
                        self.models["sac"] = result["model"]
                
                elif algo == "dqn":
                    logger.info("Attempting DQN")
                    result = self.train_dqn(env_train)
                    if result.get("model"):
                        metrics = self.evaluate_model(result["model"], env_test)
                        result["metrics"] = metrics
                        self.results["dqn"] = result
                        self.models["dqn"] = result["model"]
            
            except Exception as e:
                logger.error(f"Failed to train {algo}: {e}")
                continue
        
        self._select_best_model()
        return self.results
    
    def _select_best_model(self):
        """Select best model based on Sharpe ratio."""
        best_sharpe = -np.inf
        
        for name, result in self.results.items():
            if "metrics" in result:
                sharpe = result["metrics"].get("sharpe_ratio", -np.inf)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    self.best_model_name = name
                    self.best_model = self.models.get(name)
        
        if self.best_model_name:
            logger.info("Best RL model: %s (Sharpe: %.3f)", self.best_model_name, best_sharpe)
    
    def save_models(self, filepath: str):
        """Save trained RL models."""
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        for name, model in self.models.items():
            if model is not None:
                model_path = os.path.join(filepath, f"{name}_model")
                model.save(model_path)
                logger.info(f"Saved {name} model to {model_path}")
        
        # Save results summary
        import json
        results_summary = {}
        for name, result in self.results.items():
            if "metrics" in result:
                results_summary[name] = result["metrics"]
        
        summary_path = os.path.join(filepath, "rl_results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Saved results summary to {summary_path}")
    
    def predict_action(self, obs: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """Predict action using trained model."""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        action, _ = model.predict(obs, deterministic=True)
        return action

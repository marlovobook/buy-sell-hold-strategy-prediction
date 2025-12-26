"""
Reinforcement Learning model training for stock trading strategies.
Implements PPO, A2C, DDPG, TD3, SAC, DQN and custom reward functions.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Callable, List
import os
import pickle
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


try:
    from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.noise import NormalActionNoise
except ImportError:
    raise ImportError("Please install stable-baselines3: pip install stable-baselines3[extra]")

logger = logging.getLogger(__name__)


class TradingEnvRL(gym.Env):
    """
    Industry Standard Trading Environment.
    - Uses Target Weights for actions (Portfolio Management).
    - Normalizes observations.
    - Implements random episode starts.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 100000, 
                 commission: float = 0.001, lookback_window: int = 30, 
                 reward_func: str = "profit", max_steps: int = 1000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.lookback_window = lookback_window
        self.max_steps = max_steps
        self.reward_func_name = reward_func
        
        # Data Setup
        self.prices = self.df["close"].values.astype(np.float32)
        exclude_cols = ["close", "date", "open", "high", "low", "volume"]
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.n_features = self.features.shape[1]
        
        # Action & Observation Space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        obs_dim = self.n_features + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Index Logic
        high = len(self.df) - self.max_steps - 1
        if high <= self.lookback_window:
            self.current_step = self.lookback_window
        else:
            self.current_step = np.random.randint(self.lookback_window, high)
            
        self.stop_step = self.current_step + self.max_steps
        
        # Account State
        self.balance = self.initial_balance
        self.shares = 0
        self.total_asset = self.initial_balance
        self.prev_asset = self.initial_balance
        
        self.returns_history = []
        self.portfolio_values = [self.initial_balance]
        
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        # FIX: DATA LEAKAGE / LOOK-AHEAD BIAS
        # If we trade at 'current_step' price, we must observe data from 'current_step - 1'
        # This simulates observing yesterday's close/indicators to trade today.
        
        lagged_idx = self.current_step - 1
        
        # Safety check for index
        if lagged_idx < 0:
            lagged_idx = 0
            
        features = self.features[lagged_idx]
        
        # Account State
        current_price = self.prices[self.current_step] # Price we are about to trade at
        
        if self.total_asset <= 0:
            position_weight = 0.0
        else:
            position_weight = (self.shares * current_price) / self.total_asset
            
        pf_return = np.log(self.total_asset / self.prev_asset) if self.prev_asset > 0 else 0.0
        
        return np.concatenate([features, [position_weight, pf_return]]).astype(np.float32)

    def step(self, action: np.ndarray):
        terminated = False
        truncated = False
        
        # 1. Execute Trade at Current Price
        target_weight = np.clip(action[0], -1, 1)
        current_price = self.prices[self.current_step]
        
        target_value = self.total_asset * target_weight
        current_holding_value = self.shares * current_price
        diff_value = target_value - current_holding_value
        trade_shares = int(diff_value / current_price)
        
        if trade_shares != 0:
            cost = abs(trade_shares * current_price) * self.commission
            self.balance -= cost
            self.shares += trade_shares
            self.balance -= (trade_shares * current_price)
            
        # 2. Advance Time
        self.prev_asset = self.total_asset
        self.current_step += 1
        
        # Check termination before accessing next price
        if self.current_step >= len(self.df) - 1 or self.current_step >= self.stop_step:
            terminated = True
        
        # 3. Mark to Market (Calculate Result)
        new_price = self.prices[self.current_step] # Price at t+1
        self.total_asset = self.balance + (self.shares * new_price)
        
        # 4. Reward Calculation
        step_return = np.log(self.total_asset / self.prev_asset) if self.prev_asset > 0 else 0.0
        self.returns_history.append(step_return)
        self.portfolio_values.append(self.total_asset)
        
        reward = self._calculate_reward()
        
        if self.total_asset < self.initial_balance * 0.1:
            terminated = True
            reward = -10.0 # Ruin penalty

        return self._get_observation(), reward, terminated, truncated, {
            "total_asset": self.total_asset,
            "return": (self.total_asset - self.initial_balance)/self.initial_balance
        }
    
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
        self.vec_norm_stats = {} # Store normalization stats
    
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
    
    # def train_ppo(self, env: TradingEnvRL) -> Dict:
    #     """Train Proximal Policy Optimization agent."""
    #     try:
    #         from stable_baselines3 import PPO
    #         from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    #     except ImportError as exc:
    #         raise RuntimeError("Install stable-baselines3 for PPO") from exc
        
    #     ppo_config = self.config.get("ppo", {})
    #     timesteps = ppo_config.get("total_timesteps", 10000)
        
    #     vec_env = DummyVecEnv([lambda: env])
        
    #     # Update vec
        
    #     # 2. Apply Normalization (CRITICAL FIX)
    #     # This scales rewards and observations to a standardized range (usually -1 to 1)
    #     # It helps the agent "see" small patterns and "feel" small rewards.
    #     vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
        
        # model = PPO(
        #     "MlpPolicy",
        #     vec_env,
        #     learning_rate=ppo_config.get("learning_rate", 3e-4),
        #     n_steps=ppo_config.get("n_steps", 2048),
        #     batch_size=ppo_config.get("batch_size", 64),
        #     gamma=ppo_config.get("gamma", 0.95),
        #     gae_lambda=ppo_config.get("gae_lambda", 0.95), # Generalized Advantage Estimation for variance reduction # LOWERED: Focuses on shorter-term volatility (better for active trading)
        #     ent_coef=ppo_config.get("ent_coef", 0.05), #Entropy: Prevents model from getting stuck (e.g., always holding cash) # INCREASED: Forces exploration (prevents getting stuck in Buy&Hold)
        #     clip_range=ppo_config.get("clip_range", 0.2),
        #     verbose=0,
        # )
        
        # logger.info("Training PPO for %d timesteps", timesteps)
        # model.learn(total_timesteps=timesteps)
        
        # return {"model": model, "vec_env": vec_env, "algorithm": "PPO"}
    def train_ppo(self, env: TradingEnvRL):
        # 1. Wrap Env
        vec_env = DummyVecEnv([lambda: env])
        # 2. Normalize Env
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        ppo_config = self.config.get("ppo", {})
        timesteps = ppo_config.get("total_timesteps", 10000)
        # model = PPO("MlpPolicy", vec_env, verbose=1, **ppo_config)
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            gamma=ppo_config.get("gamma", 0.95),
            gae_lambda=ppo_config.get("gae_lambda", 0.95), # Generalized Advantage Estimation for variance reduction # LOWERED: Focuses on shorter-term volatility (better for active trading)
            ent_coef=ppo_config.get("ent_coef", 0.05), #Entropy: Prevents model from getting stuck (e.g., always holding cash) # INCREASED: Forces exploration (prevents getting stuck in Buy&Hold)
            clip_range=ppo_config.get("clip_range", 0.2),
            verbose=1,
        )
        
        logger.info("Training PPO...")
        
        model.learn(total_timesteps=timesteps)
        logger.info("Training PPO for %d timesteps", timesteps)
        # FIX: Store the environment stats so we can save them later
        self.vec_norm_stats["ppo"] = vec_env
        self.models["ppo"] = model
        
        return {"model": model, "vec_env": vec_env}
    
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
        """Train Soft Actor-Critic agent with Normalization."""
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for SAC") from exc
        
        sac_config = self.config.get("sac", {})
        timesteps = sac_config.get("total_timesteps", 10000)
        
        # 1. Wrap Environment (Vectorization)
        # SB3 algorithms expect a vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        
        # 2. Apply Normalization (CRITICAL FIX)
        # Scales observations and rewards to a standard range (e.g., -1 to 1)
        # SAC is very sensitive to the scale of rewards/observations
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=sac_config.get("learning_rate", 3e-4),
            buffer_size=sac_config.get("buffer_size", 100000),
            batch_size=sac_config.get("batch_size", 256),
            gamma=sac_config.get("gamma", 0.99),
            tau=sac_config.get("tau", 0.005),
            ent_coef=sac_config.get("ent_coef", "auto"),
            verbose=1,
        )
        
        logger.info("Training SAC for %d timesteps", timesteps)
        model.learn(total_timesteps=timesteps)
        
        # 3. Store Normalization Stats for Saving
        # We need to save these stats to use the model later!
        self.vec_norm_stats["sac"] = vec_env
        self.models["sac"] = model
        
        return {"model": model, "vec_env": vec_env, "algorithm": "SAC"}
    
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
    
    def save_models(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
            
        for name, model in self.models.items():
            # Save Model
            model.save(os.path.join(path, f"{name}_model"))
            
            # FIX: Save Normalization Statistics
            if name in self.vec_norm_stats:
                norm_env = self.vec_norm_stats[name]
                norm_path = os.path.join(path, f"{name}_vecnormalize.pkl")
                norm_env.save(norm_path)
                print(f"Saved model and normalization stats for {norm_path}")
    
    def predict_action(self, obs: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """Predict action using trained model."""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        action, _ = model.predict(obs, deterministic=True)
        return action

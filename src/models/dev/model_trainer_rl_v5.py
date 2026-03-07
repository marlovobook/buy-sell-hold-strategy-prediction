"""
Reinforcement Learning model training for stock trading strategies.
Implements PPO, A2C, DDPG, TD3, SAC, DQN and custom reward functions.
"""

from __future__ import annotations

import joblib
import logging
import os
from typing import Dict, Optional, Tuple, Callable, List

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines3
try:
    from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.noise import NormalActionNoise
except ImportError:
    pass # Will handle explicit imports in methods

logger = logging.getLogger(__name__)

# =============================================================================
# 1. TRADING ENVIRONMENT (LEAK-FREE)
# =============================================================================

class TradingEnvRL(gym.Env):
    """
    Industry Standard Trading Environment (Fixed for Data Leakage).
    - Execution: Next Day Open (eliminates look-ahead bias).
    - Valuation: Current Day Close.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000,
        commission: float = 0.001,
        reward_scaling: float = 1e-4, 
        lookback_window: int = 30,
        max_steps: int = 1000, 
        reward_func: str = "profit", 
        random_start: bool = True,
        start_index: Optional[int] = None,
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.lookback_window = lookback_window
        self.max_steps = max_steps
        self.reward_func_name = reward_func
        self.random_start = random_start
        self.start_index = start_index
        
        # --- 1. Data Setup (Fixed) ---
        # prices_close: Used for Valuation (Mark-to-Market)
        # prices_open:  Used for Execution (Trade at t+1 Open)
        self.prices_close = self.df["close"].values.astype(np.float32)
        self.prices_open = self.df["open"].values.astype(np.float32) 
        
        # Filter features to ensure no leakage (remove 'close', 'open', 'future', etc.)
        exclude_cols = ["close", "date", "open", "high", "low", "volume", "adj close"]
        leak_keywords = ['future', 'target', 'label', 'next']
        self.feature_cols = [
            col for col in self.df.columns 
            if col.lower() not in exclude_cols 
            and not any(k in col.lower() for k in leak_keywords)
        ]
        
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.n_features = self.features.shape[1]
        
        # Action Space: Target Weight [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation Space
        obs_dim = self.n_features + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Internal state
        self.returns_history = []
        self.portfolio_values = []

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)

        options = options or {}
        random_start = options.get("random_start", self.random_start)
        start_index = options.get("start_index", self.start_index)

        if not random_start:
            base_start = self.lookback_window if start_index is None else int(start_index)
            self.current_step = max(self.lookback_window, base_start)
        else:
            # Need at least 2 steps remaining for t+1 execution
            high = len(self.df) - self.max_steps - 2 
            if high <= self.lookback_window:
                self.current_step = self.lookback_window
            else:
                self.current_step = np.random.randint(self.lookback_window, high)

        if self.max_steps is None:
            self.stop_step = len(self.df) - 2
        else:
            self.stop_step = min(self.current_step + self.max_steps, len(self.df) - 2)

        self.balance = self.initial_balance
        self.shares = 0
        self.total_asset = self.initial_balance
        self.prev_asset = self.initial_balance
        self.returns_history = []
        self.portfolio_values = [self.initial_balance]

        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        # Market Features (Step t)
        features = self.features[self.current_step]
        
        # Account State (Valued at Step t Close)
        current_price = self.prices_close[self.current_step]
        
        if self.total_asset <= 0:
            position_weight = 0.0
        else:
            position_weight = (self.shares * current_price) / self.total_asset
            
        pf_return = np.log(self.total_asset / self.prev_asset) if self.prev_asset > 0 else 0.0
        
        obs = np.concatenate([features, [position_weight, pf_return]])
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray):
        target_weight = np.clip(action[0], -1, 1)
        
        # --- EXECUTION LOGIC (Next Open) ---
        # Decision made at t Close. Executed at t+1 Open.
        execution_price = self.prices_open[self.current_step + 1]
        
        # Calculate shares to trade
        target_value = self.total_asset * target_weight
        current_holding_value = self.shares * execution_price
        diff_value = target_value - current_holding_value
        trade_shares = int(diff_value / execution_price)
        
        # Execute Trade
        if trade_shares != 0:
            transaction_cost = abs(trade_shares * execution_price) * self.commission
            self.balance -= transaction_cost
            self.shares += trade_shares
            self.balance -= (trade_shares * execution_price)
            
        # --- TIME STEP UPDATE ---
        self.prev_asset = self.total_asset
        self.current_step += 1
        
        # --- VALUATION (t+1 Close) ---
        new_price = self.prices_close[self.current_step]
        self.total_asset = self.balance + (self.shares * new_price)
        
        # Track history
        step_return = np.log(self.total_asset / self.prev_asset) if self.prev_asset > 0 else 0.0
        self.returns_history.append(step_return)
        self.portfolio_values.append(self.total_asset)
        
        reward = self._calculate_reward()

        terminated = self.current_step >= self.stop_step
        if self.total_asset < self.initial_balance * 0.1:
            terminated = True
            reward = -10.0 # Ruin penalty
            
        return self._get_observation(), reward, terminated, False, {
            "total_asset": self.total_asset,
            "return": (self.total_asset - self.initial_balance) / self.initial_balance
        }
    
    def _calculate_reward(self) -> float:
        if self.reward_func_name == "profit": return self._reward_profit()
        elif self.reward_func_name == "sharpe": return self._reward_sharpe()
        elif self.reward_func_name == "sortino": return self._reward_sortino()
        elif self.reward_func_name == "cvar": return self._reward_cvar()
        elif self.reward_func_name == "max_drawdown": return self._reward_max_drawdown()
        else: return self._reward_profit()
    
    def _reward_profit(self) -> float:
        return self.returns_history[-1] * 100 if len(self.returns_history) > 0 else 0.0
    
    def _reward_sharpe(self) -> float:
        if len(self.returns_history) < 2: return 0.0
        r = np.array(self.returns_history[-30:])
        return float(np.mean(r) / (np.std(r) + 1e-6))
    
    def _reward_sortino(self) -> float:
        if len(self.returns_history) < 2: return 0.0
        r = np.array(self.returns_history[-30:])
        downside = r[r < 0]
        if len(downside) == 0: return np.mean(r) * 10
        return float(np.mean(r) / (np.std(downside) + 1e-6))

    def _reward_cvar(self, alpha=0.05) -> float:
        if len(self.returns_history) < 10: return 0.0
        r = np.array(self.returns_history[-30:])
        var = np.quantile(r, alpha)
        return float(-r[r <= var].mean() * 100)

    def _reward_max_drawdown(self) -> float:
        if len(self.portfolio_values) < 2: return 0.0
        p = np.array(self.portfolio_values[-30:])
        rmax = np.maximum.accumulate(p)
        dd = (p - rmax) / (rmax + 1e-9)
        return float(-np.min(dd) * 100)


# =============================================================================
# 2. MODEL TRAINER (SMART WRAPPING)
# =============================================================================

class ModelTrainerRL:
    """Reinforcement Learning model trainer for stock trading."""
    
    def __init__(self, config: Dict):
        self.config = config or {}
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.artifacts_dir = self.config.get("artifacts_dir", "artifacts")
        self.vecnorm_paths = {} 
        
        # We store env params to re-create envs during inference
        self.env_params = {} 
    
    def prepare_environment(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        reward_func: str = "sharpe",
    ) -> Tuple[TradingEnvRL, TradingEnvRL]:
        
        env_config = self.config.get("environment", {})
        
        # Save params for later use in load_inference_env
        self.env_params = {
            "initial_balance": env_config.get("initial_balance", 10000),
            "commission": env_config.get("commission", 0.001),
            "lookback_window": env_config.get("lookback_window", 30),
            "max_steps": env_config.get("max_steps", 1000),
            "reward_func": reward_func
        }
        
        # We DO NOT apply StandardScaler here anymore.
        # We rely on VecNormalize inside the training loop.
        
        env_train = TradingEnvRL(df_train, **self.env_params, random_start=True)
        # 2. Test Env: Needs max_steps=None
        # FIX: Create a copy of params and update the key to avoid the collision
        test_params = self.env_params.copy()
        test_params["max_steps"] = None 
        
        # Now pass the modified dict. 
        # Note: random_start is NOT in env_params, so it is safe to pass explicitly.
        env_test = TradingEnvRL(df_test, **test_params, random_start=False)
        return env_train, env_test

    def train_ppo(self, env: TradingEnvRL) -> Dict:
        """Train PPO with VecNormalize."""
        ppo_config = self.config.get("ppo", {})
        timesteps = ppo_config.get("total_timesteps", 10000)
        
        # 1. Wrap env in VecNormalize
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
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
        
        # Save stats path
        os.makedirs(self.artifacts_dir, exist_ok=True)
        vec_path = os.path.join(self.artifacts_dir, "ppo_vecnormalize.pkl")
        vec_env.save(vec_path)
        self.vecnorm_paths["ppo"] = vec_path
        
        return {"model": model, "vec_env": vec_env}

    def load_inference_env(self, df: pd.DataFrame, algorithm: str = "ppo"):
        """
        Creates a test environment and wraps it with the saved Normalization stats.
        Use this for inference/backtesting to ensure inputs match training distribution.
        """
        algo_key = algorithm.lower()
        if algo_key not in self.vecnorm_paths:
            logger.warning(f"No normalization stats found for {algorithm}. Returning raw env.")
            return TradingEnvRL(df, **self.env_params, random_start=False, max_steps=None)
        
        # 1. Create Raw Env
        # Force max_steps=None for full backtest
        params = self.env_params.copy()
        params['max_steps'] = None
        params['random_start'] = False
        raw_env = TradingEnvRL(df, **params)
        
        # 2. Wrap in DummyVecEnv
        vec_env = DummyVecEnv([lambda: raw_env])
        
        # 3. Load Stats
        vec_path = self.vecnorm_paths[algo_key]
        try:
            norm_env = VecNormalize.load(vec_path, vec_env)
            norm_env.training = False     # Do not update stats during test
            norm_env.norm_reward = False  # Return raw rewards
            logger.info(f"Loaded VecNormalize stats for {algorithm}")
            return norm_env
        except Exception as e:
            logger.error(f"Failed to load VecNormalize: {e}")
            return vec_env

    def evaluate_over_seeds(self, model, env, seeds, algorithm="ppo"):
        """Evaluates model using the correct normalization wrapper."""
        results = {"total_return": [], "sharpe_ratio": [], "max_drawdown": []}
        
        # Determine if we need to wrap the env
        is_vec = False
        if algorithm.lower() in self.vecnorm_paths:
            # Create a temporary wrapped env for evaluation using the env's dataframe
            # Note: This assumes 'env' passed here is a raw TradingEnvRL
            eval_env = self.load_inference_env(env.df, algorithm)
            is_vec = True
        else:
            eval_env = env

        for seed in seeds:
            obs = eval_env.reset()
            if not is_vec: obs = obs[0] # Handle gym API diff
            
            done = False
            while not done:
                # VecEnv returns (obs, rewards, dones, infos)
                # Gym returns (obs, reward, terminated, truncated, info)
                if is_vec:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, dones, infos = eval_env.step(action)
                    done = dones[0]
                    final_info = infos[0]
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = eval_env.step(action)
                    done = term or trunc
                    final_info = info
            
            # Calculate metrics from the environment's internal history
            # We access the underlying env (unwrapped)
            base_env = eval_env.venv.envs[0] if is_vec else eval_env
            
            # Simple manual calculation to verify
            total_ret = (base_env.total_asset - base_env.initial_balance) / base_env.initial_balance
            results["total_return"].append(total_ret)
            
            # Calculate Sharpe
            if len(base_env.returns_history) > 1:
                r = np.array(base_env.returns_history)
                sharpe = np.mean(r) / (np.std(r) + 1e-6) * np.sqrt(252)
                results["sharpe_ratio"].append(sharpe)
            else:
                results["sharpe_ratio"].append(0.0)

            # Max DD
            p = np.array(base_env.portfolio_values)
            rmax = np.maximum.accumulate(p)
            dd = (p - rmax) / (rmax + 1e-9)
            results["max_drawdown"].append(np.min(dd))

        # Aggregate
        def stats(arr):
            return {"mean": float(np.mean(arr)), "ci95": [float(np.min(arr)), float(np.max(arr))]} # Simple min/max for CI
            
        return {k: stats(v) for k, v in results.items()}

    def save_models(self, filepath: str):
        """Save models and their normalization stats."""
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
        # Save Models
        for name, model in self.models.items():
            if model:
                model.save(os.path.join(filepath, f"{name}_model"))
        
        # Copy Normalization Stats
        import shutil
        for algo, src_path in self.vecnorm_paths.items():
            dst_path = os.path.join(filepath, f"{algo}_vecnormalize.pkl")
            try:
                shutil.copyfile(src_path, dst_path)
                logger.info(f"Saved {algo} stats to {dst_path}")
            except Exception as e:
                logger.warning(f"Could not copy stats: {e}")
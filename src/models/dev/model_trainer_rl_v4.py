"""
Reinforcement Learning model training for stock trading strategies.
Implements PPO, A2C, DDPG, TD3, SAC, DQN and custom reward functions.
"""

from __future__ import annotations

import joblib
import logging
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple, Callable, List
import os

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class TradingEnvRL(gym.Env):
    """
    Industry Standard Trading Environment.
    - Uses Target Weights for actions (Portfolio Management).
    - Normalizes observations.
    - Implements random episode starts.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000,
        commission: float = 0.001,
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
        # We separate prices to prevent leakage:
        # - prices_open: USED FOR TRADING (Execution at t+1 Open)
        # - prices_close: USED FOR VALUATION (Mark-to-market at t Close)
        self.prices_close = self.df["close"].values.astype(np.float32)
        self.prices_open = self.df["open"].values.astype(np.float32) 
        
        # Filter features to ensure no leakage
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
        
        # Internal state tracking
        self.returns_history = []
        self.portfolio_values = []
        
    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)

        options = options or {}
        random_start = options.get("random_start", self.random_start)
        start_index = options.get("start_index", self.start_index)

        # Logic to handle random starts safely
        if not random_start:
            base_start = self.lookback_window if start_index is None else int(start_index)
            self.current_step = max(self.lookback_window, base_start)
        else:
            # We need at least 2 steps remaining (current close -> next open)
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
        # 1. Market Features for current step
        features = self.features[self.current_step]
        
        # 2. Account State (Using CLOSE price for valuation)
        # FIX: Changed self.prices to self.prices_close
        current_price = self.prices_close[self.current_step]
        
        if self.total_asset <= 0:
            position_weight = 0.0
        else:
            position_weight = (self.shares * current_price) / self.total_asset
            
        # 3. Portfolio Return
        pf_return = np.log(self.total_asset / self.prev_asset) if self.prev_asset > 0 else 0.0
        
        obs = np.concatenate([features, [position_weight, pf_return]])
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray):
        target_weight = np.clip(action[0], -1, 1)
        
        # --- EXECUTION LOGIC (Next Open) ---
        # We are at step t. We execute at t+1 Open.
        execution_price = self.prices_open[self.current_step + 1]
        
        # Determine trade size
        target_value = self.total_asset * target_weight
        current_holding_value = self.shares * execution_price
        diff_value = target_value - current_holding_value
        
        trade_shares = int(diff_value / execution_price)
        
        # Execute
        if trade_shares != 0:
            transaction_cost = abs(trade_shares * execution_price) * self.commission
            self.balance -= transaction_cost
            self.shares += trade_shares
            self.balance -= (trade_shares * execution_price)
            
        # --- TIME STEP UPDATE ---
        self.prev_asset = self.total_asset
        self.current_step += 1
        
        # --- VALUATION (Current Close) ---
        # FIX: Changed self.prices to self.prices_close
        new_price = self.prices_close[self.current_step]
        self.total_asset = self.balance + (self.shares * new_price)
        
        # Track history
        step_return = np.log(self.total_asset / self.prev_asset) if self.prev_asset > 0 else 0.0
        self.returns_history.append(step_return)
        self.portfolio_values.append(self.total_asset)
        
        # Calculate Reward
        reward = self._calculate_reward()

        terminated = self.current_step >= self.stop_step
        
        # Ruin condition
        if self.total_asset < self.initial_balance * 0.1:
            terminated = True
            reward = -10.0 # Heavy penalty for blowing up account
            
        return self._get_observation(), reward, terminated, False, {
            "total_asset": self.total_asset,
            "return": (self.total_asset - self.initial_balance) / self.initial_balance
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
        if len(self.returns_history) > 0:
            return self.returns_history[-1] * 100
        return 0.0
    
    def _reward_sharpe(self) -> float:
        if len(self.returns_history) < 2:
            return 0.0
        returns = np.array(self.returns_history[-30:])
        return float(np.mean(returns) / (np.std(returns) + 1e-6))
    
    def _reward_sortino(self) -> float:
        if len(self.returns_history) < 2:
            return 0.0
        returns = np.array(self.returns_history[-30:])
        downside = returns[returns < 0]
        if len(downside) == 0: return np.mean(returns) * 10
        return float(np.mean(returns) / (np.std(downside) + 1e-6))
    
    def _reward_cvar(self, alpha=0.05) -> float:
        if len(self.returns_history) < 10: return 0.0
        returns = np.array(self.returns_history[-30:])
        var = np.quantile(returns, alpha)
        cvar = returns[returns <= var].mean()
        return float(-cvar * 100)

    def _reward_max_drawdown(self) -> float:
        if len(self.portfolio_values) < 2: return 0.0
        p = np.array(self.portfolio_values[-30:])
        rmax = np.maximum.accumulate(p)
        dd = (p - rmax) / rmax
        return float(-np.min(dd) * 100)

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares}, Total: {self.total_asset:.2f}")


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
        # New: artifacts directory and normalization state
        self.artifacts_dir = self.config.get("artifacts_dir", "artifacts")
        self.vecnorm_paths: Dict[str, str] = {}
        self.scaler: Optional[StandardScaler] = None
    
    def prepare_environment(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        reward_func: str = "sharpe",
    ) -> Tuple[TradingEnvRL, TradingEnvRL]:
        """Prepare train and test environments with feature scaling."""
        env_config = self.config.get("environment", {})
        initial_balance = env_config.get("initial_balance", 10000)
        commission = env_config.get("commission", 0.001)
        lookback = env_config.get("lookback_window", 30)
        scale_features = env_config.get("scale_features", True)
        random_start = env_config.get("random_start", True)
        max_steps = env_config.get("max_steps", 1000)
        
        # Identify feature columns (must match env's logic)
        exclude_cols = ["close", "date", "open", "high", "low", "volume"]
        feat_cols = [c for c in df_train.columns if c not in exclude_cols]
        
        df_train = df_train.copy()
        df_test = df_test.copy()
        
        # # Fit scaler on train, apply to both
        # if scale_features and len(feat_cols) > 0:
        #     self.scaler = StandardScaler()
        #     self.scaler.fit(df_train[feat_cols].astype(np.float32))
        #     df_train[feat_cols] = self.scaler.transform(df_train[feat_cols].astype(np.float32))
        #     df_test[feat_cols] = self.scaler.transform(df_test[feat_cols].astype(np.float32))
        #     logger.info("Fitted and applied feature scaling (StandardScaler)")
        
        env_train = TradingEnvRL(
            df_train,
            initial_balance=initial_balance,
            commission=commission,
            reward_func=reward_func,
            lookback_window=lookback,
            random_start=True,  # Always randomize for training
            max_steps=max_steps,
        )
        
        env_test = TradingEnvRL(
            df_test,
            initial_balance=initial_balance,
            commission=commission,
            reward_func=reward_func,
            lookback_window=lookback,
            random_start=False,  # Deterministic for evaluation
            max_steps=None,  # Cover full test span
        )
        
        self.env_train = env_train
        self.env_test = env_test
        
        logger.info("Prepared training env with %d steps (scaled=%s)", len(df_train), scale_features)
        logger.info("Prepared testing env with %d steps (scaled=%s)", len(df_test), scale_features)
        
        return env_train, env_test

    def train_ppo(self, env: TradingEnvRL) -> Dict:
        """Train Proximal Policy Optimization agent."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for PPO") from exc
        
        ppo_config = self.config.get("ppo", {})
        timesteps = ppo_config.get("total_timesteps", 10000)
        
        vec_env = DummyVecEnv([lambda: env])
        
        # Update vec
        
        # 2. Apply Normalization (CRITICAL FIX)
        # This scales rewards and observations to a standardized range (usually -1 to 1)
        # It helps the agent "see" small patterns and "feel" small rewards.
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
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
            verbose=0,
        )
        
        logger.info("Training PPO for %d timesteps", timesteps)
        model.learn(total_timesteps=timesteps)
        
        return {"model": model, "vec_env": vec_env, "algorithm": "PPO"}
    
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
        """Train Deep Deterministic Policy Gradient agent with VecNormalize."""
        try:
            from stable_baselines3 import DDPG
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from stable_baselines3.common.noise import NormalActionNoise
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for DDPG") from exc
        
        ddpg_config = self.config.get("ddpg", {})
        timesteps = ddpg_config.get("total_timesteps", 10000)
        
        # Wrap in VecEnv for normalization
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # Action noise for exploration
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        model = DDPG(
            "MlpPolicy",
            vec_env,
            learning_rate=ddpg_config.get("learning_rate", 1e-3),
            buffer_size=ddpg_config.get("buffer_size", 100000),
            batch_size=ddpg_config.get("batch_size", 100),
            gamma=ddpg_config.get("gamma", 0.99),
            action_noise=action_noise,
            verbose=0,
        )
        
        logger.info("Training DDPG for %d timesteps", timesteps)
        model.learn(total_timesteps=timesteps)
        
        # Persist VecNormalize stats
        os.makedirs(self.artifacts_dir, exist_ok=True)
        vec_path = os.path.join(self.artifacts_dir, "ddpg_vecnormalize.pkl")
        vec_env.save(vec_path)
        self.vecnorm_paths["ddpg"] = vec_path
        logger.info("Saved DDPG VecNormalize stats to %s", vec_path)
        
        return {"model": model, "vec_env": vec_env, "algorithm": "DDPG"}
    
    def train_td3(self, env: TradingEnvRL) -> Dict:
        """Train Twin Delayed DDPG agent with VecNormalize."""
        try:
            from stable_baselines3 import TD3
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from stable_baselines3.common.noise import NormalActionNoise
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for TD3") from exc
        
        td3_config = self.config.get("td3", {})
        timesteps = td3_config.get("total_timesteps", 10000)
        
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        model = TD3(
            "MlpPolicy",
            vec_env,
            learning_rate=td3_config.get("learning_rate", 1e-3),
            buffer_size=td3_config.get("buffer_size", 100000),
            batch_size=td3_config.get("batch_size", 100),
            gamma=td3_config.get("gamma", 0.99),
            action_noise=action_noise,
            verbose=0,
        )
        
        logger.info("Training TD3 for %d timesteps", timesteps)
        model.learn(total_timesteps=timesteps)
        
        # Persist VecNormalize stats
        os.makedirs(self.artifacts_dir, exist_ok=True)
        vec_path = os.path.join(self.artifacts_dir, "td3_vecnormalize.pkl")
        vec_env.save(vec_path)
        self.vecnorm_paths["td3"] = vec_path
        logger.info("Saved TD3 VecNormalize stats to %s", vec_path)
        
        return {"model": model, "vec_env": vec_env, "algorithm": "TD3"}
    
    def train_sac(self, env: TradingEnvRL) -> Dict:
        """Train Soft Actor-Critic agent with VecNormalize."""
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 for SAC") from exc
        
        sac_config = self.config.get("sac", {})
        timesteps = sac_config.get("total_timesteps", 10000)
        
        vec_env = DummyVecEnv([lambda: env])
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
            verbose=0,
        )
        
        logger.info("Training SAC for %d timesteps", timesteps)
        model.learn(total_timesteps=timesteps)
        
        # Persist VecNormalize stats
        os.makedirs(self.artifacts_dir, exist_ok=True)
        vec_path = os.path.join(self.artifacts_dir, "sac_vecnormalize.pkl")
        vec_env.save(vec_path)
        self.vecnorm_paths["sac"] = vec_path
        logger.info("Saved SAC VecNormalize stats to %s", vec_path)
        
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
    
    def _wrap_eval_env_for_algorithm(self, env: TradingEnvRL, algorithm: str):
        """
        Wrap evaluation env with saved VecNormalize stats if available.
        Returns (rollout_env, base_env, is_vectorized).
        """
        if algorithm in ["PPO", "DDPG", "TD3", "SAC"] and algorithm.lower() in self.vecnorm_paths:
            try:
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                eval_vec = DummyVecEnv([lambda: env])
                # Load and apply saved normalization
                vec_path = self.vecnorm_paths[algorithm.lower()]
                eval_vec = VecNormalize.load(vec_path, eval_vec)
                eval_vec.training = False  # Disable running mean/std updates
                eval_vec.norm_reward = False  # Don't normalize rewards during eval
                base_env = eval_vec.venv.envs[0]
                logger.info("Wrapped eval env with %s VecNormalize stats", algorithm)
                return eval_vec, base_env, True
            except Exception as e:
                logger.warning("Failed to load VecNormalize for %s: %s. Falling back to raw env.", algorithm, e)
        
        # Fallback: use env directly (no vectorization)
        return env, env, False

    def _compute_metrics_from_env(self, base_env: TradingEnvRL) -> Dict:
        """Extract performance metrics from base environment."""
        final_value = base_env.total_asset
        total_return = (final_value - base_env.initial_balance) / base_env.initial_balance

        # Sharpe ratio
        if len(base_env.returns_history) > 1:
            r = np.array(base_env.returns_history)
            sharpe = float(np.mean(r) / (np.std(r) + 1e-6) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Sortino ratio
        if len(base_env.returns_history) > 1:
            r = np.array(base_env.returns_history)
            downside = r[r < 0]
            if len(downside) == 0:
                sortino = float(np.mean(r) * np.sqrt(252))
            else:
                sortino = float(np.mean(r) / (np.std(downside) + 1e-6) * np.sqrt(252))
        else:
            sortino = 0.0

        # Max drawdown
        p = np.array(base_env.portfolio_values) if len(base_env.portfolio_values) else np.array([base_env.initial_balance])
        running_max = np.maximum.accumulate(p)
        drawdown = (p - running_max) / (running_max + 1e-9)
        max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        return {
            "total_return": float(total_return),
            "final_value": float(final_value),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
        }

    def evaluate_model(
        self, 
        model, 
        env: TradingEnvRL, 
        n_episodes: int = 1, 
        seed: Optional[int] = None,
        algorithm: Optional[str] = None,
    ) -> Dict:
        """Evaluate trained RL model with VecNormalize support."""
        rollout_env, base_env, is_vec = self._wrap_eval_env_for_algorithm(env, algorithm or "")
        all_rewards = []

        for ep in range(n_episodes):
            if is_vec:
                obs = rollout_env.reset(seed=seed)
                done = np.array([False])
                ep_reward = 0.0
                while not done[0]:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, infos = rollout_env.step(action)
                    ep_reward += float(reward[0])
                all_rewards.append(ep_reward)
            else:
                obs, _ = rollout_env.reset(seed=seed)
                done = False
                ep_reward = 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _info = rollout_env.step(action)
                    done = terminated or truncated
                    ep_reward += float(reward)
                all_rewards.append(ep_reward)

        metrics = self._compute_metrics_from_env(base_env)
        metrics["avg_reward"] = float(np.mean(all_rewards))
        
        logger.info(
            "Evaluation - Return: %.2f%%, Sharpe: %.3f, Sortino: %.3f, Max DD: %.2f%%",
            metrics["total_return"] * 100,
            metrics["sharpe_ratio"],
            metrics["sortino_ratio"],
            metrics["max_drawdown"] * 100,
        )
        
        return metrics

    def _mean_ci(self, arr: List[float], alpha: float = 0.05) -> Tuple[float, float, float]:
        """Return (mean, lower, upper) for 100*(1-alpha)% confidence interval."""
        x = np.array(arr, dtype=float)
        n = len(x)
        if n <= 1:
            return float(np.mean(x)), float(np.mean(x)), float(np.mean(x))
        mean = float(np.mean(x))
        se = float(np.std(x, ddof=1) / np.sqrt(n))
        try:
            from scipy.stats import t
            tcrit = float(t.ppf(1 - alpha/2, df=n-1))
        except Exception:
            tcrit = 1.96  # Fallback
        half = tcrit * se
        return mean, mean - half, mean + half

    def evaluate_over_seeds(
        self, 
        model, 
        env: TradingEnvRL, 
        seeds: List[int],
        algorithm: Optional[str] = None,
    ) -> Dict:
        """Run multiple seeds and report mean ± 95% CI for key metrics."""
        tr, sh, so, dd = [], [], [], []
        for s in seeds:
            m = self.evaluate_model(model, env, n_episodes=1, seed=s, algorithm=algorithm)
            tr.append(m["total_return"])
            sh.append(m["sharpe_ratio"])
            so.append(m["sortino_ratio"])
            dd.append(m["max_drawdown"])
        
        def pack(vals):
            mean, lo, hi = self._mean_ci(vals)
            return {"mean": float(mean), "ci95": [float(lo), float(hi)]}
        
        result = {
            "total_return": pack(tr),
            "sharpe_ratio": pack(sh),
            "sortino_ratio": pack(so),
            "max_drawdown": pack(dd),
            "seeds": seeds,
        }
        
        logger.info(
            "Multi-seed evaluation (n=%d): Sharpe %.3f ± CI(%.3f, %.3f)",
            len(seeds),
            result["sharpe_ratio"]["mean"],
            result["sharpe_ratio"]["ci95"][0],
            result["sharpe_ratio"]["ci95"][1],
        )
        
        return result
    
    def train_all(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        reward_func: str = "sharpe",
    ) -> Dict:
        """Train all configured RL algorithms with multi-seed evaluation."""
        # Split data
        split_idx = int(len(df) * (1 - test_size))
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        
        # Prepare environments (with feature scaling)
        env_train, env_test = self.prepare_environment(df_train, df_test, reward_func)
        
        # Multi-seed evaluation config
        eval_cfg = self.config.get("evaluation", {})
        seeds = eval_cfg.get("seeds", None)
        if seeds is None:
            n = int(eval_cfg.get("n_seeds", 3))
            seeds = list(range(42, 42 + n))
        
        algorithms = self.config.get("algorithms", ["ppo", "a2c"])
        
        for algo in algorithms:
            try:
                if algo == "ppo":
                    logger.info("Training PPO")
                    result = self.train_ppo(env_train)
                    if result["model"]:
                        agg = self.evaluate_over_seeds(result["model"], env_test, seeds, algorithm="PPO")
                        result["metrics_agg"] = agg
                        single = self.evaluate_model(result["model"], env_test, algorithm="PPO", seed=seeds[0])
                        result["metrics"] = single
                        self.results["ppo"] = result
                        self.models["ppo"] = result["model"]
                
                elif algo == "a2c":
                    logger.info("Training A2C")
                    result = self.train_a2c(env_train)
                    if result["model"]:
                        agg = self.evaluate_over_seeds(result["model"], env_test, seeds, algorithm="A2C")
                        result["metrics_agg"] = agg
                        single = self.evaluate_model(result["model"], env_test, algorithm="A2C", seed=seeds[0])
                        result["metrics"] = single
                        self.results["a2c"] = result
                        self.models["a2c"] = result["model"]
                
                elif algo == "ddpg":
                    logger.info("Training DDPG")
                    result = self.train_ddpg(env_train)
                    if result["model"]:
                        agg = self.evaluate_over_seeds(result["model"], env_test, seeds, algorithm="DDPG")
                        result["metrics_agg"] = agg
                        single = self.evaluate_model(result["model"], env_test, algorithm="DDPG", seed=seeds[0])
                        result["metrics"] = single
                        self.results["ddpg"] = result
                        self.models["ddpg"] = result["model"]
                
                elif algo == "td3":
                    logger.info("Training TD3")
                    result = self.train_td3(env_train)
                    if result["model"]:
                        agg = self.evaluate_over_seeds(result["model"], env_test, seeds, algorithm="TD3")
                        result["metrics_agg"] = agg
                        single = self.evaluate_model(result["model"], env_test, algorithm="TD3", seed=seeds[0])
                        result["metrics"] = single
                        self.results["td3"] = result
                        self.models["td3"] = result["model"]
                
                elif algo == "sac":
                    logger.info("Training SAC")
                    result = self.train_sac(env_train)
                    if result["model"]:
                        agg = self.evaluate_over_seeds(result["model"], env_test, seeds, algorithm="SAC")
                        result["metrics_agg"] = agg
                        single = self.evaluate_model(result["model"], env_test, algorithm="SAC", seed=seeds[0])
                        result["metrics"] = single
                        self.results["sac"] = result
                        self.models["sac"] = result["model"]
                
                elif algo == "dqn":
                    logger.info("Attempting DQN")
                    result = self.train_dqn(env_train)
                    if result.get("model"):
                        agg = self.evaluate_over_seeds(result["model"], env_test, seeds, algorithm="DQN")
                        result["metrics_agg"] = agg
                        single = self.evaluate_model(result["model"], env_test, algorithm="DQN", seed=seeds[0])
                        result["metrics"] = single
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
        """Save trained RL models, VecNormalize stats, and feature scaler."""
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # Save models
        for name, model in self.models.items():
            if model is not None:
                model_path = os.path.join(filepath, f"{name}_model")
                model.save(model_path)
                logger.info(f"Saved {name} model to {model_path}")
        
        # Copy VecNormalize stats files
        for algo, vec_path in self.vecnorm_paths.items():
            if os.path.exists(vec_path):
                target = os.path.join(filepath, f"{algo}_vecnormalize.pkl")
                if vec_path != target:
                    try:
                        import shutil
                        shutil.copyfile(vec_path, target)
                        logger.info(f"Saved {algo} VecNormalize stats to {target}")
                    except Exception as e:
                        logger.warning(f"Failed to copy VecNormalize for {algo}: {e}")
        
        # Save feature scaler
        if self.scaler is not None:
            scaler_path = os.path.join(filepath, "feature_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved feature scaler to {scaler_path}")
        
        # Save results summary (with multi-seed metrics)
        import json
        results_summary = {}
        for name, result in self.results.items():
            if "metrics_agg" in result:
                results_summary[name] = {
                    "single_run": result.get("metrics", {}),
                    "multi_seed": result.get("metrics_agg", {}),
                }
            elif "metrics" in result:
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
    
    def walk_forward_evaluation(
        self,
        df: pd.DataFrame,
        window_size: int = 200,
        step_size: int = 50,
        reward_func: str = "sharpe",
        algorithm: str = "sac",
        seeds: Optional[List[int]] = None,
    ) -> Dict:
        """
        Walk-forward (expanding) evaluation:
        - Initial window: [0, window_size)
        - Test: [window_size, window_size + step_size)
        - Then slide by step_size and repeat
        """
        if seeds is None:
            seeds = [42, 43, 44]
        
        if len(df) < window_size + step_size:
            logger.warning("Dataset too small for walk-forward. Use regular train/test split instead.")
            return {}
        
        fold_results = []
        idx = 0
        fold_num = 0
        
        while idx + window_size + step_size <= len(df):
            fold_num += 1
            train_end = idx + window_size
            test_end = min(train_end + step_size, len(df))
            
            df_train_wf = df.iloc[idx:train_end].copy()
            df_test_wf = df.iloc[train_end:test_end].copy()
            
            logger.info(
                "Walk-forward Fold %d: Train [%d:%d], Test [%d:%d]",
                fold_num, idx, train_end, train_end, test_end
            )
            
            try:
                # Prepare fresh environments for this fold
                env_train_wf, env_test_wf = self.prepare_environment(
                    df_train_wf, df_test_wf, reward_func=reward_func
                )
                
                # Train algorithm
                result = None
                if algorithm == "ppo":
                    result = self.train_ppo(env_train_wf)
                elif algorithm == "a2c":
                    result = self.train_a2c(env_train_wf)
                elif algorithm == "ddpg":
                    result = self.train_ddpg(env_train_wf)
                elif algorithm == "td3":
                    result = self.train_td3(env_train_wf)
                elif algorithm == "sac":
                    result = self.train_sac(env_train_wf)
                else:
                    logger.error("Unknown algorithm: %s", algorithm)
                    continue
                
                if result and result.get("model"):
                    # Evaluate with multiple seeds
                    metrics_agg = self.evaluate_over_seeds(
                        result["model"],
                        env_test_wf,
                        seeds=seeds,
                        algorithm=algorithm.upper(),
                    )
                    
                    fold_results.append({
                        "fold": fold_num,
                        "train_idx": [idx, train_end],
                        "test_idx": [train_end, test_end],
                        "metrics": metrics_agg,
                    })
            
            except Exception as e:
                logger.error("Fold %d failed: %s", fold_num, e)
                continue
            
            idx += step_size
        
        # Aggregate across folds
        agg = self._aggregate_walk_forward(fold_results)
        
        return {
            "fold_results": fold_results,
            "aggregate": agg,
            "algorithm": algorithm,
            "window_size": window_size,
            "step_size": step_size,
        }
    
    def _aggregate_walk_forward(self, fold_results: List[Dict]) -> Dict:
        """Aggregate walk-forward fold results."""
        if not fold_results:
            return {}
        
        all_returns = []
        all_sharpe = []
        all_sortino = []
        all_dd = []
        
        for fold in fold_results:
            metrics = fold["metrics"]
            all_returns.append(metrics["total_return"]["mean"])
            all_sharpe.append(metrics["sharpe_ratio"]["mean"])
            all_sortino.append(metrics["sortino_ratio"]["mean"])
            all_dd.append(metrics["max_drawdown"]["mean"])
        
        def fold_pack(vals):
            mean, lo, hi = self._mean_ci(vals)
            return {"mean": float(mean), "ci95": [float(lo), float(hi)]}
        
        return {
            "n_folds": len(fold_results),
            "total_return": fold_pack(all_returns),
            "sharpe_ratio": fold_pack(all_sharpe),
            "sortino_ratio": fold_pack(all_sortino),
            "max_drawdown": fold_pack(all_dd),
        }

# Deep Reinforcement Learning Agent Performance Metrics

## Overview

This document describes the comprehensive performance evaluation metrics for Deep RL agents implemented in `model_trainer_rl_v2_2.py`.

## Functions Added

### 1. `calculate_rl_performance_metrics()`

**Purpose**: Calculate comprehensive metrics for evaluating Deep RL agent performance.

**Parameters**:
- `episode_rewards` (List[float]): List of cumulative rewards per episode
- `episode_lengths` (List[int]): List of episode lengths in steps
- `episode_returns` (List[float]): List of portfolio returns per episode
- `success_threshold` (float): Threshold for defining successful episodes (default: 0.0)

**Returns**: Dictionary with the following metric categories:

#### 1. Cumulative Reward Metrics
- `total_cumulative_reward`: Sum of all episode rewards
- `mean_episode_reward`: Average reward per episode
- `std_episode_reward`: Standard deviation of episode rewards
- `median_episode_reward`: Median episode reward
- `max_episode_reward`: Best episode reward
- `min_episode_reward`: Worst episode reward

#### 2. Average Reward Per Step
- `avg_reward_per_step`: Total reward / total steps
- `mean_step_reward`: Average reward per step across episodes
- `std_step_reward`: Variability in step rewards

#### 3. Success Rate
- `success_rate_pct`: Percentage of successful episodes
- `successful_episodes`: Count of episodes exceeding threshold
- `total_episodes`: Total number of episodes

#### 4. Episode Length Statistics
- `mean_episode_length`: Average steps per episode
- `std_episode_length`: Variability in episode length
- `max_episode_length`: Longest episode
- `min_episode_length`: Shortest episode
- `total_steps`: Cumulative steps across all episodes

#### 5. Learning Curve Metrics
- `learning_improvement`: Difference between late and early phase performance
- `learning_rate_pct`: Percentage improvement in learning
- `trend_slope`: Linear trend in rewards over time
- `early_phase_mean_reward`: Performance in first 1/3 of training
- `middle_phase_mean_reward`: Performance in middle 1/3
- `late_phase_mean_reward`: Performance in final 1/3

#### 6. Stability & Convergence Metrics
- `coefficient_of_variation`: CV% (lower = more stable)
- `variance_reduction_pct`: Variance decrease from first to second half
- `first_half_variance`: Variance in first half of training
- `second_half_variance`: Variance in second half of training

#### 7. Best/Worst Episode Performance
- `best_episode_idx`: Index of best episode
- `best_episode_reward`: Reward of best episode
- `best_episode_return`: Return of best episode
- `worst_episode_idx`: Index of worst episode
- `worst_episode_reward`: Reward of worst episode
- `worst_episode_return`: Return of worst episode

#### 8. Portfolio-Specific Metrics
- `mean_return_pct`: Average portfolio return
- `std_return_pct`: Return volatility
- `sharpe_ratio`: Risk-adjusted return metric
- `win_rate_pct`: Percentage of profitable episodes
- `max_drawdown_pct`: Maximum drawdown in returns

---

### 2. `track_episode_metrics()`

**Purpose**: Track metrics for a single episode during training.

**Parameters**:
- `episode_reward` (float): Cumulative reward for the episode
- `episode_length` (int): Number of steps in the episode
- `episode_return` (float): Portfolio return for the episode
- `timestep` (int): Current training timestep

**Usage**: Call this at the end of each training episode to build up metrics history.

---

### 3. `get_learning_curve_data()`

**Purpose**: Get data for plotting learning curves.

**Returns**: Dictionary containing:
- `episodes`: Episode numbers
- `rewards`: Raw rewards per episode
- `smoothed_rewards`: Moving average of rewards
- `lengths`: Episode lengths
- `returns`: Portfolio returns

---

## Usage Examples

### Example 1: Basic Usage in Jupyter Notebook

```python
import numpy as np
from src.models.model_trainer_rl_v2_2 import ModelTrainerRL

# Initialize trainer
config = {"environment": {"initial_balance": 10000}}
trainer = ModelTrainerRL(config)

# After training, calculate metrics
metrics = trainer.calculate_rl_performance_metrics(
    episode_rewards=[45.2, 67.8, 89.3, ...],  # Your episode rewards
    episode_lengths=[456, 512, 487, ...],      # Steps per episode
    episode_returns=[0.045, 0.067, 0.089, ...], # Portfolio returns
    success_threshold=0.0  # Episodes with positive return = success
)

# Display key metrics
print(f"Mean Reward: {metrics['mean_episode_reward']:.2f}")
print(f"Success Rate: {metrics['success_rate_pct']:.1f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Learning Improvement: {metrics['learning_improvement']:.2f}")
```

### Example 2: Track During Training

```python
# During training loop
for episode in range(n_episodes):
    obs = env.reset()
    episode_reward = 0
    step_count = 0
    
    while not done:
        action = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        step_count += 1
    
    # Track episode
    portfolio_return = info['return']
    trainer.track_episode_metrics(
        episode_reward=episode_reward,
        episode_length=step_count,
        episode_return=portfolio_return,
        timestep=total_timesteps
    )

# After training, calculate comprehensive metrics
metrics = trainer.calculate_rl_performance_metrics()
```

### Example 3: Visualize Learning Curves

```python
import matplotlib.pyplot as plt

# Get learning curve data
curve_data = trainer.get_learning_curve_data()

# Plot rewards
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(curve_data['episodes'], curve_data['rewards'], alpha=0.3, label='Raw')
plt.plot(curve_data['episodes'][9:], curve_data['smoothed_rewards'], 
         linewidth=2, label='Smoothed')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
returns_pct = np.array(curve_data['returns']) * 100
plt.plot(curve_data['episodes'], returns_pct)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Return (%)')
plt.title('Portfolio Returns')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Example 4: Compare Reward Functions

```python
# Train with different reward functions
reward_functions = ['profit', 'sharpe', 'sortino', 'cvar']
results = {}

for reward_func in reward_functions:
    env = TradingEnvRL(df_train, reward_func=reward_func)
    trainer = ModelTrainerRL(config)
    
    # Train...
    # Collect metrics...
    
    metrics = trainer.calculate_rl_performance_metrics()
    results[reward_func] = metrics

# Compare
for func, metrics in results.items():
    print(f"\n{func.upper()}:")
    print(f"  Mean Reward: {metrics['mean_episode_reward']:.2f}")
    print(f"  Success Rate: {metrics['success_rate_pct']:.1f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
```

### Example 5: Complete Notebook Workflow

```python
# Cell 1: Setup
import pandas as pd
from src.models.model_trainer_rl_v2_2 import ModelTrainerRL, TradingEnvRL

# Cell 2: Load Data
df = pd.read_csv('data/processed/AAPL_processed.csv')
split = int(len(df) * 0.8)
df_train, df_test = df[:split], df[split:]

# Cell 3: Initialize & Train
config = {"ppo": {"total_timesteps": 50000}}
trainer = ModelTrainerRL(config)
env_train, env_test = trainer.prepare_environment(df_train, df_test, reward_func="sharpe")
result = trainer.train_ppo(env_train)

# Cell 4: Evaluate
n_episodes = 20
for ep in range(n_episodes):
    # Run episode...
    trainer.track_episode_metrics(reward, length, return_val, step)

# Cell 5: Calculate & Display Metrics
metrics = trainer.calculate_rl_performance_metrics()

# Cell 6: Visualize
curve_data = trainer.get_learning_curve_data()
# Plot as shown in Example 3...

# Cell 7: Export Report
import json
with open('rl_metrics_report.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

---

## Interpreting Metrics

### Good Performance Indicators:
- **High Success Rate** (>60%): Agent finds profitable strategies frequently
- **Positive Learning Improvement**: Late phase rewards >> early phase rewards
- **Positive Trend Slope**: Consistent improvement over time
- **Low CV** (<50%): Stable, consistent performance
- **High Variance Reduction** (>50%): Agent is converging
- **High Sharpe Ratio** (>1.0): Good risk-adjusted returns

### Warning Signs:
- **Negative Trend Slope**: Agent not learning or degrading
- **High CV** (>100%): Very unstable performance
- **Negative Variance Reduction**: Becoming more erratic over time
- **Low Success Rate** (<40%): Struggling to find profitable trades

### Optimal Training Indicators:
1. **Early Phase**: High exploration, high variance
2. **Middle Phase**: Variance decreasing, rewards improving
3. **Late Phase**: Low variance, high rewards, plateau

---

## Advanced Usage

### Custom Success Thresholds

```python
# Conservative: Only count large wins
metrics = trainer.calculate_rl_performance_metrics(
    success_threshold=0.05  # 5% return required
)

# Liberal: Any non-negative return
metrics = trainer.calculate_rl_performance_metrics(
    success_threshold=-0.01  # Allow small losses
)
```

### Comparing Agents

```python
agents = ['ppo', 'sac', 'td3']
comparison = []

for agent_name in agents:
    # Load agent metrics...
    metrics = trainer.calculate_rl_performance_metrics(...)
    comparison.append({
        'Agent': agent_name,
        'Mean Reward': metrics['mean_episode_reward'],
        'Sharpe': metrics['sharpe_ratio'],
        'Success Rate': metrics['success_rate_pct']
    })

df_comparison = pd.DataFrame(comparison)
print(df_comparison)
```

---

## Full Example Script

See `example_rl_metrics_evaluation.py` for complete, runnable examples including:
- Simulated data evaluation
- Real training integration
- Comprehensive visualizations
- Multi-agent comparisons
- Report generation

Run in terminal:
```bash
python example_rl_metrics_evaluation.py
```

Or import in Jupyter:
```python
from example_rl_metrics_evaluation import (
    example_1_basic_evaluation,
    example_3_visualize_performance,
    example_5_export_metrics_report
)
```

---

## References

- **Cumulative Reward**: Total sum of rewards (primary RL metric)
- **Sharpe Ratio**: (Mean Return) / (Std Return) - risk-adjusted metric
- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
- **CVaR**: Conditional Value at Risk - tail risk measure
- **Coefficient of Variation**: (Std / Mean) * 100 - normalized stability
- **Learning Curve**: Plot of performance over training time

---

## Notes

1. All metrics are calculated automatically and logged to console
2. Metrics can be calculated on-the-fly or from stored training history
3. Works with any reward function (profit, sharpe, sortino, cvar, max_drawdown)
4. Compatible with all RL algorithms (PPO, SAC, TD3, A2C, DDPG, DQN)
5. Handles variable episode lengths automatically
6. Provides both raw and smoothed learning curves
7. Includes financial metrics specific to trading (Sharpe, drawdown, win rate)

---

*For questions or issues, refer to the inline documentation in `model_trainer_rl_v2_2.py`*

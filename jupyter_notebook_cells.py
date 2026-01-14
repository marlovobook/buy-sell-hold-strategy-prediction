# ============================================================================
# JUPYTER NOTEBOOK: RL Agent Performance Evaluation
# Copy these cells into your notebook for quick evaluation
# ============================================================================

# CELL 1: Imports
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.model_trainer_rl_v2_2 import ModelTrainerRL, TradingEnvRL
import yaml

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


# CELL 2: Load Configuration and Data
# ============================================================================
# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data
df = pd.read_csv('data/processed/AAPL_processed.csv', parse_dates=['date'])
split_idx = int(len(df) * 0.8)
df_train = df[:split_idx].copy()
df_test = df[split_idx:].copy()

print(f"Training data: {len(df_train)} rows")
print(f"Testing data: {len(df_test)} rows")


# CELL 3: Initialize Trainer and Environments
# ============================================================================
trainer = ModelTrainerRL(config.get('rl', {}))
env_train, env_test = trainer.prepare_environment(
    df_train, 
    df_test, 
    reward_func="sharpe"  # Options: "profit", "sharpe", "sortino", "cvar", "max_drawdown"
)


# CELL 4: Train Agent (PPO Example)
# ============================================================================
print("🚀 Training PPO Agent...")
result = trainer.train_ppo(env_train)
model = result['model']
vec_env = result['vec_env']
print("✅ Training complete!")


# CELL 5: Evaluate Agent and Collect Metrics
# ============================================================================
n_eval_episodes = 20
episode_rewards = []
episode_lengths = []
episode_returns = []

print(f"\n📊 Evaluating agent over {n_eval_episodes} episodes...")

for episode in range(n_eval_episodes):
    obs = env_test.reset()[0]
    done = False
    episode_reward = 0
    step_count = 0
    initial_balance = env_test.initial_balance
    
    while not done:
        # Normalize observation
        obs_normalized = vec_env.normalize_obs(obs)
        action, _ = model.predict(obs_normalized, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        step_count += 1
    
    final_balance = info.get('total_asset', initial_balance)
    portfolio_return = (final_balance - initial_balance) / initial_balance
    
    episode_rewards.append(episode_reward)
    episode_lengths.append(step_count)
    episode_returns.append(portfolio_return)
    
    print(f"Episode {episode+1:2d}/{n_eval_episodes}: "
          f"Reward={episode_reward:7.2f}, "
          f"Return={portfolio_return*100:6.2f}%, "
          f"Steps={step_count:4d}")

print("✅ Evaluation complete!")


# CELL 6: Calculate Comprehensive Metrics
# ============================================================================
metrics = trainer.calculate_rl_performance_metrics(
    episode_rewards=episode_rewards,
    episode_lengths=episode_lengths,
    episode_returns=episode_returns,
    success_threshold=0.0  # Episodes with positive return = success
)

# The metrics are automatically printed by the function
# Access specific metrics like:
print(f"\n🎯 KEY TAKEAWAYS:")
print(f"Mean Episode Reward: {metrics['mean_episode_reward']:.2f}")
print(f"Success Rate: {metrics['success_rate_pct']:.1f}%")
print(f"Portfolio Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Learning Improvement: {metrics['learning_improvement']:.2f}")


# CELL 7: Visualize Performance
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Deep RL Agent Performance Analysis', fontsize=16, fontweight='bold')

episodes = np.arange(len(episode_rewards))

# 1. Learning Curve
ax1 = axes[0, 0]
ax1.plot(episodes, episode_rewards, alpha=0.3, label='Raw Rewards')
window = 5
if len(episode_rewards) >= window:
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax1.plot(episodes[window-1:], smoothed, linewidth=2, label=f'{window}-Episode MA', color='red')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Cumulative Reward')
ax1.set_title('Learning Curve: Episode Rewards')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Portfolio Returns
ax2 = axes[0, 1]
returns_pct = np.array(episode_returns) * 100
ax2.plot(episodes, returns_pct, marker='o', markersize=3, linewidth=1)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Return (%)')
ax2.set_title('Portfolio Returns per Episode')
ax2.grid(True, alpha=0.3)

# 3. Episode Lengths
ax3 = axes[0, 2]
ax3.plot(episodes, episode_lengths, color='green', marker='o', markersize=3, linewidth=1)
ax3.set_xlabel('Episode')
ax3.set_ylabel('Steps')
ax3.set_title('Episode Length Evolution')
ax3.grid(True, alpha=0.3)

# 4. Reward Distribution
ax4 = axes[1, 0]
ax4.hist(episode_rewards, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
ax4.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
ax4.set_xlabel('Episode Reward')
ax4.set_ylabel('Frequency')
ax4.set_title('Reward Distribution')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Return Distribution
ax5 = axes[1, 1]
ax5.hist(returns_pct, bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
ax5.axvline(np.mean(returns_pct), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(returns_pct):.2f}%')
ax5.set_xlabel('Episode Return (%)')
ax5.set_ylabel('Frequency')
ax5.set_title('Return Distribution')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Cumulative Success Rate
ax6 = axes[1, 2]
success_mask = np.array(episode_returns) > 0
cumulative_success_rate = np.cumsum(success_mask) / (episodes + 1) * 100
ax6.plot(episodes, cumulative_success_rate, color='purple', linewidth=2)
ax6.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
ax6.set_xlabel('Episode')
ax6.set_ylabel('Success Rate (%)')
ax6.set_title('Cumulative Success Rate')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('rl_agent_performance_analysis.png', dpi=300, bbox_inches='tight')
print("📊 Visualization saved as 'rl_agent_performance_analysis.png'")
plt.show()


# CELL 8: Display Metrics Table
# ============================================================================
metrics_df = pd.DataFrame([
    {'Metric': 'Total Episodes', 'Value': metrics['total_episodes']},
    {'Metric': 'Mean Episode Reward', 'Value': f"{metrics['mean_episode_reward']:.2f}"},
    {'Metric': 'Success Rate (%)', 'Value': f"{metrics['success_rate_pct']:.1f}"},
    {'Metric': 'Mean Return (%)', 'Value': f"{metrics['mean_return_pct']:.2f}"},
    {'Metric': 'Sharpe Ratio', 'Value': f"{metrics['sharpe_ratio']:.3f}"},
    {'Metric': 'Win Rate (%)', 'Value': f"{metrics['win_rate_pct']:.1f}"},
    {'Metric': 'Max Drawdown (%)', 'Value': f"{metrics['max_drawdown_pct']:.2f}"},
    {'Metric': 'Learning Improvement', 'Value': f"{metrics['learning_improvement']:.2f}"},
    {'Metric': 'Trend Slope', 'Value': f"{metrics['trend_slope']:.6f}"},
    {'Metric': 'Coefficient of Variation (%)', 'Value': f"{metrics['coefficient_of_variation']:.2f}"},
])

print("\n📊 KEY PERFORMANCE METRICS:")
print(metrics_df.to_string(index=False))


# CELL 9: Export Metrics to JSON
# ============================================================================
import json

# Save to file
with open('rl_performance_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ Metrics exported to 'rl_performance_metrics.json'")


# CELL 10: Compare with Buy & Hold (Optional)
# ============================================================================
# Calculate Buy & Hold return
bh_return = (df_test['close'].iloc[-1] - df_test['close'].iloc[0]) / df_test['close'].iloc[0]
bh_return_pct = bh_return * 100

# Calculate Strategy return
strategy_return_pct = metrics['mean_return_pct']

print(f"\n📈 STRATEGY vs BUY & HOLD:")
print(f"Strategy Return:      {strategy_return_pct:.2f}%")
print(f"Buy & Hold Return:    {bh_return_pct:.2f}%")
print(f"Outperformance:       {strategy_return_pct - bh_return_pct:.2f}%")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Strategy', 'Buy & Hold']
values = [strategy_return_pct, bh_return_pct]
colors = ['green' if v > 0 else 'red' for v in values]
bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top',
            fontsize=12, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Return (%)', fontsize=12)
ax.set_title('Strategy Performance vs Buy & Hold', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('strategy_vs_buyhold.png', dpi=300, bbox_inches='tight')
print("📊 Comparison chart saved as 'strategy_vs_buyhold.png'")
plt.show()


# ============================================================================
# OPTIONAL: Advanced Analysis
# ============================================================================

# CELL 11: Statistical Tests (Optional)
# ============================================================================
from scipy import stats

# T-test: Are returns significantly different from zero?
t_stat, p_value = stats.ttest_1samp(episode_returns, 0)
print(f"\n📊 STATISTICAL SIGNIFICANCE:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("✅ Returns are statistically significant (p < 0.05)")
else:
    print("⚠️ Returns are NOT statistically significant")

# Normality test
_, norm_p_value = stats.shapiro(episode_returns)
print(f"\nNormality test p-value: {norm_p_value:.4f}")
if norm_p_value > 0.05:
    print("✅ Returns are approximately normally distributed")
else:
    print("⚠️ Returns are NOT normally distributed")


# CELL 12: Risk Metrics (Optional)
# ============================================================================
returns_array = np.array(episode_returns)

# Value at Risk (VaR) at 95% confidence
var_95 = np.percentile(returns_array, 5)
print(f"\n📉 RISK METRICS:")
print(f"Value at Risk (95%): {var_95*100:.2f}%")
print(f"  → 5% chance of losing more than {abs(var_95)*100:.2f}%")

# Conditional VaR (CVaR) / Expected Shortfall
cvar_95 = returns_array[returns_array <= var_95].mean()
print(f"Conditional VaR (95%): {cvar_95*100:.2f}%")
print(f"  → Expected loss in worst 5% of cases")

# Downside deviation
downside_returns = returns_array[returns_array < 0]
downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
print(f"Downside Deviation: {downside_std*100:.2f}%")


# ============================================================================
# END OF NOTEBOOK
# ============================================================================

print("\n" + "="*70)
print("✅ ALL ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated Files:")
print("  📊 rl_agent_performance_analysis.png")
print("  📊 strategy_vs_buyhold.png")
print("  📄 rl_performance_metrics.json")
print("\nNext Steps:")
print("  1. Review the performance metrics")
print("  2. Analyze the visualizations")
print("  3. Tune hyperparameters if needed")
print("  4. Test with different reward functions")
print("  5. Run on different stocks/timeframes")
print("="*70)

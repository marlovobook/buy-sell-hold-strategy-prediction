"""
Example: Deep Reinforcement Learning Agent Performance Evaluation
==================================================================

This script demonstrates how to use the RL performance metrics functions
to evaluate and visualize the performance of a trained RL agent.

Use this code in your Jupyter notebooks for comprehensive agent evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import yaml

# Import your RL trainer
from src.models.model_trainer_rl_v2_2 import ModelTrainerRL, TradingEnvRL

# Set style for better visualizations
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


# ============================================================================
# EXAMPLE 1: Evaluate RL Agent with Simulated Training Data
# ============================================================================

def example_1_basic_evaluation():
    """
    Basic example: Calculate RL metrics from simulated episode data.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic RL Performance Evaluation")
    print("=" * 70)
    
    # Simulate episode data from training
    # In real usage, these would come from your actual training runs
    np.random.seed(42)
    
    # Simulate 100 episodes with learning progression
    n_episodes = 100
    episode_rewards = []
    episode_lengths = []
    episode_returns = []
    
    for i in range(n_episodes):
        # Simulate improvement over time
        base_reward = -50 + (i / n_episodes) * 150  # -50 to 100
        noise = np.random.randn() * 30
        reward = base_reward + noise
        episode_rewards.append(reward)
        
        # Episode lengths vary
        length = np.random.randint(200, 1000)
        episode_lengths.append(length)
        
        # Portfolio returns (scaled from rewards)
        portfolio_return = reward / 1000  # Convert to return percentage
        episode_returns.append(portfolio_return)
    
    # Initialize trainer (just for metrics calculation)
    config = {"environment": {"initial_balance": 10000}}
    trainer = ModelTrainerRL(config)
    
    # Calculate comprehensive metrics
    metrics = trainer.calculate_rl_performance_metrics(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        episode_returns=episode_returns,
        success_threshold=0.0  # Episodes with positive return = success
    )
    
    # Display key metrics
    print("\n📊 KEY PERFORMANCE INDICATORS:")
    print("-" * 70)
    print(f"Total Episodes:              {metrics['total_episodes']}")
    print(f"Mean Episode Reward:         {metrics['mean_episode_reward']:.2f} ± {metrics['std_episode_reward']:.2f}")
    print(f"Success Rate:                {metrics['success_rate_pct']:.1f}%")
    print(f"Average Reward per Step:     {metrics['avg_reward_per_step']:.4f}")
    print(f"Mean Episode Length:         {metrics['mean_episode_length']:.0f} steps")
    print()
    print(f"📈 LEARNING PROGRESS:")
    print(f"Early Phase Avg Reward:      {metrics['early_phase_mean_reward']:.2f}")
    print(f"Late Phase Avg Reward:       {metrics['late_phase_mean_reward']:.2f}")
    print(f"Learning Improvement:        {metrics['learning_improvement']:.2f} ({metrics['learning_rate_pct']:.1f}%)")
    print(f"Trend Slope:                 {metrics['trend_slope']:.4f}")
    print()
    print(f"🎯 STABILITY & CONVERGENCE:")
    print(f"Coefficient of Variation:    {metrics['coefficient_of_variation']:.2f}%")
    print(f"Variance Reduction:          {metrics['variance_reduction_pct']:.1f}%")
    print()
    print(f"💼 PORTFOLIO PERFORMANCE:")
    print(f"Mean Return:                 {metrics['mean_return_pct']:.2f}%")
    print(f"Sharpe Ratio:                {metrics['sharpe_ratio']:.3f}")
    print(f"Win Rate:                    {metrics['win_rate_pct']:.1f}%")
    print(f"Max Drawdown:                {metrics['max_drawdown_pct']:.2f}%")
    print("=" * 70)
    
    return metrics


# ============================================================================
# EXAMPLE 2: Evaluate RL Agent from Real Training Session
# ============================================================================

def example_2_real_training_evaluation():
    """
    Realistic example: Train an agent and evaluate its performance.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Real Training with Performance Tracking")
    print("=" * 70)
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load your data
    df = pd.read_csv('data/processed/AAPL_processed.csv', parse_dates=['date'])
    
    # Split data
    split_idx = int(len(df) * 0.8)
    df_train = df[:split_idx].copy()
    df_test = df[split_idx:].copy()
    
    # Initialize trainer
    trainer = ModelTrainerRL(config.get('rl', {}))
    
    # Prepare environments
    env_train, env_test = trainer.prepare_environment(
        df_train, 
        df_test, 
        reward_func="sharpe"  # or "profit", "sortino", "cvar"
    )
    
    # Train PPO agent (this will automatically track metrics if implemented)
    print("\n🚀 Training PPO Agent...")
    result = trainer.train_ppo(env_train)
    model = result['model']
    vec_env = result['vec_env']
    
    # Manually collect episode metrics during evaluation
    print("\n📊 Evaluating trained agent...")
    n_eval_episodes = 20
    episode_rewards = []
    episode_lengths = []
    episode_returns = []
    
    for episode in range(n_eval_episodes):
        obs = env_test.reset()[0]
        done = False
        episode_reward = 0
        step_count = 0
        initial_balance = env_test.initial_balance
        
        while not done:
            # Normalize observation for prediction
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
        
        print(f"Episode {episode+1}/{n_eval_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Return={portfolio_return*100:.2f}%, "
              f"Steps={step_count}")
    
    # Calculate comprehensive metrics
    metrics = trainer.calculate_rl_performance_metrics(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        episode_returns=episode_returns,
        success_threshold=0.0
    )
    
    return trainer, metrics


# ============================================================================
# EXAMPLE 3: Visualize Learning Curves and Performance
# ============================================================================

def example_3_visualize_performance(episode_rewards: List[float], 
                                   episode_returns: List[float],
                                   episode_lengths: List[int]):
    """
    Create comprehensive visualizations of RL agent performance.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Performance Visualization")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Deep RL Agent Performance Analysis', fontsize=16, fontweight='bold')
    
    episodes = np.arange(len(episode_rewards))
    
    # 1. Learning Curve (Rewards)
    ax1 = axes[0, 0]
    ax1.plot(episodes, episode_rewards, alpha=0.3, label='Raw Rewards')
    
    # Moving average
    window = min(10, len(episode_rewards) // 5)
    if window > 1:
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
    ax4.hist(episode_rewards, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Return Distribution
    ax5 = axes[1, 1]
    ax5.hist(returns_pct, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
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


# ============================================================================
# EXAMPLE 4: Compare Multiple Reward Functions
# ============================================================================

def example_4_compare_reward_functions():
    """
    Compare performance across different reward functions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Comparing Reward Functions")
    print("=" * 70)
    
    # Simulate different reward function results
    reward_functions = ['profit', 'sharpe', 'sortino', 'cvar']
    
    comparison_data = []
    
    for reward_func in reward_functions:
        # Simulate training with different reward functions
        np.random.seed(42 + len(reward_func))  # Different seed per function
        
        n_episodes = 50
        episode_rewards = np.random.randn(n_episodes) * 20 + (50 if reward_func == 'sharpe' else 30)
        episode_returns = episode_rewards / 1000
        episode_lengths = np.random.randint(200, 800, n_episodes)
        
        # Calculate metrics
        trainer = ModelTrainerRL({"environment": {"initial_balance": 10000}})
        metrics = trainer.calculate_rl_performance_metrics(
            episode_rewards=list(episode_rewards),
            episode_lengths=list(episode_lengths),
            episode_returns=list(episode_returns)
        )
        
        comparison_data.append({
            'Reward Function': reward_func,
            'Mean Reward': metrics['mean_episode_reward'],
            'Success Rate (%)': metrics['success_rate_pct'],
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Learning Improvement': metrics['learning_improvement'],
            'Stability (CV%)': metrics['coefficient_of_variation']
        })
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n📊 REWARD FUNCTION COMPARISON:")
    print(df_comparison.to_string(index=False))
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Mean Reward
    axes[0].bar(df_comparison['Reward Function'], df_comparison['Mean Reward'], color='skyblue')
    axes[0].set_title('Mean Episode Reward')
    axes[0].set_ylabel('Reward')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Success Rate
    axes[1].bar(df_comparison['Reward Function'], df_comparison['Success Rate (%)'], color='lightgreen')
    axes[1].set_title('Success Rate')
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Sharpe Ratio
    axes[2].bar(df_comparison['Reward Function'], df_comparison['Sharpe Ratio'], color='coral')
    axes[2].set_title('Sharpe Ratio')
    axes[2].set_ylabel('Sharpe Ratio')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('reward_function_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Comparison chart saved as 'reward_function_comparison.png'")
    plt.show()
    
    return df_comparison


# ============================================================================
# EXAMPLE 5: Export Metrics to Report
# ============================================================================

def example_5_export_metrics_report(metrics: Dict, filename: str = "rl_performance_report.txt"):
    """
    Export comprehensive metrics to a text report.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Exporting Performance Report")
    print("=" * 70)
    
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DEEP REINFORCEMENT LEARNING AGENT PERFORMANCE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. CUMULATIVE REWARD METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Cumulative Reward:     {metrics['total_cumulative_reward']:.2f}\n")
        f.write(f"Mean Episode Reward:         {metrics['mean_episode_reward']:.2f}\n")
        f.write(f"Std Episode Reward:          {metrics['std_episode_reward']:.2f}\n")
        f.write(f"Median Episode Reward:       {metrics['median_episode_reward']:.2f}\n")
        f.write(f"Max Episode Reward:          {metrics['max_episode_reward']:.2f}\n")
        f.write(f"Min Episode Reward:          {metrics['min_episode_reward']:.2f}\n\n")
        
        f.write("2. AVERAGE REWARD PER STEP\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Reward per Step:     {metrics['avg_reward_per_step']:.6f}\n")
        f.write(f"Mean Step Reward:            {metrics['mean_step_reward']:.6f}\n")
        f.write(f"Std Step Reward:             {metrics['std_step_reward']:.6f}\n\n")
        
        f.write("3. SUCCESS RATE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Success Rate:                {metrics['success_rate_pct']:.2f}%\n")
        f.write(f"Successful Episodes:         {metrics['successful_episodes']}\n")
        f.write(f"Total Episodes:              {metrics['total_episodes']}\n\n")
        
        f.write("4. EPISODE LENGTH STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Episode Length:         {metrics['mean_episode_length']:.0f} steps\n")
        f.write(f"Std Episode Length:          {metrics['std_episode_length']:.2f}\n")
        f.write(f"Max Episode Length:          {metrics['max_episode_length']:.0f}\n")
        f.write(f"Min Episode Length:          {metrics['min_episode_length']:.0f}\n")
        f.write(f"Total Steps:                 {metrics['total_steps']}\n\n")
        
        f.write("5. LEARNING CURVE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Early Phase Mean Reward:     {metrics['early_phase_mean_reward']:.2f}\n")
        f.write(f"Middle Phase Mean Reward:    {metrics['middle_phase_mean_reward']:.2f}\n")
        f.write(f"Late Phase Mean Reward:      {metrics['late_phase_mean_reward']:.2f}\n")
        f.write(f"Learning Improvement:        {metrics['learning_improvement']:.2f}\n")
        f.write(f"Learning Rate:               {metrics['learning_rate_pct']:.2f}%\n")
        f.write(f"Trend Slope:                 {metrics['trend_slope']:.6f}\n\n")
        
        f.write("6. STABILITY & CONVERGENCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Coefficient of Variation:    {metrics['coefficient_of_variation']:.2f}%\n")
        f.write(f"Variance Reduction:          {metrics['variance_reduction_pct']:.2f}%\n")
        f.write(f"First Half Variance:         {metrics['first_half_variance']:.2f}\n")
        f.write(f"Second Half Variance:        {metrics['second_half_variance']:.2f}\n\n")
        
        f.write("7. PORTFOLIO PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Return:                 {metrics['mean_return_pct']:.2f}%\n")
        f.write(f"Std Return:                  {metrics['std_return_pct']:.2f}%\n")
        f.write(f"Sharpe Ratio:                {metrics['sharpe_ratio']:.4f}\n")
        f.write(f"Win Rate:                    {metrics['win_rate_pct']:.2f}%\n")
        f.write(f"Max Drawdown:                {metrics['max_drawdown_pct']:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Performance report exported to '{filename}'")


# ============================================================================
# MAIN EXECUTION (for Jupyter Notebook)
# ============================================================================

if __name__ == "__main__":
    """
    Run this in your Jupyter notebook cell-by-cell:
    
    # Cell 1: Basic evaluation with simulated data
    metrics1 = example_1_basic_evaluation()
    
    # Cell 2: Real training evaluation (requires data)
    # trainer, metrics2 = example_2_real_training_evaluation()
    
    # Cell 3: Visualize performance
    # Assuming you have episode data from training:
    episode_rewards = trainer.training_metrics['episode_rewards']
    episode_returns = trainer.training_metrics['episode_returns']
    episode_lengths = trainer.training_metrics['episode_lengths']
    example_3_visualize_performance(episode_rewards, episode_returns, episode_lengths)
    
    # Cell 4: Compare reward functions
    df_comparison = example_4_compare_reward_functions()
    
    # Cell 5: Export report
    example_5_export_metrics_report(metrics1)
    """
    
    # Run Example 1 (works without real data)
    print("\n🚀 Running Example 1: Basic Evaluation with Simulated Data\n")
    metrics = example_1_basic_evaluation()
    
    # Visualize simulated data
    np.random.seed(42)
    n_episodes = 100
    episode_rewards = [-50 + (i / n_episodes) * 150 + np.random.randn() * 30 for i in range(n_episodes)]
    episode_returns = [r / 1000 for r in episode_rewards]
    episode_lengths = [np.random.randint(200, 1000) for _ in range(n_episodes)]
    
    example_3_visualize_performance(episode_rewards, episode_returns, episode_lengths)
    
    # Compare reward functions
    df_comparison = example_4_compare_reward_functions()
    
    # Export report
    example_5_export_metrics_report(metrics)
    
    print("\n✅ All examples completed successfully!")
    print("📂 Check your directory for generated visualizations and reports.")

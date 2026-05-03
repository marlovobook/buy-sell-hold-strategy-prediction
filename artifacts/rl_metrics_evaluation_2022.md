# RL Metrics Evaluation Report (processed_2022)

Generated at: 2026-04-21T20:26:26

## Executive Summary (One Page)

### Study Design

This study evaluates a PPO-based reinforcement learning policy on eight equities (AAPL, AMZN, TSLA, BAC, MDU, CWCO, NEE, DUK) using `*_processed_2022.csv` datasets.

The protocol applies an 80/20 train-test split per asset, 10,000 PPO training timesteps per asset, and 20 out-of-sample evaluation episodes. Statistical validation includes bootstrap 95% confidence intervals and permutation-based hypothesis tests.

### Principal Findings

- All eight evaluations completed successfully.
- Most assets show positive mean return with significant one-sample tests for $H_1: \mu_{return} > 0$.
- Under this 2022 configuration, RL underperforms buy-and-hold across assets based on paired benchmark tests (negative mean return differences with significant p-values).
- Variance characteristics are asset-dependent; several assets show substantial reward variance, while near-zero return variance in others produces numerically inflated Sharpe ratios.

### Relevance to Reviewer Feedback

- Statistical validation is explicitly reported via confidence intervals and hypothesis tests.
- RL variance analysis is explicitly reported via reward variance, late/early variance ratio, and rolling variance trend slope.

### Interpretation and Limitations

- The policy can be profitable in absolute terms for multiple assets, but comparative performance against buy-and-hold is weaker in this setup.
- A single-year horizon (2022) limits generalizability across market regimes.
- Near-zero variance episodes can destabilize ratio-based metrics (especially Sharpe).

### Recommended Robustness Extensions

1. Multi-seed evaluation with pooled confidence intervals.
2. Walk-forward multi-year testing.
3. Additional non-RL baselines.
4. Transaction-cost and slippage sensitivity analysis.

---

## Setup

- Stocks: AAPL, AMZN, TSLA, BAC, MDU, CWCO, NEE, DUK
- Dataset pattern: data/processed/{TICKER}_processed_2022.csv
- Model: PPO (from model_trainer_rl_v2_3_buyhold)
- PPO total timesteps: 10,000 (batch-eval runtime setting)
- Evaluation episodes per stock: 20
- Statistical validation: bootstrap CI + permutation tests

## Summary Table

| Stock | Mean Return % | Std Return % | Sharpe | Success % | Reward Var | p-value (Return > 0) | p-value (RL vs B&H) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 16.4277 | 0.2581 | 63.6452 | 100.0 | 16.8373 | 0.000200 | 0.000200 |
| AMZN | 5.0787 | 0.0210 | 242.3845 | 100.0 | 8.0728 | 0.000200 | 0.000200 |
| TSLA | 0.3541 | 0.0000 | 354093.3199 | 100.0 | 0.0825 | 0.000200 | 0.000200 |
| BAC | 9.1285 | 0.2638 | 34.5995 | 100.0 | 4.4929 | 0.000200 | 0.000200 |
| MDU | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | 1.000000 | 0.000200 |
| CWCO | 3.7009 | 0.0882 | 41.9417 | 100.0 | 4.6770 | 0.000200 | 0.000200 |
| NEE | 13.3259 | 0.7816 | 17.0489 | 100.0 | 141.0666 | 0.000200 | 0.000200 |
| DUK | 0.0966 | 0.0000 | 96572.8688 | 100.0 | 0.0000 | 0.000200 | 0.000200 |

## Per-Stock Statistical Validation

### AAPL

- Mean episode reward: 28.6933 ± 4.1033
- Mean return: 16.4277% (std 0.2581%)
- Sharpe ratio: 63.6452
- Max drawdown: 0.0000%
- Variance ratio (late/early): 3.7829
- Rolling variance trend slope: 1.700529
- Reward mean 95% CI: [27.044358, 30.672971]
- Return mean 95% CI: [0.163315, 0.165462]
- One-sample permutation test (H1: mean return > 0): p = 0.000200
- Paired benchmark test (RL vs B&H): p = 0.000200
- Mean difference (RL - B&H) 95% CI: [-0.201810, -0.183951]

### AMZN

- Mean episode reward: 16.0494 ± 2.8413
- Mean return: 5.0787% (std 0.0210%)
- Sharpe ratio: 242.3845
- Max drawdown: 0.0000%
- Variance ratio (late/early): 0.8130
- Rolling variance trend slope: -0.043224
- Reward mean 95% CI: [14.866358, 17.212606]
- Return mean 95% CI: [0.050702, 0.050873]
- One-sample permutation test (H1: mean return > 0): p = 0.000200
- Paired benchmark test (RL vs B&H): p = 0.000200
- Mean difference (RL - B&H) 95% CI: [-0.100381, -0.066985]

### TSLA

- Mean episode reward: 11.8782 ± 0.2873
- Mean return: 0.3541% (std 0.0000%)
- Sharpe ratio: 354093.3199
- Max drawdown: 0.0000%
- Variance ratio (late/early): 1.3932
- Rolling variance trend slope: 0.002090
- Reward mean 95% CI: [11.754838, 12.003533]
- Return mean 95% CI: [0.003541, 0.003541]
- One-sample permutation test (H1: mean return > 0): p = 0.000200
- Paired benchmark test (RL vs B&H): p = 0.000200
- Mean difference (RL - B&H) 95% CI: [-0.153730, -0.133160]

### BAC

- Mean episode reward: 26.8173 ± 2.1197
- Mean return: 9.1285% (std 0.2638%)
- Sharpe ratio: 34.5995
- Max drawdown: 0.0000%
- Variance ratio (late/early): 4.1223
- Rolling variance trend slope: 0.185285
- Reward mean 95% CI: [25.810448, 27.709549]
- Return mean 95% CI: [0.090152, 0.092418]
- One-sample permutation test (H1: mean return > 0): p = 0.000200
- Paired benchmark test (RL vs B&H): p = 0.000200
- Mean difference (RL - B&H) 95% CI: [-0.123606, -0.105134]

### MDU

- Mean episode reward: 0.0000 ± 0.0000
- Mean return: 0.0000% (std 0.0000%)
- Sharpe ratio: 0.0000
- Max drawdown: 0.0000%
- Variance ratio (late/early): 0.0000
- Rolling variance trend slope: 0.000000
- Reward mean 95% CI: [0.000000, 0.000000]
- Return mean 95% CI: [0.000000, 0.000000]
- One-sample permutation test (H1: mean return > 0): p = 1.000000
- Paired benchmark test (RL vs B&H): p = 0.000200
- Mean difference (RL - B&H) 95% CI: [-0.227544, -0.214971]

### CWCO

- Mean episode reward: 11.7148 ± 2.1626
- Mean return: 3.7009% (std 0.0882%)
- Sharpe ratio: 41.9417
- Max drawdown: 0.0000%
- Variance ratio (late/early): 0.2913
- Rolling variance trend slope: -0.193251
- Reward mean 95% CI: [10.813732, 12.759845]
- Return mean 95% CI: [0.036709, 0.037453]
- One-sample permutation test (H1: mean return > 0): p = 0.000200
- Paired benchmark test (RL vs B&H): p = 0.000200
- Mean difference (RL - B&H) 95% CI: [-0.268916, -0.247934]

### NEE

- Mean episode reward: 47.4689 ± 11.8771
- Mean return: 13.3259% (std 0.7816%)
- Sharpe ratio: 17.0489
- Max drawdown: 0.0000%
- Variance ratio (late/early): 0.0193
- Rolling variance trend slope: -15.842258
- Reward mean 95% CI: [42.655544, 52.911815]
- Return mean 95% CI: [0.129805, 0.136516]
- One-sample permutation test (H1: mean return > 0): p = 0.000200
- Paired benchmark test (RL vs B&H): p = 0.000200
- Mean difference (RL - B&H) 95% CI: [-0.122845, -0.105943]

### DUK

- Mean episode reward: 4.5583 ± 0.0000
- Mean return: 0.0966% (std 0.0000%)
- Sharpe ratio: 96572.8688
- Max drawdown: 0.0000%
- Variance ratio (late/early): 0.0000
- Rolling variance trend slope: 0.000000
- Reward mean 95% CI: [4.558279, 4.558279]
- Return mean 95% CI: [0.000966, 0.000966]
- One-sample permutation test (H1: mean return > 0): p = 0.000200
- Paired benchmark test (RL vs B&H): p = 0.000200
- Mean difference (RL - B&H) 95% CI: [-0.093372, -0.075995]

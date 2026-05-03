# RL Metrics Evaluation 2022: Executive Summary

Generated at: 2026-04-21T20:26:26

## Study Design

This evaluation assesses a PPO-based reinforcement learning (RL) trading policy on eight equities (AAPL, AMZN, TSLA, BAC, MDU, CWCO, NEE, DUK) using 2022 processed datasets at `data/processed/{TICKER}_processed_2022.csv`.

The protocol uses an 80/20 train-test split per asset, 10,000 PPO training timesteps per asset, and 20 out-of-sample evaluation episodes. Statistical inference includes bootstrap 95% confidence intervals and permutation-based hypothesis testing.

## Main Findings

All eight experiments completed successfully. Most assets exhibit positive mean returns with statistically significant one-sample tests for $H_1: \mu_{return} > 0$.

However, paired benchmark tests indicate that, under this 2022 configuration, the RL strategy underperforms buy-and-hold across the evaluated assets, with negative mean return differences and significant p-values.

Variance behavior is heterogeneous across assets. Several assets show substantial reward variance, while others exhibit near-zero return variance, which induces numerically inflated Sharpe ratios.

## Methodological Contribution Relative to Reviewer Comments

This report explicitly addresses both requested points:

1. Statistical validation is provided through confidence intervals and hypothesis tests.
2. RL variance analysis is provided through reward variance, late-versus-early variance ratio, and rolling variance trend slope.

## Interpretation

The policy demonstrates profitability in absolute terms on multiple assets, but relative performance against buy-and-hold remains unfavorable in this single-year setup. Accordingly, the current evidence supports the inclusion of statistical rigor and variance analysis, while also motivating stronger robustness checks before final claims on comparative superiority.

## Limitations

- Single-year horizon (2022) may not represent broader market regimes.
- Some metrics (notably Sharpe) are unstable when return variance is near zero.
- Single-configuration evaluation is less robust than multi-seed, multi-period designs.

## Recommended Extensions for Camera-Ready Version

1. Multi-seed experiments with pooled confidence intervals.
2. Walk-forward multi-year validation.
3. Additional baselines beyond buy-and-hold.
4. Explicit transaction-cost and slippage sensitivity analysis.

## Related Artifacts

- Full report: artifacts/rl_metrics_evaluation_2022.md
- Raw outputs: artifacts/rl_metrics_evaluation_2022.json

# Recording learning and loss curves

Before this change neither a learning curve nor a loss curve was stored:

* `train_ppo()` / `train_sac()` wrapped the environment in `DummyVecEnv` + `VecNormalize`
  but **not** in `Monitor`, so Stable-Baselines3 had no episode buffer and never logged
  `rollout/ep_rew_mean`. The `time/` block of the training output only showed `fps`,
  `iterations` and `total_timesteps`.
* `track_episode_metrics()` and `get_learning_curve_data()` existed but were never called,
  so `training_metrics` stayed empty.
* Losses were printed to stdout by `verbose=1`, but nothing kept them; the notebook read
  `model.logger.name_to_value` only *after* training, which holds the last values only.

## What changed

In `src/models/model_trainer_rl_v2_4_buyhold.py`:

1. The raw environment is now wrapped in `Monitor` **inside** the vectorized env, so the
   recorded episode rewards are the real ones, before reward normalization:
   `DummyVecEnv([lambda: Monitor(env)])`.
2. A new `TrainingCurveCallback` records, during training:
   * one row per finished episode — timestep, episode reward, episode length, portfolio return;
   * a snapshot of every `train/*` value the SB3 logger holds, every `log_every` timesteps
     (`value_loss`, `policy_gradient_loss`, `entropy_loss`, `approx_kl`, `clip_fraction`,
     `explained_variance` for PPO; `actor_loss`, `critic_loss`, `ent_coef_loss` for SAC).
   It also forwards every episode to `track_episode_metrics()`, so `training_metrics` and
   `get_learning_curve_data()` finally return real data.
3. `train_ppo()` and `train_sac()` take `record_curves=True` by default, pass the callback to
   `model.learn()`, store it in `self.training_curves[algo]`, and return it under the
   `"callback"` key.
4. Both accept an optional `tensorboard_log` entry from the config.
5. Two new methods: `save_training_curves(path, tag)` writes the CSVs, and
   `plot_training_curves(path, tag)` draws the learning curve next to the loss curve.

`config/config.yaml` gained optional `tensorboard_log` entries (commented out) and
`sac.curve_log_every`.

## How to use it

The notebooks currently import the older trainer:

```python
from src.models.model_trainer_rl_v2_2_buyhold import ModelTrainerRL, TradingEnvRL
```

Change that line to the patched version:

```python
from src.models.model_trainer_rl_v2_4_buyhold import ModelTrainerRL, TradingEnvRL
```

Then, inside `run_ppo_trading_pipeline()` (and the SAC equivalent), after training:

```python
result = trainer.train_ppo(env_train)          # curves are recorded by default

trainer.save_training_curves("artifacts/curves", tag=stock_symbol)
trainer.plot_training_curves("artifacts/curves", tag=stock_symbol)
```

This writes, per stock:

```
artifacts/curves/ppo_AAPL_learning_curve.csv   timestep, episode_reward, episode_length, portfolio_return
artifacts/curves/ppo_AAPL_loss_curve.csv       timestep, value_loss, policy_gradient_loss, ...
artifacts/curves/training_curves_AAPL.png
```

To disable recording for a quick run: `trainer.train_ppo(env_train, record_curves=False)`.

For TensorBoard, uncomment `tensorboard_log` in `config/config.yaml`, `pip install tensorboard`,
then `tensorboard --logdir artifacts/tb`.

## Notes

* Episode rewards are the sum of the per-step rewards, which for the `profit` reward are
  `100 * log(V_t / V_{t-1})`, i.e. percentage points of portfolio growth.
* Because training uses random episode start indices, consecutive episodes cover different
  market windows; the moving average in the plot is the informative line, not the raw series.
* An episode ends at the data boundary, at `max_steps`, or on the ruin condition
  (portfolio below 10% of the initial balance), which returns a reward of -10. A cluster of
  short episodes with very negative reward in the learning curve therefore indicates ruin.
* The existing PPO/SAC results were produced without these changes; re-running is required
  to obtain curves for the current models.

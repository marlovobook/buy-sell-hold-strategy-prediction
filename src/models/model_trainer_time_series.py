"""
Time-series model training utilities for stock forecasting.
Supports Prophet, ARIMA, GARCH, NeuralProphet, and simple baselines.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.time_series_features import prophet_decompose_series

logger = logging.getLogger(__name__)


class ModelTrainerTimeSeries:
    def __init__(self, config: Dict):
        base_cfg = config or {}
        # Accept either the full app config or the already-scoped time_series block
        self.config = base_cfg.get("time_series", base_cfg)
        self.results: Dict[str, Dict] = {}
        self.models: Dict[str, object] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[object] = None
        self.metric_cfg = self.config.get("metrics", {})
        self.model_scalers: Dict[str, Optional[object]] = {}

    def prepare_series(
        self,
        df: pd.DataFrame,
        target_column: str = "close",
        date_column: str = "date",
    ) -> pd.Series:
        if date_column not in df.columns:
            raise ValueError(f"Missing date column '{date_column}' in dataframe")
        if target_column not in df.columns:
            raise ValueError(f"Missing target column '{target_column}' in dataframe")

        series = df[[date_column, target_column]].copy()
        series[date_column] = pd.to_datetime(series[date_column])
        series.sort_values(date_column, inplace=True)
        series.set_index(date_column, inplace=True)
        series = series[target_column].asfreq(self.config.get("freq", "D"))
        series = series.ffill().bfill()
        logger.info("Prepared time series with %d observations", len(series))
        return series

    def split_series(
        self, series: pd.Series, test_size: float = 0.2
    ) -> Tuple[pd.Series, pd.Series]:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        split_idx = max(1, int(len(series) * (1 - test_size)))
        train, test = series.iloc[:split_idx], series.iloc[split_idx:]
        logger.info("Train size: %d, Test size: %d", len(train), len(test))
        return train, test

    def _make_scaler(self, train_vals: np.ndarray):
        """Create forward and inverse scaling functions based on config.

        Supported methods:
        - none: identity
        - multiply: y' = y * factor
        - divide: y' = y / denom (default denom=max|y|)
        - standardize: y' = (y - mean) / std
        - log1p: y' = log1p(y), requires non-negative y
        """
        sc_cfg = self.config.get("scaling", {})
        method = (sc_cfg.get("method") or "none").lower()
        eps = float(self.metric_cfg.get("epsilon", 1e-8))

        if method == "none":
            return (lambda x: x, lambda x: x)

        if method == "multiply":
            factor = float(sc_cfg.get("factor", 0.1))
            return (lambda x: x * factor, lambda x: x / factor)

        if method == "divide":
            denom = float(sc_cfg.get("denom", float(np.max(np.abs(train_vals)) or 1.0)))
            if denom == 0.0:
                denom = 1.0
            return (lambda x: x / denom, lambda x: x * denom)

        if method == "standardize":
            mean = float(np.mean(train_vals))
            std = float(np.std(train_vals))
            std = std if std > eps else 1.0
            return (lambda x: (x - mean) / std, lambda x: x * std + mean)

        if method == "log1p":
            if (train_vals < 0).any():
                raise ValueError("log1p scaling requires non-negative targets")
            return (lambda x: np.log1p(x), lambda x: np.expm1(x))

        # Fallback to identity for unknown methods
        return (lambda x: x, lambda x: x)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        eps = float(self.metric_cfg.get("epsilon", 1e-8))
        exclude_zero_mape = bool(self.metric_cfg.get("exclude_zero_mape", True))

        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))

        if exclude_zero_mape:
            mask = np.abs(y_true) > eps
            if not np.any(mask):
                mape = 0.0
            else:
                mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.maximum(np.abs(y_true[mask]), eps))) * 100)
        else:
            mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)

        smape_den = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
        smape = float(np.mean(np.abs(y_true - y_pred) / smape_den) * 100)

        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "smape": smape}

    def _build_lagged_features(
        self,
        series: pd.Series,
        n_lags: int,
        extra_features: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
        """Transform a univariate series into a supervised matrix using past lags.

        Optionally, additional exogenous features can be provided via
        ``extra_features``. These must be a DataFrame indexed by the same
        DatetimeIndex as ``series`` (or at least reindex-able to it). The
        exogenous columns are concatenated to the lag features before
        training LightGBM.

        Parameters
        ----------
        series : pd.Series
            Full time series (train + test) with a DatetimeIndex.
        n_lags : int
            Number of past observations to use as features.
        extra_features : Optional[pd.DataFrame]
            Additional time-aligned covariates to include as features.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix with columns lag_1, ..., lag_n_lags and any
            exogenous feature columns.
        y : np.ndarray
            Target values aligned with X rows.
        idx : pd.DatetimeIndex
            Index corresponding to each row in X/y.
        """
        if n_lags <= 0:
            raise ValueError("n_lags must be a positive integer for LGBM")

        df = pd.DataFrame({"y": series.astype(float)})
        for lag in range(1, n_lags + 1):
            df[f"lag_{lag}"] = df["y"].shift(lag)

        # Attach any exogenous features, aligned on the same index
        exog_cols: Optional[list[str]] = None
        if extra_features is not None:
            # Reindex to match the temporary frame (before dropna so that
            # we can drop rows with missing lags or exog in one step).
            aligned_exog = extra_features.reindex(df.index)
            df = pd.concat([df, aligned_exog], axis=1)
            exog_cols = list(aligned_exog.columns)

        df = df.dropna()

        feature_cols = [f"lag_{lag}" for lag in range(1, n_lags + 1)]
        if exog_cols:
            feature_cols.extend(exog_cols)

        X = df[feature_cols].copy()
        y = df["y"].to_numpy(dtype=float)
        idx = df.index

        return X, y, idx

    def train_prophet(self, train: pd.Series, test: pd.Series, freq: str) -> Dict:
        try:
            from prophet import Prophet
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install prophet to use the Prophet model") from exc

        config = self.config.get("prophet", {})
        model = Prophet(
            seasonality_mode=config.get("seasonality_mode", "additive"),
            changepoint_prior_scale=config.get("changepoint_prior_scale", 0.05),
            yearly_seasonality=config.get("yearly_seasonality", True),
            weekly_seasonality=config.get("weekly_seasonality", True),
            daily_seasonality=config.get("daily_seasonality", False),
        )

        train_df = train.reset_index().rename(columns={train.index.name: "ds", 0: "y", train.name: "y"})
        model.fit(train_df)

        future = model.make_future_dataframe(periods=len(test), freq=freq, include_history=False)
        forecast_df = model.predict(future)
        y_pred = forecast_df["yhat"].values
        metrics = self._compute_metrics(test.values, y_pred)

        return {
            "model": model,
            "metrics": metrics,
            "predictions": y_pred,
            "forecast_df": forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        }

    def train_arima(self, train: pd.Series, test: pd.Series) -> Dict:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install statsmodels to use the ARIMA model") from exc

        order = tuple(self.config.get("arima", {}).get("order", (1, 1, 1)))
        fwd, inv = self._make_scaler(train.values)
        model = ARIMA(fwd(train.values), order=order)
        fitted = model.fit()
        y_pred_scaled = fitted.forecast(steps=len(test))
        y_pred = inv(y_pred_scaled)
        metrics = self._compute_metrics(test.values, y_pred)
        self.model_scalers["arima"] = inv
        return {"model": fitted, "metrics": metrics, "predictions": y_pred}

    def train_garch(self, train: pd.Series, test: pd.Series) -> Dict:
        try:
            from arch import arch_model
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install arch to use the GARCH model") from exc

        garch_cfg = self.config.get("garch", {})
        p = garch_cfg.get("p", 1)
        q = garch_cfg.get("q", 1)
        fwd, inv = self._make_scaler(train.values)
        model = arch_model(fwd(train.values), vol="Garch", p=p, q=q, dist=garch_cfg.get("dist", "normal"))
        res = model.fit(disp="off")
        fc = res.forecast(horizon=len(test))
        y_pred_scaled = fc.mean.iloc[-1].values
        y_pred = inv(y_pred_scaled)
        metrics = self._compute_metrics(test.values, y_pred)
        self.model_scalers["garch"] = inv
        return {"model": res, "metrics": metrics, "predictions": y_pred}

    def train_poisson(self, train: pd.Series, test: pd.Series) -> Dict:
        """Count-friendly GLM Poisson with time index as regressor."""
        try:
            import statsmodels.api as sm
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install statsmodels to use the Poisson model") from exc

        # Guard against negative targets which Poisson cannot handle well
        if (train.values < 0).any():
            raise ValueError("Poisson model requires non-negative targets")

        x_train = np.arange(len(train)).reshape(-1, 1)
        x_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

        glm = sm.GLM(train.values, sm.add_constant(x_train), family=sm.families.Poisson())
        fitted = glm.fit()
        y_pred = fitted.predict(sm.add_constant(x_test))

        metrics = self._compute_metrics(test.values, y_pred)
        return {"model": fitted, "metrics": metrics, "predictions": y_pred}

    def train_neuralprophet(self, train: pd.Series, test: pd.Series, freq: str) -> Dict:
        try:
            from neuralprophet import NeuralProphet
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install neuralprophet to use the NeuralProphet model") from exc

        # Allowlist NeuralProphet classes for torch.load when weights_only=True (PyTorch 2.6+)
        try:
            from torch import serialization
            from neuralprophet.configure import (
                ConfigSeasonality,
                Season,
                Train,
            )

            serialization.add_safe_globals([
                ConfigSeasonality,
                Season,
                Train,
            ])
        except Exception as exc:  # pragma: no cover - avoid breaking if torch API changes
            logger.warning("Could not allowlist NeuralProphet globals for safe loading: %s", exc)

        np_cfg = self.config.get("neuralprophet", {})
        model = NeuralProphet(
            n_changepoints=np_cfg.get("n_changepoints", 5),
            yearly_seasonality=np_cfg.get("yearly_seasonality", True),
            weekly_seasonality=np_cfg.get("weekly_seasonality", True),
            daily_seasonality=np_cfg.get("daily_seasonality", False),
            learning_rate=0.01
        )

        train_df = train.reset_index().rename(columns={train.index.name: "ds", 0: "y", train.name: "y"})
        model.fit(train_df, freq=freq, progress="off")

        future = model.make_future_dataframe(train_df, periods=len(test), n_historic_predictions=False)
        forecast_df = model.predict(future)
        y_pred = forecast_df.tail(len(test))["yhat1"].values
        metrics = self._compute_metrics(test.values, y_pred) 
        return {"model": model, "metrics": metrics, "predictions": y_pred}

    def train_lgbm(
        self,
        series: pd.Series,
        train: pd.Series,
        test: pd.Series,
        extra_features: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Gradient boosting (LightGBM) model using lag-based features.

        The model is purely univariate: it learns to predict the next value
        from a configurable number of past observations.
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install lightgbm to use the LGBM time-series model") from exc

        lgbm_cfg = self.config.get("lgbm", {})
        n_lags = int(lgbm_cfg.get("n_lags", 14))
        use_prophet_feats = bool(lgbm_cfg.get("use_prophet_features", False))

        # Base: lagged features on the FULL series (train + test) so that
        # early test points can use history from the training window only.
        X_all, y_all, idx_all = self._build_lagged_features(
            series,
            n_lags=n_lags,
            extra_features=extra_features,
        )

        # LightGBM expects numeric or categorical dtypes; convert any
        # remaining object-typed columns (e.g. string labels) to
        # pandas "category" so they can be used as categorical
        # features instead of raising dtype errors.
        obj_cols = X_all.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            X_all[obj_cols] = X_all[obj_cols].astype("category")

        # Explicitly capture categorical feature columns so that the
        # same set is used for both training and any internal
        # validation data. This avoids LightGBM errors such as
        # "train and valid dataset categorical_feature do not match"
        # that can occur when relying on automatic detection.
        categorical_cols = list(X_all.select_dtypes(include=["category"]).columns)

        # Optionally enrich with Prophet decomposition features (trend/seasonal/residual)
        if use_prophet_feats:
            prophet_cfg = self.config.get("prophet", {})
            dec_df = prophet_decompose_series(series, prophet_cfg=prophet_cfg, prefix="ts_")
            # Align decomposition rows to the supervised matrix index
            dec_aligned = dec_df.reindex(idx_all)
            dec_aligned = dec_aligned.ffill().bfill()
            X_all = pd.concat([X_all, dec_aligned], axis=1)

        # Align rows with train/test indices (after dropping first n_lags rows)
        train_mask = idx_all.isin(train.index)
        test_mask = idx_all.isin(test.index)

        if not np.any(train_mask):
            raise ValueError("Not enough historical data to build lag features for LGBM train set")
        if not np.any(test_mask):
            logging.warning("No test rows with sufficient history for LGBM; metrics will be skipped")

        X_train, y_train = X_all.loc[train_mask], y_all[train_mask]
        X_test, y_test = X_all.loc[test_mask], y_all[test_mask]

        params = {
            "objective": "regression",
            "num_leaves": int(lgbm_cfg.get("num_leaves", 31)),
            "learning_rate": float(lgbm_cfg.get("learning_rate", 0.05)),
            "n_estimators": int(lgbm_cfg.get("n_estimators", 300)),
            "subsample": float(lgbm_cfg.get("subsample", 0.8)),
            "colsample_bytree": float(lgbm_cfg.get("colsample_bytree", 0.8)),
            "random_state": int(lgbm_cfg.get("random_state", 42)),
            "metric": lgbm_cfg.get("metric", ["mae"]),
            # --- ADDED TO FIX WARNINGS ---
            "min_child_samples": int(lgbm_cfg.get("min_child_samples", 5)),       # (Default is 20) Allows leaves to have as few as 5 data points
            "min_child_weight": float(lgbm_cfg.get("min_child_weight", 0.001)),    # (Default is 0.001) Keep low to allow splits on small groups
            "verbose": int(lgbm_cfg.get("verbose", -1)),                # Silence Python warnings
            "verbosity": int(lgbm_cfg.get("verbosity", -1)),            # Silence C++ backend warnings
        }

        model = lgb.LGBMRegressor(**params)

        fit_kwargs = {}
        if categorical_cols:
            fit_kwargs["categorical_feature"] = categorical_cols

        model.fit(X_train, y_train, **fit_kwargs)

        # Predict on test rows where we have sufficient history
        if np.any(test_mask):
            y_pred_test = model.predict(X_test)
            pred_series = pd.Series(y_pred_test, index=idx_all[test_mask])
            # Align predictions to the original test index (may contain NaNs for
            # the first few test points that don't have enough lag history).
            aligned_pred = pred_series.reindex(test.index)

            # Compute metrics only on points where predictions exist
            valid_mask = aligned_pred.notna().to_numpy()
            if np.any(valid_mask):
                metrics = self._compute_metrics(test.values[valid_mask], aligned_pred.values[valid_mask])
            else:
                metrics = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "mape": float("nan"), "smape": float("nan")}

            y_pred_out = aligned_pred.values
        else:
            y_pred_out = np.full_like(test.values, np.nan, dtype=float)
            metrics = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "mape": float("nan"), "smape": float("nan")}

        # Persist configuration needed for out-of-sample forecasting
        self.model_scalers["lgbm_n_lags"] = n_lags

        return {
            "model": model,
            "metrics": metrics,
            "predictions": y_pred_out,
            # Expose feature names so callers can build
            # consistent out-of-sample design matrices.
            "feature_names": list(X_all.columns),
        }

    def train_naive(self, train: pd.Series, test: pd.Series) -> Dict:
        y_pred = np.repeat(train.iloc[-1], len(test))
        metrics = self._compute_metrics(test.values, y_pred)
        return {"model": "naive_last_value", "metrics": metrics, "predictions": y_pred}

    def train_all(
        self,
        df: pd.DataFrame,
        target_column: str = "close",
        date_column: str = "date",
    ) -> Dict[str, Dict]:
        freq = self.config.get("freq", "D")
        series = self.prepare_series(df, target_column=target_column, date_column=date_column)
        train, test = self.split_series(series, test_size=self.config.get("test_size", 0.2))
        
        # Store for plotting
        self.train_series = train
        self.test_series = test
        self.full_series = series

        algorithms = self.config.get(
            "algorithms",
            ["prophet", "arima", "garch", "poisson", "neuralprophet", "naive", "lgbm"],
        )

        if "prophet" in algorithms:
            logger.info("Training Prophet")
            self.results["prophet"] = self.train_prophet(train, test, freq)
            self.models["prophet"] = self.results["prophet"]["model"]

        if "arima" in algorithms:
            logger.info("Training ARIMA")
            self.results["arima"] = self.train_arima(train, test)
            self.models["arima"] = self.results["arima"]["model"]

        if "garch" in algorithms:
            logger.info("Training GARCH")
            self.results["garch"] = self.train_garch(train, test)
            self.models["garch"] = self.results["garch"]["model"]

        if "poisson" in algorithms:
            logger.info("Training Poisson GLM")
            self.results["poisson"] = self.train_poisson(train, test)
            self.models["poisson"] = self.results["poisson"]["model"]

        if "neuralprophet" in algorithms:
            logger.info("Training NeuralProphet")
            self.results["neuralprophet"] = self.train_neuralprophet(train, test, freq)
            self.models["neuralprophet"] = self.results["neuralprophet"]["model"]

        if "naive" in algorithms:
            logger.info("Training naive baseline")
            self.results["naive"] = self.train_naive(train, test)
            self.models["naive"] = self.results["naive"]["model"]

        if "lgbm" in algorithms:
            logger.info("Training LGBM (lag-based)")
            self.results["lgbm"] = self.train_lgbm(series, train, test)
            self.models["lgbm"] = self.results["lgbm"]["model"]

        self._select_best_model()
        self.test_index = test.index
        return self.results

    def _select_best_model(self):
        best_mae = float("inf")
        for name, result in self.results.items():
            mae = result["metrics"].get("mae", float("inf"))
            if mae < best_mae:
                best_mae = mae
                self.best_model_name = name
                self.best_model = self.models.get(name)
        logger.info("Best time-series model: %s (MAE %.4f)", self.best_model_name, best_mae)

    def get_prediction_intervals(
        self, model_name: Optional[str] = None, alpha: float = 0.1
    ) -> pd.DataFrame:
        """Get predictions with confidence intervals for the test period.
        
        Returns a DataFrame with columns:
        - ds: datetime index
        - y: actual values (if available)
        - yhat: point forecast
        - yhat_lower: lower bound (default 90% CI)
        - yhat_upper: upper bound (default 90% CI)
        - p10, p50, p90: percentile forecasts
        
        Parameters
        ----------
        model_name : str, optional
            Model to use. If None, uses best model.
        alpha : float, default 0.1
            Significance level for confidence intervals (0.1 = 90% CI)
        
        Example
        -------
        >>> trainer = ModelTrainerTimeSeries(config)
        >>> results = trainer.train_all(df, date_column="date", target_column="value")
        >>> intervals_df = trainer.get_prediction_intervals("prophet")
        >>> print(intervals_df.head())
        """
        if model_name is None:
            model_name = self.best_model_name
        if model_name is None:
            raise RuntimeError("No trained model available")

        if not hasattr(self, "test_series"):
            raise RuntimeError("Must call train_all() first")

        test_dates = self.test_series.index
        test_vals = self.test_series.values
        
        result_dict = self.results.get(model_name)
        if not result_dict:
            raise ValueError(f"Model '{model_name}' not found in results")

        y_pred = result_dict["predictions"]

        # Build base dataframe
        df_out = pd.DataFrame({
            "ds": test_dates,
            "y": test_vals,
            "yhat": y_pred,
        })

        # Model-specific intervals
        if model_name == "prophet":
            forecast_df = result_dict.get("forecast_df")
            if forecast_df is not None:
                df_out["yhat_lower"] = forecast_df["yhat_lower"].values
                df_out["yhat_upper"] = forecast_df["yhat_upper"].values
            else:
                # Fallback: assume ±20% bands
                df_out["yhat_lower"] = y_pred * 0.8
                df_out["yhat_upper"] = y_pred * 1.2

        elif model_name == "arima":
            try:
                from scipy import stats
                model = self.models["arima"]
                inv = self.model_scalers.get("arima", lambda x: x)
                # Get forecast with intervals in scaled space
                fc = model.get_forecast(steps=len(test_vals))
                fc_summary = fc.summary_frame(alpha=alpha)
                df_out["yhat_lower"] = inv(fc_summary["mean_ci_lower"].values)
                df_out["yhat_upper"] = inv(fc_summary["mean_ci_upper"].values)
            except Exception as exc:
                logger.warning("Could not compute ARIMA intervals: %s", exc)
                df_out["yhat_lower"] = y_pred * 0.9
                df_out["yhat_upper"] = y_pred * 1.1

        elif model_name == "garch":
            # GARCH gives volatility forecast; use for symmetric bands
            try:
                model = self.models["garch"]
                inv = self.model_scalers.get("garch", lambda x: x)
                fc = model.forecast(horizon=len(test_vals))
                vol = np.sqrt(fc.variance.iloc[-1].values)
                # Approximate 90% CI with ±1.645*sigma
                z_score = 1.645
                df_out["yhat_lower"] = inv(y_pred - z_score * vol)
                df_out["yhat_upper"] = inv(y_pred + z_score * vol)
            except Exception as exc:
                logger.warning("Could not compute GARCH intervals: %s", exc)
                df_out["yhat_lower"] = y_pred * 0.9
                df_out["yhat_upper"] = y_pred * 1.1

        else:
            # Fallback for poisson, naive, lgbm, etc: use empirical ±std from residuals if available
            residuals = test_vals - y_pred
            mask = np.isfinite(residuals)
            if np.any(mask):
                resid_std = np.std(residuals[mask]) if np.sum(mask) > 1 else np.std(test_vals[mask])
            else:
                resid_std = 0.0
            if resid_std == 0 or not np.isfinite(resid_std):
                resid_std = np.abs(np.nanmean(y_pred)) * 0.1 if np.isfinite(np.nanmean(y_pred)) else 1.0
            z_score = 1.645  # 90% CI
            df_out["yhat_lower"] = y_pred - z_score * resid_std
            df_out["yhat_upper"] = y_pred + z_score * resid_std

        # Compute percentiles from intervals (approximate)
        df_out["p10"] = df_out["yhat_lower"]
        df_out["p50"] = df_out["yhat"]
        df_out["p90"] = df_out["yhat_upper"]

        return df_out

    def plot_forecast(
        self,
        model_name: Optional[str] = None,
        show_train: bool = True,
        figsize: Tuple[int, int] = (14, 6),
    ):
        """Plot forecast with actual values and prediction intervals.
        
        Parameters
        ----------
        model_name : str, optional
            Model to plot. If None, uses best model.
        show_train : bool, default True
            Whether to show training data in the plot.
        figsize : tuple, default (14, 6)
            Figure size (width, height).
        
        Returns
        -------
        fig, ax : matplotlib figure and axis
        
        Example
        -------
        >>> trainer = ModelTrainerTimeSeries(config)
        >>> results = trainer.train_all(df, date_column="date", target_column="value")
        >>> # Plot best model
        >>> fig, ax = trainer.plot_forecast()
        >>> plt.show()
        >>> 
        >>> # Plot specific model
        >>> fig, ax = trainer.plot_forecast(model_name="prophet")
        >>> plt.savefig("forecast_prophet.png")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("Install matplotlib to use plotting: pip install matplotlib") from exc

        if model_name is None:
            model_name = self.best_model_name
        if model_name is None:
            raise RuntimeError("No trained model available")

        intervals_df = self.get_prediction_intervals(model_name=model_name)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot training data if requested
        if show_train and hasattr(self, "train_series"):
            ax.plot(
                self.train_series.index,
                self.train_series.values,
                "o-",
                label="Training Data",
                color="#1f77b4",
                markersize=3,
                alpha=0.6,
            )

        # Plot test actuals
        ax.plot(
            intervals_df["ds"],
            intervals_df["y"],
            "o",
            label="Test Actuals",
            color="black",
            markersize=4,
        )

        # Plot forecast
        ax.plot(
            intervals_df["ds"],
            intervals_df["yhat"],
            "-",
            label=f"Forecast ({model_name})",
            color="#ff7f0e",
            linewidth=2,
        )

        # Plot confidence intervals
        ax.fill_between(
            intervals_df["ds"],
            intervals_df["yhat_lower"],
            intervals_df["yhat_upper"],
            alpha=0.2,
            color="#ff7f0e",
            label="90% Confidence Interval",
        )

        # Add vertical line at train/test split
        if hasattr(self, "train_series") and len(self.train_series) > 0:
            split_date = self.train_series.index[-1]
            ax.axvline(x=split_date, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Train/Test Split")

        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title(f"Time Series Forecast - {model_name.upper()}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig, ax

    def forecast(self, recent_series: pd.Series, steps: int, model_name: Optional[str] = None) -> np.ndarray:
        if model_name is None:
            model_name = self.best_model_name
        if model_name is None:
            raise RuntimeError("No trained model available for forecasting")

        model = self.models.get(model_name)
        freq = self.config.get("freq", "D")

        if model_name == "prophet":
            future_df = model.make_future_dataframe(periods=steps, freq=freq, include_history=False)
            forecast_df = model.predict(future_df)
            return forecast_df["yhat"].values

        if model_name == "neuralprophet":
            recent_df = recent_series.reset_index().rename(columns={recent_series.index.name: "ds", 0: "y", recent_series.name: "y"})
            future = model.make_future_dataframe(recent_df, periods=steps, n_historic_predictions=False)
            forecast_df = model.predict(future)
            return forecast_df.tail(steps)["yhat1"].values

        if model_name == "arima":
            inv = self.model_scalers.get("arima", None)
            preds = model.forecast(steps=steps)
            return inv(preds) if callable(inv) else preds

        if model_name == "garch":
            fc = model.forecast(horizon=steps)
            preds = fc.mean.iloc[-1].values
            inv = self.model_scalers.get("garch", None)
            return inv(preds) if callable(inv) else preds

        if model_name == "naive":
            return np.repeat(recent_series.iloc[-1], steps)

        if model_name == "poisson":
            import numpy as _np  # local import to avoid hard dependency if unused
            x_future = _np.arange(len(recent_series), len(recent_series) + steps).reshape(-1, 1)
            import statsmodels.api as sm

            return model.predict(sm.add_constant(x_future))

        if model_name == "lgbm":
            n_lags = int(self.model_scalers.get("lgbm_n_lags", 14))
            if len(recent_series) < n_lags:
                raise ValueError(f"recent_series must have at least {n_lags} points for LGBM forecasting")

            history = list(np.asarray(recent_series.values, dtype=float))
            preds = []

            # We don't need actual future timestamps here; forecast() returns only values
            model_lgbm = model
            for _ in range(steps):
                lags = [history[-lag] for lag in range(1, n_lags + 1)]
                x_next = np.array(lags, dtype=float).reshape(1, -1)
                next_val = float(model_lgbm.predict(x_next)[0])
                preds.append(next_val)
                history.append(next_val)

            return np.array(preds, dtype=float)

        raise ValueError(f"Unknown model '{model_name}' for forecasting")

    def plot_forecast_all(
        self,
        show_train: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        models_to_plot: Optional[list[str]] = None,
    ):
        """Plot forecasts from all trained models in a grid layout.
        
        Parameters
        ----------
        show_train : bool, default False
            Whether to show training data in each subplot.
        figsize : tuple, optional
            Figure size (width, height). If None, auto-calculated based on number of models.
        models_to_plot : list[str], optional
            List of model names to plot. If None, plots all available models.
        
        Returns
        -------
        fig, axes : matplotlib figure and axes array
        
        Example
        -------
        >>> trainer = ModelTrainerTimeSeries(config)
        >>> results = trainer.train_all(df, date_column="date", target_column="value")
        >>> # Plot all models
        >>> fig, axes = trainer.plot_forecast_all()
        >>> plt.show()
        >>> 
        >>> # Plot specific models only
        >>> fig, axes = trainer.plot_forecast_all(models_to_plot=["prophet", "neuralprophet"])
        >>> plt.savefig("forecast_comparison.png")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("Install matplotlib to use plotting: pip install matplotlib") from exc

        if not self.results:
            raise RuntimeError("No trained models available. Call train_all() first.")

        # Determine which models to plot
        if models_to_plot is None:
            available_models = list(self.results.keys())
        else:
            available_models = [m for m in models_to_plot if m in self.results]
        
        if not available_models:
            raise ValueError("No valid models to plot")

        n_models = len(available_models)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division

        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (16, 5 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle('Forecast Comparison Across Models', fontsize=16)

        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        for idx, model_name in enumerate(available_models):
            ax = axes_flat[idx]
            intervals = self.get_prediction_intervals(model_name=model_name)
            
            # Plot training data if requested
            if show_train and hasattr(self, "train_series"):
                ax.plot(
                    self.train_series.index,
                    self.train_series.values,
                    "o-",
                    label="Training Data",
                    color="#1f77b4",
                    markersize=2,
                    alpha=0.4,
                )
            
            # Plot test actuals
            ax.plot(
                intervals['ds'],
                intervals['y'],
                'o',
                label='Actuals',
                color='black',
                markersize=3,
            )
            
            # Plot forecast
            ax.plot(
                intervals['ds'],
                intervals['yhat'],
                '-',
                label='Forecast',
                color='#ff7f0e',
                linewidth=2,
            )
            
            # Plot confidence intervals
            ax.fill_between(
                intervals['ds'],
                intervals['yhat_lower'],
                intervals['yhat_upper'],
                alpha=0.2,
                color='#ff7f0e',
                label='90% CI',
            )
            
            # Add metrics to title
            metrics = self.results[model_name]["metrics"]
            title = f'{model_name.upper()}\nMAE: {metrics["mae"]:.2f} | RMSE: {metrics["rmse"]:.2f}'
            ax.set_title(title, fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Date', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
            plt.setp(ax.yaxis.get_majorticklabels(), fontsize=8)

        # Hide unused subplots
        for idx in range(n_models, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()
        return fig, axes
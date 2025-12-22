"""
Time-series model training utilities for stock forecasting.
Supports Prophet, ARIMA, GARCH, NeuralProphet, and simple baselines.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelTrainerTimeSeries:
    def __init__(self, config: Dict):
        self.config = config or {}
        self.results: Dict[str, Dict] = {}
        self.models: Dict[str, object] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[object] = None

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

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100)
        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}

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

        return {"model": model, "metrics": metrics, "predictions": y_pred}

    def train_arima(self, train: pd.Series, test: pd.Series) -> Dict:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install statsmodels to use the ARIMA model") from exc

        order = tuple(self.config.get("arima", {}).get("order", (1, 1, 1)))
        model = ARIMA(train.values, order=order)
        fitted = model.fit()
        y_pred = fitted.forecast(steps=len(test))
        metrics = self._compute_metrics(test.values, y_pred)
        return {"model": fitted, "metrics": metrics, "predictions": y_pred}

    def train_garch(self, train: pd.Series, test: pd.Series) -> Dict:
        try:
            from arch import arch_model
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install arch to use the GARCH model") from exc

        garch_cfg = self.config.get("garch", {})
        p = garch_cfg.get("p", 1)
        q = garch_cfg.get("q", 1)
        model = arch_model(train.values, vol="Garch", p=p, q=q, dist=garch_cfg.get("dist", "normal"))
        res = model.fit(disp="off")
        fc = res.forecast(horizon=len(test))
        y_pred = fc.mean.iloc[-1].values
        metrics = self._compute_metrics(test.values, y_pred)
        return {"model": res, "metrics": metrics, "predictions": y_pred}

    def train_neuralprophet(self, train: pd.Series, test: pd.Series, freq: str) -> Dict:
        try:
            from neuralprophet import NeuralProphet
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install neuralprophet to use the NeuralProphet model") from exc

        np_cfg = self.config.get("neuralprophet", {})
        model = NeuralProphet(
            n_changepoints=np_cfg.get("n_changepoints", 5),
            yearly_seasonality=np_cfg.get("yearly_seasonality", True),
            weekly_seasonality=np_cfg.get("weekly_seasonality", True),
            daily_seasonality=np_cfg.get("daily_seasonality", False),
        )

        train_df = train.reset_index().rename(columns={train.index.name: "ds", 0: "y", train.name: "y"})
        model.fit(train_df, freq=freq, progress="off")

        future = model.make_future_dataframe(train_df, periods=len(test), n_historic_predictions=False)
        forecast_df = model.predict(future)
        y_pred = forecast_df.tail(len(test))["yhat1"].values
        metrics = self._compute_metrics(test.values, y_pred)
        return {"model": model, "metrics": metrics, "predictions": y_pred}

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

        algorithms = self.config.get("algorithms", ["prophet", "arima", "garch", "neuralprophet", "naive"])

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

        if "neuralprophet" in algorithms:
            logger.info("Training NeuralProphet")
            self.results["neuralprophet"] = self.train_neuralprophet(train, test, freq)
            self.models["neuralprophet"] = self.results["neuralprophet"]["model"]

        if "naive" in algorithms:
            logger.info("Training naive baseline")
            self.results["naive"] = self.train_naive(train, test)
            self.models["naive"] = self.results["naive"]["model"]

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
            return model.forecast(steps=steps)

        if model_name == "garch":
            fc = model.forecast(horizon=steps)
            return fc.mean.iloc[-1].values

        if model_name == "naive":
            return np.repeat(recent_series.iloc[-1], steps)

        raise ValueError(f"Unknown model '{model_name}' for forecasting")

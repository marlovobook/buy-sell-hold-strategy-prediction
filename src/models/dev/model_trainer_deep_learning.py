"""
Deep learning model training utilities for stock price forecasting.
Supports LSTM, Bidirectional LSTM, LSTM with Attention, MLP, GRU, CNN-LSTM, and more.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class ModelTrainerDeepLearning:
    """Deep learning models for time-series stock forecasting."""

    def __init__(self, config: Dict):
        self.config = config or {}
        self.results: Dict[str, Dict] = {}
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, object] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[object] = None

    def prepare_sequences(
        self,
        series: pd.Series,
        lookback: int = 60,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[MinMaxScaler]]:
        """Create sequences for training deep learning models."""
        data = series.values.reshape(-1, 1)
        scaler = None

        if normalize:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i : i + lookback])
            y.append(data[i + lookback, 0])

        X, y = np.array(X), np.array(y)
        logger.info("Created %d sequences with lookback=%d", len(X), lookback)
        return X, y, scaler

    def split_sequences(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split sequences into train and test sets."""
        split_idx = max(1, int(len(X) * (1 - test_size)))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        logger.info("Train: %d, Test: %d sequences", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mape = float(
            np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        )
        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}

    def train_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train a Multi-Layer Perceptron."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install tensorflow to use MLP model") from exc

        mlp_cfg = self.config.get("mlp", {})
        units = mlp_cfg.get("units", [128, 64, 32])
        dropout = mlp_cfg.get("dropout", 0.2)
        epochs = mlp_cfg.get("epochs", 50)
        batch_size = mlp_cfg.get("batch_size", 32)

        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        for unit in units:
            model.add(layers.Dense(unit, activation="relu"))
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = self._compute_metrics(y_test, y_pred)

        logger.info("MLP MAE: %.4f", metrics["mae"])
        return {"model": model, "metrics": metrics, "predictions": y_pred, "history": history}

    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train a standard LSTM model."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install tensorflow to use LSTM model") from exc

        lstm_cfg = self.config.get("lstm", {})
        units = lstm_cfg.get("units", 50)
        dropout = lstm_cfg.get("dropout", 0.2)
        epochs = lstm_cfg.get("epochs", 50)
        batch_size = lstm_cfg.get("batch_size", 32)

        model = keras.Sequential(
            [
                layers.LSTM(units, activation="relu", input_shape=(X_train.shape[1], 1)),
                layers.Dropout(dropout),
                layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = self._compute_metrics(y_test, y_pred)

        logger.info("LSTM MAE: %.4f", metrics["mae"])
        return {"model": model, "metrics": metrics, "predictions": y_pred, "history": history}

    def train_bilstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train a Bidirectional LSTM (W-LSTM variant)."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install tensorflow to use Bidirectional LSTM model") from exc

        bilstm_cfg = self.config.get("bilstm", {})
        units = bilstm_cfg.get("units", 50)
        dropout = bilstm_cfg.get("dropout", 0.2)
        epochs = bilstm_cfg.get("epochs", 50)
        batch_size = bilstm_cfg.get("batch_size", 32)

        model = keras.Sequential(
            [
                layers.Bidirectional(
                    layers.LSTM(units, activation="relu", return_sequences=False),
                    input_shape=(X_train.shape[1], 1),
                ),
                layers.Dropout(dropout),
                layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = self._compute_metrics(y_test, y_pred)

        logger.info("Bidirectional LSTM MAE: %.4f", metrics["mae"])
        return {"model": model, "metrics": metrics, "predictions": y_pred, "history": history}

    def train_lstm_attention(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train LSTM with Attention mechanism (LSTM-ARO variant)."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install tensorflow to use LSTM-Attention model") from exc

        attn_cfg = self.config.get("lstm_attention", {})
        units = attn_cfg.get("units", 50)
        dropout = attn_cfg.get("dropout", 0.2)
        epochs = attn_cfg.get("epochs", 50)
        batch_size = attn_cfg.get("batch_size", 32)

        # Build LSTM with attention
        inputs = keras.Input(shape=(X_train.shape[1], 1))
        lstm_out = layers.LSTM(units, activation="relu", return_sequences=True)(inputs)
        lstm_out = layers.Dropout(dropout)(lstm_out)

        # Simple attention
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=units // 4)(
            lstm_out, lstm_out
        )
        attention = layers.GlobalAveragePooling1D()(attention)
        outputs = layers.Dense(1)(attention)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = self._compute_metrics(y_test, y_pred)

        logger.info("LSTM-Attention MAE: %.4f", metrics["mae"])
        return {"model": model, "metrics": metrics, "predictions": y_pred, "history": history}

    def train_gru(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train a GRU (Gated Recurrent Unit) model."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install tensorflow to use GRU model") from exc

        gru_cfg = self.config.get("gru", {})
        units = gru_cfg.get("units", 50)
        dropout = gru_cfg.get("dropout", 0.2)
        epochs = gru_cfg.get("epochs", 50)
        batch_size = gru_cfg.get("batch_size", 32)

        model = keras.Sequential(
            [
                layers.GRU(units, activation="relu", input_shape=(X_train.shape[1], 1)),
                layers.Dropout(dropout),
                layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = self._compute_metrics(y_test, y_pred)

        logger.info("GRU MAE: %.4f", metrics["mae"])
        return {"model": model, "metrics": metrics, "predictions": y_pred, "history": history}

    def train_cnn_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train a CNN-LSTM hybrid model."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install tensorflow to use CNN-LSTM model") from exc

        cnn_cfg = self.config.get("cnn_lstm", {})
        cnn_filters = cnn_cfg.get("cnn_filters", 32)
        lstm_units = cnn_cfg.get("lstm_units", 50)
        dropout = cnn_cfg.get("dropout", 0.2)
        epochs = cnn_cfg.get("epochs", 50)
        batch_size = cnn_cfg.get("batch_size", 32)

        model = keras.Sequential(
            [
                layers.Conv1D(cnn_filters, 3, activation="relu", input_shape=(X_train.shape[1], 1)),
                layers.MaxPooling1D(2),
                layers.LSTM(lstm_units, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = self._compute_metrics(y_test, y_pred)

        logger.info("CNN-LSTM MAE: %.4f", metrics["mae"])
        return {"model": model, "metrics": metrics, "predictions": y_pred, "history": history}

    def train_stacked_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train a stacked LSTM with multiple layers."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install tensorflow to use Stacked LSTM model") from exc

        stacked_cfg = self.config.get("stacked_lstm", {})
        units_list = stacked_cfg.get("units", [50, 50])
        dropout = stacked_cfg.get("dropout", 0.2)
        epochs = stacked_cfg.get("epochs", 50)
        batch_size = stacked_cfg.get("batch_size", 32)

        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1], 1)))

        for i, units in enumerate(units_list):
            return_seq = i < len(units_list) - 1
            model.add(layers.LSTM(units, activation="relu", return_sequences=return_seq))
            model.add(layers.Dropout(dropout))

        model.add(layers.Dense(1))

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = self._compute_metrics(y_test, y_pred)

        logger.info("Stacked LSTM MAE: %.4f", metrics["mae"])
        return {"model": model, "metrics": metrics, "predictions": y_pred, "history": history}

    def train_all(
        self,
        df: pd.DataFrame,
        target_column: str = "close",
        date_column: str = "date",
    ) -> Dict[str, Dict]:
        """Train all configured deep learning models."""
        lookback = self.config.get("lookback", 60)
        test_size = self.config.get("test_size", 0.2)

        # Prepare time-series
        series = df[[target_column]].squeeze()
        X, y, scaler = self.prepare_sequences(series, lookback=lookback, normalize=True)
        self.scalers["primary"] = scaler

        X_train, X_test, y_train, y_test = self.split_sequences(X, y, test_size=test_size)

        # Reshape for RNN models (add channel dimension)
        X_train_rnn = X_train.reshape(*X_train.shape, 1)
        X_test_rnn = X_test.reshape(*X_test.shape, 1)

        algorithms = self.config.get("algorithms", ["mlp", "lstm", "gru", "cnn_lstm"])

        if "mlp" in algorithms:
            logger.info("Training MLP")
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            self.results["mlp"] = self.train_mlp(X_train_flat, y_train, X_test_flat, y_test)
            self.models["mlp"] = self.results["mlp"]["model"]

        if "lstm" in algorithms:
            logger.info("Training LSTM")
            self.results["lstm"] = self.train_lstm(X_train_rnn, y_train, X_test_rnn, y_test)
            self.models["lstm"] = self.results["lstm"]["model"]

        if "bilstm" in algorithms:
            logger.info("Training Bidirectional LSTM")
            self.results["bilstm"] = self.train_bilstm(X_train_rnn, y_train, X_test_rnn, y_test)
            self.models["bilstm"] = self.results["bilstm"]["model"]

        if "lstm_attention" in algorithms:
            logger.info("Training LSTM with Attention")
            self.results["lstm_attention"] = self.train_lstm_attention(
                X_train_rnn, y_train, X_test_rnn, y_test
            )
            self.models["lstm_attention"] = self.results["lstm_attention"]["model"]

        if "gru" in algorithms:
            logger.info("Training GRU")
            self.results["gru"] = self.train_gru(X_train_rnn, y_train, X_test_rnn, y_test)
            self.models["gru"] = self.results["gru"]["model"]

        if "cnn_lstm" in algorithms:
            logger.info("Training CNN-LSTM")
            self.results["cnn_lstm"] = self.train_cnn_lstm(
                X_train_rnn, y_train, X_test_rnn, y_test
            )
            self.models["cnn_lstm"] = self.results["cnn_lstm"]["model"]

        if "stacked_lstm" in algorithms:
            logger.info("Training Stacked LSTM")
            self.results["stacked_lstm"] = self.train_stacked_lstm(
                X_train_rnn, y_train, X_test_rnn, y_test
            )
            self.models["stacked_lstm"] = self.results["stacked_lstm"]["model"]

        self._select_best_model()
        self.X_test = X_test_rnn
        self.y_test = y_test
        return self.results

    def _select_best_model(self):
        """Select best model based on MAE."""
        best_mae = float("inf")
        for name, result in self.results.items():
            mae = result["metrics"].get("mae", float("inf"))
            if mae < best_mae:
                best_mae = mae
                self.best_model_name = name
                self.best_model = self.models.get(name)
        logger.info("Best deep learning model: %s (MAE %.4f)", self.best_model_name, best_mae)

    def forecast(
        self,
        recent_data: np.ndarray,
        steps: int,
        model_name: Optional[str] = None,
        scaler: Optional[MinMaxScaler] = None,
    ) -> np.ndarray:
        """Generate forecasts using trained model."""
        if model_name is None:
            model_name = self.best_model_name
        if model_name is None:
            raise RuntimeError("No trained model available")

        model = self.models.get(model_name)
        scaler = scaler or self.scalers.get("primary")

        predictions = []
        current_seq = recent_data.copy()

        for _ in range(steps):
            if model_name == "mlp":
                seq_flat = current_seq.reshape(1, -1)
                next_pred = model.predict(seq_flat, verbose=0)[0, 0]
            else:
                seq_reshaped = current_seq.reshape(1, current_seq.shape[0], 1)
                next_pred = model.predict(seq_reshaped, verbose=0)[0, 0]

            predictions.append(next_pred)
            current_seq = np.append(current_seq[1:], [[next_pred]], axis=0)

        forecasts = np.array(predictions)

        # Inverse transform if scaler available
        if scaler is not None:
            forecasts = scaler.inverse_transform(forecasts.reshape(-1, 1)).flatten()

        return forecasts

"""Time-series feature utilities (e.g., Prophet-based decompositions)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def prophet_decompose_series(
    series: pd.Series,
    prophet_cfg: Optional[Dict] = None,
    prefix: str = "ts_",
) -> pd.DataFrame:
    """Fit Prophet on a univariate series and return decomposition features.

    The returned DataFrame is indexed like the input series and contains:
    - ``{prefix}trend``
    - ``{prefix}seasonal``
    - optional seasonal components (``{prefix}yearly``, ``{prefix}weekly``, ``{prefix}daily``)
    - ``{prefix}residual`` = actual - fitted

    Parameters
    ----------
    series : pd.Series
        Time series with a DatetimeIndex.
    prophet_cfg : dict, optional
        Configuration for Prophet; expects the same keys as the time_series.prophet
        block (seasonality_mode, changepoint_prior_scale, yearly_seasonality,
        weekly_seasonality, daily_seasonality).
    prefix : str, default "ts_"
        Prefix for created feature column names.
    """
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install prophet to use Prophet-based decomposition features") from exc

    if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError("prophet_decompose_series expects a series with a DatetimeIndex or PeriodIndex")

    # Ensure DatetimeIndex for Prophet
    s = series.copy()
    s.index = pd.to_datetime(s.index)

    cfg = prophet_cfg or {}
    model = Prophet(
        seasonality_mode=cfg.get("seasonality_mode", "additive"),
        changepoint_prior_scale=cfg.get("changepoint_prior_scale", 0.05),
        yearly_seasonality=cfg.get("yearly_seasonality", True),
        weekly_seasonality=cfg.get("weekly_seasonality", True),
        daily_seasonality=cfg.get("daily_seasonality", False),
    )

    df = s.reset_index()
    df.columns = ["ds", "y"]

    model.fit(df)

    forecast = model.predict(df[["ds"]])

    # Build features aligned to original index
    feat = pd.DataFrame(index=pd.DatetimeIndex(df["ds"]))
    feat[f"{prefix}trend"] = forecast["trend"].values

    # Some Prophet versions don't expose a combined "seasonal" column.
    # If it's missing, approximate seasonal as the sum of available
    # per-season components (yearly/weekly/daily/holidays/additive_terms).
    if "seasonal" in forecast.columns:
        seasonal_vals = forecast["seasonal"].values
    else:
        seasonal_sources = [
            c
            for c in (
                "yearly",
                "weekly",
                "daily",
                "holidays",
                "additive_terms",
            )
            if c in forecast.columns
        ]
        if seasonal_sources:
            seasonal_vals = forecast[seasonal_sources].sum(axis=1).values
        else:
            seasonal_vals = 0.0
    feat[f"{prefix}seasonal"] = seasonal_vals

    for comp in ("yearly", "weekly", "daily"):
        if comp in forecast.columns:
            feat[f"{prefix}{comp}"] = forecast[comp].values

    # Residual = actual - fitted yhat
    if "yhat" in forecast.columns:
        feat[f"{prefix}residual"] = df["y"].values - forecast["yhat"].values
    else:
        feat[f"{prefix}residual"] = 0.0

    # Reindex to exactly match the input series index ordering
    feat = feat.reindex(s.index)

    return feat


def prophet_decompose_train_and_forecast(
    train: pd.Series,
    test: pd.Series,
    prophet_cfg: Optional[Dict] = None,
    prefix: str = "ts_",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit Prophet on the *training* slice and decompose train + future horizon.

    This helper is intended for leakage-safe model evaluation:

    - Prophet is fit **only** on ``train``.
    - Components for the training period are computed by predicting on the
      training timestamps and comparing against the observed values.
    - Components for the test period are obtained by forecasting into the
      future for ``len(test)`` steps. Residuals for the test horizon are not
      defined (there is no observed ``y`` at forecast time), so they are
      filled with NaN.

    Both returned DataFrames use DatetimeIndex and contain at least:

    - ``{prefix}trend``
    - ``{prefix}seasonal``
    - optional ``{prefix}yearly``, ``{prefix}weekly``, ``{prefix}daily``
    - ``{prefix}residual`` (train only; NaN on the test horizon)
    """
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install prophet to use Prophet-based decomposition features"
        ) from exc

    if not isinstance(train.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError(
            "prophet_decompose_train_and_forecast expects train with a DatetimeIndex or PeriodIndex"
        )
    if not isinstance(test.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError(
            "prophet_decompose_train_and_forecast expects test with a DatetimeIndex or PeriodIndex"
        )

    # Ensure DatetimeIndex for Prophet
    train_s = train.copy()
    train_s.index = pd.to_datetime(train_s.index)
    test_index = pd.to_datetime(test.index)

    cfg = prophet_cfg or {}
    model = Prophet(
        seasonality_mode=cfg.get("seasonality_mode", "additive"),
        changepoint_prior_scale=cfg.get("changepoint_prior_scale", 0.05),
        yearly_seasonality=cfg.get("yearly_seasonality", True),
        weekly_seasonality=cfg.get("weekly_seasonality", True),
        daily_seasonality=cfg.get("daily_seasonality", False),
    )

    train_df = train_s.reset_index()
    train_df.columns = ["ds", "y"]

    model.fit(train_df)

    # 1) Decomposition for the training window
    fc_train = model.predict(train_df[["ds"]])
    feat_train = pd.DataFrame(index=pd.DatetimeIndex(train_df["ds"]))
    feat_train[f"{prefix}trend"] = fc_train["trend"].values

    if "seasonal" in fc_train.columns:
        seasonal_vals = fc_train["seasonal"].values
    else:
        seasonal_sources = [
            c
            for c in (
                "yearly",
                "weekly",
                "daily",
                "holidays",
                "additive_terms",
            )
            if c in fc_train.columns
        ]
        seasonal_vals = fc_train[seasonal_sources].sum(axis=1).values if seasonal_sources else 0.0
    feat_train[f"{prefix}seasonal"] = seasonal_vals

    for comp in ("yearly", "weekly", "daily"):
        if comp in fc_train.columns:
            feat_train[f"{prefix}{comp}"] = fc_train[comp].values

    if "yhat" in fc_train.columns:
        feat_train[f"{prefix}residual"] = train_df["y"].values - fc_train["yhat"].values
    else:
        feat_train[f"{prefix}residual"] = 0.0

    # 2) Decomposition for the test horizon via future forecast
    # We build a future frame whose ds values match the test index.
    # Prophet's make_future_dataframe assumes regular spacing; because the
    # input series has already been converted to a fixed freq earlier,
    # using the same horizon length will align with the test index.
    horizon = len(test_index)
    if horizon > 0:
        # Include history=False so we only get future points
        future = model.make_future_dataframe(
            periods=horizon,
            freq=pd.infer_freq(train_s.index) or "D",
            include_history=False,
        )
        fc_test = model.predict(future)
        # Force index to the test index ordering
        feat_test = pd.DataFrame(index=test_index)
        # Align Prophet outputs to the test index length
        fc_test = fc_test.iloc[:horizon]
        feat_test[f"{prefix}trend"] = fc_test["trend"].values

        if "seasonal" in fc_test.columns:
            seasonal_vals_test = fc_test["seasonal"].values
        else:
            seasonal_sources_test = [
                c
                for c in (
                    "yearly",
                    "weekly",
                    "daily",
                    "holidays",
                    "additive_terms",
                )
                if c in fc_test.columns
            ]
            seasonal_vals_test = (
                fc_test[seasonal_sources_test].sum(axis=1).values if seasonal_sources_test else 0.0
            )
        feat_test[f"{prefix}seasonal"] = seasonal_vals_test

        for comp in ("yearly", "weekly", "daily"):
            if comp in fc_test.columns:
                feat_test[f"{prefix}{comp}"] = fc_test[comp].values

        # Residuals are undefined for the future horizon; mark as NaN
        feat_test[f"{prefix}residual"] = float("nan")
    else:
        feat_test = pd.DataFrame(index=test_index)

    return feat_train, feat_test


def prophet_decompose_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    date_col: str,
    target_col: str,
    prophet_cfg: Optional[Dict] = None,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Add Prophet decomposition features per group to a dataframe.

    Fits a separate Prophet model for each group defined by ``group_cols`` on
    the target column over time, and merges the resulting components back
    into the original dataframe.

    Each row in the returned dataframe will have, per group and date:
    - ``{prefix}trend``
    - ``{prefix}seasonal``
    - optional ``{prefix}yearly``, ``{prefix}weekly``, ``{prefix}daily``
    - ``{prefix}residual``

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing group columns, a date column, and target column.
    group_cols : sequence of str
        Columns that define each group, e.g. ["store_name", "sub_department"].
    date_col : str
        Name of the date/time column.
    target_col : str
        Name of the numeric target column to decompose.
    prophet_cfg : dict, optional
        Passed through to ``prophet_decompose_series`` / Prophet.
    prefix : str, optional
        Prefix for created feature column names. If None, ``f"{target_col}_"``
        is used.
    """
    if isinstance(group_cols, (str, bytes)):
        group_cols = [group_cols]  # type: ignore[assignment]

    group_cols = list(group_cols)
    pfx = prefix or f"{target_col}_"

    # Ensure date column is datetime for consistent behaviour
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])

    pieces: List[pd.DataFrame] = []
    for keys, g in work[group_cols + [date_col, target_col]].dropna(subset=[target_col]).groupby(group_cols):
        # keys is a scalar or tuple depending on length of group_cols
        series = g.sort_values(date_col).set_index(date_col)[target_col]

        dec = prophet_decompose_series(series, prophet_cfg=prophet_cfg, prefix=pfx)
        dec_reset = dec.reset_index()
        dec_reset = dec_reset.rename(columns={dec_reset.columns[0]: date_col})

        # Attach group columns back
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(group_cols, keys):
            dec_reset[col] = val

        pieces.append(dec_reset)

    if not pieces:
        # Nothing to decompose; return original frame unchanged
        return work

    feat_all = pd.concat(pieces, ignore_index=True)

    # Merge back to original df using group_cols + date_col as key
    merge_keys = group_cols + [date_col]
    out = work.merge(feat_all, on=merge_keys, how="left")
    return out


def prophet_plot_components_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    date_col: str,
    target_col: str,
    prophet_cfg: Optional[Dict] = None,
) -> None:
    """Plot Prophet decomposed components for each group.

    For every group defined by ``group_cols``, this function fits a Prophet
    model on ``target_col`` over ``date_col`` and calls
    ``model.plot_components(forecast)`` to show the trend and seasonal
    components.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing group columns, a date column, and target column.
    group_cols : sequence of str
        Columns that define each group, e.g. ["store_name", "sub_department"].
    date_col : str
        Name of the date/time column.
    target_col : str
        Name of the numeric target column to model.
    prophet_cfg : dict, optional
        Configuration for Prophet; expects the same keys as the
        ``time_series.prophet`` block (seasonality_mode, changepoint_prior_scale,
        yearly_seasonality, weekly_seasonality, daily_seasonality).
    """
    try:  # Local import so module doesn't hard-depend on matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install matplotlib to plot Prophet decompositions") from exc

    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install prophet to use Prophet-based decomposition features") from exc

    if isinstance(group_cols, (str, bytes)):
        group_cols = [group_cols]  # type: ignore[assignment]

    group_cols = list(group_cols)

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])

    cfg = prophet_cfg or {}

    for keys, g in work[group_cols + [date_col, target_col]].dropna(subset=[target_col]).groupby(group_cols):
        series = g.sort_values(date_col).set_index(date_col)[target_col]

        model = Prophet(
            seasonality_mode=cfg.get("seasonality_mode", "additive"),
            changepoint_prior_scale=cfg.get("changepoint_prior_scale", 0.05),
            yearly_seasonality=cfg.get("yearly_seasonality", True),
            weekly_seasonality=cfg.get("weekly_seasonality", True),
            daily_seasonality=cfg.get("daily_seasonality", False),
        )

        df_model = series.reset_index()
        df_model.columns = ["ds", "y"]

        model.fit(df_model)

        forecast = model.predict(df_model[["ds"]])

        fig = model.plot_components(forecast)

        # Add a helpful title per group
        if not isinstance(keys, tuple):
            keys = (keys,)
        title_parts = [f"{col}={val}" for col, val in zip(group_cols, keys)]
        fig.suptitle("; ".join(title_parts))

        plt.show()


def add_technical_indicators(
    df: pd.DataFrame,
    value_col: str,
    group_cols: Optional[Sequence[str]] = None,
    date_col: str = "date",
    sma_periods: Optional[List[int]] = None,
    ema_periods: Optional[List[int]] = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    roc_period: int = 1,
    atr_period: int = 14,
    fibo_window: int = 252,
    prefix: str = "",
) -> pd.DataFrame:
    """Add technical indicators to a time series dataframe.
    
    This function calculates various technical indicators similar to those used
    in trading analysis, but designed for any time series data (e.g., sales,
    demand, inventory). Indicators are computed per group if group_cols is provided.
    
    Indicators added:
    - Simple Moving Averages (SMA)
    - Exponential Moving Averages (EMA)
    - Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Bollinger Bands (upper, middle, lower)
    - Rate of Change (ROC)
    - Average True Range (ATR)
    - Fibonacci retracement levels
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data.
    value_col : str
        Name of the column to calculate indicators from (e.g., 'total_qty', 'close').
    group_cols : sequence of str, optional
        Column names to group by. If provided, indicators are calculated per group.
    date_col : str, default "date"
        Name of the date/time column for sorting.
    sma_periods : list of int, optional
        Periods for Simple Moving Averages. Default [7, 14, 20, 50].
    ema_periods : list of int, optional
        Periods for Exponential Moving Averages. Default [7, 14, 30, 50].
    rsi_period : int, default 14
        Period for RSI calculation.
    macd_fast : int, default 12
        Fast EMA period for MACD.
    macd_slow : int, default 26
        Slow EMA period for MACD.
    macd_signal : int, default 9
        Signal line EMA period for MACD.
    bb_period : int, default 20
        Period for Bollinger Bands.
    bb_std : float, default 2.0
        Number of standard deviations for Bollinger Bands.
    roc_period : int, default 1
        Period for Rate of Change.
    atr_period : int, default 14
        Period for Average True Range.
    fibo_window : int, default 252
        Rolling window for Fibonacci retracement levels.
    prefix : str, default ""
        Prefix for indicator column names.
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with added indicator columns.
    
    Examples
    --------
    >>> # Add indicators to sales data without grouping
    >>> df_with_indicators = add_technical_indicators(
    ...     df=df,
    ...     value_col="total_qty",
    ...     date_col="sale_date",
    ...     sma_periods=[7, 14, 30],
    ... )
    
    >>> # Add indicators per store/department group
    >>> df_with_indicators = add_technical_indicators(
    ...     df=df,
    ...     value_col="total_qty",
    ...     group_cols=["store_name", "sub_department"],
    ...     date_col="sale_date",
    ...     prefix="qty_",
    ... )
    """
    if sma_periods is None:
        sma_periods = [7, 14, 20, 50]
    if ema_periods is None:
        ema_periods = [7, 14, 30, 50]
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if group_cols is None:
        # Process entire dataframe as one group
        df = df.sort_values(date_col).reset_index(drop=True)
        df = _calculate_indicators(
            df=df,
            value_col=value_col,
            sma_periods=sma_periods,
            ema_periods=ema_periods,
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            bb_period=bb_period,
            bb_std=bb_std,
            roc_period=roc_period,
            atr_period=atr_period,
            fibo_window=fibo_window,
            prefix=prefix,
        )
    else:
        # Process each group separately
        if isinstance(group_cols, (str, bytes)):
            group_cols = [group_cols]  # type: ignore[assignment]
        
        group_cols = list(group_cols)
        pieces = []
        
        for _, g in df.groupby(group_cols):
            g = g.sort_values(date_col).reset_index(drop=True)
            g_with_indicators = _calculate_indicators(
                df=g,
                value_col=value_col,
                sma_periods=sma_periods,
                ema_periods=ema_periods,
                rsi_period=rsi_period,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                bb_period=bb_period,
                bb_std=bb_std,
                roc_period=roc_period,
                atr_period=atr_period,
                fibo_window=fibo_window,
                prefix=prefix,
            )
            pieces.append(g_with_indicators)
        
        df = pd.concat(pieces, ignore_index=True)
    
    return df


def _calculate_indicators(
    df: pd.DataFrame,
    value_col: str,
    sma_periods: List[int],
    ema_periods: List[int],
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    bb_period: int,
    bb_std: float,
    roc_period: int,
    atr_period: int,
    fibo_window: int,
    prefix: str,
) -> pd.DataFrame:
    """Calculate technical indicators for a single group or entire dataframe."""
    df = df.copy()
    values = df[value_col].astype(float)
    
    # Simple Moving Averages
    for period in sma_periods:
        df[f"{prefix}SMA_{period}"] = values.rolling(window=period, min_periods=1).mean()
    
    # Exponential Moving Averages
    for period in ema_periods:
        df[f"{prefix}EMA_{period}"] = values.ewm(span=period, adjust=False, min_periods=1).mean()
    
    # Relative Strength Index (RSI)
    df[f"{prefix}RSI_{rsi_period}"] = _calculate_rsi(values, period=rsi_period)
    
    # MACD (Moving Average Convergence Divergence)
    macd, macd_signal_line, macd_hist = _calculate_macd(
        values, fast=macd_fast, slow=macd_slow, signal=macd_signal
    )
    df[f"{prefix}MACD"] = macd
    df[f"{prefix}MACD_signal"] = macd_signal_line
    df[f"{prefix}MACD_hist"] = macd_hist
    
    # Bollinger Bands
    bb_middle, bb_upper, bb_lower = _calculate_bollinger_bands(
        values, period=bb_period, num_std=bb_std
    )
    df[f"{prefix}BB_upper"] = bb_upper
    df[f"{prefix}BB_middle"] = bb_middle
    df[f"{prefix}BB_lower"] = bb_lower
    df[f"{prefix}BB_width"] = bb_upper - bb_lower
    
    # Rate of Change (ROC)
    df[f"{prefix}ROC_{roc_period}"] = _calculate_roc(values, period=roc_period)
    
    # Average True Range (ATR)
    # For non-OHLC data, use simple range approximation
    df[f"{prefix}ATR_{atr_period}"] = _calculate_atr_simple(values, period=atr_period)
    
    # Fibonacci Retracement Levels
    fibo_cols = _calculate_fibonacci_levels(values, window=fibo_window, prefix=prefix)
    for col_name, col_values in fibo_cols.items():
        df[col_name] = col_values
    
    return df


def _calculate_rsi(values: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = values.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Neutral RSI when no data


def _calculate_macd(
    values: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = values.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = values.ewm(span=slow, adjust=False, min_periods=1).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def _calculate_bollinger_bands(
    values: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    middle = values.rolling(window=period, min_periods=1).mean()
    std = values.rolling(window=period, min_periods=1).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return middle, upper, lower


def _calculate_roc(values: pd.Series, period: int = 1) -> pd.Series:
    """Calculate Rate of Change (ROC)."""
    roc = ((values - values.shift(period)) / values.shift(period).replace(0, np.nan)) * 100
    return roc.fillna(0)


def _calculate_atr_simple(values: pd.Series, period: int = 14) -> pd.Series:
    """Calculate simplified Average True Range for univariate data.
    
    For traditional ATR, you need High, Low, Close. For single-value series,
    we approximate using rolling range (max - min) over the period.
    """
    rolling_max = values.rolling(window=period, min_periods=1).max()
    rolling_min = values.rolling(window=period, min_periods=1).min()
    true_range = rolling_max - rolling_min
    
    # Apply exponential smoothing similar to traditional ATR
    atr = true_range.ewm(span=period, adjust=False, min_periods=1).mean()
    
    return atr


def _calculate_fibonacci_levels(
    values: pd.Series,
    window: int = 252,
    prefix: str = "",
) -> Dict[str, pd.Series]:
    """Calculate Fibonacci retracement levels based on rolling window."""
    rolling_high = values.rolling(window=window, min_periods=1).max()
    rolling_low = values.rolling(window=window, min_periods=1).min()
    diff = rolling_high - rolling_low
    
    fibo_levels = {
        f"{prefix}Fibo_0.0": rolling_high,
        f"{prefix}Fibo_23.6": rolling_high - 0.236 * diff,
        f"{prefix}Fibo_38.2": rolling_high - 0.382 * diff,
        f"{prefix}Fibo_50.0": rolling_high - 0.5 * diff,
        f"{prefix}Fibo_61.8": rolling_high - 0.618 * diff,
        f"{prefix}Fibo_78.6": rolling_high - 0.786 * diff,
        f"{prefix}Fibo_100.0": rolling_low,
        f"{prefix}Fibo_161.8": rolling_high - 1.618 * diff,
        f"{prefix}Rolling_High": rolling_high,
        f"{prefix}Rolling_Low": rolling_low,
    }
    
    return fibo_levels

"""
from src.utils.time_series_features import add_technical_indicators

# Add indicators per group
df_with_indicators = add_technical_indicators(
    df=df,
    value_col="total_qty",
    group_cols=["store_name", "sub_department"],
    date_col="sale_date",
    sma_periods=[7, 14, 30],
    ema_periods=[5, 10, 20, 50, 100],
    rsi_period=14,
    prefix="qty_",
)

# Or for entire dataset (no grouping)
df_with_indicators = add_technical_indicators(
    df=df,
    value_col="total_qty",
    date_col="sale_date",
    prefix="",
)
"""
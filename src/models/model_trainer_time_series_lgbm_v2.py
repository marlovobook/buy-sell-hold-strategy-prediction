"""
LGBM model training utilities for time-series forecasting with grouped data.
This module provides specialized functions for training LGBM models on grouped time series data.

Usage Examples
--------------

Example 1: Train LGBM models for multiple groups without PDF report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    import pandas as pd
    from src.models.model_trainer_time_series import ModelTrainerTimeSeries
    from src.models.model_trainer_time_series_lgbm import LGBMGroupTrainer
    from src.utils.config_loader import load_config
    
    # Load configuration
    config = load_config("config/config.yaml")
    time_series_config = config['time_series']
    
    # Initialize base trainer
    time_series_config["lgbm"]["use_prophet_features"] = False
    time_series_config["lgbm"]["verbose"] = -1
    trainer = ModelTrainerTimeSeries(time_series_config)
    
    # Initialize LGBM group trainer
    lgbm_trainer = LGBMGroupTrainer(trainer, time_series_config)
    
    # Load your data
    df = pd.read_parquet("data/feature_v1.parquet")
    
    # Train models for each group
    group_metrics, ypred_list = lgbm_trainer.train_lgbm_group(
        df=df,
        group_cols=["store_name", "sub_department"],
        date_col="sale_date",
        target_col="qty_cy",
        forecast_horizon=30,
        show_plots=True,  # Display plots in notebook
    )
    
    # Combine all predictions
    ypred_df = pd.concat(ypred_list, ignore_index=True)
    
    # Access metrics for a specific group
    print(group_metrics[("Store A", "Department 1")])


Example 2: Train LGBM models and generate PDF report with metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    import pandas as pd
    from src.models.model_trainer_time_series import ModelTrainerTimeSeries
    from src.models.model_trainer_time_series_lgbm import LGBMGroupTrainer
    from src.utils.config_loader import load_config
    
    # Load configuration
    config = load_config("config/config.yaml")
    time_series_config = config['time_series']
    
    # Initialize base trainer
    time_series_config["lgbm"]["use_prophet_features"] = False
    time_series_config["lgbm"]["verbose"] = -1
    trainer = ModelTrainerTimeSeries(time_series_config)
    
    # Initialize LGBM group trainer
    lgbm_trainer = LGBMGroupTrainer(trainer, time_series_config)
    
    # Load your data
    df = pd.read_parquet("data/feature_v1.parquet")
    
    # Train models and generate PDF report
    group_metrics, ypred_list = lgbm_trainer.train_lgbm_group_report(
        df=df,
        group_cols=["store_name", "sub_department"],
        date_col="sale_date",
        target_col="qty_cy",
        pdf_path="data/figure/lgbm_forecasts.pdf",
        forecast_horizon=30,
    )
    
    # All plots saved to PDF with metrics displayed
    # Combine all predictions
    ypred_df = pd.concat(ypred_list, ignore_index=True)
    ypred_df.to_csv("data/predictions.csv", index=False)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas.tseries.frequencies import to_offset

from .model_trainer_time_series import ModelTrainerTimeSeries

logger = logging.getLogger(__name__)


class LGBMGroupTrainer:
    """
    Trainer class for LGBM models on grouped time-series data.
    
    This class provides methods to train LGBM models on multiple groups of time series,
    generate forecasts, and create visualizations with optional PDF export.
    """
    
    def __init__(self, trainer: ModelTrainerTimeSeries, time_series_config: Dict):
        """
        Initialize the LGBM group trainer.
        
        Parameters
        ----------
        trainer : ModelTrainerTimeSeries
            An instance of the base time series model trainer.
        time_series_config : Dict
            Configuration dictionary containing LGBM and time series settings.
        """
        self.trainer = trainer
        self.config = time_series_config
        self.test_size = time_series_config.get("test_size", 0.2)
        self.n_lags = time_series_config.get("lgbm", {}).get("n_lags", 14)
    
    def train_lgbm_group(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        date_col: str,
        target_col: str,
        forecast_horizon: int = 30,
        show_plots: bool = True,
    ) -> Tuple[Dict, List[pd.DataFrame]]:
        """
        Train LGBM models for each group in the dataset without generating PDF reports.
        
        This function iterates through each group, trains an LGBM model, generates
        forecasts, and optionally displays plots in the notebook/console.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing time series data for multiple groups.
        group_cols : List[str]
            Column names to group by (e.g., ['store_name', 'sub_department']).
        date_col : str
            Name of the date column.
        target_col : str
            Name of the target column to forecast.
        forecast_horizon : int, default=30
            Number of future time steps to forecast.
        show_plots : bool, default=True
            Whether to display plots using plt.show().
        
        Returns
        -------
        group_metrics : Dict
            Dictionary mapping group keys to their respective metrics.
        ypred_list : List[pd.DataFrame]
            List of dataframes containing predictions for each group.
        
        Examples
        --------
        >>> # Train models for multiple store-department combinations
        >>> group_metrics, ypred_list = lgbm_trainer.train_lgbm_group(
        ...     df=df,
        ...     group_cols=["store_name", "sub_department"],
        ...     date_col="sale_date",
        ...     target_col="qty_cy",
        ...     forecast_horizon=30,
        ...     show_plots=True,
        ... )
        >>> 
        >>> # Combine all predictions
        >>> ypred_df = pd.concat(ypred_list, ignore_index=True)
        >>> 
        >>> # Check metrics for a specific group
        >>> store_dept_key = ("Store A", "Department 1")
        >>> print(f"MAE: {group_metrics[store_dept_key]['mae']:.2f}")
        >>> print(f"RMSE: {group_metrics[store_dept_key]['rmse']:.2f}")
        """
        group_metrics = {}
        ypred_list = []
        
        for keys, g in df.groupby(group_cols):
            result_df = self._train_single_group(
                g=g,
                keys=keys,
                group_cols=group_cols,
                date_col=date_col,
                target_col=target_col,
                forecast_horizon=forecast_horizon,
                group_metrics=group_metrics,
            )
            
            if result_df is not None:
                ypred_list.append(result_df)
                
                if show_plots:
                    # Create and show plot for this group
                    self._create_plot(
                        result_df=result_df,
                        keys=keys,
                        metrics=group_metrics[keys],
                        show_metrics=True,
                    )
                    plt.show()
        
        return group_metrics, ypred_list
    
    def train_lgbm_group_report(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        date_col: str,
        target_col: str,
        pdf_path: str,
        forecast_horizon: int = 30,
    ) -> Tuple[Dict, List[pd.DataFrame]]:
        """
        Train LGBM models for each group and generate a PDF report with plots and metrics.
        
        This function iterates through each group, trains an LGBM model, generates
        forecasts, and saves all plots with metrics to a single PDF file.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing time series data for multiple groups.
        group_cols : List[str]
            Column names to group by (e.g., ['store_name', 'sub_department']).
        date_col : str
            Name of the date column.
        target_col : str
            Name of the target column to forecast.
        pdf_path : str
            Path where the PDF report should be saved.
        forecast_horizon : int, default=30
            Number of future time steps to forecast.
        
        Returns
        -------
        group_metrics : Dict
            Dictionary mapping group keys to their respective metrics.
        ypred_list : List[pd.DataFrame]
            List of dataframes containing predictions for each group.
        
        Examples
        --------
        >>> # Train models and generate PDF report
        >>> group_metrics, ypred_list = lgbm_trainer.train_lgbm_group_report(
        ...     df=df,
        ...     group_cols=["store_name", "sub_department"],
        ...     date_col="sale_date",
        ...     target_col="qty_cy",
        ...     pdf_path="data/figure/lgbm_forecasts.pdf",
        ...     forecast_horizon=30,
        ... )
        >>> # All plots saved to: data/figure/lgbm_forecasts.pdf
        >>> 
        >>> # Save predictions to CSV
        >>> ypred_df = pd.concat(ypred_list, ignore_index=True)
        >>> ypred_df.to_csv("data/predictions.csv", index=False)
        >>> 
        >>> # Filter predictions for a specific group and time range
        >>> specific_group = ypred_df[
        ...     (ypred_df["store_name"] == "Store A") &
        ...     (ypred_df["sub_department"] == "Department 1") &
        ...     (ypred_df["yhat_future"].notnull())
        ... ]
        """
        group_metrics = {}
        ypred_list = []
        
        os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

        with PdfPages(pdf_path) as pdf:
            for keys, g in df.groupby(group_cols):
                result_df = self._train_single_group(
                    g=g,
                    keys=keys,
                    group_cols=group_cols,
                    date_col=date_col,
                    target_col=target_col,
                    forecast_horizon=forecast_horizon,
                    group_metrics=group_metrics,
                )
                
                if result_df is not None:
                    ypred_list.append(result_df)
                    
                    # Create plot with metrics and save to PDF
                    fig = self._create_plot(
                        result_df=result_df,
                        keys=keys,
                        metrics=group_metrics[keys],
                        show_metrics=True,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)
        
        print(f"\nAll plots saved to: {pdf_path}")
        return group_metrics, ypred_list
    
    def _train_single_group(
        self,
        g: pd.DataFrame,
        keys: Tuple,
        group_cols: List[str],
        date_col: str,
        target_col: str,
        forecast_horizon: int,
        group_metrics: Dict,
    ) -> Optional[pd.DataFrame]:
        """
        Train LGBM model for a single group and generate forecasts.
        
        Parameters
        ----------
        g : pd.DataFrame
            Group data.
        keys : Tuple
            Group keys (e.g., (store_name, sub_department)).
        group_cols : List[str]
            Column names used for grouping.
        date_col : str
            Name of the date column.
        target_col : str
            Name of the target column.
        forecast_horizon : int
            Number of future steps to forecast.
        group_metrics : Dict
            Dictionary to store metrics for this group.
        
        Returns
        -------
        pd.DataFrame or None
            DataFrame containing historical and forecast data for this group.
        """
        # Determine feature columns
        feature_cols = [
            c for c in g.columns
            if c not in [date_col, target_col] + group_cols
        ]
        print(f"Processing group: {keys}")
        print(f"Feature columns: {feature_cols}")
        
        # Aggregate to one row per date
        agg_dict = {target_col: "sum"}
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(g[c]):
                agg_dict[c] = "mean"
            else:
                agg_dict[c] = "first"
        
        g_agg = (
            g[[date_col] + [target_col] + feature_cols]
            .groupby(date_col, as_index=False)
            .agg(agg_dict)
        )
        
        # Prepare univariate target series
        series_g = self.trainer.prepare_series(
            g_agg[[date_col, target_col]],
            target_column=target_col,
            date_column=date_col,
        )
        train_g, test_g = self.trainer.split_series(series_g, test_size=self.test_size)
        
        # Build exogenous feature frame
        g_feat = (
            g_agg[[date_col] + feature_cols]
            .assign(**{date_col: pd.to_datetime(g_agg[date_col])})
            .sort_values(date_col)
            .set_index(date_col)
        )
        g_feat = g_feat.reindex(series_g.index).ffill().bfill()
        
        # Identify categorical features
        cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(g_feat[c])]
        cat_categories = {c: pd.Categorical(g_feat[c]).categories for c in cat_cols}
        
        # Train LGBM; skip groups that lack enough history for lag features
        try:
            lgbm_g = self.trainer.train_lgbm(
                series_g,
                train_g,
                test_g,
                extra_features=g_feat,
            )
        except ValueError as exc:
            if "Not enough historical data to build lag features for LGBM train set" in str(exc):
                msg = (
                    f"Skipping group {keys}: insufficient history for lag features "
                    f"(n_lags={self.n_lags}, rows={len(series_g)})"
                )
                print(msg)
                logger.warning(msg)
                return None
            raise
        
        group_metrics[keys] = lgbm_g["metrics"]
        model = lgbm_g["model"]
        feature_names = lgbm_g.get("feature_names", None)
        
        # Print metrics
        print(f"\nMetrics for group {keys}:")
        for k, v in lgbm_g["metrics"].items():
            print(f"  {k}: {v:.4f}" if v == v else f"  {k}: NaN")
        
        # Generate predictions
        pred_test = pd.Series(lgbm_g["predictions"], index=test_g.index)
        pred_insample = pred_test.reindex(series_g.index)
        
        # Generate future forecasts
        pred_future = self._generate_future_forecast(
            series_g=series_g,
            model=model,
            feature_names=feature_names,
            feature_cols=feature_cols,
            g_feat=g_feat,
            cat_cols=cat_cols,
            cat_categories=cat_categories,
            forecast_horizon=forecast_horizon,
        )
        
        # Collect results into dataframe
        result_df = self._create_result_dataframe(
            keys=keys,
            group_cols=group_cols,
            series_g=series_g,
            pred_insample=pred_insample,
            pred_future=pred_future,
        )
        
        return result_df
    
    def _generate_future_forecast(
        self,
        series_g: pd.Series,
        model,
        feature_names: Optional[List[str]],
        feature_cols: List[str],
        g_feat: pd.DataFrame,
        cat_cols: List[str],
        cat_categories: Dict,
        forecast_horizon: int,
    ) -> pd.Series:
        """Generate future forecasts using iterative prediction with dynamic exogenous features."""
        
        hist_vals = series_g.values.astype(float)
        last_window = hist_vals[-self.n_lags:].copy()
        
        # Infer frequency for future index
        freq = series_g.index.freq or pd.infer_freq(series_g.index) or "D"
        future_index = pd.date_range(
            series_g.index[-1] + to_offset(freq),
            periods=forecast_horizon,
            freq=freq,
        )
        
        future_vals = []
        for current_date in future_index:
            row_dict = {}
            
            # 1. Lag features (Autoregressive part)
            for i in range(self.n_lags):
                row_dict[f"lag_{i + 1}"] = float(last_window[-(i + 1)])
            
            # 2. Dynamic Exogenous features 
            # Look up features for the specific 'current_date' if they exist in g_feat
            if current_date in g_feat.index:
                current_exog = g_feat.loc[current_date]
            else:
                # Fallback to the last available features if the future date is missing
                current_exog = g_feat.iloc[-1] if not g_feat.empty else None
            
            if current_exog is not None:
                for c in feature_cols:
                    row_dict[c] = current_exog[c]
            
            # 3. Model Prediction
            X_future = pd.DataFrame([row_dict])
            if feature_names is not None:
                # Ensure column order matches training
                X_future = X_future.reindex(columns=feature_names)
            
            # Ensure categorical features use same categories as training
            for c in cat_cols:
                if c in X_future.columns:
                    X_future[c] = pd.Categorical(X_future[c], categories=cat_categories[c])
            
            y_hat = model.predict(X_future)[0]
            future_vals.append(y_hat)
            
            # 4. Update lag window for the next iteration
            last_window[:-1] = last_window[1:]
            last_window[-1] = y_hat
        
        return pd.Series(future_vals, index=future_index)
    
    def _create_result_dataframe(
        self,
        keys: Tuple,
        group_cols: List[str],
        series_g: pd.Series,
        pred_insample: pd.Series,
        pred_future: pd.Series,
    ) -> pd.DataFrame:
        """Create a dataframe with historical and forecast results."""
        # Create column dict for group keys
        group_dict = {col: key for col, key in zip(group_cols, keys)}
        
        # Historical data
        hist_df = pd.DataFrame({
            **group_dict,
            "date": series_g.index,
            "actual": series_g.values,
            "yhat_test": pred_insample.reindex(series_g.index).values,
        })
        
        # Future forecast data
        future_df = pd.DataFrame({
            **group_dict,
            "date": pred_future.index,
            "actual": np.nan,
            "yhat_test": np.nan,
            "yhat_future": pred_future.values,
        })
        
        # Ensure same columns
        hist_df["yhat_future"] = np.nan
        
        return pd.concat([hist_df, future_df], ignore_index=True)
    
    def _create_plot(
        self,
        result_df: pd.DataFrame,
        keys: Tuple,
        metrics: Dict,
        show_metrics: bool = True,
        figsize: Tuple[int, int] = (12, 4),
    ):
        """
        Create a plot for a single group's forecast.
        
        Parameters
        ----------
        result_df : pd.DataFrame
            DataFrame containing the forecast results.
        keys : Tuple
            Group keys for the plot title.
        metrics : Dict
            Dictionary of evaluation metrics.
        show_metrics : bool, default=True
            Whether to display metrics on the plot.
        figsize : Tuple[int, int], default=(12, 4)
            Figure size for the plot.
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual values
        actual_data = result_df[result_df["actual"].notnull()]
        ax.plot(actual_data["date"], actual_data["actual"], 
                label="Actual", color="tab:blue")
        
        # Plot test predictions
        test_data = result_df[result_df["yhat_test"].notnull()]
        if not test_data.empty:
            ax.plot(test_data["date"], test_data["yhat_test"],
                    label="LGBM (test period)", color="tab:orange")
        
        # Plot future forecast
        future_data = result_df[result_df["yhat_future"].notnull()]
        if not future_data.empty:
            ax.plot(future_data["date"], future_data["yhat_future"],
                    label="LGBM forecast", color="tab:green", linestyle="--")
        
        ax.legend()
        ax.set_title(f"LGBM: {' - '.join(map(str, keys))}")
        
        # Add metrics text box if requested
        if show_metrics:
            metrics_text = "\n".join([
                f"MAE: {metrics.get('mae', np.nan):.4f}",
                f"MSE: {metrics.get('mse', np.nan):.4f}",
                f"RMSE: {metrics.get('rmse', np.nan):.4f}",
                f"MAPE: {metrics.get('mape', np.nan):.4f}",
                f"SMAPE: {metrics.get('smape', np.nan):.4f}"
            ])
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig

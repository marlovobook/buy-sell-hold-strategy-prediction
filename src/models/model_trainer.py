"""
Model training and selection module.
Implements multiple ML algorithms and model selection logic.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import logging
from typing import Dict, Tuple, List
import json

import matplotlib.pyplot as plt
import seaborn as sns

import os
        

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates multiple ML models for buy/sell/hold prediction."""
    
    def __init__(self, config: Dict):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
    
    def prepare_features(self, df: pd.DataFrame, target_variable) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Args:
            df: DataFrame with all data including labels
            
        Returns:
            Tuple of (features, target)
        """
        # Select feature columns (exclude target and non-feature columns)
        exclude_cols = [target_variable]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns]
        #X = df.drop(columns=[target_variable])
        y = df[target_variable]
        
        logger.info(f"Prepared {len(X)} samples with {len(self.feature_columns)} features")
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train Random Forest classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Random Forest model...")
        
        rf_config = self.config.get('random_forest', {})
        model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 10),
            random_state=rf_config.get('random_state', 42),
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
        
        # return {
        #     'model': model,
        #     'accuracy': accuracy,
        #     'report': report,
        #     'predictions': y_pred
        # }
        
        res = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred
        }
        
        self.results['random_forest'] = res
        self.models['random_forest'] = model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train XGBoost classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training XGBoost model...")
        
        xgb_config = self.config.get('xgboost', {})
        model = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 100),
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            random_state=xgb_config.get('random_state', 42),
            eval_metric='mlogloss'
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        # get precision, recall, f1_score from report for class '1' (Hold)
        # plot confusion matrix using seaborn heatmap
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - XGBoost')
        plt.show()
        
        logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
        
        # return {
        #     'model': model,
        #     'accuracy': accuracy,
        #     'report': report,
        #     'predictions': y_pred,
        #     'confusion_matrix': cm
        # }
        
        res = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
        
        self.results['xgboost_classification'] = res
        self.models['xgboost_classification'] = model
        
        return res
    
    def train_keras(self, X_train, y_train, X_test, y_test):
        """
        Train Keras neural network.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with model and metrics
        """
        # Lazy-import TensorFlow/Keras
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as e:
            raise RuntimeError("TensorFlow not available. Install tensorflow-cpu or skip Keras training.") from e
        
        logger.info("Training Keras model...")
        
        keras_config = self.config.get('keras', {})
        
        # Build Keras model
        model = keras.Sequential([
            layers.Dense(keras_config.get('units', 64), activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(keras_config.get('dropout', 0.5)),
            layers.Dense(3, activation='softmax')  # 3 classes: Sell, Hold, Buy
        ])
        
        model.compile(
            optimizer=keras_config.get('optimizer', 'adam'),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=keras_config.get('epochs', 50),
            batch_size=keras_config.get('batch_size', 32),
            validation_split=0.1,
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Keras Accuracy: {accuracy:.4f}")
        
        # return {
        #     'model': model,
        #     'accuracy': accuracy,
        #     'report': report,
        #     'predictions': y_pred,
        #     'history': history.history
        # }
        
        res = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred,
            'history': history.history
        }
        
        self.results['keras'] = res
        self.models['keras'] = model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            X: Feature array
            y: Target array
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []
        
        # For each valid starting position, create a sequence
        # Last valid i: len(X) - sequence_length - 1, which gives y[len(X) - 1] (last element)
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_all_models(self, df: pd.DataFrame, target_variable: str = None) -> Dict:
        """
        Train all configured models and compare performance.
        
        Args:
            df: Prepared DataFrame with features and labels
            
        Returns:
            Dictionary with all model results
        """
        logger.info("Starting training of all models...")
        # Use target_variable from config if not provided
        if target_variable is None:
            target_variable = self.config.get('target_variable', 'payer_coverage')
            
        # Prepare data
        X, y = self.prepare_features(df, target_variable=target_variable)
        X_train, X_test, y_train, y_test = self.split_data(
            X, y, self.config.get('test_size', 0.2)
        )
        
        algorithms = self.config.get('algorithms', ['random_forest'])
        
        # Train each model
        if 'random_forest' in algorithms:
            self.results['random_forest'] = self.train_random_forest(
                X_train, y_train, X_test, y_test
            )
            self.models['random_forest'] = self.results['random_forest']['model']
        
        if 'xgboost' in algorithms:
            self.results['xgboost'] = self.train_xgboost(
                X_train, y_train, X_test, y_test
            )
            self.models['xgboost'] = self.results['xgboost']['model']
        
        if 'keras' in algorithms:
            self.results['keras'] = self.train_keras(
                X_train, y_train, X_test, y_test
            )
            self.models['keras'] = self.results['keras']['model']
            
        if 'xgboost_regressor' in algorithms:
            self.results['xgboost_regressor'] = self.train_xgboost_regressor(
                X_train, y_train, X_test, y_test
            )
            self.models['xgboost_regressor'] = self.results['xgboost_regressor']['model']
        
        # Select best model
        self._select_best_model()
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
        
        return self.results
    
    def _select_best_model(self):
        """Select the best performing model based on accuracy."""
        best_accuracy = 0
        
        for name, result in self.results.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        logger.info(f"Best model: {self.best_model_name} with accuracy {best_accuracy:.4f}")
    
    def save_models(self, filepath: str):
        """
        Save trained models to disk.
        
        Args:
            filepath: Directory to save models
        """
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
        for name, model in self.models.items():
            if name == 'keras':
                model.save(f"{filepath}/keras_model.h5")
            else:
                joblib.dump(model, f"{filepath}/{name}_model.pkl")
            
            logger.info(f"Saved {name} model to {filepath}")
        
        # create {filepath}/feature_columns.json if doesn't exist
        feature_columns_path = f"{filepath}/feature_columns.json"
        if not os.path.exists(feature_columns_path):
            os.makedirs(os.path.dirname(feature_columns_path), exist_ok=True)
        
        # Save feature columns
        with open(f"{filepath}/feature_columns.json", 'w') as f:
            json.dump(self.feature_columns, f)
        
        
        
        # create {filepath}/regression_results_summary.json if doesn't exist
        regression_results_summary_path = f"{filepath}/regression_results_summary.json"
        if not os.path.exists(regression_results_summary_path):
            os.makedirs(os.path.dirname(regression_results_summary_path), exist_ok=True)
        # create {filepath}/classification_results_summary.json if doesn't exist
        classification_results_summary_path = f"{filepath}/classification_results_summary.json"
        if not os.path.exists(classification_results_summary_path):
            os.makedirs(os.path.dirname(classification_results_summary_path), exist_ok=True) 
            
        # Build results summaries
        regression_summary = {}
        classification_summary = {}

        
        for name, result in self.results.items():
            lname = name.lower()
            if "regressor" in lname:
                regression_summary[name] = {
                    "mse": result.get("mse"),
                    "r2": result.get("r2"),
                }
            elif "classifier" in lname or lname in ["random_forest", "xgboost", "keras"]:
                classification_summary[name] = {
                    "accuracy": result.get("accuracy"),
                    "report": result.get("report"),
                }

        if regression_summary:
            with open(f"{filepath}/regression_results_summary.json", "w") as f:
                json.dump(regression_summary, f, indent=2)
            logger.info("Saved regression results summary.")

        if classification_summary:
            with open(f"{filepath}/classification_results_summary.json", "w") as f:
                json.dump(classification_summary, f, indent=2)
            logger.info("Saved classification results summary.")        
        
        
        # ##if model name contain classification or regression, save results accordingly
        # if "regressor" in self.best_model_name:
        #     # Save regression results
        #     results_summary = {
        #         name: {
        #             'mse': result['mse'],
        #             'r2': result['r2']
        #         }
        #         for name, result in self.results.items()
        #     }
        #     with open(f"{filepath}/regression_results_summary.json", 'w') as f:
        #         json.dump(results_summary, f, indent=2)
        # elif "classifier" in self.best_model_name or self.best_model_name in ['random_forest', 'xgboost', 'keras']:
        #     # Save classification results
        #     results_summary = {
        #         name: {
        #             'accuracy': result['accuracy'],
        #             'report': result['report']
        #         }
        #         for name, result in self.results.items()
        #     }
        #     with open(f"{filepath}/classification_results_summary.json", 'w') as f:
        #         json.dump(results_summary, f, indent=2)
                
        # else: None
            
        # # Save results summary
        # results_summary = {
        #     name: {
        #         'accuracy': result['accuracy'],
        #         'report': result['report']
        #     }
        #     for name, result in self.results.items()
        # }
        
        # with open(f"{filepath}/results_summary.json", 'w') as f:
        #     json.dump(results_summary, f, indent=2)
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Features to predict
            model_name: Name of model to use (default: best model)
            
        Returns:
            Array of predictions
        """
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        return model.predict(X[self.feature_columns])
    
    def train_xgboost_regressor(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train XGBoost regressor for continuous target prediction.
        """
        logger.info("Training XGBoost Regressor model...")
        
        xgb_config = self.config.get('xgboost', {})
        model = xgb.XGBRegressor(
            n_estimators=xgb_config.get('n_estimators', 100),
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            random_state=xgb_config.get('random_state', 42),
            enable_categorical=True
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"XGBoost MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # return {
        #     'model': model,
        #     'mse': mse,
        #     'r2': r2,
        #     'predictions': y_pred
        # }
        
        # Create the result dictionary
        res = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred
        }
        
        # KEY FIX: Store it in the class results so the plotter can see it
        self.results['xgboost_regressor'] = res
        self.models['xgboost_regressor'] = model
        
        return res
        
    def plot_model_performance(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series):
            """
            Generalized visualization for both Regression and Classification models.
            """
            if model_name not in self.results:
                # Try checking the models dict if it's not in results
                logger.error(f"Model '{model_name}' not found in results. Available keys: {list(self.results.keys())}")
                return

            result = self.results[model_name]
            y_pred = result['predictions']
            model = result['model']
            
            # We use 3 subplots for Regression (Fit, Residuals, Importance) 
            # and 2 for Classification (CM, Importance)
            is_regression = 'r2' in result
            num_cols = 3 if is_regression else 2
            fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5))

            # --- PLOT 1: Primary Performance ---
            if is_regression:
                # Predicted vs Actual
                sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=axes[0])
                line_coords = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                axes[0].plot(line_coords, line_coords, color='red', lw=2, linestyle='--')
                axes[0].set_title(f"Predicted vs Actual (R²: {result['r2']:.4f})")
                
                # --- PLOT 2: Residual Analysis (Specific to Regression) ---
                residuals = y_test - y_pred
                sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=axes[1])
                axes[1].axhline(y=0, color='red', lw=2, linestyle='--')
                axes[1].set_title("Residuals (Error Distribution)")
                axes[1].set_xlabel("Predicted Values")
                axes[1].set_ylabel("Error")
                
                importance_ax = axes[2]
            else:
                # Classification: Confusion Matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
                axes[0].set_title(f"Confusion Matrix (Acc: {result['accuracy']:.4f})")
                importance_ax = axes[1]

            # --- FINAL PLOT: Feature Importance ---
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[-10:] # Top 10
                importance_ax.barh(range(len(indices)), importances[indices], color='teal')
                importance_ax.set_yticks(range(len(indices)))
                importance_ax.set_yticklabels([self.feature_columns[i] for i in indices])
                importance_ax.set_title("Top 10 Feature Importance")
            
            plt.tight_layout()
            plt.show()
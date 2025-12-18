"""
Model training and selection module.
Implements multiple ML algorithms and model selection logic.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import logging
from typing import Dict, Tuple, List
import json

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
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Args:
            df: DataFrame with all data including labels
            
        Returns:
            Tuple of (features, target)
        """
        # Select feature columns (exclude target and non-feature columns)
        exclude_cols = ['Label', 'Future_Return', 'Dividends', 'Stock Splits']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns]
        y = df['Label']
        
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
        
        return {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred
        }
    
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
        
        logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred
        }
    
    def train_lstm(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   sequence_length: int = 60) -> Dict:
        """
        Train LSTM neural network.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            sequence_length: Length of input sequences
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training LSTM model...")
        
        lstm_config = self.config.get('lstm', {})
        
        # Reshape data for LSTM (samples, timesteps, features)
        n_features = X_train.shape[1]
        
        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(
            X_train.values, y_train.values, sequence_length
        )
        X_test_seq, y_test_seq = self._create_sequences(
            X_test.values, y_test.values, sequence_length
        )
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(
                lstm_config.get('units', 50),
                return_sequences=True,
                input_shape=(sequence_length, n_features)
            ),
            layers.Dropout(lstm_config.get('dropout', 0.2)),
            layers.LSTM(lstm_config.get('units', 50)),
            layers.Dropout(lstm_config.get('dropout', 0.2)),
            layers.Dense(3, activation='softmax')  # 3 classes: Sell, Hold, Buy
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=lstm_config.get('epochs', 50),
            batch_size=lstm_config.get('batch_size', 32),
            validation_split=0.1,
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test_seq)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test_seq, y_pred)
        report = classification_report(y_test_seq, y_pred, output_dict=True)
        
        logger.info(f"LSTM Accuracy: {accuracy:.4f}")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred,
            'history': history.history
        }
    
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
    
    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """
        Train all configured models and compare performance.
        
        Args:
            df: Prepared DataFrame with features and labels
            
        Returns:
            Dictionary with all model results
        """
        # Prepare data
        X, y = self.prepare_features(df)
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
        
        if 'lstm' in algorithms:
            self.results['lstm'] = self.train_lstm(
                X_train, y_train, X_test, y_test
            )
            self.models['lstm'] = self.results['lstm']['model']
        
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
        for name, model in self.models.items():
            if name == 'lstm':
                model.save(f"{filepath}/lstm_model.h5")
            else:
                joblib.dump(model, f"{filepath}/{name}_model.pkl")
            
            logger.info(f"Saved {name} model to {filepath}")
        
        # Save feature columns
        with open(f"{filepath}/feature_columns.json", 'w') as f:
            json.dump(self.feature_columns, f)
        
        # Save results summary
        results_summary = {
            name: {
                'accuracy': result['accuracy'],
                'report': result['report']
            }
            for name, result in self.results.items()
        }
        
        with open(f"{filepath}/results_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
    
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

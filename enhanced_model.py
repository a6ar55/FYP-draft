"""
Enhanced LSTM Model for Stock Market Prediction
Incorporates:
- Stock price data (OHLCV)
- Technical indicators (RSI, MACD, Moving Averages, etc.)
- News sentiment features
- Attention mechanism for better feature weighting
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Concatenate, Layer, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from data_preprocessing import StockDataPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class AttentionLayer(Layer):
    """Custom attention layer for LSTM outputs"""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)


class EnhancedStockPredictor:
    """
    Enhanced LSTM model with attention mechanism and multi-feature support
    """

    def __init__(self, lookback_days: int = 60, epochs: int = 100,
                 batch_size: int = 32, learning_rate: float = 0.001):
        """
        Initialize model

        Args:
            lookback_days: Number of days to look back
            epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        self.lookback_days = lookback_days
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_simple_lstm(self, n_features: int) -> Model:
        """
        Build simple LSTM model (similar to original model.py)

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.lookback_days, n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def build_attention_lstm(self, n_features: int) -> Model:
        """
        Build LSTM model with attention mechanism

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=(self.lookback_days, n_features))

        # First LSTM layer
        lstm1 = LSTM(units=100, return_sequences=True)(inputs)
        dropout1 = Dropout(0.3)(lstm1)

        # Second LSTM layer
        lstm2 = LSTM(units=100, return_sequences=True)(dropout1)
        dropout2 = Dropout(0.3)(lstm2)

        # Attention layer
        attention_output = AttentionLayer()(dropout2)

        # Dense layers
        dense1 = Dense(units=50, activation='relu')(attention_output)
        dropout3 = Dropout(0.2)(dense1)
        dense2 = Dense(units=25, activation='relu')(dropout3)
        outputs = Dense(units=1)(dense2)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def build_bidirectional_lstm(self, n_features: int) -> Model:
        """
        Build Bidirectional LSTM model (reads sequences forward and backward)

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Bidirectional(LSTM(units=64, return_sequences=True),
                         input_shape=(self.lookback_days, n_features)),
            Dropout(0.3),
            Bidirectional(LSTM(units=64, return_sequences=False)),
            Dropout(0.3),
            Dense(units=32, activation='relu'),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dense(units=1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model_type: str = 'attention') -> None:
        """
        Train the model

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            model_type: Type of model ('simple', 'attention', 'bidirectional')
        """
        n_features = X_train.shape[2]
        logger.info(f"Building {model_type} LSTM model with {n_features} features...")

        # Build model based on type
        if model_type == 'simple':
            self.model = self.build_simple_lstm(n_features)
        elif model_type == 'attention':
            self.model = self.build_attention_lstm(n_features)
        elif model_type == 'bidirectional':
            self.model = self.build_bidirectional_lstm(n_features)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Model architecture:")
        self.model.summary(print_fn=logger.info)

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        # Train model
        logger.info(f"Training model for up to {self.epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )

        logger.info("Training completed!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input sequences

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 ticker: str) -> Dict[str, float]:
        """
        Calculate evaluation metrics

        Args:
            y_true: True values
            y_pred: Predicted values
            ticker: Stock ticker symbol

        Returns:
            Dictionary of metrics
        """
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAE
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # R2 Score
        r2 = r2_score(y_true, y_pred)

        # Directional Accuracy
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100

        metrics = {
            'ticker': ticker,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2_score': r2,
            'directional_accuracy': directional_accuracy
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation Metrics for {ticker}")
        logger.info(f"{'='*60}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
        logger.info(f"{'='*60}")

        return metrics

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        dates: np.ndarray, ticker: str,
                        save_path: str = None) -> None:
        """
        Plot actual vs predicted prices

        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Date values
            ticker: Stock ticker symbol
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(15, 6))

        plt.plot(dates, y_true, label='Actual Price', color='blue', linewidth=1.5)
        plt.plot(dates, y_pred, label='Predicted Price', color='red', linewidth=1.5, alpha=0.7)

        plt.title(f'{ticker} - Actual vs Predicted Stock Prices', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (₹)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    def plot_training_history(self, save_path: str = None) -> None:
        """
        Plot training history (loss curves)

        Args:
            save_path: Path to save plot (optional)
        """
        if self.history is None:
            logger.warning("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        ax2.plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        ax2.set_title('Model MAE During Training', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()

    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save!")

        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        self.model = load_model(filepath, custom_objects={'AttentionLayer': AttentionLayer})
        logger.info(f"Model loaded from {filepath}")


def train_single_stock(ticker: str, model_type: str = 'attention',
                       lookback_days: int = 60, epochs: int = 100) -> Dict:
    """
    Complete training pipeline for a single stock

    Args:
        ticker: Stock ticker symbol
        model_type: Type of model ('simple', 'attention', 'bidirectional')
        lookback_days: Number of days to look back
        epochs: Maximum training epochs

    Returns:
        Dictionary containing results and metrics
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"Training model for {ticker}")
    logger.info(f"{'#'*60}\n")

    # Preprocess data
    preprocessor = StockDataPreprocessor(
        stock_data_dir="stock_data",
        news_db_path="news_database.db",
        lookback_days=lookback_days
    )

    data_dict = preprocessor.preprocess_for_training(ticker, test_split=0.2)

    # Create validation split from training data
    val_split = 0.2
    split_idx = int(len(data_dict['X_train']) * (1 - val_split))

    X_train = data_dict['X_train'][:split_idx]
    y_train = data_dict['y_train'][:split_idx]
    X_val = data_dict['X_train'][split_idx:]
    y_val = data_dict['y_train'][split_idx:]

    # Train model
    predictor = EnhancedStockPredictor(
        lookback_days=lookback_days,
        epochs=epochs,
        batch_size=32
    )

    predictor.train(X_train, y_train, X_val, y_val, model_type=model_type)

    # Make predictions on test set
    y_pred_scaled = predictor.predict(data_dict['X_test'])

    # Inverse transform to original scale
    y_test_original = preprocessor.inverse_transform_predictions(
        ticker, data_dict['y_test'], data_dict['target_col_idx']
    )
    y_pred_original = preprocessor.inverse_transform_predictions(
        ticker, y_pred_scaled, data_dict['target_col_idx']
    )

    # Evaluate
    metrics = predictor.evaluate(y_test_original, y_pred_original, ticker)

    # Plot results
    os.makedirs('results', exist_ok=True)
    predictor.plot_predictions(
        y_test_original, y_pred_original,
        data_dict['test_dates'], ticker,
        save_path=f'results/{ticker}_predictions.png'
    )
    predictor.plot_training_history(
        save_path=f'results/{ticker}_training_history.png'
    )

    # Save model
    os.makedirs('saved_models', exist_ok=True)
    predictor.save_model(f'saved_models/{ticker}_model.keras')

    return {
        'ticker': ticker,
        'metrics': metrics,
        'model': predictor,
        'data': data_dict
    }


def train_all_stocks(model_type: str = 'attention', lookback_days: int = 60,
                     epochs: int = 100) -> List[Dict]:
    """
    Train models for all stocks

    Args:
        model_type: Type of model
        lookback_days: Number of days to look back
        epochs: Maximum training epochs

    Returns:
        List of result dictionaries
    """
    tickers = [
        'INFY.NS', 'ITC.NS', 'BHARTIARTL.NS', 'TCS.NS', 'HINDUNILVR.NS',
        'LICI.NS', 'SBIN.NS', 'RELIANCE.NS', 'ICICIBANK.NS', 'HDFCBANK.NS'
    ]

    all_results = []

    for ticker in tickers:
        try:
            result = train_single_stock(ticker, model_type, lookback_days, epochs)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary metrics
    summary_df = pd.DataFrame([r['metrics'] for r in all_results])
    summary_df.to_csv('results/training_summary.csv', index=False)

    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    logger.info(summary_df.to_string(index=False))

    # Calculate average metrics
    logger.info("\n" + "="*60)
    logger.info("Average Metrics Across All Stocks")
    logger.info("="*60)
    for col in ['rmse', 'mae', 'mape', 'r2_score', 'directional_accuracy']:
        logger.info(f"{col}: {summary_df[col].mean():.4f}")

    return all_results


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Train enhanced stock prediction model')
    parser.add_argument('--ticker', type=str, default='all',
                        help='Stock ticker (or "all" for all stocks)')
    parser.add_argument('--model', choices=['simple', 'attention', 'bidirectional'],
                        default='attention', help='Model type')
    parser.add_argument('--lookback', type=int, default=60,
                        help='Lookback days')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum epochs')

    args = parser.parse_args()

    if args.ticker.lower() == 'all':
        train_all_stocks(args.model, args.lookback, args.epochs)
    else:
        train_single_stock(args.ticker, args.model, args.lookback, args.epochs)


if __name__ == "__main__":
    main()

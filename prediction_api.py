"""
Prediction API for Stock Market Forecasting
Provides real-time predictions using trained models
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from tensorflow.keras.models import load_model

from data_preprocessing import StockDataPreprocessor
from enhanced_model import AttentionLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockPredictor:
    """
    Real-time stock price predictor using trained models
    """

    def __init__(self, models_dir: str = 'saved_models',
                 stock_data_dir: str = 'stock_data',
                 news_db_path: str = 'news_database.db',
                 lookback_days: int = 60):
        """
        Initialize predictor

        Args:
            models_dir: Directory containing saved models
            stock_data_dir: Directory with stock CSV files
            news_db_path: Path to news database
            lookback_days: Number of days to look back
        """
        self.models_dir = models_dir
        self.stock_data_dir = stock_data_dir
        self.news_db_path = news_db_path
        self.lookback_days = lookback_days

        self.models = {}
        self.preprocessor = StockDataPreprocessor(
            stock_data_dir=stock_data_dir,
            news_db_path=news_db_path,
            lookback_days=lookback_days
        )

    def load_model_for_ticker(self, ticker: str) -> bool:
        """
        Load trained model for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if successful, False otherwise
        """
        model_path = os.path.join(self.models_dir, f'{ticker}_model.keras')

        if not os.path.exists(model_path):
            logger.error(f"Model not found for {ticker}: {model_path}")
            return False

        try:
            model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
            self.models[ticker] = model
            logger.info(f"Loaded model for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Error loading model for {ticker}: {e}")
            return False

    def predict_next_day(self, ticker: str) -> Optional[Dict]:
        """
        Predict next day's closing price

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with prediction results
        """
        # Load model if not already loaded
        if ticker not in self.models:
            if not self.load_model_for_ticker(ticker):
                return None

        try:
            # Get latest data
            df = self.preprocessor.merge_all_data(ticker)
            features_df, feature_cols = self.preprocessor.prepare_features(df)

            # Get last lookback_days of data
            if len(features_df) < self.lookback_days:
                logger.error(f"Insufficient data for {ticker}: need {self.lookback_days}, have {len(features_df)}")
                return None

            # Prepare data for prediction
            dates = features_df['Date'].values
            last_date = dates[-1]

            features_df = features_df.drop('Date', axis=1)
            data = features_df.values.astype(np.float32)

            # Get last sequence
            last_sequence = data[-self.lookback_days:]

            # Scale data
            if ticker not in self.preprocessor.scalers:
                # If scaler not available, fit on all data
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(data)
                self.preprocessor.scalers[ticker] = scaler

            scaler = self.preprocessor.scalers[ticker]
            last_sequence_scaled = scaler.transform(last_sequence)

            # Reshape for model input
            X = last_sequence_scaled.reshape(1, self.lookback_days, len(feature_cols))

            # Make prediction
            model = self.models[ticker]
            prediction_scaled = model.predict(X, verbose=0)[0][0]

            # Inverse transform
            target_col_idx = feature_cols.index('Close')
            dummy = np.zeros((1, len(feature_cols)))
            dummy[0, target_col_idx] = prediction_scaled
            prediction_original = scaler.inverse_transform(dummy)[0, target_col_idx]

            # Get current price
            current_price = data[-1, target_col_idx]

            # Calculate change
            price_change = prediction_original - current_price
            price_change_pct = (price_change / current_price) * 100

            # Prediction date (next trading day)
            prediction_date = last_date + timedelta(days=1)

            result = {
                'ticker': ticker,
                'current_date': str(last_date)[:10],
                'prediction_date': str(prediction_date)[:10],
                'current_price': float(current_price),
                'predicted_price': float(prediction_original),
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'direction': 'UP' if price_change > 0 else 'DOWN'
            }

            return result

        except Exception as e:
            logger.error(f"Error predicting for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_multiple_days(self, ticker: str, days: int = 7) -> Optional[List[Dict]]:
        """
        Predict multiple days ahead (iterative prediction)

        Args:
            ticker: Stock ticker symbol
            days: Number of days to predict

        Returns:
            List of prediction dictionaries

        Note: Accuracy decreases with each iterative prediction
        """
        if ticker not in self.models:
            if not self.load_model_for_ticker(ticker):
                return None

        try:
            # Get latest data
            df = self.preprocessor.merge_all_data(ticker)
            features_df, feature_cols = self.preprocessor.prepare_features(df)

            dates = features_df['Date'].values
            last_date = dates[-1]

            features_df = features_df.drop('Date', axis=1)
            data = features_df.values.astype(np.float32)

            # Initialize with actual data
            extended_data = data.copy()

            scaler = self.preprocessor.scalers.get(ticker)
            if scaler is None:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(data)

            target_col_idx = feature_cols.index('Close')
            model = self.models[ticker]

            predictions = []

            for day in range(days):
                # Get last sequence
                last_sequence = extended_data[-self.lookback_days:]
                last_sequence_scaled = scaler.transform(last_sequence)

                # Reshape for model
                X = last_sequence_scaled.reshape(1, self.lookback_days, len(feature_cols))

                # Predict
                prediction_scaled = model.predict(X, verbose=0)[0][0]

                # Inverse transform
                dummy = np.zeros((1, len(feature_cols)))
                dummy[0, target_col_idx] = prediction_scaled
                prediction_original = scaler.inverse_transform(dummy)[0, target_col_idx]

                # Create new row (copy last row and update Close price)
                new_row = extended_data[-1].copy()
                new_row[target_col_idx] = prediction_original

                # Append to extended data
                extended_data = np.vstack([extended_data, new_row])

                # Save prediction
                pred_date = last_date + timedelta(days=day+1)
                current_price = extended_data[-2, target_col_idx]
                price_change = prediction_original - current_price
                price_change_pct = (price_change / current_price) * 100

                predictions.append({
                    'ticker': ticker,
                    'prediction_date': str(pred_date)[:10],
                    'predicted_price': float(prediction_original),
                    'price_change': float(price_change),
                    'price_change_pct': float(price_change_pct),
                    'direction': 'UP' if price_change > 0 else 'DOWN',
                    'confidence': 'low' if day > 3 else 'medium' if day > 1 else 'high'
                })

            return predictions

        except Exception as e:
            logger.error(f"Error predicting multiple days for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def batch_predict(self, tickers: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Predict for multiple tickers

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dictionary mapping ticker to prediction result
        """
        results = {}

        for ticker in tickers:
            logger.info(f"Predicting for {ticker}...")
            result = self.predict_next_day(ticker)
            results[ticker] = result

        return results

    def get_summary(self, predictions: Dict[str, Optional[Dict]]) -> pd.DataFrame:
        """
        Get summary dataframe of predictions

        Args:
            predictions: Dictionary of predictions

        Returns:
            DataFrame with summary
        """
        summary_data = []

        for ticker, pred in predictions.items():
            if pred:
                summary_data.append({
                    'Ticker': ticker,
                    'Current Price (₹)': f"{pred['current_price']:.2f}",
                    'Predicted Price (₹)': f"{pred['predicted_price']:.2f}",
                    'Change (₹)': f"{pred['price_change']:.2f}",
                    'Change (%)': f"{pred['price_change_pct']:.2f}%",
                    'Direction': pred['direction']
                })
            else:
                summary_data.append({
                    'Ticker': ticker,
                    'Current Price (₹)': 'N/A',
                    'Predicted Price (₹)': 'N/A',
                    'Change (₹)': 'N/A',
                    'Change (%)': 'N/A',
                    'Direction': 'N/A'
                })

        return pd.DataFrame(summary_data)


def main():
    """Demo usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Stock Price Prediction API')
    parser.add_argument('--ticker', type=str, help='Stock ticker to predict')
    parser.add_argument('--all', action='store_true', help='Predict all stocks')
    parser.add_argument('--days', type=int, default=1, help='Number of days to predict')

    args = parser.parse_args()

    predictor = StockPredictor()

    if args.all:
        tickers = [
            'INFY.NS', 'ITC.NS', 'BHARTIARTL.NS', 'TCS.NS', 'HINDUNILVR.NS',
            'LICI.NS', 'SBIN.NS', 'RELIANCE.NS', 'ICICIBANK.NS', 'HDFCBANK.NS'
        ]

        logger.info("Predicting for all stocks...")
        predictions = predictor.batch_predict(tickers)

        # Print summary
        summary_df = predictor.get_summary(predictions)
        print("\n" + "="*80)
        print("STOCK PRICE PREDICTIONS")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)

        # Save to CSV
        summary_df.to_csv('predictions_summary.csv', index=False)
        logger.info("Summary saved to predictions_summary.csv")

    elif args.ticker:
        if args.days == 1:
            # Single day prediction
            result = predictor.predict_next_day(args.ticker)

            if result:
                print("\n" + "="*60)
                print(f"Prediction for {args.ticker}")
                print("="*60)
                print(f"Current Date: {result['current_date']}")
                print(f"Prediction Date: {result['prediction_date']}")
                print(f"Current Price: ₹{result['current_price']:.2f}")
                print(f"Predicted Price: ₹{result['predicted_price']:.2f}")
                print(f"Change: ₹{result['price_change']:.2f} ({result['price_change_pct']:.2f}%)")
                print(f"Direction: {result['direction']}")
                print("="*60)
        else:
            # Multiple days prediction
            results = predictor.predict_multiple_days(args.ticker, days=args.days)

            if results:
                print("\n" + "="*60)
                print(f"{args.days}-Day Predictions for {args.ticker}")
                print("="*60)

                for i, result in enumerate(results, 1):
                    print(f"\nDay {i} - {result['prediction_date']}")
                    print(f"  Predicted Price: ₹{result['predicted_price']:.2f}")
                    print(f"  Change: ₹{result['price_change']:.2f} ({result['price_change_pct']:.2f}%)")
                    print(f"  Direction: {result['direction']}")
                    print(f"  Confidence: {result['confidence']}")

                print("="*60)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

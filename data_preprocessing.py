"""
Data Preprocessing Pipeline for Stock Market Prediction
Combines stock prices, technical indicators, and news sentiment
Prepares data for LSTM model training
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import sqlite3
from sklearn.preprocessing import MinMaxScaler
import logging
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for stock data"""

    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index
        Measures momentum, ranges from 0-100
        >70 = overbought, <30 = oversold
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence
        Returns: macd_line, signal_line, histogram
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        Returns: middle_band, upper_band, lower_band
        """
        middle_band = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()

        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        return middle_band, upper_band, lower_band

    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        Returns: %K, %D
        """
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()

        return k_percent, d_percent

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range (volatility indicator)
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()

        return atr

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume
        Cumulative volume indicator
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def calculate_momentum(data: pd.Series, window: int = 10) -> pd.Series:
        """Price momentum"""
        return data.diff(window)

    @staticmethod
    def calculate_roc(data: pd.Series, window: int = 10) -> pd.Series:
        """
        Rate of Change
        Percentage change over n periods
        """
        return ((data - data.shift(window)) / data.shift(window)) * 100


class StockDataPreprocessor:
    """
    Main preprocessing class that combines everything:
    - Stock OHLCV data
    - Technical indicators
    - News sentiment
    - Creates LSTM-ready sequences
    """

    def __init__(self, stock_data_dir: str = "stock_data",
                 news_db_path: str = "news_database.db",
                 lookback_days: int = 60):
        """
        Initialize preprocessor

        Args:
            stock_data_dir: Directory containing stock CSV files
            news_db_path: Path to news database
            lookback_days: Number of days to look back for LSTM sequences
        """
        self.stock_data_dir = stock_data_dir
        self.news_db_path = news_db_path
        self.lookback_days = lookback_days
        self.scalers = {}  # Store scalers for each ticker
        self.feature_columns = []

    def load_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Load stock data from CSV

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with stock data
        """
        # Find the file matching the ticker
        for filename in os.listdir(self.stock_data_dir):
            if ticker in filename and filename.endswith('.csv'):
                filepath = os.path.join(self.stock_data_dir, filename)
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                return df

        raise FileNotFoundError(f"Stock data file not found for ticker: {ticker}")

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to stock dataframe

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        ti = TechnicalIndicators()

        # Moving Averages
        df['SMA_5'] = ti.calculate_sma(df['Close'], 5)
        df['SMA_10'] = ti.calculate_sma(df['Close'], 10)
        df['SMA_20'] = ti.calculate_sma(df['Close'], 20)
        df['SMA_50'] = ti.calculate_sma(df['Close'], 50)
        df['SMA_200'] = ti.calculate_sma(df['Close'], 200)

        df['EMA_12'] = ti.calculate_ema(df['Close'], 12)
        df['EMA_26'] = ti.calculate_ema(df['Close'], 26)

        # RSI
        df['RSI_14'] = ti.calculate_rsi(df['Close'], 14)

        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = ti.calculate_macd(df['Close'])

        # Bollinger Bands
        df['BB_middle'], df['BB_upper'], df['BB_lower'] = ti.calculate_bollinger_bands(df['Close'])
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = ti.calculate_stochastic(df['High'], df['Low'], df['Close'])

        # ATR (volatility)
        df['ATR_14'] = ti.calculate_atr(df['High'], df['Low'], df['Close'])

        # OBV
        df['OBV'] = ti.calculate_obv(df['Close'], df['Volume'])

        # Momentum indicators
        df['Momentum_10'] = ti.calculate_momentum(df['Close'], 10)
        df['ROC_10'] = ti.calculate_roc(df['Close'], 10)

        # Price-based features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']

        # Volume features
        df['Volume_SMA_20'] = ti.calculate_sma(df['Volume'], 20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

        # Volatility
        df['Volatility_20'] = df['Close'].rolling(window=20).std()
        df['Volatility_50'] = df['Close'].rolling(window=50).std()

        logger.info(f"Added {len([col for col in df.columns if col not in ['Date', 'Company', 'Ticker', 'Dividends', 'Stock Splits', 'Adj Close']])} technical indicators")

        return df

    def load_sentiment_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load sentiment data from database

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with sentiment data or None if not available
        """
        if not os.path.exists(self.news_db_path):
            logger.warning(f"News database not found: {self.news_db_path}")
            return None

        try:
            conn = sqlite3.connect(self.news_db_path)

            query = """
                SELECT
                    published_date as Date,
                    AVG(sentiment_score) as sentiment_mean,
                    COUNT(*) as article_count,
                    SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                    SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count
                FROM news_articles
                WHERE company_ticker = ? AND sentiment_score IS NOT NULL
                GROUP BY published_date
                ORDER BY published_date
            """

            df = pd.read_sql_query(query, conn, params=(ticker,))
            conn.close()

            if df.empty:
                logger.warning(f"No sentiment data found for {ticker}")
                return None

            df['Date'] = pd.to_datetime(df['Date'])

            # Calculate additional sentiment features
            df['sentiment_momentum'] = df['sentiment_mean'].diff()
            df['sentiment_ma_7'] = df['sentiment_mean'].rolling(7, min_periods=1).mean()
            df['sentiment_ma_30'] = df['sentiment_mean'].rolling(30, min_periods=1).mean()
            df['positive_ratio'] = df['positive_count'] / df['article_count']
            df['negative_ratio'] = df['negative_count'] / df['article_count']
            df['neutral_ratio'] = df['neutral_count'] / df['article_count']

            logger.info(f"Loaded sentiment data for {ticker}: {len(df)} days")
            return df

        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
            return None

    def merge_all_data(self, ticker: str) -> pd.DataFrame:
        """
        Merge stock data, technical indicators, and sentiment

        Args:
            ticker: Stock ticker symbol

        Returns:
            Combined DataFrame
        """
        # Load stock data
        df = self.load_stock_data(ticker)
        logger.info(f"Loaded stock data for {ticker}: {len(df)} rows")

        # Add technical indicators
        df = self.add_technical_indicators(df)

        # Load and merge sentiment data
        sentiment_df = self.load_sentiment_data(ticker)
        if sentiment_df is not None:
            df = df.merge(sentiment_df, on='Date', how='left')

            # Forward fill sentiment for days without news
            sentiment_cols = [col for col in sentiment_df.columns if col != 'Date']
            df[sentiment_cols] = df[sentiment_cols].fillna(method='ffill')

            # Fill remaining NaNs with neutral values
            df['sentiment_mean'].fillna(0, inplace=True)
            df['article_count'].fillna(0, inplace=True)
            df['positive_count'].fillna(0, inplace=True)
            df['neutral_count'].fillna(0, inplace=True)
            df['negative_count'].fillna(0, inplace=True)
            df['positive_ratio'].fillna(0, inplace=True)
            df['negative_ratio'].fillna(0, inplace=True)
            df['neutral_ratio'].fillna(1, inplace=True)  # Default to neutral
            df['sentiment_momentum'].fillna(0, inplace=True)
            df['sentiment_ma_7'].fillna(0, inplace=True)
            df['sentiment_ma_30'].fillna(0, inplace=True)

            logger.info(f"Merged sentiment data for {ticker}")
        else:
            logger.warning(f"No sentiment data available for {ticker}, continuing without it")

        # Remove any remaining NaN values (from technical indicators at the beginning)
        df = df.fillna(method='bfill')
        df = df.fillna(0)

        # Sort by date
        df = df.sort_values('Date')

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select and prepare features for modeling

        Args:
            df: Combined dataframe

        Returns:
            DataFrame with selected features and list of feature names
        """
        # Define feature columns (exclude non-feature columns)
        exclude_cols = ['Date', 'Company', 'Ticker', 'Dividends', 'Stock Splits', 'Adj Close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Keep Date for reference but use other columns as features
        features_df = df[['Date'] + feature_cols].copy()

        logger.info(f"Selected {len(feature_cols)} features for modeling")
        logger.info(f"Features: {', '.join(feature_cols[:10])}...")  # Print first 10

        return features_df, feature_cols

    def create_sequences(self, data: np.ndarray, target: np.ndarray,
                        lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM

        Args:
            data: Feature data (normalized)
            target: Target values (Close price, normalized)
            lookback: Number of timesteps to look back

        Returns:
            X (sequences), y (targets)
        """
        X, y = [], []

        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def preprocess_for_training(self, ticker: str, test_split: float = 0.2) -> Dict:
        """
        Complete preprocessing pipeline for a single ticker

        Args:
            ticker: Stock ticker symbol
            test_split: Proportion of data to use for testing

        Returns:
            Dictionary containing all processed data and metadata
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Preprocessing {ticker}")
        logger.info(f"{'='*60}")

        # Merge all data
        df = self.merge_all_data(ticker)

        # Prepare features
        features_df, feature_cols = self.prepare_features(df)
        self.feature_columns = feature_cols

        # Split date and features
        dates = features_df['Date'].values
        features_df = features_df.drop('Date', axis=1)

        # Identify target column (Close price)
        target_col_idx = feature_cols.index('Close')

        # Convert to numpy
        data = features_df.values.astype(np.float32)

        # Time-based train-test split (no shuffling!)
        split_idx = int(len(data) * (1 - test_split))

        train_data = data[:split_idx]
        test_data = data[split_idx:]

        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]

        # Scale data (fit on train only!)
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Store scaler for inverse transform later
        self.scalers[ticker] = scaler

        # Create sequences
        X_train, y_train = self.create_sequences(
            train_scaled,
            train_scaled[:, target_col_idx],
            self.lookback_days
        )

        X_test, y_test = self.create_sequences(
            test_scaled,
            test_scaled[:, target_col_idx],
            self.lookback_days
        )

        # Adjust dates (remove first lookback_days dates)
        train_dates_adj = train_dates[self.lookback_days:]
        test_dates_adj = test_dates[self.lookback_days:]

        logger.info(f"Training sequences: {X_train.shape}")
        logger.info(f"Testing sequences: {X_test.shape}")
        logger.info(f"Number of features: {X_train.shape[2]}")

        return {
            'ticker': ticker,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_dates': train_dates_adj,
            'test_dates': test_dates_adj,
            'scaler': scaler,
            'feature_columns': feature_cols,
            'target_col_idx': target_col_idx,
            'original_train_data': train_data,
            'original_test_data': test_data
        }

    def inverse_transform_predictions(self, ticker: str, predictions: np.ndarray,
                                     target_col_idx: int) -> np.ndarray:
        """
        Inverse transform predictions back to original scale

        Args:
            ticker: Stock ticker symbol
            predictions: Normalized predictions
            target_col_idx: Index of target column

        Returns:
            Predictions in original scale
        """
        scaler = self.scalers[ticker]

        # Create dummy array with same shape as original features
        dummy = np.zeros((len(predictions), len(self.feature_columns)))
        dummy[:, target_col_idx] = predictions

        # Inverse transform
        inverse_transformed = scaler.inverse_transform(dummy)

        return inverse_transformed[:, target_col_idx]


def main():
    """Test the preprocessing pipeline"""
    preprocessor = StockDataPreprocessor(
        stock_data_dir="stock_data",
        news_db_path="news_database.db",
        lookback_days=60
    )

    # Test with one ticker
    test_ticker = "INFY.NS"

    try:
        data_dict = preprocessor.preprocess_for_training(test_ticker)

        print(f"\nPreprocessing successful for {test_ticker}!")
        print(f"Training samples: {len(data_dict['X_train'])}")
        print(f"Testing samples: {len(data_dict['X_test'])}")
        print(f"Features: {data_dict['X_train'].shape[2]}")
        print(f"\nFeature list:")
        for i, col in enumerate(data_dict['feature_columns'][:20]):
            print(f"  {i+1}. {col}")
        if len(data_dict['feature_columns']) > 20:
            print(f"  ... and {len(data_dict['feature_columns'])-20} more")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

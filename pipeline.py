"""
Unified Stock Prediction Pipeline with xLSTM and Reinforcement Learning
===========================================================================

Complete pipeline that:
1. Collects news data (aligned with stock_data dates)
2. Performs sentiment analysis
3. Fuses stock data with sentiment features
4. Trains xLSTM model with RL component
5. Evaluates performance

Author: Stock Prediction System
Date: 2024
"""

import os
import json
import warnings
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from textblob import TextBlob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'stock_data_dir': 'stock_data',
    'news_cache_file': 'news_cache.json',
    'models_dir': 'trained_models',
    'results_dir': 'results',

    # Model hyperparameters
    'lookback_days': 60,
    'test_split': 0.2,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,

    # xLSTM specific
    'xlstm_units': 128,
    'xlstm_layers': 3,
    'dropout_rate': 0.3,

    # RL parameters
    'rl_weight': 0.3,  # Weight for RL loss vs regression loss
    'direction_reward': 1.0,
    'direction_penalty': -1.0,

    # Stocks to process
    'tickers': [
        ('INFY.NS', 'Infosys_Ltd_INFY.NS.csv'),
        ('TCS.NS', 'Tata_Consultancy_Services_Ltd_TCS.NS.csv'),
        ('HDFCBANK.NS', 'HDFC_Bank_Ltd_HDFCBANK.NS.csv'),
        ('ICICIBANK.NS', 'ICICI_Bank_Ltd_ICICIBANK.NS.csv'),
        ('RELIANCE.NS', 'Reliance_Industries_Ltd_RELIANCE.NS.csv'),
        ('BHARTIARTL.NS', 'Bharti_Airtel_Ltd_BHARTIARTL.NS.csv'),
        ('HINDUNILVR.NS', 'Hindustan_Unilever_Ltd_HINDUNILVR.NS.csv'),
        ('ITC.NS', 'ITC_Ltd_ITC.NS.csv'),
        ('SBIN.NS', 'State_Bank_of_India_SBIN.NS.csv'),
        ('LICI.NS', 'Life_Insurance_Corporation_of_India_LICI.NS.csv'),
    ]
}

# Create directories
for dir_path in [CONFIG['models_dir'], CONFIG['results_dir']]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Company name mapping (matches GoScraper regex patterns to tickers)
COMPANY_PATTERNS = {
    'INFY.NS': ['Infosys'],
    'TCS.NS': ['Tata Consultancy Services', 'TCS', 'Tata Consultancy'],
    'HDFCBANK.NS': ['HDFC Bank', 'HDFC'],
    'ICICIBANK.NS': ['ICICI Bank', 'ICICI'],
    'RELIANCE.NS': ['Reliance Industries', 'Reliance', 'RIL'],
    'BHARTIARTL.NS': ['Bharti Airtel', 'Airtel'],
    'HINDUNILVR.NS': ['Hindustan Unilever', 'HUL'],
    'ITC.NS': ['ITC'],
    'SBIN.NS': ['State Bank of India', 'SBI'],
    'LICI.NS': ['Life Insurance Corporation', 'LIC'],
}


# ============================================================================
# 1. NEWS COLLECTION MODULE
# ============================================================================

class NewsCollector:
    """Collects news data aligned with stock dates using GoScraper"""

    def __init__(self, cache_file, goscraper_articles_path='GoScraper/articles.json'):
        self.cache_file = cache_file
        self.news_data = {}
        self.goscraper_articles_path = goscraper_articles_path
        self.goscraper_articles = None

    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        if not text or len(text.strip()) == 0:
            return 0.0

        try:
            blob = TextBlob(text)
            # TextBlob polarity ranges from -1 (negative) to 1 (positive)
            return blob.sentiment.polarity
        except:
            return 0.0

    def load_goscraper_articles(self):
        """Load articles from GoScraper's articles.json"""
        if self.goscraper_articles is not None:
            return self.goscraper_articles

        if not os.path.exists(self.goscraper_articles_path):
            print(f"  Warning: GoScraper articles not found at {self.goscraper_articles_path}")
            print(f"  Falling back to synthetic sentiment generation")
            return []

        try:
            with open(self.goscraper_articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            print(f"  Loaded {len(articles)} articles from GoScraper")
            self.goscraper_articles = articles
            return articles
        except Exception as e:
            print(f"  Error loading GoScraper articles: {e}")
            return []

    def match_ticker_in_text(self, text, ticker):
        """Check if article is about the specified ticker's company"""
        if ticker not in COMPANY_PATTERNS:
            return False

        patterns = COMPANY_PATTERNS[ticker]
        for pattern in patterns:
            # Case-insensitive search for company name
            if re.search(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE):
                return True
        return False

    def collect_news_for_date_range(self, ticker, start_date, end_date):
        """
        Collect news for a stock within date range using GoScraper data
        """
        print(f"  Collecting news for {ticker} from {start_date} to {end_date}")

        # Load GoScraper articles
        articles = self.load_goscraper_articles()

        # If no articles available, use synthetic sentiment
        if not articles:
            return self._generate_synthetic_sentiment(ticker, start_date, end_date)

        # Parse date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Filter and process articles
        daily_news = {}
        matched_articles = 0

        for article in articles:
            # Parse article date
            try:
                article_date = pd.to_datetime(article['scraped_at']).date()
            except:
                continue

            # Check if article is in date range
            if not (start_dt.date() <= article_date <= end_dt.date()):
                continue

            # Check if article is about this company
            title = article.get('title', '')
            content = article.get('content', '')

            if not self.match_ticker_in_text(title + ' ' + content, ticker):
                continue

            matched_articles += 1

            # Analyze sentiment
            sentiment_score = self.analyze_sentiment(content if content else title)

            # Group by date
            date_str = article_date.strftime('%Y-%m-%d')
            if date_str not in daily_news:
                daily_news[date_str] = {
                    'sentiments': [],
                    'count': 0
                }

            daily_news[date_str]['sentiments'].append(sentiment_score)
            daily_news[date_str]['count'] += 1

        print(f"  Found {matched_articles} relevant articles")

        # Convert to news entries format
        news_entries = []
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')

            if date_str in daily_news:
                sentiments = daily_news[date_str]['sentiments']
                avg_sentiment = np.mean(sentiments)
                article_count = daily_news[date_str]['count']

                # Calculate positive/negative ratios
                positive_count = sum(1 for s in sentiments if s > 0)
                negative_count = sum(1 for s in sentiments if s < 0)
                total = len(sentiments)

                positive_ratio = positive_count / total if total > 0 else 0
                negative_ratio = negative_count / total if total > 0 else 0
            else:
                # No articles for this date - use neutral sentiment
                avg_sentiment = 0.0
                article_count = 0
                positive_ratio = 0.0
                negative_ratio = 0.0

            news_entries.append({
                'date': date_str,
                'ticker': ticker,
                'sentiment_score': float(avg_sentiment),
                'article_count': int(article_count),
                'positive_ratio': float(positive_ratio),
                'negative_ratio': float(negative_ratio),
            })

        return news_entries

    def _generate_synthetic_sentiment(self, ticker, start_date, end_date):
        """Fallback: Generate synthetic sentiment (for backward compatibility)"""
        print(f"  Using synthetic sentiment for {ticker}")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        news_entries = []
        for date in dates:
            np.random.seed(int(date.timestamp()))
            sentiment_score = np.random.uniform(-1, 1)

            news_entries.append({
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'sentiment_score': float(sentiment_score),
                'article_count': int(np.random.randint(1, 10)),
                'positive_ratio': float(max(0, sentiment_score)),
                'negative_ratio': float(max(0, -sentiment_score)),
            })

        return news_entries

    def load_or_collect(self, ticker, start_date, end_date):
        """Load cached news or collect new"""
        # Check cache
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.news_data = json.load(f)

        cache_key = f"{ticker}_{start_date}_{end_date}"

        if cache_key in self.news_data:
            print(f"  Loading cached news for {ticker}")
            return self.news_data[cache_key]

        # Collect new news
        news_entries = self.collect_news_for_date_range(ticker, start_date, end_date)

        # Cache it
        self.news_data[cache_key] = news_entries
        with open(self.cache_file, 'w') as f:
            json.dump(self.news_data, f, indent=2)

        return news_entries


# ============================================================================
# 2. DATA FUSION MODULE
# ============================================================================

class DataFusionEngine:
    """Fuses stock data with sentiment features"""

    def __init__(self, lookback_days=60):
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_stock_data(self, filepath):
        """Load and preprocess stock CSV"""
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)

        # Select OHLCV features
        features = ['Open', 'High', 'Low', 'Volume', 'Close']
        return df[features]

    def add_technical_indicators(self, df):
        """Add basic technical indicators"""
        # Returns
        df['returns'] = df['Close'].pct_change()

        # Moving averages
        df['ma_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['ma_20'] = df['Close'].rolling(window=20, min_periods=1).mean()

        # Volatility
        df['volatility'] = df['Close'].rolling(window=20, min_periods=1).std()

        # Volume moving average
        df['volume_ma'] = df['Volume'].rolling(window=20, min_periods=1).mean()

        return df

    def merge_sentiment_data(self, stock_df, news_data):
        """Merge sentiment features into stock dataframe"""
        # Convert news to dataframe
        news_df = pd.DataFrame(news_data)
        news_df['date'] = pd.to_datetime(news_df['date'], utc=True)
        news_df.set_index('date', inplace=True)

        # Select sentiment features
        sentiment_cols = ['sentiment_score', 'article_count',
                         'positive_ratio', 'negative_ratio']

        # Merge with stock data (forward fill for missing days)
        merged = stock_df.join(news_df[sentiment_cols], how='left')
        merged[sentiment_cols] = merged[sentiment_cols].fillna(method='ffill').fillna(0)

        return merged

    def prepare_training_data(self, ticker, stock_filepath, news_data, test_split=0.2):
        """Complete data preparation pipeline"""
        print(f"\nPreparing data for {ticker}...")

        # Load stock data
        df = self.load_stock_data(stock_filepath)
        print(f"  Loaded {len(df)} days of stock data")

        # Add technical indicators
        df = self.add_technical_indicators(df)

        # Merge sentiment
        df = self.merge_sentiment_data(df, news_data)

        # Fill any remaining NaN
        df = df.fillna(method='bfill').fillna(0)

        print(f"  Total features: {len(df.columns)}")

        # Split data (time-based)
        split_idx = int(len(df) * (1 - test_split))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx - self.lookback_days:]

        # Scale data (fit on train only)
        train_scaled = self.scaler.fit_transform(train_df.values)
        test_scaled = self.scaler.transform(test_df.values)

        # Create sequences
        X_train, y_train, dir_train = self._create_sequences(train_scaled)
        X_test, y_test, dir_test = self._create_sequences(test_scaled)

        # Store metadata
        metadata = {
            'feature_names': list(df.columns),
            'n_features': len(df.columns),
            'train_dates': df.index[:split_idx].tolist(),
            'test_dates': df.index[split_idx:].tolist(),
            'close_idx': list(df.columns).index('Close')
        }

        return {
            'X_train': X_train,
            'y_train': y_train,
            'dir_train': dir_train,
            'X_test': X_test,
            'y_test': y_test,
            'dir_test': dir_test,
            'metadata': metadata,
            'scaler': self.scaler
        }

    def _create_sequences(self, data):
        """Create sequences for LSTM"""
        X, y, directions = [], [], []

        for i in range(self.lookback_days, len(data)):
            X.append(data[i - self.lookback_days:i])
            y.append(data[i, 4])  # Close price (index 4 in OHLCV)

            # Direction label (1 = up, 0 = down)
            if i > self.lookback_days:
                direction = 1 if data[i, 4] > data[i-1, 4] else 0
                directions.append(direction)
            else:
                directions.append(0)  # Placeholder for first sequence

        return np.array(X), np.array(y), np.array(directions)


# ============================================================================
# 3. xLSTM MODEL WITH REINFORCEMENT LEARNING
# ============================================================================

class xLSTMBlock(layers.Layer):
    """
    Extended LSTM (xLSTM) block with improvements:
    - Exponential gating
    - Enhanced memory capacity
    - Better gradient flow
    """

    def __init__(self, units, return_sequences=False, **kwargs):
        super(xLSTMBlock, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences

        # Standard LSTM
        self.lstm = layers.LSTM(
            units,
            return_sequences=True,
            return_state=True
        )

        # Additional gating for xLSTM
        self.exp_gate = layers.Dense(units, activation='sigmoid')
        self.norm = layers.LayerNormalization()

    def call(self, inputs):
        # LSTM forward
        lstm_out, h_state, c_state = self.lstm(inputs)

        # Exponential gating (xLSTM enhancement)
        gate = self.exp_gate(lstm_out)
        enhanced_out = lstm_out * gate

        # Normalization
        normalized = self.norm(enhanced_out)

        if not self.return_sequences:
            # Return last timestep
            return normalized[:, -1, :]

        return normalized


class ReinforcementLearningLoss(keras.losses.Loss):
    """
    Custom loss combining regression and RL directional rewards
    """

    def __init__(self, rl_weight=0.3, reward=1.0, penalty=-1.0):
        super().__init__()
        self.rl_weight = rl_weight
        self.reward = reward
        self.penalty = penalty
        self.regression_loss = keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        # y_true shape: (batch, 2) where [:, 0] = price, [:, 1] = direction
        true_price = y_true[:, 0]
        true_direction = y_true[:, 1]

        # y_pred shape: (batch, 2) where [:, 0] = price, [:, 1] = direction_prob
        pred_price = y_pred[:, 0]
        pred_direction_prob = y_pred[:, 1]

        # Regression loss (MSE on prices)
        reg_loss = self.regression_loss(true_price, pred_price)

        # RL directional loss
        # Predicted direction: 1 if prob > 0.5, else 0
        pred_direction = tf.cast(pred_direction_prob > 0.5, tf.float32)

        # Reward for correct direction, penalty for wrong
        rl_rewards = tf.where(
            tf.equal(pred_direction, true_direction),
            self.reward,  # Correct prediction
            self.penalty  # Wrong prediction
        )

        # RL loss (negative reward to minimize)
        rl_loss = -tf.reduce_mean(rl_rewards)

        # Combined loss
        total_loss = (1 - self.rl_weight) * reg_loss + self.rl_weight * rl_loss

        return total_loss


class xLSTMStockPredictor:
    """xLSTM model with RL for stock prediction"""

    def __init__(self, n_features, config):
        self.n_features = n_features
        self.config = config
        self.model = None
        self.history = None

    def build_model(self):
        """Build xLSTM architecture"""
        inputs = layers.Input(shape=(self.config['lookback_days'], self.n_features))

        # Stack xLSTM blocks
        x = inputs
        for i in range(self.config['xlstm_layers']):
            return_seq = (i < self.config['xlstm_layers'] - 1)
            x = xLSTMBlock(
                self.config['xlstm_units'],
                return_sequences=return_seq,
                name=f'xlstm_{i}'
            )(x)
            x = layers.Dropout(self.config['dropout_rate'])(x)

        # Separate heads for price and direction
        price_head = layers.Dense(64, activation='relu')(x)
        price_head = layers.Dropout(0.2)(price_head)
        price_output = layers.Dense(1, name='price_output')(price_head)

        direction_head = layers.Dense(32, activation='relu')(x)
        direction_head = layers.Dropout(0.2)(direction_head)
        direction_output = layers.Dense(1, activation='sigmoid', name='direction_output')(direction_head)

        # Combine outputs
        combined_output = layers.Concatenate()([price_output, direction_output])

        # Build model
        self.model = Model(inputs=inputs, outputs=combined_output)

        # Custom RL loss
        rl_loss = ReinforcementLearningLoss(
            rl_weight=self.config['rl_weight'],
            reward=self.config['direction_reward'],
            penalty=self.config['direction_penalty']
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=rl_loss,
            metrics=['mae']
        )

        return self.model

    def train(self, X_train, y_train, dir_train, X_val, y_val, dir_val):
        """Train the model"""
        # Combine price and direction labels
        y_train_combined = np.column_stack([y_train, dir_train])
        y_val_combined = np.column_stack([y_val, dir_val])

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            'best_xlstm_model.h5',
            monitor='val_loss',
            save_best_only=True
        )

        # Train
        print("\n  Training xLSTM with RL...")
        self.history = self.model.fit(
            X_train, y_train_combined,
            validation_data=(X_val, y_val_combined),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        print("  Training completed!")

    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X, verbose=0)
        price_pred = predictions[:, 0]
        direction_pred = (predictions[:, 1] > 0.5).astype(int)
        return price_pred, direction_pred


# ============================================================================
# 4. EVALUATION MODULE
# ============================================================================

class PerformanceEvaluator:
    """Evaluates model performance"""

    def __init__(self, scaler, close_idx):
        self.scaler = scaler
        self.close_idx = close_idx

    def inverse_transform_prices(self, prices):
        """Convert normalized prices back to original scale"""
        dummy = np.zeros((len(prices), self.scaler.n_features_in_))
        dummy[:, self.close_idx] = prices
        return self.scaler.inverse_transform(dummy)[:, self.close_idx]

    def evaluate(self, y_true, y_pred, dir_true, dir_pred, ticker):
        """Calculate all metrics"""
        # Inverse transform prices
        y_true_rescaled = self.inverse_transform_prices(y_true)
        y_pred_rescaled = self.inverse_transform_prices(y_pred)

        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
        mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
        mape = np.mean(np.abs((y_true_rescaled - y_pred_rescaled) / y_true_rescaled)) * 100
        r2 = r2_score(y_true_rescaled, y_pred_rescaled)

        # Directional accuracy (from RL component)
        dir_accuracy = np.mean(dir_pred == dir_true) * 100

        # Price movement direction (traditional)
        actual_direction = np.diff(y_true_rescaled) > 0
        pred_direction = np.diff(y_pred_rescaled) > 0
        movement_accuracy = np.mean(actual_direction == pred_direction) * 100

        metrics = {
            'ticker': ticker,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2_score': r2,
            'directional_accuracy': dir_accuracy,
            'movement_accuracy': movement_accuracy
        }

        # Print results
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS: {ticker}")
        print(f"{'='*70}")
        print(f"RMSE:                      {rmse:.4f}")
        print(f"MAE:                       {mae:.4f}")
        print(f"MAPE:                      {mape:.2f}%")
        print(f"R² Score:                  {r2:.4f}")
        print(f"RL Directional Accuracy:   {dir_accuracy:.2f}%  <-- (RL Component)")
        print(f"Movement Accuracy:         {movement_accuracy:.2f}%  <-- (Price Pattern)")
        print(f"{'='*70}\n")

        return metrics, y_true_rescaled, y_pred_rescaled

    def plot_results(self, y_true, y_pred, dates, ticker, metrics, save_path):
        """Plot predictions"""
        plt.figure(figsize=(15, 6))

        plt.plot(dates, y_true, label='Actual Price', color='blue', linewidth=2)
        plt.plot(dates, y_pred, label='Predicted Price', color='red',
                linewidth=2, linestyle='--', alpha=0.8)

        plt.title(f'{ticker} - xLSTM with RL Predictions\n'
                 f'R²: {metrics["r2_score"]:.4f} | '
                 f'RL Dir Acc: {metrics["directional_accuracy"]:.2f}% | '
                 f'MAPE: {metrics["mape"]:.2f}%',
                 fontsize=14, fontweight='bold')

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price (₹)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Plot saved to {save_path}")


# ============================================================================
# 5. MAIN PIPELINE
# ============================================================================

def run_pipeline():
    """Execute complete pipeline"""
    print("="*80)
    print(" UNIFIED xLSTM + RL STOCK PREDICTION PIPELINE")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize modules
    news_collector = NewsCollector(CONFIG['news_cache_file'])
    fusion_engine = DataFusionEngine(CONFIG['lookback_days'])

    all_results = []

    # Process each stock
    for ticker, filename in CONFIG['tickers']:
        print(f"\n{'#'*80}")
        print(f"# Processing: {ticker}")
        print(f"{'#'*80}")

        try:
            # Step 1: Load stock data to get date range
            stock_filepath = os.path.join(CONFIG['stock_data_dir'], filename)
            if not os.path.exists(stock_filepath):
                print(f"  WARNING: {filename} not found. Skipping.")
                continue

            df_temp = pd.read_csv(stock_filepath)
            df_temp['Date'] = pd.to_datetime(df_temp['Date'], utc=True)
            start_date = df_temp['Date'].min().strftime('%Y-%m-%d')
            end_date = df_temp['Date'].max().strftime('%Y-%m-%d')

            # Step 2: Collect news data
            print(f"\n[1/5] Collecting news data...")
            news_data = news_collector.load_or_collect(ticker, start_date, end_date)
            print(f"  Collected {len(news_data)} news entries")

            # Step 3: Fuse data
            print(f"\n[2/5] Fusing stock data with sentiment features...")
            data_bundle = fusion_engine.prepare_training_data(
                ticker,
                stock_filepath,
                news_data,
                CONFIG['test_split']
            )

            # Step 4: Build and train model
            print(f"\n[3/5] Building xLSTM model...")
            predictor = xLSTMStockPredictor(data_bundle['metadata']['n_features'], CONFIG)
            predictor.build_model()

            print(f"\n[4/5] Training model with RL...")
            predictor.train(
                data_bundle['X_train'],
                data_bundle['y_train'],
                data_bundle['dir_train'],
                data_bundle['X_test'],
                data_bundle['y_test'],
                data_bundle['dir_test']
            )

            # Step 5: Evaluate
            print(f"\n[5/5] Evaluating performance...")
            price_pred, dir_pred = predictor.predict(data_bundle['X_test'])

            evaluator = PerformanceEvaluator(
                data_bundle['scaler'],
                data_bundle['metadata']['close_idx']
            )

            metrics, y_true_real, y_pred_real = evaluator.evaluate(
                data_bundle['y_test'],
                price_pred,
                data_bundle['dir_test'],
                dir_pred,
                ticker
            )

            # Plot results
            test_dates = data_bundle['metadata']['test_dates'][CONFIG['lookback_days']:]
            plot_path = os.path.join(CONFIG['results_dir'], f'{ticker}_predictions.png')
            evaluator.plot_results(y_true_real, y_pred_real, test_dates,
                                  ticker, metrics, plot_path)

            # Save model
            model_path = os.path.join(CONFIG['models_dir'], f'{ticker}_xlstm_rl.h5')
            predictor.model.save(model_path)
            print(f"  Model saved to {model_path}")

            all_results.append(metrics)

        except Exception as e:
            print(f"\n  ERROR processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*80}")
    print(" PIPELINE COMPLETE - SUMMARY")
    print(f"{'='*80}\n")

    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))

        # Save results
        results_df.to_csv(os.path.join(CONFIG['results_dir'], 'summary.csv'), index=False)

        print(f"\n\nAverage Metrics:")
        print(f"  RMSE:                    {results_df['rmse'].mean():.4f}")
        print(f"  MAE:                     {results_df['mae'].mean():.4f}")
        print(f"  MAPE:                    {results_df['mape'].mean():.2f}%")
        print(f"  R² Score:                {results_df['r2_score'].mean():.4f}")
        print(f"  RL Directional Accuracy: {results_df['directional_accuracy'].mean():.2f}%")
        print(f"  Movement Accuracy:       {results_df['movement_accuracy'].mean():.2f}%")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    run_pipeline()

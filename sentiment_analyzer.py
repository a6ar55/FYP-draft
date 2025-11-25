"""
Sentiment Analysis Module for Financial News
Uses FinBERT (BERT fine-tuned for financial sentiment analysis)
Processes news articles and assigns sentiment scores
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinancialSentimentAnalyzer:
    """
    Sentiment analyzer using FinBERT for financial text
    FinBERT is trained on financial news and achieves better results than generic sentiment models
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", db_path: str = "news_database.db"):
        """
        Initialize sentiment analyzer

        Args:
            model_name: HuggingFace model name (default: FinBERT)
            db_path: Path to SQLite database with news articles
        """
        self.db_path = db_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load FinBERT model and tokenizer
        logger.info(f"Loading {model_name} model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # FinBERT outputs: negative, neutral, positive
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def analyze_text(self, text: str, max_length: int = 512) -> Tuple[float, str]:
        """
        Analyze sentiment of a single text

        Args:
            text: Input text to analyze
            max_length: Maximum token length (BERT limit: 512)

        Returns:
            Tuple of (sentiment_score, sentiment_label)
            sentiment_score: -1 (negative) to +1 (positive)
            sentiment_label: 'negative', 'neutral', or 'positive'
        """
        if not text or len(text.strip()) == 0:
            return 0.0, 'neutral'

        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Get predicted class
            predicted_class = np.argmax(probabilities)
            sentiment_label = self.label_mapping[predicted_class]

            # Calculate sentiment score: weighted by probabilities
            # negative (-1), neutral (0), positive (+1)
            sentiment_score = (
                probabilities[0] * -1.0 +  # negative
                probabilities[1] * 0.0 +    # neutral
                probabilities[2] * 1.0      # positive
            )

            return float(sentiment_score), sentiment_label

        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return 0.0, 'neutral'

    def analyze_database_articles(self, batch_size: int = 16, update_existing: bool = False):
        """
        Analyze all articles in the database and update sentiment scores

        Args:
            batch_size: Number of articles to process in each batch
            update_existing: If True, re-analyze articles that already have sentiment
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get articles without sentiment (or all if update_existing=True)
        if update_existing:
            query = "SELECT id, title, content FROM news_articles"
        else:
            query = "SELECT id, title, content FROM news_articles WHERE sentiment_score IS NULL"

        cursor.execute(query)
        articles = cursor.fetchall()

        if not articles:
            logger.info("No articles to analyze")
            conn.close()
            return

        logger.info(f"Analyzing sentiment for {len(articles)} articles...")

        # Process in batches
        for i in tqdm(range(0, len(articles), batch_size), desc="Processing batches"):
            batch = articles[i:i + batch_size]

            for article_id, title, content in batch:
                # Combine title and content for analysis
                text = f"{title}. {content}" if content else title

                # Analyze sentiment
                sentiment_score, sentiment_label = self.analyze_text(text)

                # Update database
                cursor.execute('''
                    UPDATE news_articles
                    SET sentiment_score = ?, sentiment_label = ?
                    WHERE id = ?
                ''', (sentiment_score, sentiment_label, article_id))

            conn.commit()

        logger.info("Sentiment analysis completed successfully")
        self.print_sentiment_statistics(conn)
        conn.close()

    def print_sentiment_statistics(self, conn: sqlite3.Connection):
        """Print statistics about sentiment distribution"""
        cursor = conn.cursor()

        # Overall sentiment distribution
        cursor.execute("""
            SELECT sentiment_label, COUNT(*) as count
            FROM news_articles
            WHERE sentiment_label IS NOT NULL
            GROUP BY sentiment_label
        """)

        logger.info("\nOverall sentiment distribution:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]}")

        # Average sentiment by company
        cursor.execute("""
            SELECT company_ticker, AVG(sentiment_score) as avg_sentiment, COUNT(*) as count
            FROM news_articles
            WHERE sentiment_score IS NOT NULL
            GROUP BY company_ticker
            ORDER BY avg_sentiment DESC
        """)

        logger.info("\nAverage sentiment by company:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]:.3f} ({row[2]} articles)")

    def aggregate_daily_sentiment(self, output_dir: str = "processed_news_data"):
        """
        Aggregate sentiment scores by company and date for merging with stock data

        Args:
            output_dir: Directory to save aggregated sentiment data
        """
        os.makedirs(output_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)

        # Aggregate sentiment by company and date
        query = """
            SELECT
                company_ticker,
                published_date,
                AVG(sentiment_score) as avg_sentiment,
                COUNT(*) as article_count,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                STDEV(sentiment_score) as sentiment_volatility
            FROM news_articles
            WHERE sentiment_score IS NOT NULL
            GROUP BY company_ticker, published_date
            ORDER BY company_ticker, published_date
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Note: SQLite doesn't have STDEV, so we'll calculate it in pandas
        full_query = """
            SELECT company_ticker, published_date, sentiment_score
            FROM news_articles
            WHERE sentiment_score IS NOT NULL
            ORDER BY company_ticker, published_date
        """

        conn = sqlite3.connect(self.db_path)
        full_df = pd.read_sql_query(full_query, conn)
        conn.close()

        # Calculate sentiment volatility (standard deviation) per day
        sentiment_volatility = full_df.groupby(['company_ticker', 'published_date'])['sentiment_score'].std().reset_index()
        sentiment_volatility.columns = ['company_ticker', 'published_date', 'sentiment_volatility']
        sentiment_volatility['sentiment_volatility'].fillna(0, inplace=True)

        # Merge with aggregated data
        df = df.merge(sentiment_volatility, on=['company_ticker', 'published_date'], how='left', suffixes=('', '_new'))
        if 'sentiment_volatility_new' in df.columns:
            df['sentiment_volatility'] = df['sentiment_volatility_new']
            df.drop('sentiment_volatility_new', axis=1, inplace=True)

        # Save aggregated sentiment data
        output_path = os.path.join(output_dir, "daily_sentiment_aggregated.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Aggregated sentiment data saved to {output_path}")

        # Also save per-company files
        for ticker in df['company_ticker'].unique():
            ticker_df = df[df['company_ticker'] == ticker]
            ticker_path = os.path.join(output_dir, f"sentiment_{ticker}.csv")
            ticker_df.to_csv(ticker_path, index=False)
            logger.info(f"Saved {len(ticker_df)} days of sentiment data for {ticker}")

        return df

    def create_sentiment_features(self, lookback_days: int = 7) -> pd.DataFrame:
        """
        Create advanced sentiment features including rolling averages and trends

        Args:
            lookback_days: Number of days to look back for rolling features

        Returns:
            DataFrame with sentiment features
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT company_ticker, published_date, sentiment_score
            FROM news_articles
            WHERE sentiment_score IS NOT NULL
            ORDER BY company_ticker, published_date
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Convert date to datetime
        df['published_date'] = pd.to_datetime(df['published_date'])

        features_list = []

        for ticker in df['company_ticker'].unique():
            ticker_df = df[df['company_ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('published_date')

            # Daily aggregation
            daily = ticker_df.groupby('published_date')['sentiment_score'].agg([
                ('sentiment_mean', 'mean'),
                ('sentiment_std', 'std'),
                ('sentiment_min', 'min'),
                ('sentiment_max', 'max'),
                ('article_count', 'count')
            ]).reset_index()

            daily['sentiment_std'].fillna(0, inplace=True)

            # Rolling features
            for window in [3, 7, 14, 30]:
                daily[f'sentiment_ma_{window}d'] = daily['sentiment_mean'].rolling(window, min_periods=1).mean()
                daily[f'sentiment_vol_{window}d'] = daily['sentiment_mean'].rolling(window, min_periods=1).std()

            # Sentiment momentum (change from previous day)
            daily['sentiment_momentum_1d'] = daily['sentiment_mean'].diff()
            daily['sentiment_momentum_7d'] = daily['sentiment_mean'].diff(7)

            # Add ticker
            daily['company_ticker'] = ticker

            features_list.append(daily)

        # Combine all features
        all_features = pd.concat(features_list, ignore_index=True)

        # Fill NaN values
        all_features.fillna(method='ffill', inplace=True)
        all_features.fillna(0, inplace=True)

        return all_features


class SimpleSentimentAnalyzer:
    """
    Lightweight sentiment analyzer using VADER (no GPU required)
    Use this as a fallback if FinBERT is too slow or resource-intensive
    """

    def __init__(self, db_path: str = "news_database.db"):
        """Initialize VADER sentiment analyzer"""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()
        self.db_path = db_path
        logger.info("VADER sentiment analyzer initialized")

    def analyze_text(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment using VADER

        Args:
            text: Input text

        Returns:
            Tuple of (sentiment_score, sentiment_label)
        """
        if not text or len(text.strip()) == 0:
            return 0.0, 'neutral'

        scores = self.analyzer.polarity_scores(text)
        compound_score = scores['compound']

        # Determine label based on compound score
        if compound_score >= 0.05:
            label = 'positive'
        elif compound_score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        return compound_score, label

    def analyze_database_articles(self, batch_size: int = 100, update_existing: bool = False):
        """Analyze all articles in database using VADER"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if update_existing:
            query = "SELECT id, title, content FROM news_articles"
        else:
            query = "SELECT id, title, content FROM news_articles WHERE sentiment_score IS NULL"

        cursor.execute(query)
        articles = cursor.fetchall()

        if not articles:
            logger.info("No articles to analyze")
            conn.close()
            return

        logger.info(f"Analyzing sentiment for {len(articles)} articles using VADER...")

        for article_id, title, content in tqdm(articles, desc="Processing"):
            text = f"{title}. {content}" if content else title
            sentiment_score, sentiment_label = self.analyze_text(text)

            cursor.execute('''
                UPDATE news_articles
                SET sentiment_score = ?, sentiment_label = ?
                WHERE id = ?
            ''', (sentiment_score, sentiment_label, article_id))

            if article_id % batch_size == 0:
                conn.commit()

        conn.commit()
        logger.info("VADER sentiment analysis completed")
        conn.close()


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze sentiment of news articles')
    parser.add_argument('--model', choices=['finbert', 'vader'], default='finbert',
                        help='Sentiment analysis model to use')
    parser.add_argument('--db', default='news_database.db',
                        help='Path to news database')
    parser.add_argument('--update', action='store_true',
                        help='Re-analyze articles that already have sentiment')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for processing')

    args = parser.parse_args()

    # Check if database exists
    if not os.path.exists(args.db):
        logger.error(f"Database not found: {args.db}")
        logger.info("Please run historical_news_scraper.py first to create the database")
        return

    # Initialize analyzer
    if args.model == 'finbert':
        try:
            analyzer = FinancialSentimentAnalyzer(db_path=args.db)
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            logger.info("Falling back to VADER...")
            analyzer = SimpleSentimentAnalyzer(db_path=args.db)
    else:
        analyzer = SimpleSentimentAnalyzer(db_path=args.db)

    # Analyze articles
    analyzer.analyze_database_articles(
        batch_size=args.batch_size,
        update_existing=args.update
    )

    # Aggregate daily sentiment (only for FinancialSentimentAnalyzer)
    if isinstance(analyzer, FinancialSentimentAnalyzer):
        logger.info("\nAggregating daily sentiment data...")
        analyzer.aggregate_daily_sentiment()


if __name__ == "__main__":
    main()

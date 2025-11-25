"""
Integrated Pipeline for Stock Market Prediction
Orchestrates the complete workflow:
1. Historical news collection
2. Sentiment analysis
3. Data preprocessing (stock + sentiment + technical indicators)
4. Model training
5. Prediction and evaluation
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Optional
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class IntegratedPipeline:
    """
    Complete pipeline for stock market prediction
    """

    def __init__(self, config: Dict):
        """
        Initialize pipeline with configuration

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}

    def step1_collect_news(self) -> bool:
        """
        Step 1: Collect historical news data

        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Collecting Historical News Data")
        logger.info("="*80 + "\n")

        try:
            from historical_news_scraper import HistoricalNewsScraper

            scraper = HistoricalNewsScraper(db_path=self.config['news_db_path'])

            # Import existing GoScraper articles
            scraper.import_goscraper_articles(
                articles_json_path=self.config.get('goscraper_articles', 'GoScraper/articles.json')
            )

            # Scrape historical data using GDELT
            if self.config.get('collect_historical_news', True):
                scraper.scrape_all_historical_data(
                    newsapi_key=self.config.get('newsapi_key', None),
                    use_gdelt=True,
                    use_goscraper=False  # Already imported above
                )

            # Export to CSV
            scraper.export_to_csv(output_dir=self.config.get('processed_news_dir', 'processed_news_data'))

            scraper.close()

            logger.info("\n✓ Step 1 completed successfully\n")
            return True

        except Exception as e:
            logger.error(f"✗ Step 1 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step2_analyze_sentiment(self) -> bool:
        """
        Step 2: Analyze sentiment of news articles

        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Analyzing News Sentiment")
        logger.info("="*80 + "\n")

        try:
            from sentiment_analyzer import FinancialSentimentAnalyzer, SimpleSentimentAnalyzer

            # Choose sentiment analyzer based on config
            use_finbert = self.config.get('use_finbert', True)

            if use_finbert:
                try:
                    analyzer = FinancialSentimentAnalyzer(db_path=self.config['news_db_path'])
                    logger.info("Using FinBERT for sentiment analysis")
                except Exception as e:
                    logger.warning(f"Failed to load FinBERT: {e}")
                    logger.info("Falling back to VADER...")
                    analyzer = SimpleSentimentAnalyzer(db_path=self.config['news_db_path'])
            else:
                analyzer = SimpleSentimentAnalyzer(db_path=self.config['news_db_path'])
                logger.info("Using VADER for sentiment analysis")

            # Analyze articles
            analyzer.analyze_database_articles(
                batch_size=self.config.get('sentiment_batch_size', 16),
                update_existing=False
            )

            # Aggregate daily sentiment (only for FinancialSentimentAnalyzer)
            if hasattr(analyzer, 'aggregate_daily_sentiment'):
                analyzer.aggregate_daily_sentiment(
                    output_dir=self.config.get('processed_news_dir', 'processed_news_data')
                )

            logger.info("\n✓ Step 2 completed successfully\n")
            return True

        except Exception as e:
            logger.error(f"✗ Step 2 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step3_preprocess_data(self) -> bool:
        """
        Step 3: Preprocess and merge all data

        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Preprocessing Data (Stock + Sentiment + Technical Indicators)")
        logger.info("="*80 + "\n")

        try:
            from data_preprocessing import StockDataPreprocessor

            preprocessor = StockDataPreprocessor(
                stock_data_dir=self.config['stock_data_dir'],
                news_db_path=self.config['news_db_path'],
                lookback_days=self.config['lookback_days']
            )

            # Test preprocessing on one stock
            test_ticker = self.config['tickers'][0]
            logger.info(f"Testing preprocessing on {test_ticker}...")

            data_dict = preprocessor.preprocess_for_training(test_ticker)

            logger.info(f"\nPreprocessing successful!")
            logger.info(f"Features created: {data_dict['X_train'].shape[2]}")
            logger.info(f"Training samples: {len(data_dict['X_train'])}")
            logger.info(f"Testing samples: {len(data_dict['X_test'])}")

            logger.info("\n✓ Step 3 completed successfully\n")
            return True

        except Exception as e:
            logger.error(f"✗ Step 3 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step4_train_models(self) -> bool:
        """
        Step 4: Train models for all stocks

        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Training Models")
        logger.info("="*80 + "\n")

        try:
            from enhanced_model import train_all_stocks, train_single_stock

            if self.config.get('train_all_stocks', False):
                results = train_all_stocks(
                    model_type=self.config.get('model_type', 'attention'),
                    lookback_days=self.config['lookback_days'],
                    epochs=self.config['epochs']
                )
            else:
                # Train individual stocks
                results = []
                for ticker in self.config['tickers']:
                    result = train_single_stock(
                        ticker=ticker,
                        model_type=self.config.get('model_type', 'attention'),
                        lookback_days=self.config['lookback_days'],
                        epochs=self.config['epochs']
                    )
                    results.append(result)

            self.results['training_results'] = results

            logger.info("\n✓ Step 4 completed successfully\n")
            return True

        except Exception as e:
            logger.error(f"✗ Step 4 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_full_pipeline(self) -> bool:
        """
        Run the complete pipeline

        Returns:
            True if all steps successful, False otherwise
        """
        logger.info("\n" + "#"*80)
        logger.info("#" + " "*78 + "#")
        logger.info("#" + " "*20 + "INTEGRATED STOCK PREDICTION PIPELINE" + " "*22 + "#")
        logger.info("#" + " "*78 + "#")
        logger.info("#"*80 + "\n")

        start_time = datetime.now()

        steps = [
            ('Collect Historical News', self.step1_collect_news),
            ('Analyze Sentiment', self.step2_analyze_sentiment),
            ('Preprocess Data', self.step3_preprocess_data),
            ('Train Models', self.step4_train_models)
        ]

        # Allow skipping certain steps
        skip_steps = self.config.get('skip_steps', [])

        for i, (step_name, step_func) in enumerate(steps, 1):
            if i in skip_steps:
                logger.info(f"\n⊘ Skipping Step {i}: {step_name}\n")
                continue

            success = step_func()

            if not success and self.config.get('stop_on_error', True):
                logger.error(f"\nPipeline stopped at Step {i} due to error")
                return False

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "#"*80)
        logger.info("#" + " "*78 + "#")
        logger.info("#" + " "*25 + "PIPELINE COMPLETED SUCCESSFULLY" + " "*22 + "#")
        logger.info("#" + " "*78 + "#")
        logger.info("#"*80 + "\n")

        logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

        return True


def load_config(config_file: Optional[str] = None) -> Dict:
    """
    Load configuration from file or use defaults

    Args:
        config_file: Path to JSON config file

    Returns:
        Configuration dictionary
    """
    default_config = {
        'stock_data_dir': 'stock_data',
        'news_db_path': 'news_database.db',
        'processed_news_dir': 'processed_news_data',
        'goscraper_articles': 'GoScraper/articles.json',
        'tickers': [
            'INFY.NS', 'ITC.NS', 'BHARTIARTL.NS', 'TCS.NS', 'HINDUNILVR.NS',
            'LICI.NS', 'SBIN.NS', 'RELIANCE.NS', 'ICICIBANK.NS', 'HDFCBANK.NS'
        ],
        'lookback_days': 60,
        'epochs': 100,
        'model_type': 'attention',
        'use_finbert': True,
        'sentiment_batch_size': 16,
        'collect_historical_news': True,
        'train_all_stocks': False,
        'skip_steps': [],  # List of step numbers to skip (1, 2, 3, or 4)
        'stop_on_error': False,
        'newsapi_key': None
    }

    if config_file and os.path.exists(config_file):
        logger.info(f"Loading configuration from {config_file}")
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            default_config.update(user_config)

    return default_config


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Integrated Stock Prediction Pipeline')
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--skip-news', action='store_true', help='Skip news collection step')
    parser.add_argument('--skip-sentiment', action='store_true', help='Skip sentiment analysis step')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--only-train', action='store_true', help='Only run training step')
    parser.add_argument('--ticker', type=str, help='Train only this ticker')
    parser.add_argument('--model', choices=['simple', 'attention', 'bidirectional'],
                        default='attention', help='Model type')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line arguments
    skip_steps = []
    if args.skip_news:
        skip_steps.append(1)
    if args.skip_sentiment:
        skip_steps.append(2)
    if args.skip_preprocess:
        skip_steps.append(3)
    if args.only_train:
        skip_steps.extend([1, 2, 3])

    config['skip_steps'] = skip_steps

    if args.ticker:
        config['tickers'] = [args.ticker]
        config['train_all_stocks'] = False

    if args.model:
        config['model_type'] = args.model

    if args.epochs:
        config['epochs'] = args.epochs

    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs(config['processed_news_dir'], exist_ok=True)

    # Run pipeline
    pipeline = IntegratedPipeline(config)
    success = pipeline.run_full_pipeline()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

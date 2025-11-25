"""
Historical News Scraper for Stock Market Data
Collects historical news articles for companies from multiple sources
and aligns them with stock price dates (2015-present)
"""

import os
import json
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from time import sleep
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistoricalNewsScraper:
    """Scraper for collecting historical news data aligned with stock dates"""

    def __init__(self, db_path: str = "news_database.db"):
        """
        Initialize the scraper with database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.companies = {
            'INFY.NS': ['Infosys', 'INFY'],
            'ITC.NS': ['ITC'],
            'BHARTIARTL.NS': ['Bharti Airtel', 'Airtel'],
            'TCS.NS': ['Tata Consultancy Services', 'TCS', 'Tata Consultancy'],
            'HINDUNILVR.NS': ['Hindustan Unilever', 'HUL'],
            'LICI.NS': ['Life Insurance Corporation', 'LIC'],
            'SBIN.NS': ['State Bank of India', 'SBI'],
            'RELIANCE.NS': ['Reliance Industries', 'Reliance', 'RIL'],
            'ICICIBANK.NS': ['ICICI Bank', 'ICICI'],
            'HDFCBANK.NS': ['HDFC Bank', 'HDFC']
        }
        self.setup_database()

    def setup_database(self):
        """Create SQLite database schema for storing news articles"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Create news articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_ticker TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                url TEXT UNIQUE,
                published_date DATE NOT NULL,
                source TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company_ticker, title, published_date)
            )
        ''')

        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_company_date
            ON news_articles(company_ticker, published_date)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_published_date
            ON news_articles(published_date)
        ''')

        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def scrape_newsapi(self, api_key: str, start_date: str, end_date: str, company_ticker: str):
        """
        Scrape news from NewsAPI (limited to last 30 days for free tier)

        Args:
            api_key: NewsAPI key
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            company_ticker: Stock ticker symbol
        """
        company_names = self.companies.get(company_ticker, [])

        for company_name in company_names:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': company_name,
                'from': start_date,
                'to': end_date,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': api_key,
                'pageSize': 100
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get('status') == 'ok':
                    articles = data.get('articles', [])
                    logger.info(f"Found {len(articles)} articles for {company_name} ({company_ticker})")

                    for article in articles:
                        self.save_article(
                            company_ticker=company_ticker,
                            title=article.get('title', ''),
                            content=article.get('description', '') + ' ' + article.get('content', ''),
                            url=article.get('url', ''),
                            published_date=article.get('publishedAt', '')[:10],
                            source=article.get('source', {}).get('name', 'NewsAPI')
                        )

                    sleep(1)  # Rate limiting
                else:
                    logger.warning(f"NewsAPI error: {data.get('message', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error scraping NewsAPI for {company_name}: {e}")

    def scrape_gdelt(self, start_date: str, end_date: str, company_ticker: str):
        """
        Scrape news from GDELT Project (Global Database of Events, Language, and Tone)
        GDELT provides historical news data going back to 2015

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            company_ticker: Stock ticker symbol
        """
        company_names = self.companies.get(company_ticker, [])

        for company_name in company_names:
            # GDELT DOC 2.0 API endpoint
            url = "https://api.gdeltproject.org/api/v2/doc/doc"
            params = {
                'query': f'{company_name} AND (stock OR shares OR market OR earnings)',
                'mode': 'ArtList',
                'maxrecords': 250,
                'format': 'json',
                'startdatetime': start_date.replace('-', '') + '000000',
                'enddatetime': end_date.replace('-', '') + '235959',
                'sort': 'datedesc'
            }

            try:
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()

                articles = data.get('articles', [])
                logger.info(f"Found {len(articles)} GDELT articles for {company_name} ({company_ticker})")

                for article in articles:
                    pub_date = article.get('seendate', '')
                    if len(pub_date) >= 8:
                        formatted_date = f"{pub_date[:4]}-{pub_date[4:6]}-{pub_date[6:8]}"
                    else:
                        formatted_date = start_date

                    self.save_article(
                        company_ticker=company_ticker,
                        title=article.get('title', ''),
                        content=article.get('title', ''),  # GDELT doesn't provide full content
                        url=article.get('url', ''),
                        published_date=formatted_date,
                        source=article.get('domain', 'GDELT')
                    )

                sleep(2)  # Be respectful with rate limiting

            except Exception as e:
                logger.error(f"Error scraping GDELT for {company_name}: {e}")

    def import_goscraper_articles(self, articles_json_path: str = "GoScraper/articles.json"):
        """
        Import existing articles from GoScraper's articles.json

        Args:
            articles_json_path: Path to articles.json file
        """
        if not os.path.exists(articles_json_path):
            logger.warning(f"Articles file not found: {articles_json_path}")
            return

        try:
            with open(articles_json_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)

            logger.info(f"Importing {len(articles)} articles from GoScraper")

            for article in articles:
                # Try to match article to company
                title = article.get('title', '').lower()
                content = article.get('content', '').lower()
                text = title + ' ' + content

                # Find matching companies
                matched_companies = []
                for ticker, names in self.companies.items():
                    if any(name.lower() in text for name in names):
                        matched_companies.append(ticker)

                # Save for each matched company
                for ticker in matched_companies:
                    scraped_at = article.get('scraped_at', '')
                    pub_date = scraped_at[:10] if scraped_at else datetime.now().strftime('%Y-%m-%d')

                    self.save_article(
                        company_ticker=ticker,
                        title=article.get('title', ''),
                        content=article.get('content', ''),
                        url=article.get('url', ''),
                        published_date=pub_date,
                        source='GoScraper'
                    )

            logger.info("GoScraper articles imported successfully")

        except Exception as e:
            logger.error(f"Error importing GoScraper articles: {e}")

    def save_article(self, company_ticker: str, title: str, content: str,
                     url: str, published_date: str, source: str):
        """
        Save article to database

        Args:
            company_ticker: Stock ticker symbol
            title: Article title
            content: Article content
            url: Article URL
            published_date: Publication date (YYYY-MM-DD)
            source: News source name
        """
        if not title or not published_date:
            return

        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO news_articles
                (company_ticker, title, content, url, published_date, source)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (company_ticker, title, content, url, published_date, source))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")

    def get_date_ranges_from_stock_data(self, stock_data_dir: str = "stock_data") -> Dict[str, tuple]:
        """
        Get date ranges from stock CSV files

        Args:
            stock_data_dir: Directory containing stock CSV files

        Returns:
            Dictionary mapping ticker to (min_date, max_date)
        """
        date_ranges = {}

        for filename in os.listdir(stock_data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(stock_data_dir, filename)
                ticker = filename.split('_')[-1].replace('.csv', '')

                try:
                    df = pd.read_csv(filepath)
                    df['Date'] = pd.to_datetime(df['Date'])
                    min_date = df['Date'].min().strftime('%Y-%m-%d')
                    max_date = df['Date'].max().strftime('%Y-%m-%d')
                    date_ranges[ticker] = (min_date, max_date)
                    logger.info(f"{ticker}: {min_date} to {max_date}")
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")

        return date_ranges

    def scrape_all_historical_data(self, newsapi_key: Optional[str] = None,
                                   use_gdelt: bool = True,
                                   use_goscraper: bool = True):
        """
        Main method to scrape all historical news data aligned with stock dates

        Args:
            newsapi_key: NewsAPI key (optional, for recent news)
            use_gdelt: Whether to use GDELT for historical data
            use_goscraper: Whether to import existing GoScraper articles
        """
        # Import existing GoScraper articles first
        if use_goscraper:
            logger.info("Importing existing GoScraper articles...")
            self.import_goscraper_articles()

        # Get date ranges from stock data
        date_ranges = self.get_date_ranges_from_stock_data()

        for ticker, (start_date, end_date) in date_ranges.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {ticker}: {start_date} to {end_date}")
            logger.info(f"{'='*60}")

            # For historical data, we need to break into chunks
            # GDELT works best with monthly chunks
            if use_gdelt:
                current_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

                while current_date < end_datetime:
                    chunk_end = min(current_date + timedelta(days=30), end_datetime)

                    logger.info(f"GDELT: {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                    self.scrape_gdelt(
                        start_date=current_date.strftime('%Y-%m-%d'),
                        end_date=chunk_end.strftime('%Y-%m-%d'),
                        company_ticker=ticker
                    )

                    current_date = chunk_end + timedelta(days=1)
                    sleep(3)  # Rate limiting between chunks

            # NewsAPI for recent data (last 30 days only on free tier)
            if newsapi_key:
                recent_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                logger.info(f"NewsAPI: {recent_start} to {end_date}")
                self.scrape_newsapi(newsapi_key, recent_start, end_date, ticker)

        logger.info("\n" + "="*60)
        logger.info("Historical news scraping completed!")
        logger.info("="*60)
        self.print_statistics()

    def print_statistics(self):
        """Print database statistics"""
        cursor = self.conn.cursor()

        # Total articles
        cursor.execute("SELECT COUNT(*) FROM news_articles")
        total = cursor.fetchone()[0]
        logger.info(f"\nTotal articles in database: {total}")

        # Articles per company
        cursor.execute("""
            SELECT company_ticker, COUNT(*) as count
            FROM news_articles
            GROUP BY company_ticker
            ORDER BY count DESC
        """)

        logger.info("\nArticles per company:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]}")

        # Date range
        cursor.execute("SELECT MIN(published_date), MAX(published_date) FROM news_articles")
        date_range = cursor.fetchone()
        logger.info(f"\nDate range: {date_range[0]} to {date_range[1]}")

    def export_to_csv(self, output_dir: str = "processed_news_data"):
        """
        Export news data to CSV files (one per company)

        Args:
            output_dir: Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)

        for ticker in self.companies.keys():
            query = """
                SELECT company_ticker, title, content, url, published_date,
                       source, sentiment_score, sentiment_label
                FROM news_articles
                WHERE company_ticker = ?
                ORDER BY published_date
            """

            df = pd.read_sql_query(query, self.conn, params=(ticker,))

            if not df.empty:
                output_path = os.path.join(output_dir, f"news_{ticker}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Exported {len(df)} articles for {ticker} to {output_path}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Main execution function"""
    # Initialize scraper
    scraper = HistoricalNewsScraper()

    # Optional: Add your NewsAPI key here (free tier: 100 requests/day, last 30 days only)
    # Get free key at: https://newsapi.org/
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', None)

    # Scrape all historical data
    # Note: GDELT is free and provides historical data back to 2015
    # NewsAPI free tier only provides last 30 days
    scraper.scrape_all_historical_data(
        newsapi_key=NEWSAPI_KEY,
        use_gdelt=True,
        use_goscraper=True
    )

    # Export to CSV files
    scraper.export_to_csv()

    # Close connection
    scraper.close()


if __name__ == "__main__":
    main()

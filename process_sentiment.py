"""
News Sentiment Analysis Script
================================

Performs sentiment analysis on collected news.
Extracts sentiment and impact features relevant to each stock/date.

Usage:
    python3 process_sentiment.py
"""

import os
import json
import re
import pandas as pd
from textblob import TextBlob
from datetime import datetime

# Configuration
INPUT_FILE = 'collected_news.json'
OUTPUT_FILE = 'news_sentiment.json'

# Company name patterns (matches articles to tickers)
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

print("="*70)
print("NEWS SENTIMENT ANALYSIS")
print("="*70)

# ============================================================================
# Step 1: Load collected news
# ============================================================================
print("\n[1/3] Loading collected news...")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found. Run collect_news.py first.")
    exit(1)

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    news_data = json.load(f)

articles = news_data.get('articles', [])
print(f"  ✓ Loaded {len(articles)} articles")

# ============================================================================
# Step 2: Sentiment Analysis
# ============================================================================
print("\n[2/3] Performing sentiment analysis...")

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    if not text or len(text.strip()) == 0:
        return 0.0
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity  # -1 (negative) to +1 (positive)
    except:
        return 0.0

def match_ticker_in_text(text, ticker):
    """Check if article is about the specified ticker's company"""
    if ticker not in COMPANY_PATTERNS:
        return False

    patterns = COMPANY_PATTERNS[ticker]
    for pattern in patterns:
        if re.search(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE):
            return True
    return False

# Process each article
processed_articles = []
ticker_stats = {ticker: {'count': 0, 'pos': 0, 'neg': 0, 'neu': 0}
                for ticker in COMPANY_PATTERNS.keys()}

for article in articles:
    title = article.get('title', '')
    content = article.get('content', '')
    scraped_at = article.get('scraped_at', '')

    # Calculate sentiment
    text_to_analyze = content if content else title
    sentiment_score = analyze_sentiment(text_to_analyze)

    # Classify sentiment
    if sentiment_score > 0.1:
        sentiment_label = 'positive'
    elif sentiment_score < -0.1:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'

    # Match to tickers
    matched_tickers = []
    full_text = title + ' ' + content

    for ticker in COMPANY_PATTERNS.keys():
        if match_ticker_in_text(full_text, ticker):
            matched_tickers.append(ticker)
            ticker_stats[ticker]['count'] += 1

            # Update stats
            if sentiment_label == 'positive':
                ticker_stats[ticker]['pos'] += 1
            elif sentiment_label == 'negative':
                ticker_stats[ticker]['neg'] += 1
            else:
                ticker_stats[ticker]['neu'] += 1

    # Store processed article
    processed_articles.append({
        'url': article.get('url', ''),
        'title': title,
        'content': content,
        'scraped_at': scraped_at,
        'sentiment_score': round(sentiment_score, 4),
        'sentiment_label': sentiment_label,
        'matched_tickers': matched_tickers,
    })

print(f"  ✓ Analyzed sentiment for {len(processed_articles)} articles")

# Show ticker distribution
print("\n  Articles per ticker:")
for ticker, stats in ticker_stats.items():
    if stats['count'] > 0:
        print(f"    {ticker:15s}: {stats['count']:3d} articles "
              f"(+{stats['pos']} / ={stats['neu']} / -{stats['neg']})")

# ============================================================================
# Step 3: Aggregate by ticker and date
# ============================================================================
print("\n[3/3] Aggregating sentiment features by ticker and date...")

# Group by ticker and date
ticker_daily_sentiment = {}

for article in processed_articles:
    try:
        date = pd.to_datetime(article['scraped_at']).date()
        date_str = str(date)

        for ticker in article['matched_tickers']:
            key = f"{ticker}_{date_str}"

            if key not in ticker_daily_sentiment:
                ticker_daily_sentiment[key] = {
                    'ticker': ticker,
                    'date': date_str,
                    'sentiments': [],
                    'articles': []
                }

            ticker_daily_sentiment[key]['sentiments'].append(article['sentiment_score'])
            ticker_daily_sentiment[key]['articles'].append({
                'title': article['title'][:100],
                'sentiment': article['sentiment_score']
            })

    except Exception as e:
        continue

# Calculate aggregated features
aggregated_data = []

for key, data in ticker_daily_sentiment.items():
    sentiments = data['sentiments']

    # Calculate features
    avg_sentiment = sum(sentiments) / len(sentiments)
    article_count = len(sentiments)
    positive_count = sum(1 for s in sentiments if s > 0.1)
    negative_count = sum(1 for s in sentiments if s < -0.1)

    positive_ratio = positive_count / article_count if article_count > 0 else 0
    negative_ratio = negative_count / article_count if article_count > 0 else 0

    aggregated_data.append({
        'ticker': data['ticker'],
        'date': data['date'],
        'sentiment_score': round(avg_sentiment, 4),
        'article_count': article_count,
        'positive_ratio': round(positive_ratio, 4),
        'negative_ratio': round(negative_ratio, 4),
        'sample_articles': data['articles'][:3]  # Top 3 for reference
    })

print(f"  ✓ Created {len(aggregated_data)} ticker-date sentiment records")

# ============================================================================
# Step 4: Save processed sentiment data
# ============================================================================
output_data = {
    'processed_date': datetime.now().isoformat(),
    'total_articles': len(processed_articles),
    'ticker_stats': ticker_stats,
    'processed_articles': processed_articles,
    'aggregated_sentiment': aggregated_data
}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n  ✓ Saved to {OUTPUT_FILE}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SENTIMENT ANALYSIS COMPLETE")
print("="*70)
print(f"Total articles analyzed:  {len(processed_articles)}")
print(f"Ticker-date records:      {len(aggregated_data)}")
print(f"Output file:              {OUTPUT_FILE}")
print("="*70)
print("\nNext step: Fuse sentiment data with stock data")
print("  python3 fuse_data.py")
print("="*70)

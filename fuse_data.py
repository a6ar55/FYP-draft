"""
Data Fusion Script
==================

Aligns stock data and processed news by date.
Produces a combined training-ready dataset.

Usage:
    python3 fuse_data.py
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
STOCK_DATA_DIR = 'stock_data'
SENTIMENT_FILE = 'news_sentiment.json'
OUTPUT_DIR = 'fused_data'
OUTPUT_FILE = f'{OUTPUT_DIR}/training_data.csv'

# Ticker to filename mapping
TICKER_FILES = {
    'INFY.NS': 'Infosys_Ltd_INFY.NS.csv',
    'TCS.NS': 'Tata_Consultancy_Services_Ltd_TCS.NS.csv',
    'HDFCBANK.NS': 'HDFC_Bank_Ltd_HDFCBANK.NS.csv',
    'ICICIBANK.NS': 'ICICI_Bank_Ltd_ICICIBANK.NS.csv',
    'RELIANCE.NS': 'Reliance_Industries_Ltd_RELIANCE.NS.csv',
    'BHARTIARTL.NS': 'Bharti_Airtel_Ltd_BHARTIARTL.NS.csv',
    'HINDUNILVR.NS': 'Hindustan_Unilever_Ltd_HINDUNILVR.NS.csv',
    'ITC.NS': 'ITC_Ltd_ITC.NS.csv',
    'SBIN.NS': 'State_Bank_of_India_SBIN.NS.csv',
    'LICI.NS': 'Life_Insurance_Corporation_of_India_LICI.NS.csv',
}

print("="*70)
print("DATA FUSION (STOCK + SENTIMENT)")
print("="*70)

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================================
# Step 1: Load sentiment data
# ============================================================================
print("\n[1/3] Loading sentiment data...")

if not os.path.exists(SENTIMENT_FILE):
    print(f"ERROR: {SENTIMENT_FILE} not found. Run process_sentiment.py first.")
    exit(1)

with open(SENTIMENT_FILE, 'r', encoding='utf-8') as f:
    sentiment_data = json.load(f)

aggregated_sentiment = sentiment_data.get('aggregated_sentiment', [])
print(f"  ✓ Loaded {len(aggregated_sentiment)} sentiment records")

# Convert to DataFrame for easier merging
sentiment_df = pd.DataFrame(aggregated_sentiment)
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], utc=True)
# Normalize to date only (remove time component) for matching
sentiment_df['date'] = sentiment_df['date'].dt.normalize()

# Select only the features we need
sentiment_features = ['ticker', 'date', 'sentiment_score', 'article_count',
                      'positive_ratio', 'negative_ratio']
sentiment_df = sentiment_df[sentiment_features]

print(f"  ✓ Sentiment features: {list(sentiment_df.columns[2:])}")

# ============================================================================
# Step 2: Load and merge stock data
# ============================================================================
print("\n[2/3] Loading and merging stock data...")

all_fused_data = []
ticker_counts = {}

for ticker, filename in TICKER_FILES.items():
    filepath = os.path.join(STOCK_DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"  WARNING: {filename} not found, skipping")
        continue

    # Load stock data
    stock_df = pd.read_csv(filepath)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)
    # Normalize to date only (remove time component) for matching
    stock_df['Date'] = stock_df['Date'].dt.normalize()
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)

    # Select OHLCV features
    stock_features = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    stock_df = stock_df[stock_features]

    # Add technical indicators
    stock_df['returns'] = stock_df['Close'].pct_change()
    stock_df['ma_5'] = stock_df['Close'].rolling(window=5, min_periods=1).mean()
    stock_df['ma_20'] = stock_df['Close'].rolling(window=20, min_periods=1).mean()
    stock_df['volatility'] = stock_df['Close'].rolling(window=20, min_periods=1).std()
    stock_df['volume_ma'] = stock_df['Volume'].rolling(window=20, min_periods=1).mean()

    # Get sentiment for this ticker
    ticker_sentiment = sentiment_df[sentiment_df['ticker'] == ticker].copy()

    # Debug: Show date range of sentiment data for this ticker
    if len(ticker_sentiment) > 0:
        sentiment_dates = ticker_sentiment['date'].min(), ticker_sentiment['date'].max()
        # print(f"    {ticker} sentiment dates: {sentiment_dates[0].date()} to {sentiment_dates[1].date()}")

    ticker_sentiment = ticker_sentiment.drop('ticker', axis=1)
    ticker_sentiment.rename(columns={'date': 'Date'}, inplace=True)

    # Merge stock and sentiment by date
    merged_df = stock_df.merge(ticker_sentiment, on='Date', how='left')

    # Fill missing sentiment values (days without news)
    # Forward fill first, then fill remaining with 0
    sentiment_cols = ['sentiment_score', 'article_count', 'positive_ratio', 'negative_ratio']
    merged_df[sentiment_cols] = merged_df[sentiment_cols].ffill().fillna(0)

    # Fill any remaining NaN values
    merged_df = merged_df.bfill().fillna(0)

    # Add ticker column
    merged_df['ticker'] = ticker

    all_fused_data.append(merged_df)

    ticker_counts[ticker] = {
        'total_days': len(merged_df),
        'days_with_news': int((merged_df['article_count'] > 0).sum())
    }

    print(f"  ✓ {ticker:15s}: {len(merged_df)} days, "
          f"{ticker_counts[ticker]['days_with_news']} with news")

# Combine all tickers
fused_df = pd.concat(all_fused_data, ignore_index=True)

print(f"\n  ✓ Total fused records: {len(fused_df)}")
print(f"  ✓ Total features: {len(fused_df.columns)}")

# ============================================================================
# Step 3: Save fused dataset
# ============================================================================
print("\n[3/3] Saving fused training data...")

# Reorder columns for clarity
column_order = ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                'returns', 'ma_5', 'ma_20', 'volatility', 'volume_ma',
                'sentiment_score', 'article_count', 'positive_ratio', 'negative_ratio']

fused_df = fused_df[column_order]

# Save to CSV
fused_df.to_csv(OUTPUT_FILE, index=False)

print(f"  ✓ Saved to {OUTPUT_FILE}")
print(f"  ✓ File size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")

# Also save metadata
metadata = {
    'creation_date': pd.Timestamp.now().isoformat(),
    'total_records': len(fused_df),
    'total_features': len(fused_df.columns),
    'feature_names': list(fused_df.columns),
    'ticker_counts': ticker_counts,
    'date_range': {
        'min': fused_df['Date'].min().isoformat(),
        'max': fused_df['Date'].max().isoformat()
    }
}

metadata_file = f'{OUTPUT_DIR}/metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Saved metadata to {metadata_file}")

# Show sample
print("\n  Sample fused data (first 5 rows):")
sample_cols = ['ticker', 'Date', 'Close', 'sentiment_score', 'article_count']
print(fused_df[sample_cols].head().to_string(index=False))

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("DATA FUSION COMPLETE")
print("="*70)
print(f"Total records:      {len(fused_df)}")
print(f"Total features:     {len(fused_df.columns)}")
print(f"Features:           {', '.join(fused_df.columns.tolist())}")
print(f"Output file:        {OUTPUT_FILE}")
print("="*70)
print("\nTraining-ready dataset created!")
print("You can now use this data to train your xLSTM model.")
print("="*70)

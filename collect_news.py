"""
News Data Collection Script
============================

Uses GoScraper to collect news data and filters articles by stock_data date ranges.
Stores news data once per execution.

Usage:
    python3 collect_news.py
"""

import os
import json
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuration
STOCK_DATA_DIR = 'stock_data'
GOSCRAPER_DIR = 'GoScraper'
GOSCRAPER_ARTICLES = 'GoScraper/articles.json'
OUTPUT_FILE = 'collected_news.json'

print("="*70)
print("NEWS DATA COLLECTION")
print("="*70)

# ============================================================================
# Step 1: Get date range from stock_data
# ============================================================================
print("\n[1/4] Analyzing stock_data directory for date ranges...")

stock_files = [f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.csv')]

if not stock_files:
    print(f"ERROR: No CSV files found in {STOCK_DATA_DIR}/")
    exit(1)

all_dates = []
stock_date_ranges = {}

for file in stock_files:
    filepath = os.path.join(STOCK_DATA_DIR, file)
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    min_date = df['Date'].min()
    max_date = df['Date'].max()

    all_dates.append(min_date)
    all_dates.append(max_date)

    # Extract ticker from filename
    ticker = file.split('_')[-1].replace('.csv', '')
    stock_date_ranges[ticker] = {
        'min': min_date,
        'max': max_date,
        'days': len(df)
    }

global_min_date = min(all_dates)
global_max_date = max(all_dates)

print(f"\n✓ Analyzed {len(stock_files)} stock files")
print(f"  Global date range: {global_min_date.date()} to {global_max_date.date()}")
print(f"  Total span: {(global_max_date - global_min_date).days} days")

# ============================================================================
# Step 2: Run GoScraper to collect latest news
# ============================================================================
print(f"\n[2/4] Running GoScraper to collect news articles...")

if not os.path.exists(GOSCRAPER_DIR):
    print(f"ERROR: GoScraper directory not found at {GOSCRAPER_DIR}/")
    exit(1)

# Check if Go is installed
try:
    subprocess.run(['go', 'version'], capture_output=True, check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print("ERROR: Go is not installed. Install Go from https://golang.org/")
    exit(1)

print("  Running GoScraper (this may take a minute)...")

try:
    result = subprocess.run(
        ['go', 'run', '.'],
        cwd=GOSCRAPER_DIR,
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode == 0:
        print("  ✓ GoScraper completed successfully")
    else:
        print(f"  WARNING: GoScraper returned code {result.returncode}")
        print(f"  Output: {result.stdout[-200:]}")

except subprocess.TimeoutExpired:
    print("  WARNING: GoScraper timed out (120s)")
except Exception as e:
    print(f"  ERROR running GoScraper: {e}")

# ============================================================================
# Step 3: Load and filter articles by stock date range
# ============================================================================
print(f"\n[3/4] Filtering articles by stock_data date range...")

if not os.path.exists(GOSCRAPER_ARTICLES):
    print(f"ERROR: GoScraper articles file not found at {GOSCRAPER_ARTICLES}")
    exit(1)

# Load all articles from GoScraper
with open(GOSCRAPER_ARTICLES, 'r', encoding='utf-8') as f:
    all_articles = json.load(f)

print(f"  Loaded {len(all_articles)} articles from GoScraper")

# Filter articles by date range
filtered_articles = []
skipped_count = 0

for article in all_articles:
    try:
        # Parse article date
        scraped_at = article.get('scraped_at', '')
        article_date = pd.to_datetime(scraped_at, utc=True)

        # Check if article falls within stock data date range
        if global_min_date <= article_date <= global_max_date:
            filtered_articles.append(article)
        else:
            skipped_count += 1

    except Exception as e:
        # Skip articles with invalid dates
        skipped_count += 1
        continue

print(f"  ✓ Filtered articles: {len(filtered_articles)}")
print(f"  ✓ Skipped (out of range): {skipped_count}")

# Show date distribution
if filtered_articles:
    article_dates = [pd.to_datetime(a['scraped_at']).date() for a in filtered_articles]
    print(f"  Article date range: {min(article_dates)} to {max(article_dates)}")

# ============================================================================
# Step 4: Save filtered news data
# ============================================================================
print(f"\n[4/4] Saving filtered news data...")

# Prepare output data
output_data = {
    'collection_date': datetime.now().isoformat(),
    'stock_date_range': {
        'min': global_min_date.isoformat(),
        'max': global_max_date.isoformat()
    },
    'total_articles_scraped': len(all_articles),
    'filtered_articles_count': len(filtered_articles),
    'articles': filtered_articles
}

# Save to file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"  ✓ Saved to {OUTPUT_FILE}")
print(f"  ✓ Total size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("NEWS COLLECTION COMPLETE")
print("="*70)
print(f"Stock date range:     {global_min_date.date()} to {global_max_date.date()}")
print(f"Articles collected:   {len(filtered_articles)}")
print(f"Output file:          {OUTPUT_FILE}")
print("="*70)
print("\nNext step: Run your pipeline with this collected news data")
print("  python3 pipeline.py")
print("="*70)

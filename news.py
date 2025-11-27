"""
Simple News Data Loader
========================

Loads news articles from GoScraper and organizes them by company and date
aligned with stock_data date ranges.

Usage:
    python3 news.py
"""

import os
import json
import re
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Company name patterns for matching articles
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

# Stock file mapping
STOCK_FILES = {
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
print("NEWS DATA LOADER FROM GOSCRAPER")
print("="*70)

# ============================================================================
# Step 1: Get stock data date ranges
# ============================================================================
print("\n[1/4] Loading stock data date ranges...")

stock_date_ranges = {}
for ticker, filename in STOCK_FILES.items():
    filepath = os.path.join('stock_data', filename)
    if not os.path.exists(filepath):
        continue

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])

    stock_date_ranges[ticker] = {
        'min': df['Date'].min(),
        'max': df['Date'].max(),
        'count': len(df)
    }

print(f"  ✓ Loaded {len(stock_date_ranges)} stock files")

# Get global date range
all_dates = []
for info in stock_date_ranges.values():
    all_dates.extend([info['min'], info['max']])

global_min = min(all_dates)
global_max = max(all_dates)

print(f"  Global date range: {global_min.date()} to {global_max.date()}")

# ============================================================================
# Step 2: Load GoScraper articles
# ============================================================================
print("\n[2/4] Loading articles from GoScraper...")

articles_file = 'GoScraper/articles.json'
if not os.path.exists(articles_file):
    print(f"  ERROR: {articles_file} not found")
    exit(1)

with open(articles_file, 'r', encoding='utf-8') as f:
    all_articles = json.load(f)

print(f"  ✓ Loaded {len(all_articles)} articles from GoScraper")

# Filter out placeholder articles
real_articles = [a for a in all_articles if a.get('title', '') != 'Just a moment...']
print(f"  ✓ Real articles: {len(real_articles)}")

# ============================================================================
# Step 3: Match articles to companies and dates
# ============================================================================
print("\n[3/4] Matching articles to companies...")

def match_company(text, ticker):
    """Check if text mentions the company"""
    patterns = COMPANY_PATTERNS.get(ticker, [])
    for pattern in patterns:
        if re.search(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE):
            return True
    return False

# Organize: ticker -> date -> [articles]
company_news = {ticker: defaultdict(list) for ticker in COMPANY_PATTERNS.keys()}

matched_count = 0
date_filtered_count = 0

for article in real_articles:
    # Parse article date
    try:
        article_date = pd.to_datetime(article['scraped_at'])
    except:
        continue

    # Check if article is within global stock date range
    if not (global_min <= article_date <= global_max):
        date_filtered_count += 1
        continue

    # Match to companies
    title = article.get('title', '')
    content = article.get('content', '')
    full_text = title + ' ' + content

    matched_to_any = False
    for ticker in COMPANY_PATTERNS.keys():
        if match_company(full_text, ticker):
            date_str = article_date.strftime('%Y-%m-%d')

            company_news[ticker][date_str].append({
                'title': title,
                'url': article.get('url', ''),
                'content': content[:500],  # First 500 chars
                'scraped_at': article['scraped_at']
            })

            matched_count += 1
            matched_to_any = True

print(f"  ✓ Matched {matched_count} article-company pairs")
print(f"  ✓ Filtered {date_filtered_count} articles outside date range")

# Show counts per company
print("\n  Articles per company:")
for ticker in sorted(COMPANY_PATTERNS.keys()):
    article_count = sum(len(articles) for articles in company_news[ticker].values())
    date_count = len(company_news[ticker])
    if article_count > 0:
        print(f"    {ticker:15s}: {article_count:3d} articles across {date_count} dates")

# ============================================================================
# Step 4: Save organized news data
# ============================================================================
print("\n[4/4] Saving organized news data...")

# Convert defaultdict to regular dict for JSON serialization
output_data = {
    'generated_at': datetime.now().isoformat(),
    'stock_date_range': {
        'min': global_min.isoformat(),
        'max': global_max.isoformat()
    },
    'total_articles': len(real_articles),
    'matched_articles': matched_count,
    'company_news': {
        ticker: dict(dates) for ticker, dates in company_news.items()
    }
}

# Save to file
output_file = 'company_news.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"  ✓ Saved to {output_file}")

# Create a summary
summary = []
for ticker, dates in company_news.items():
    if dates:
        article_count = sum(len(articles) for articles in dates.values())
        date_list = sorted(dates.keys())
        summary.append({
            'ticker': ticker,
            'total_articles': article_count,
            'date_count': len(dates),
            'date_range': {
                'min': date_list[0] if date_list else None,
                'max': date_list[-1] if date_list else None
            }
        })

# Save summary
summary_file = 'company_news_summary.json'
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump({
        'summary': summary,
        'total_companies_with_news': len([s for s in summary if s['total_articles'] > 0])
    }, f, indent=2)

print(f"  ✓ Saved summary to {summary_file}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("NEWS DATA LOADING COMPLETE")
print("="*70)
print(f"Total articles from GoScraper: {len(real_articles)}")
print(f"Matched article-company pairs:  {matched_count}")
print(f"Companies with news:             {len([s for s in summary if s['total_articles'] > 0])}")
print(f"\nOutput files:")
print(f"  - {output_file}        (full news data by company/date)")
print(f"  - {summary_file}  (summary statistics)")
print("="*70)

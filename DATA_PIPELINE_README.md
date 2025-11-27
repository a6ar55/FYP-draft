# Data Pipeline Documentation

## Overview

This pipeline collects news data using GoScraper, performs sentiment analysis, and fuses it with stock data to create a training-ready dataset.

## Architecture

```
┌─────────────────┐
│  GoScraper (Go) │ → Scrapes news from RSS feeds
└────────┬────────┘
         ↓
┌─────────────────┐
│ collect_news.py │ → Filters by stock_data date ranges
└────────┬────────┘
         ↓ collected_news.json
┌─────────────────────┐
│ process_sentiment.py│ → Sentiment analysis (TextBlob)
└────────┬────────────┘
         ↓ news_sentiment.json
┌──────────────┐
│ fuse_data.py │ → Merges stock + sentiment
└──────┬───────┘
       ↓ fused_data/training_data.csv
```

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
python3 run_data_pipeline.py
```

This runs all 3 steps automatically:
1. News collection
2. Sentiment analysis
3. Data fusion

### Option 2: Run Steps Individually

```bash
# Step 1: Collect news
python3 collect_news.py

# Step 2: Analyze sentiment
python3 process_sentiment.py

# Step 3: Fuse data
python3 fuse_data.py
```

## Scripts

### 1. `collect_news.py`

**Purpose**: Collects news from GoScraper and filters by stock_data date ranges

**Process**:
- Analyzes stock_data/ to find date range (2015-2025)
- Runs GoScraper to scrape latest news
- Filters articles within stock date range
- Stores once per execution (not incremental)

**Output**: `collected_news.json`

**Features**:
- Automatic date range detection
- Filters out-of-range articles
- Single execution storage

---

### 2. `process_sentiment.py`

**Purpose**: Performs sentiment analysis on collected news

**Process**:
- Loads collected_news.json
- Analyzes sentiment using TextBlob (-1 to +1 scale)
- Matches articles to company tickers
- Aggregates by ticker and date

**Output**: `news_sentiment.json`

**Features Extracted**:
- `sentiment_score`: Average sentiment (-1 to +1)
- `article_count`: Number of articles per day
- `positive_ratio`: % of positive articles
- `negative_ratio`: % of negative articles

---

### 3. `fuse_data.py`

**Purpose**: Aligns stock data and sentiment by date

**Process**:
- Loads news_sentiment.json
- Loads stock CSV files
- Adds technical indicators (MA, volatility, returns)
- Merges stock + sentiment by date
- Forward fills missing sentiment (days without news)

**Output**: `fused_data/training_data.csv`

**Final Features** (16 total):
- Stock: Open, High, Low, Close, Volume
- Technical: returns, ma_5, ma_20, volatility, volume_ma
- Sentiment: sentiment_score, article_count, positive_ratio, negative_ratio
- Metadata: ticker, Date

---

## Output Files

### `collected_news.json`
```json
{
  "collection_date": "2025-11-27T...",
  "stock_date_range": {"min": "2015-01-01", "max": "2025-11-04"},
  "total_articles_scraped": 457,
  "filtered_articles_count": 340,
  "articles": [...]
}
```

### `news_sentiment.json`
```json
{
  "processed_date": "2025-11-27T...",
  "total_articles": 340,
  "ticker_stats": {...},
  "aggregated_sentiment": [
    {
      "ticker": "RELIANCE.NS",
      "date": "2025-10-09",
      "sentiment_score": 0.1234,
      "article_count": 5,
      "positive_ratio": 0.8,
      "negative_ratio": 0.2
    },
    ...
  ]
}
```

### `fused_data/training_data.csv`
```
ticker,Date,Open,High,Low,Close,Volume,returns,ma_5,ma_20,volatility,volume_ma,sentiment_score,article_count,positive_ratio,negative_ratio
INFY.NS,2015-01-01,503.3,505.2,498.1,503.3,1234567,0.01,500.2,495.3,12.5,1000000,0.15,3,0.67,0.0
...
```

**Stats**:
- 24,952 rows (10 tickers × ~2,677 days each)
- 16 features
- Ready for training

---

## Company Tickers Supported

| Ticker | Company |
|--------|---------|
| INFY.NS | Infosys Ltd |
| TCS.NS | Tata Consultancy Services |
| HDFCBANK.NS | HDFC Bank |
| ICICIBANK.NS | ICICI Bank |
| RELIANCE.NS | Reliance Industries |
| BHARTIARTL.NS | Bharti Airtel |
| HINDUNILVR.NS | Hindustan Unilever |
| ITC.NS | ITC Ltd |
| SBIN.NS | State Bank of India |
| LICI.NS | Life Insurance Corporation |

---

## Data Flow

1. **GoScraper** collects news from:
   - Economic Times
   - CNBC TV18
   - Investing.com
   - Hindu Business Line
   - MoneyControl

2. **collect_news.py** filters articles matching:
   - Date range from stock_data (2015-2025)
   - Company names in article text

3. **process_sentiment.py** extracts:
   - Sentiment polarity using TextBlob
   - Positive/negative ratios
   - Article counts per day

4. **fuse_data.py** creates:
   - Stock OHLCV features
   - Technical indicators
   - Aligned sentiment features
   - Training-ready CSV

---

## Requirements

```
pandas
textblob
numpy
```

Install:
```bash
pip install pandas textblob numpy
```

For GoScraper (already setup):
```bash
cd GoScraper
go run .
```

---

## Notes

- **Storage**: News data stored once per execution (not continuous)
- **Date Filtering**: Only articles within stock_data date ranges
- **Missing Sentiment**: Forward-filled for days without news
- **GoScraper**: Automatically called by collect_news.py

---

## Troubleshooting

### No articles collected
- Check if GoScraper is working: `cd GoScraper && go run .`
- Verify GoScraper/articles.json exists

### Sentiment file not found
- Run `collect_news.py` first
- Check `collected_news.json` exists

### Fusion fails
- Run `process_sentiment.py` first
- Check `news_sentiment.json` exists
- Verify stock_data/ CSVs are present

---

## Next Steps

After running the pipeline, use `fused_data/training_data.csv` to train your model:

```python
import pandas as pd

# Load fused data
df = pd.read_csv('fused_data/training_data.csv')

# Your training code here
# The dataset includes stock + sentiment features aligned by date
```

Or use the existing `pipeline.py` which already has xLSTM model implementation.

import pandas as pd
import json
import os
import re
from textblob import TextBlob
from datetime import datetime, timedelta
import numpy as np

# Configuration
STOCK_DIR = 'stock_data'
NEWS_FILE = 'GoScraper/articles.json'
OUTPUT_FILE = 'combined_data.csv'

def parse_date(date_str):
    # Handle various date formats
    # 2025-10-09T09:15:00.631147978+05:30
    try:
        return pd.to_datetime(date_str, utc=True)
    except:
        return None

def extract_title_from_url(url):
    # Extract slug from URL
    # https://www.investing.com/news/company-news/slug-here-12345
    try:
        parts = url.split('/')
        slug = parts[-1]
        # Remove trailing numbers often found in news slugs
        slug = re.sub(r'-\d+$', '', slug)
        # Replace hyphens with spaces
        return slug.replace('-', ' ')
    except:
        return ""

def load_and_process_news():
    print("Loading news data...")
    if not os.path.exists(NEWS_FILE):
        print(f"Warning: {NEWS_FILE} not found. Proceeding without news.")
        return pd.DataFrame(columns=['Date', 'Sentiment', 'Subjectivity'])

    with open(NEWS_FILE, 'r') as f:
        try:
            data = json.load(f)
        except:
            print("Error reading JSON")
            return pd.DataFrame(columns=['Date', 'Sentiment', 'Subjectivity'])

    processed_data = []
    
    for item in data:
        date = parse_date(item.get('scraped_at') or item.get('time'))
        if date is None:
            continue
            
        title = item.get('title', '')
        content = item.get('content', '')
        url = item.get('url', '')
        
        # Fallback if title is blocked
        if not title or "Just a moment" in title:
            title = extract_title_from_url(url)
            
        text = f"{title} {content}".strip()
        if not text:
            continue
            
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        processed_data.append({
            'Date': date.normalize(), # Keep only date part for merging
            'Sentiment': sentiment,
            'Subjectivity': subjectivity
        })
        
    df = pd.DataFrame(processed_data)
    if df.empty:
        return df
        
    # Aggregate by date
    daily_news = df.groupby('Date').agg({
        'Sentiment': 'mean',
        'Subjectivity': 'mean',
        'Date': 'count' # Rename to NewsCount
    }).rename(columns={'Date': 'NewsCount'})
    
    return daily_news

def process_stock_data(news_df):
    print("Processing stock data...")
    combined_frames = []
    
    files = [f for f in os.listdir(STOCK_DIR) if f.endswith('.csv')]
    
    for file in files:
        ticker = file.split('_')[-1].replace('.csv', '')
        path = os.path.join(STOCK_DIR, file)
        
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.normalize()
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Merge with news
        if not news_df.empty:
            df = df.join(news_df, how='left')
            
        # Fill missing news data with neutral values
        df['Sentiment'] = df['Sentiment'].fillna(0)
        df['Subjectivity'] = df['Subjectivity'].fillna(0.5)
        df['NewsCount'] = df['NewsCount'].fillna(0)
        
        df['Ticker'] = ticker
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        combined_frames.append(df)
        
    if not combined_frames:
        return None
        
    full_df = pd.concat(combined_frames, ignore_index=True)
    return full_df

def main():
    news_df = load_and_process_news()
    print(f"Processed {len(news_df)} daily news records.")
    
    final_df = process_stock_data(news_df)
    
    if final_df is not None:
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved combined data to {OUTPUT_FILE} with {len(final_df)} rows.")
    else:
        print("No stock data found.")

if __name__ == "__main__":
    main()

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

def add_technical_indicators(df):
    # Ensure sorted by date
    df = df.sort_index()
    
    # 1. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # 2. MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. Bollinger Bands (20, 2)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['BB_Std'] * 2)
    
    # 4. ATR (14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # 5. OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # 6. SMA Trends
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Fill NaNs (caused by rolling windows)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True) # Final fallback
    
    return df

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
        
        # Calculate Technical Indicators BEFORE merging
        df = add_technical_indicators(df)
        
        # Merge with news
        
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

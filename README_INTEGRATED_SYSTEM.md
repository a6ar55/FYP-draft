# Integrated Stock Market Prediction System

A comprehensive stock market prediction system that combines **historical stock prices**, **technical indicators**, and **news sentiment analysis** to predict future stock prices using deep learning (LSTM with attention mechanism).

## 🎯 Overview

This system predicts stock prices for 10 major Indian companies listed on NSE:
- **INFY.NS** - Infosys Ltd
- **ITC.NS** - ITC Ltd
- **BHARTIARTL.NS** - Bharti Airtel Ltd
- **TCS.NS** - Tata Consultancy Services
- **HINDUNILVR.NS** - Hindustan Unilever Ltd
- **LICI.NS** - Life Insurance Corporation
- **SBIN.NS** - State Bank of India
- **RELIANCE.NS** - Reliance Industries Ltd
- **ICICIBANK.NS** - ICICI Bank Ltd
- **HDFCBANK.NS** - HDFC Bank Ltd

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                         │
├──────────────────────┬──────────────────────────────────────────┤
│  Historical Stock    │  News & Sentiment Data                   │
│  Prices (OHLCV)      │  - GoScraper (Real-time)                 │
│  - 2015 to Present   │  - GDELT (Historical)                    │
│  - Daily granularity │  - NewsAPI (Recent)                      │
└──────────────────────┴──────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PREPROCESSING & FEATURE ENGINEERING              │
├──────────────────────────────────────────────────────────────────┤
│  • Technical Indicators (40+ features):                          │
│    - Moving Averages (SMA, EMA)                                  │
│    - Momentum Indicators (RSI, ROC, MACD)                        │
│    - Volatility Indicators (Bollinger Bands, ATR)               │
│    - Volume Indicators (OBV)                                     │
│  • Sentiment Features:                                           │
│    - Daily sentiment scores (FinBERT)                            │
│    - Sentiment momentum & moving averages                        │
│    - Article counts & positive/negative ratios                   │
│  • Price Features:                                               │
│    - Returns, log returns, ratios                                │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     LSTM MODEL WITH ATTENTION                    │
├──────────────────────────────────────────────────────────────────┤
│  • Architecture:                                                 │
│    - Bidirectional LSTM layers (100 units each)                 │
│    - Attention mechanism for feature weighting                  │
│    - Dropout layers for regularization                          │
│  • Training:                                                     │
│    - 60-day lookback window                                     │
│    - Early stopping & learning rate reduction                   │
│    - 80/20 train-test split (time-based)                        │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PREDICTION & EVALUATION                     │
├──────────────────────────────────────────────────────────────────┤
│  • Metrics: RMSE, MAE, MAPE, R², Directional Accuracy           │
│  • Visualizations: Price predictions, training curves            │
│  • API: Real-time predictions for next day or multiple days     │
└──────────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
fyp_int/
│
├── stock_data/                          # Historical stock price CSV files
│   ├── Infosys_Ltd_INFY.NS.csv
│   ├── HDFC_Bank_Ltd_HDFCBANK.NS.csv
│   └── ...
│
├── GoScraper/                           # Real-time news scraper (Go)
│   ├── main.go
│   ├── news_feeds.go
│   ├── reddit.go
│   ├── parser.go
│   └── articles.json
│
├── historical_news_scraper.py           # ✨ NEW: Historical news collection
├── sentiment_analyzer.py                # ✨ NEW: FinBERT sentiment analysis
├── data_preprocessing.py                # ✨ NEW: Data preprocessing pipeline
├── enhanced_model.py                    # ✨ NEW: Enhanced LSTM with attention
├── integrated_pipeline.py               # ✨ NEW: Complete orchestration pipeline
├── prediction_api.py                    # ✨ NEW: Real-time prediction API
│
├── model.py                             # Original LSTM model (baseline)
│
├── news_database.db                     # SQLite database for news articles
├── processed_news_data/                 # Processed sentiment data (CSV)
├── saved_models/                        # Trained model files (.keras)
├── results/                             # Plots and evaluation results
│
├── requirements.txt                     # Python dependencies
└── README_INTEGRATED_SYSTEM.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# For GoScraper (optional, for real-time news updates)
cd GoScraper
go mod download
```

### 2. Run the Complete Pipeline

The easiest way to get started:

```bash
python integrated_pipeline.py
```

This will:
1. ✅ Collect historical news from GDELT and GoScraper
2. ✅ Analyze sentiment using FinBERT
3. ✅ Preprocess data and merge with stock prices
4. ✅ Train models for all stocks

**Total time:** ~2-4 hours depending on your hardware (GPU recommended)

### 3. Make Predictions

After training, use the prediction API:

```bash
# Predict next day for all stocks
python prediction_api.py --all

# Predict next day for a specific stock
python prediction_api.py --ticker INFY.NS

# Predict 7 days ahead (iterative prediction)
python prediction_api.py --ticker RELIANCE.NS --days 7
```

## 📚 Detailed Usage

### Step-by-Step Execution

If you want more control, run each step individually:

#### Step 1: Collect Historical News

```bash
python historical_news_scraper.py
```

**What it does:**
- Imports existing GoScraper articles
- Scrapes historical news from GDELT (2015-present)
- Optionally uses NewsAPI for recent news (requires API key)
- Stores in SQLite database (`news_database.db`)

**Configuration:**
```python
# Set NewsAPI key (optional, for last 30 days)
export NEWSAPI_KEY="your_api_key_here"
```

#### Step 2: Analyze Sentiment

```bash
# Using FinBERT (recommended, requires GPU for speed)
python sentiment_analyzer.py --model finbert

# Using VADER (faster, no GPU required)
python sentiment_analyzer.py --model vader
```

**What it does:**
- Analyzes sentiment of all articles
- Assigns scores: -1 (negative) to +1 (positive)
- Aggregates daily sentiment by company
- Creates sentiment features (momentum, moving averages)

#### Step 3: Preprocess Data

```bash
python data_preprocessing.py
```

**What it does:**
- Loads stock OHLCV data
- Calculates 40+ technical indicators
- Merges with sentiment data
- Creates 60-day sequences for LSTM
- Applies MinMax scaling

#### Step 4: Train Models

```bash
# Train all stocks with attention LSTM
python enhanced_model.py --ticker all --model attention --epochs 100

# Train single stock with bidirectional LSTM
python enhanced_model.py --ticker INFY.NS --model bidirectional --epochs 50

# Train with simple LSTM (original architecture)
python enhanced_model.py --ticker TCS.NS --model simple
```

**Model Types:**
- `simple`: Original 2-layer LSTM (baseline)
- `attention`: LSTM with attention mechanism (recommended)
- `bidirectional`: Bidirectional LSTM (reads sequences both ways)

### Advanced Pipeline Options

```bash
# Skip certain steps (if already completed)
python integrated_pipeline.py --skip-news --skip-sentiment --only-train

# Train only specific ticker
python integrated_pipeline.py --ticker RELIANCE.NS --model attention --epochs 100

# Use custom configuration file
python integrated_pipeline.py --config my_config.json
```

**Configuration File Format** (`config.json`):
```json
{
  "stock_data_dir": "stock_data",
  "news_db_path": "news_database.db",
  "tickers": ["INFY.NS", "TCS.NS"],
  "lookback_days": 60,
  "epochs": 100,
  "model_type": "attention",
  "use_finbert": true,
  "newsapi_key": "your_key_here"
}
```

## 📊 Features

### Technical Indicators (40+)

**Trend Indicators:**
- Simple Moving Average (SMA): 5, 10, 20, 50, 200 days
- Exponential Moving Average (EMA): 12, 26 days
- MACD (Moving Average Convergence Divergence)

**Momentum Indicators:**
- RSI (Relative Strength Index)
- ROC (Rate of Change)
- Momentum (10-day)
- Stochastic Oscillator (%K, %D)

**Volatility Indicators:**
- Bollinger Bands (upper, middle, lower, width, position)
- ATR (Average True Range)
- Historical Volatility (20, 50 days)

**Volume Indicators:**
- OBV (On-Balance Volume)
- Volume Moving Average
- Volume Ratio

**Price Features:**
- Daily returns
- Log returns
- High/Low ratio
- Close/Open ratio

### Sentiment Features (12+)

- Average daily sentiment score
- Sentiment momentum (1-day, 7-day changes)
- Sentiment moving averages (7, 30 days)
- Article counts per day
- Positive/Neutral/Negative article ratios
- Sentiment volatility

## 📈 Model Performance

Expected performance metrics (varies by stock):

| Metric | Typical Range |
|--------|---------------|
| **RMSE** | 15-50 ₹ |
| **MAE** | 10-35 ₹ |
| **MAPE** | 1-3% |
| **R² Score** | 0.85-0.98 |
| **Directional Accuracy** | 55-75% |

**Note:** Directional accuracy is often more important than absolute price accuracy for trading strategies.

## 🔧 Customization

### Adding New Stocks

1. Add stock CSV file to `stock_data/` directory
2. Update company mapping in `historical_news_scraper.py`:

```python
self.companies = {
    'YOUR_TICKER.NS': ['Company Name', 'Alternate Name'],
    # ... existing companies
}
```

3. Run the pipeline with the new ticker

### Adding Custom Features

Edit `data_preprocessing.py` in the `add_technical_indicators()` method:

```python
def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ti = TechnicalIndicators()

    # Add your custom indicator
    df['My_Indicator'] = your_calculation_function(df['Close'])

    return df
```

### Modifying Model Architecture

Edit `enhanced_model.py` to create your own architecture:

```python
def build_custom_model(self, n_features: int) -> Model:
    model = Sequential([
        # Your custom layers here
    ])
    return model
```

## 🐛 Troubleshooting

### Issue: "Database not found"
**Solution:** Run `historical_news_scraper.py` first to create the database

### Issue: "FinBERT is too slow"
**Solution:** Use VADER instead: `python sentiment_analyzer.py --model vader`

### Issue: "Out of memory error"
**Solution:** Reduce batch size in training:
```python
python enhanced_model.py --ticker INFY.NS --batch-size 16
```

### Issue: "No sentiment data for ticker"
**Solution:** News data may not be available for all date ranges. The model will still work with technical indicators only.

### Issue: "Model not converging"
**Solution:**
- Try different learning rates
- Increase epochs
- Use simpler model architecture
- Check for data quality issues

## 📖 API Reference

### StockPredictor Class

```python
from prediction_api import StockPredictor

predictor = StockPredictor(
    models_dir='saved_models',
    stock_data_dir='stock_data',
    news_db_path='news_database.db',
    lookback_days=60
)

# Predict next day
result = predictor.predict_next_day('INFY.NS')
print(f"Predicted price: ₹{result['predicted_price']:.2f}")

# Predict multiple days
results = predictor.predict_multiple_days('RELIANCE.NS', days=7)
for i, result in enumerate(results, 1):
    print(f"Day {i}: ₹{result['predicted_price']:.2f}")

# Batch prediction
predictions = predictor.batch_predict(['INFY.NS', 'TCS.NS', 'HDFC.NS'])
```

## 🎓 Research & Theory

### Why LSTM?
- Captures temporal dependencies in sequential data
- Maintains long-term memory through cell states
- Handles variable-length sequences naturally

### Why Attention Mechanism?
- Automatically learns which features are most important
- Improves performance on long sequences
- Provides interpretability (can visualize attention weights)

### Why FinBERT?
- Specifically trained on financial text (10K, earnings calls, news)
- Outperforms generic sentiment models on finance tasks
- Pre-trained on large financial corpus

### Data Leakage Prevention
- Strict time-based train-test split (no shuffling)
- Scaler fitted only on training data
- Forward-fill sentiment (no future information)
- No look-ahead bias in technical indicators

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@misc{stock_prediction_integrated,
  title={Integrated Stock Market Prediction System},
  author={Your Name},
  year={2024},
  institution={Your Institution}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more news sources
- Implement transformer-based models
- Add real-time data streaming
- Create web dashboard
- Implement trading strategy backtesting

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**⚠️ Disclaimer:** This system is for educational and research purposes only. Stock market predictions are inherently uncertain. Do not use this for actual trading without proper risk management and additional validation.

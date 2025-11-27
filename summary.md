# Unified Stock Prediction Pipeline with xLSTM & RL

## Overview
This project implements a unified pipeline for stock price prediction that integrates:
1.  **Historical Stock Data** (2015-2025)
2.  **Sentiment-Enhanced News Data** (collected via GoScraper)
3.  **xLSTM-based Model** with **Reinforcement Learning (RL)** feedback.

## Pipeline Architecture
1.  **Data Collection**:
    -   `GoScraper`: Fetches news articles and performs sentiment analysis.
    -   `stock_data/`: Contains historical stock CSVs.
2.  **Data Processing (`data_processor.py`)**:
    -   Cleans and normalizes stock data.
    -   Extracts sentiment from news (using Title/Content or URL fallback).
    -   Aligns news sentiment with stock data by date.
    -   Produces `combined_data.csv`.
3.  **Model Training (`model.py`)**:
    -   Loads combined data.
    -   Trains an xLSTM-inspired model (Res-LSTM + LayerNorm).
    -   Optimizes a custom **Directional RL Loss** (`MSE + lambda * Directional_Penalty`).

## How to Run

### 1. Install Dependencies
```bash
python3 -m pip install tensorflow pandas numpy scikit-learn matplotlib textblob
```

### 2. Data Collection (Optional)
If you need to fetch fresh news data:
```bash
cd GoScraper
go run .
cd ..
```
*Note: `articles.json` is already present, so this step is optional unless you want new data.*

### 3. Data Processing
Process the stock and news data into a combined dataset:
```bash
python3 data_processor.py
```
This will create `combined_data.csv`.

### 4. Train & Evaluate Model
Run the training loop with RL-enhanced xLSTM:
```bash
python3 model.py
```
This will train the model on each stock ticker and output:
-   **RMSE** (Regression Error)
-   **Directional Accuracy** (RL Target)

## Model Design
-   **Input**: Price (OHLCV) + Sentiment (Polarity, Subjectivity).
-   **Architecture**: Stacked LSTM with Residuals & LayerNorm (simulating xLSTM).
-   **Loss**: `MSE + 2.0 * Directional_Penalty`.

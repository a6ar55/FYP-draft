import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Add, LayerNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
LOOKBACK = 60
TEST_SPLIT = 0.2
EPOCHS = 50
BATCH_SIZE = 32
RL_LAMBDA = 2.0  # Weight for directional loss

# ==========================================
# 2. CUSTOM LOSS (RL COMPONENT)
# ==========================================
def directional_loss(y_true, y_pred):
    """
    Reinforcement Learning-inspired loss.
    Penalizes predictions that have the wrong direction compared to the previous step.
    """
    # y_true and y_pred are shape (batch, 1)
    # We need the difference from the previous time step.
    # Since we don't have t-1 in the loss function easily without state,
    # we can approximate direction by comparing y_pred vs y_true directly if they are returns.
    # BUT, we are predicting prices.
    # A common trick is to use the difference between adjacent elements in the batch
    # OR pass the previous price as an input.
    # For simplicity in this Keras implementation, we will assume the batch is sequential
    # and approximate diffs, OR we focus on the sign of the error relative to the trend.
    
    # Better approach for "RL":
    # We want to minimize MSE but ALSO maximize directional accuracy.
    # Loss = MSE + lambda * (1 - Directional_Match)
    
    # Differentiable approximation of direction matching:
    # diff_true = y_true[t] - y_true[t-1]
    # diff_pred = y_pred[t] - y_pred[t-1]
    # match = sign(diff_true) * sign(diff_pred)
    # We want match to be positive.
    
    # Implementation using batch slicing (t vs t-1)
    y_true_next = y_true[1:]
    y_true_prev = y_true[:-1]
    y_pred_next = y_pred[1:]
    y_pred_prev = y_pred[:-1]
    
    diff_true = y_true_next - y_true_prev
    diff_pred = y_pred_next - y_pred_prev
    
    # Tanh as soft sign
    sign_true = tf.math.tanh(diff_true * 10) # Steep tanh
    sign_pred = tf.math.tanh(diff_pred * 10)
    
    # Product: 1 if same sign, -1 if opposite
    # We want to minimize: 1 - product (range 0 to 2)
    dir_penalty = tf.reduce_mean(1 - (sign_true * sign_pred))
    
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    return mse + (RL_LAMBDA * dir_penalty)

# ==========================================
# 3. DATA LOADING & PREP
# ==========================================
def load_data(file_path, ticker):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
        
    df = pd.read_csv(file_path)
    df = df[df['Ticker'] == ticker].copy()
    
    if df.empty:
        print(f"No data for {ticker}")
        return None
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # Features: OHLCV + Sentiment
    # Ensure columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 'Subjectivity']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[required_cols]

def create_sequences(data, target, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

# ==========================================
# 4. MODEL (xLSTM-inspired)
# ==========================================
def build_model(input_shape):
    # Inputs
    inputs = Input(shape=input_shape)
    
    # xLSTM Block 1 (Simulated with Res-LSTM + LayerNorm)
    x = LSTM(64, return_sequences=True)(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # xLSTM Block 2
    x = LSTM(64, return_sequences=False)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Dense Heads
    # 1. Price Prediction (Regression)
    price_out = Dense(32, activation='relu')(x)
    price_out = Dense(1, name='price')(price_out)
    
    # 2. Direction Prediction (Classification/Auxiliary)
    # We implicitly train this via the custom loss on price, 
    # but we could add an explicit head. 
    # For this task, we stick to the single output with RL loss.
    
    model = Model(inputs=inputs, outputs=price_out)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=directional_loss)
    return model

# ==========================================
# 5. TRAINING & EVALUATION
# ==========================================
def train_and_evaluate(ticker):
    print(f"\nProcessing {ticker}...")
    
    # Load
    df = load_data('combined_data.csv', ticker)
    if df is None: return
    
    data = df.values
    
    # Split
    train_len = int(len(data) * (1 - TEST_SPLIT))
    train_data = data[:train_len]
    test_data = data[train_len - LOOKBACK:]
    
    # Scale
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create Sequences
    # Target is Close price (index 3)
    X_train, y_train = create_sequences(train_scaled, train_scaled[:, 3], LOOKBACK)
    X_test, y_test = create_sequences(test_scaled, test_scaled[:, 3], LOOKBACK)
    
    # Build
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Predict
    preds = model.predict(X_test)
    
    # Inverse Transform
    # We need to inverse transform only the target column.
    # Create dummy array
    dummy_preds = np.zeros((len(preds), data.shape[1]))
    dummy_preds[:, 3] = preds.flatten()
    preds_inv = scaler.inverse_transform(dummy_preds)[:, 3]
    
    dummy_y = np.zeros((len(y_test), data.shape[1]))
    dummy_y[:, 3] = y_test.flatten()
    y_test_inv = scaler.inverse_transform(dummy_y)[:, 3]
    
    # Metrics
    rmse = math.sqrt(mean_squared_error(y_test_inv, preds_inv))
    mae = mean_absolute_error(y_test_inv, preds_inv)
    
    # Directional Accuracy
    diff_true = np.diff(y_test_inv)
    diff_pred = np.diff(preds_inv)
    correct = np.sum(np.sign(diff_true) == np.sign(diff_pred))
    dir_acc = (correct / len(diff_true)) * 100
    
    print(f"Results for {ticker}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Directional Accuracy: {dir_acc:.2f}%")
    
    return {
        'Ticker': ticker,
        'RMSE': rmse,
        'DirAcc': dir_acc
    }

if __name__ == "__main__":
    # Get list of tickers from CSV
    try:
        df = pd.read_csv('combined_data.csv')
        tickers = df['Ticker'].unique()
        
        results = []
        for ticker in tickers:
            res = train_and_evaluate(ticker)
            if res: results.append(res)
            
        print("\nFinal Summary:")
        for r in results:
            print(f"{r['Ticker']}: DirAcc={r['DirAcc']:.2f}%, RMSE={r['RMSE']:.2f}")
            
    except Exception as e:
        print(f"Error: {e}")


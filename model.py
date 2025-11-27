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
from tensorflow.keras import mixed_precision

# ==========================================
# 1. CONFIGURATION
# ==========================================
LOOKBACK = 60
TEST_SPLIT = 0.2
EPOCHS = 50
BATCH_SIZE = 1024  # Increased for L4 GPU (24GB VRAM)
RL_LAMBDA = 2.0

# ==========================================
# GPU SETUP (L4 Optimization)
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). Using GPU acceleration.")
        
        # Enable Mixed Precision for L4 (Ada Lovelace) to use Tensor Cores
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed Precision (float16) enabled.")
    except RuntimeError as e:
        print(f"GPU Error: {e}")
else:
    print("No GPU found. Using CPU.")

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

# ==========================================
# 6. PORTFOLIO OPTIMIZATION
# ==========================================
class PortfolioManager:
    def __init__(self, tickers, lookback=60):
        self.tickers = tickers
        self.lookback = lookback
        self.models = {}
        self.scalers = {}
        self.data_store = {}
        
    def train_all(self):
        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)
        for ticker in self.tickers:
            print(f"Training model for {ticker}...")
            # Load and prep data
            df = load_data('combined_data.csv', ticker)
            if df is None: continue
            
            data = df.values
            train_len = int(len(data) * (1 - TEST_SPLIT))
            train_data = data[:train_len]
            
            scaler = MinMaxScaler()
            scaler.fit(train_data)
            
            # Store for inference
            self.scalers[ticker] = scaler
            self.data_store[ticker] = df # Store full DF for validation
            
            # Train
            train_scaled = scaler.transform(train_data)
            X_train, y_train = create_sequences(train_scaled, train_scaled[:, 3], self.lookback)
            
            model = build_model((X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            self.models[ticker] = model
            print(f"✓ {ticker} Ready")

    def simulate_trading(self):
        print("\n" + "="*50)
        print("PORTFOLIO SIMULATION (TEST PHASE)")
        print("="*50)
        
        # We need to align dates across all tickers in the test set
        # Find common dates in test range
        common_dates = None
        
        for ticker in self.tickers:
            if ticker not in self.data_store: continue
            df = self.data_store[ticker]
            train_len = int(len(df) * (1 - TEST_SPLIT))
            test_dates = df.index[train_len:]
            
            if common_dates is None:
                common_dates = test_dates
            else:
                common_dates = common_dates.intersection(test_dates)
        
        if common_dates is None or len(common_dates) == 0:
            print("No overlapping test dates found.")
            return

        print(f"Simulating over {len(common_dates)} trading days...")
        
        total_profit = 0.0
        history = []
        
        # Simulation Loop
        for date in common_dates:
            daily_predictions = []
            
            for ticker in self.tickers:
                if ticker not in self.models: continue
                
                # Get sequence for this date
                # We need lookback window ending at 'date'
                df = self.data_store[ticker]
                
                # Find integer location of date
                try:
                    loc = df.index.get_loc(date)
                except KeyError:
                    continue
                    
                if loc < self.lookback: continue
                
                # Extract sequence
                raw_seq = df.iloc[loc-self.lookback:loc].values
                prev_close = raw_seq[-1, 3] # Close is index 3
                
                # Scale
                scaler = self.scalers[ticker]
                seq_scaled = scaler.transform(raw_seq)
                X_in = seq_scaled.reshape(1, self.lookback, seq_scaled.shape[1])
                
                # Predict
                pred_scaled = self.models[ticker].predict(X_in, verbose=0)
                
                # Inverse scale target only
                dummy = np.zeros((1, raw_seq.shape[1]))
                dummy[:, 3] = pred_scaled
                pred_price = scaler.inverse_transform(dummy)[0, 3]
                
                # Calculate Expected Return
                exp_return = (pred_price - prev_close) / prev_close
                
                daily_predictions.append({
                    'ticker': ticker,
                    'pred_return': exp_return,
                    'prev_close': prev_close,
                    'actual_close': df.iloc[loc]['Close'] # Cheating? No, this is for evaluation "after the fact"
                })
            
            if not daily_predictions: continue
            
            # RANKING / SELECTION POLICY
            # Sort by expected return
            daily_predictions.sort(key=lambda x: x['pred_return'], reverse=True)
            
            best_buy = daily_predictions[0]
            best_sell = daily_predictions[-1]
            
            # Calculate Realized Profit
            # Buy Profit = (Actual - Prev) / Prev
            buy_profit = (best_buy['actual_close'] - best_buy['prev_close']) / best_buy['prev_close']
            
            # Sell Profit = (Prev - Actual) / Prev  [Short Selling]
            sell_profit = (best_sell['prev_close'] - best_sell['actual_close']) / best_sell['prev_close']
            
            daily_total = buy_profit + sell_profit
            total_profit += daily_total
            
            history.append({
                'Date': date,
                'Buy': best_buy['ticker'],
                'Buy_Pred%': best_buy['pred_return']*100,
                'Buy_Actual%': buy_profit*100,
                'Sell': best_sell['ticker'],
                'Sell_Pred%': best_sell['pred_return']*100,
                'Sell_Actual%': sell_profit*100,
                'Daily_Profit%': daily_total*100
            })
        
        # Summary
        hist_df = pd.DataFrame(history)
        print("\nSimulation Results (Sample):")
        print(hist_df.head())
        print(f"\nTotal Cumulative Return: {total_profit*100:.2f}%")
        print(f"Average Daily Return: {(total_profit/len(common_dates))*100:.2f}%")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(hist_df['Date'], hist_df['Daily_Profit%'].cumsum(), label='Cumulative Portfolio Strategy')
        plt.title('Portfolio Optimization: Cumulative Profit (Long Best + Short Worst)')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig('portfolio_performance.png')
        print("Performance plot saved to portfolio_performance.png")

if __name__ == "__main__":
    try:
        df = pd.read_csv('combined_data.csv')
        tickers = df['Ticker'].unique()
        
        pm = PortfolioManager(tickers)
        pm.train_all()
        pm.simulate_trading()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


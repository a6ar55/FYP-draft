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
PREDICTION_HORIZON = 30 # Predict 30 days into the future
TEST_SPLIT = 0.2
EPOCHS = 50
BATCH_SIZE = 1024
RL_LAMBDA = 2.0

# ... (GPU Setup remains same) ...

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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ... (Configuration) ...

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
    
    # Features: OHLCV + Sentiment + Tech Indicators
    # Ensure columns exist
    required_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'Sentiment', 'Subjectivity',
        'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'ATR', 'OBV', 'SMA_50', 'SMA_200'
    ]
    for col in required_cols:
        if col not in df.columns:
            # If columns are missing (e.g. if data_processor wasn't run yet), warn user
            if col not in ['Sentiment', 'Subjectivity']: # News cols might be missing legitimately
                print(f"Warning: Column {col} missing for {ticker}. Run data_processor.py first.")
            df[col] = 0.0
            
    return df[required_cols]

# ... (create_sequences remains same) ...

# ... (build_model remains same) ...

# ... (PortfolioManager init remains same) ...

    def train_all(self):
        print("\n" + "="*50)
        print(f"TRAINING PHASE (Horizon={self.horizon} days)")
        print("="*50)
        
        # Callbacks
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-6)
        
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
            self.data_store[ticker] = df 
            
            # Train
            train_scaled = scaler.transform(train_data)
            # Create sequences with HORIZON
            X_train, y_train = create_sequences(train_scaled, train_scaled[:, 3], self.lookback, self.horizon)
            
            if len(X_train) == 0:
                print(f"Not enough data for {ticker}")
                continue

            model = build_model((X_train.shape[1], X_train.shape[2]))
            model.fit(
                X_train, y_train, 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                verbose=0,
                callbacks=[early_stop, reduce_lr]
            )
            self.models[ticker] = model
            print(f"✓ {ticker} Ready")

    def simulate_trading(self):
        print("\n" + "="*50)
        print("PORTFOLIO SIMULATION (TEST PHASE)")
        print("="*50)
        
        # 1. BATCH PREDICTION
        # Pre-calculate predictions for all tickers to avoid tf.function retracing in loop
        print("Generating batch predictions...")
        all_predictions = {} # Key: ticker, Value: DataFrame(Date, PredPrice)
        
        common_dates = None
        
        for ticker in self.tickers:
            if ticker not in self.models: continue
            
            df = self.data_store[ticker]
            train_len = int(len(df) * (1 - TEST_SPLIT))
            
            # We want to predict for the test period
            # Input sequences start from train_len - lookback
            # to produce predictions starting at train_len
            
            # However, we need to be careful. 
            # We want to simulate day-by-day.
            # For day 't', we use data [t-lookback : t] to predict t+horizon.
            
            test_data_start = train_len - self.lookback
            if test_data_start < 0: test_data_start = 0
            
            raw_test_data = df.iloc[test_data_start:].values
            scaler = self.scalers[ticker]
            test_scaled = scaler.transform(raw_test_data)
            
            # Create X_test (no y needed for prediction)
            X_test = []
            valid_dates = []
            
            # We iterate through the test portion
            # The first prediction corresponds to date = df.index[train_len]
            # The input for that is df[train_len-lookback : train_len]
            
            for i in range(self.lookback, len(test_scaled)):
                X_test.append(test_scaled[i-self.lookback:i])
                # The date this prediction is made ON is df.index[test_data_start + i - 1]
                # (The last data point used is i-1)
                # Wait, let's align with the simulation loop.
                # In simulation: loc is the current day index. We use [loc-lookback:loc].
                # So here, 'i' corresponds to 'loc'.
                # The date is df.index[test_data_start + i] ?? No.
                # Let's just store by Date index.
                
                # Correct indexing:
                # i goes from lookback to end.
                # The sequence ends at index i-1 (relative to test_scaled).
                # The corresponding date in the original DF is:
                # absolute_index = test_data_start + i - 1
                # This is the "Current Date" (t)
                
                abs_idx = test_data_start + i 
                if abs_idx >= len(df): break # Should not happen
                
                # Actually, simulate_trading uses: df.iloc[loc-self.lookback:loc]
                # So 'loc' corresponds to 'i' in this loop relative to raw_test_data?
                # Yes.
                
                valid_dates.append(df.index[abs_idx-1]) # Date t
            
            if not X_test: continue
            
            X_test = np.array(X_test)
            
            # Predict batch
            preds_scaled = self.models[ticker].predict(X_test, batch_size=BATCH_SIZE, verbose=0)
            
            # Inverse transform
            dummy = np.zeros((len(preds_scaled), raw_test_data.shape[1]))
            dummy[:, 3] = preds_scaled.flatten()
            preds_price = scaler.inverse_transform(dummy)[:, 3]
            
            # Store
            pred_df = pd.DataFrame({
                'Date': valid_dates,
                'PredPrice_Horizon': preds_price
            })
            pred_df.set_index('Date', inplace=True)
            all_predictions[ticker] = pred_df
            
            if common_dates is None:
                common_dates = pred_df.index
            else:
                common_dates = common_dates.intersection(pred_df.index)

        if common_dates is None or len(common_dates) == 0:
            print("No overlapping test dates found.")
            return

        print(f"Simulating over {len(common_dates)} trading days...")
        
        total_profit = 0.0
        history = []
        
        # 2. SIMULATION LOOP (Fast)
        for date in common_dates:
            candidates = []
            
            for ticker in self.tickers:
                if ticker not in all_predictions: continue
                
                try:
                    # Get predicted price at t+horizon
                    pred_future = all_predictions[ticker].loc[date, 'PredPrice_Horizon']
                    
                    # Get current price at t
                    # We need to look up in data_store
                    curr_price = self.data_store[ticker].loc[date, 'Close']
                    
                    # Expected Return over Horizon
                    exp_return = (pred_future - curr_price) / curr_price
                    
                    candidates.append({
                        'ticker': ticker,
                        'exp_return': exp_return,
                        'curr_price': curr_price
                    })
                except KeyError:
                    continue
            
            if not candidates: continue
            
            # STRATEGY: Long-Term Trend Following
            # Pick Top 1 stock with best 30-day potential
            candidates.sort(key=lambda x: x['exp_return'], reverse=True)
            best_pick = candidates[0]
            
            # We "hold" this stock for 1 day (until next re-evaluation)
            # Calculate 1-day return for this stock
            # We need price at t+1
            
            # Find next date for this ticker
            df = self.data_store[best_pick['ticker']]
            try:
                loc = df.index.get_loc(date)
                if loc + 1 >= len(df): continue # End of data
                
                next_price = df.iloc[loc+1]['Close']
                daily_return = (next_price - best_pick['curr_price']) / best_pick['curr_price']
                
                total_profit += daily_return
                
                history.append({
                    'Date': date,
                    'Selected': best_pick['ticker'],
                    'Predicted_30Day_Return%': best_pick['exp_return']*100,
                    'Actual_1Day_Return%': daily_return*100,
                    'Cumulative_Return%': total_profit*100
                })
                
            except Exception:
                continue
        
        # Summary
        hist_df = pd.DataFrame(history)
        if hist_df.empty:
            print("No trades executed.")
            return

        print("\nSimulation Results (Sample):")
        print(hist_df.head())
        print(f"\nTotal Cumulative Return: {total_profit*100:.2f}%")
        print(f"Average Daily Return: {(total_profit/len(common_dates))*100:.2f}%")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(hist_df['Date'], hist_df['Cumulative_Return%'], label='Long-Term Trend Strategy')
        plt.title(f'Portfolio Performance (Horizon={self.horizon} Days)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig('portfolio_performance.png')
        print("Performance plot saved to portfolio_performance.png")

if __name__ == "__main__":
    try:
        df = pd.read_csv('combined_data.csv')
        tickers = df['Ticker'].unique()
        
        # Initialize with Horizon=30
        pm = PortfolioManager(tickers, lookback=LOOKBACK, horizon=PREDICTION_HORIZON)
        pm.train_all()
        pm.simulate_trading()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


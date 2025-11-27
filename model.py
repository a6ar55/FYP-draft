import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os

# ==========================================
# 0. REPRODUCIBILITY SETUP
# ==========================================
def set_global_determinism(seed=42):
    """
    Sets all random seeds and flags for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    print(f"Random seed set to {seed}")

set_global_determinism(42)

# Suppress TF Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Add, LayerNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from tensorflow.keras import mixed_precision

# ... (Configuration remains same) ...

# ... (directional_loss remains same) ...

# ... (load_data remains same) ...

# ... (create_sequences remains same) ...

# ... (build_model remains same) ...

class PortfolioManager:
    def __init__(self, tickers, lookback=60, horizon=30):
        self.tickers = tickers
        self.lookback = lookback
        self.horizon = horizon
        self.models = {}
        self.scalers = {}
        self.data_store = {}
        
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
            
            # --- FIX: Check for sufficient data ---
            if len(train_data) < (self.lookback + self.horizon + 10):
                print(f"Skipping {ticker}: Insufficient training data ({len(train_data)} samples).")
                continue
            
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
                print(f"Not enough sequences for {ticker}")
                continue

            model = build_model((X_train.shape[1], X_train.shape[2]))
            model.fit(
                X_train, y_train, 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                verbose=0,
                callbacks=[early_stop, reduce_lr],
                shuffle=True # Seeded shuffle
            )
            self.models[ticker] = model
            print(f"✓ {ticker} Ready")

    def simulate_trading(self):
        print("\n" + "="*50)
        print("PORTFOLIO SIMULATION (TEST PHASE)")
        print("="*50)
        
        # 1. BATCH PREDICTION
        print("Generating batch predictions...")
        all_predictions = {} 
        
        common_dates = None
        
        for ticker in self.tickers:
            if ticker not in self.models: continue
            
            df = self.data_store[ticker]
            train_len = int(len(df) * (1 - TEST_SPLIT))
            
            test_data_start = train_len - self.lookback
            if test_data_start < 0: test_data_start = 0
            
            raw_test_data = df.iloc[test_data_start:].values
            scaler = self.scalers[ticker]
            test_scaled = scaler.transform(raw_test_data)
            
            X_test = []
            valid_dates = []
            
            for i in range(self.lookback, len(test_scaled)):
                X_test.append(test_scaled[i-self.lookback:i])
                abs_idx = test_data_start + i 
                if abs_idx >= len(df): break
                valid_dates.append(df.index[abs_idx-1]) # Date t
            
            if not X_test: continue
            
            X_test = np.array(X_test)
            preds_scaled = self.models[ticker].predict(X_test, batch_size=BATCH_SIZE, verbose=0)
            
            dummy = np.zeros((len(preds_scaled), raw_test_data.shape[1]))
            dummy[:, 3] = preds_scaled.flatten()
            preds_price = scaler.inverse_transform(dummy)[:, 3]
            
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
        
        # State Variables for Risk Management
        current_holding = None # Ticker
        stop_price = 0.0
        ATR_MULTIPLIER = 2.0
        
        # 2. SIMULATION LOOP
        for date in common_dates:
            # --- A. Check Stop Loss on Current Holding ---
            stop_triggered = False
            if current_holding:
                try:
                    df_hold = self.data_store[current_holding]
                    curr_close = df_hold.loc[date, 'Close']
                    curr_low = df_hold.loc[date, 'Low']
                    curr_atr = df_hold.loc[date, 'ATR']
                    
                    # Check if Low dipped below Stop Price
                    if curr_low < stop_price:
                        # STOP LOSS TRIGGERED
                        # We assume execution at Stop Price (or Close if gap down, but let's say Stop Price for simplicity)
                        # Actually, to be conservative, let's use min(Stop Price, Close)
                        exit_price = min(stop_price, curr_close) 
                        
                        # Calculate return from yesterday's close (since we mark-to-market daily)
                        prev_close = df_hold.iloc[df_hold.index.get_loc(date)-1]['Close']
                        daily_return = (exit_price - prev_close) / prev_close
                        
                        total_profit += daily_return
                        history.append({
                            'Date': date,
                            'Action': 'STOP_LOSS',
                            'Ticker': current_holding,
                            'Return%': daily_return*100,
                            'Cumulative%': total_profit*100
                        })
                        
                        current_holding = None # Go to Cash
                        stop_triggered = True
                    else:
                        # Update Trailing Stop (Only move UP)
                        new_stop = curr_close - (ATR_MULTIPLIER * curr_atr)
                        stop_price = max(stop_price, new_stop)
                        
                except Exception as e:
                    print(f"Error checking stop loss: {e}")
                    current_holding = None

            if stop_triggered:
                continue # Stay in cash for rest of day
                
            # --- B. Ranking & Selection ---
            candidates = []
            for ticker in self.tickers:
                if ticker not in all_predictions: continue
                try:
                    pred_future = all_predictions[ticker].loc[date, 'PredPrice_Horizon']
                    curr_price = self.data_store[ticker].loc[date, 'Close']
                    exp_return = (pred_future - curr_price) / curr_price
                    candidates.append({'ticker': ticker, 'exp_return': exp_return})
                except: continue
            
            if not candidates: continue
            
            # Pick Top 1
            candidates.sort(key=lambda x: x['exp_return'], reverse=True)
            best_pick = candidates[0]['ticker']
            
            # --- C. Rebalancing ---
            # If we are holding something different, switch
            if current_holding != best_pick:
                # Sell old (if any) - already accounted for in daily return tracking?
                # Wait, we need to track daily return of 'current_holding' if it wasn't stopped out.
                
                if current_holding:
                    # We held 'current_holding' through today. Calculate today's return.
                    # We sell at Close to switch.
                    df_hold = self.data_store[current_holding]
                    curr_close = df_hold.loc[date, 'Close']
                    prev_close = df_hold.iloc[df_hold.index.get_loc(date)-1]['Close']
                    daily_return = (curr_close - prev_close) / prev_close
                    total_profit += daily_return
                    
                    history.append({
                        'Date': date,
                        'Action': 'SWITCH_SELL',
                        'Ticker': current_holding,
                        'Return%': daily_return*100,
                        'Cumulative%': total_profit*100
                    })
                
                # Buy new
                current_holding = best_pick
                # Initialize Stop Price for new holding
                df_new = self.data_store[current_holding]
                curr_close = df_new.loc[date, 'Close']
                curr_atr = df_new.loc[date, 'ATR']
                stop_price = curr_close - (ATR_MULTIPLIER * curr_atr)
                
                # We don't record "Buy" return today, we capture it tomorrow.
                
            else:
                # Holding the same stock. Record daily return (Mark-to-Market).
                df_hold = self.data_store[current_holding]
                curr_close = df_hold.loc[date, 'Close']
                prev_close = df_hold.iloc[df_hold.index.get_loc(date)-1]['Close']
                daily_return = (curr_close - prev_close) / prev_close
                total_profit += daily_return
                
                history.append({
                    'Date': date,
                    'Action': 'HOLD',
                    'Ticker': current_holding,
                    'Return%': daily_return*100,
                    'Cumulative%': total_profit*100
                })
        
        # Summary
        hist_df = pd.DataFrame(history)
        if hist_df.empty:
            print("No trades executed.")
            return

        print("\nSimulation Results (Sample):")
        print(hist_df.tail())
        print(f"\nTotal Cumulative Return: {total_profit*100:.2f}%")
        print(f"Average Daily Return: {(total_profit/len(common_dates))*100:.2f}%")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(hist_df['Date'], hist_df['Cumulative%'], label='Risk-Managed Strategy (ATR Stop)')
        plt.title(f'Portfolio Performance (Horizon={self.horizon}d, ATR Stop={ATR_MULTIPLIER}x)')
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


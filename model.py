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
# ... (Loss function remains same) ...

# ==========================================
# 3. DATA LOADING & PREP
# ==========================================
# ... (load_data remains same) ...

def create_sequences(data, target, lookback, horizon=1):
    X, y = [], []
    # We need to ensure we have data for i+horizon
    for i in range(lookback, len(data) - horizon):
        X.append(data[i-lookback:i])
        y.append(target[i + horizon]) # Target is 'horizon' steps ahead
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
    X_train, y_train = create_sequences(train_scaled, train_scaled[:, 3], LOOKBACK, PREDICTION_HORIZON)
    X_test, y_test = create_sequences(test_scaled, test_scaled[:, 3], LOOKBACK, PREDICTION_HORIZON)
    
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
# ==========================================
# 6. PORTFOLIO OPTIMIZATION
# ==========================================
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
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
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


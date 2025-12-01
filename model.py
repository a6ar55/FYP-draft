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
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Add, LayerNormalization, MultiHeadAttention, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from tensorflow.keras import mixed_precision

# ==========================================
# 1. CONFIGURATION
# ==========================================
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

LOOKBACK = 60
PREDICTION_HORIZON = 30  # Predict 30 days into the future
TEST_SPLIT = 0.2
EPOCHS = 50
BATCH_SIZE = 1024
RL_LAMBDA = 2.0

# Attention Configuration
NUM_ATTENTION_HEADS = 4
ATTENTION_KEY_DIM = 16

# ==========================================
# 2. CUSTOM ATTENTION LAYER
# ==========================================
class TemporalAttention(tf.keras.layers.Layer):
    """
    Custom temporal attention mechanism for time series.
    Computes attention weights over the time dimension.
    """
    def __init__(self, units, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(TemporalAttention, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch, time_steps, features)
        # Compute attention scores
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        
        # Attention weights
        attention_weights = tf.nn.softmax(ait, axis=1)
        attention_weights = tf.expand_dims(attention_weights, -1)
        
        # Weighted sum
        weighted_input = x * attention_weights
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output, attention_weights

# ==========================================
# 3. CUSTOM LOSS (RL COMPONENT)
# ==========================================
def directional_loss(y_true, y_pred):
    """
    Reinforcement Learning-inspired loss.
    Penalizes predictions that have the wrong direction compared to the previous step.
    """
    y_true_next = y_true[1:]
    y_true_prev = y_true[:-1]
    y_pred_next = y_pred[1:]
    y_pred_prev = y_pred[:-1]
    
    diff_true = y_true_next - y_true_prev
    diff_pred = y_pred_next - y_pred_prev
    
    # Tanh as soft sign
    sign_true = tf.math.tanh(diff_true * 10)  # Steep tanh
    sign_pred = tf.math.tanh(diff_pred * 10)
    
    # Product: 1 if same sign, -1 if opposite
    dir_penalty = tf.reduce_mean(1 - (sign_true * sign_pred))
    
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    return mse + (RL_LAMBDA * dir_penalty)

# ==========================================
# 4. DATA LOADING & PREP
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
    required_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Sentiment', 'Subjectivity',
        'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower',
        'ATR', 'OBV', 'SMA_50', 'SMA_200'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            if col not in ['Sentiment', 'Subjectivity']:
                print(f"Warning: Column {col} missing for {ticker}. Run data_processor.py first.")
            df[col] = 0.0
    
    return df[required_cols]

def create_sequences(data, target, lookback, horizon=1):
    X, y = [], []
    for i in range(lookback, len(data) - horizon):
        X.append(data[i-lookback:i])
        y.append(target[i + horizon])
    return np.array(X), np.array(y)

# ==========================================
# 5. ENHANCED MODEL WITH ATTENTION
# ==========================================
def build_model(input_shape):
    """
    Enhanced xLSTM-style architecture with Multi-Head Attention
    """
    inputs = Input(shape=input_shape)
    
    # ====== BIDIRECTIONAL LSTM LAYERS ======
    # First LSTM layer (bidirectional for better context)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # ====== MULTI-HEAD ATTENTION ======
    # Self-attention to focus on important time steps
    attention_output = MultiHeadAttention(
        num_heads=NUM_ATTENTION_HEADS,
        key_dim=ATTENTION_KEY_DIM,
        dropout=0.1
    )(x, x)
    
    # Add & Norm (Residual connection)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    
    # ====== CUSTOM TEMPORAL ATTENTION ======
    # Additional attention mechanism for weighted temporal pooling
    attention_vec, attention_weights = TemporalAttention(units=128)(x)
    
    # ====== DENSE PREDICTION HEAD ======
    x = Dense(64, activation='relu')(attention_vec)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    
    # Output: Price prediction
    price_out = Dense(1, name='price')(x)
    
    model = Model(inputs=inputs, outputs=price_out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=directional_loss
    )
    
    return model

# ==========================================
# 6. PORTFOLIO MANAGER CLASS
# ==========================================
class PortfolioManager:
    def __init__(self, tickers, lookback=60, horizon=30):
        self.tickers = tickers
        self.lookback = lookback
        self.horizon = horizon
        self.models = {}
        self.scalers = {}
        self.data_store = {}
        
        # Create models directory
        if not os.path.exists('models'):
            os.makedirs('models')
    
    def train_all(self):
        print("\n" + "="*50)
        print(f"TRAINING PHASE (Horizon={self.horizon} days)")
        print("="*50)
        
        for ticker in self.tickers:
            print(f"Training model for {ticker}...")
            
            # Load and prep data
            df = load_data('combined_data.csv', ticker)
            if df is None:
                continue
            
            data = df.values
            train_len = int(len(data) * (1 - TEST_SPLIT))
            train_data = data[:train_len]
            
            # Check for sufficient data
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
            X_train, y_train = create_sequences(
                train_scaled, 
                train_scaled[:, 3],  # Close price
                self.lookback, 
                self.horizon
            )
            
            if len(X_train) == 0:
                print(f"Not enough sequences for {ticker}")
                continue
            
            model = build_model((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            early_stop = EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
            checkpoint = ModelCheckpoint(
                f'models/{ticker}.keras',
                monitor='loss',
                save_best_only=True,
                save_weights_only=False
            )
            
            model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[early_stop, reduce_lr, checkpoint],
                shuffle=True
            )
            
            self.models[ticker] = model
            print(f"✓ {ticker} Ready (Attention-Enhanced Model)")
    
    def simulate_trading(self):
        print("\n" + "="*50)
        print("PORTFOLIO SIMULATION (TEST PHASE)")
        print("="*50)
        
        # 1. BATCH PREDICTION
        print("Generating batch predictions...")
        all_predictions = {}
        common_dates = None
        
        for ticker in self.tickers:
            if ticker not in self.models:
                continue
            
            df = self.data_store[ticker]
            train_len = int(len(df) * (1 - TEST_SPLIT))
            test_data_start = train_len - self.lookback
            
            if test_data_start < 0:
                test_data_start = 0
            
            raw_test_data = df.iloc[test_data_start:].values
            scaler = self.scalers[ticker]
            test_scaled = scaler.transform(raw_test_data)
            
            X_test = []
            valid_dates = []
            
            for i in range(self.lookback, len(test_scaled)):
                X_test.append(test_scaled[i-self.lookback:i])
                abs_idx = test_data_start + i
                if abs_idx >= len(df):
                    break
                valid_dates.append(df.index[abs_idx-1])
            
            if not X_test:
                continue
            
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
        
        # Strategy State
        current_holding = None
        stop_price = 0.0
        days_held = 0
        
        # Strategy Parameters
        ATR_MULTIPLIER = 2.0
        MIN_RETURN_THRESHOLD = 0.02  # 2%
        MAX_HOLDING_DAYS = 30
        
        # 2. SIMULATION LOOP
        for date in common_dates:
            # --- STAGE 1: MANAGEMENT (If Holding) ---
            if current_holding:
                try:
                    df_hold = self.data_store[current_holding]
                    curr_close = df_hold.loc[date, 'Close']
                    curr_low = df_hold.loc[date, 'Low']
                    curr_atr = df_hold.loc[date, 'ATR']
                    
                    # 1. Check Stop Loss
                    if curr_low < stop_price:
                        exit_price = min(stop_price, curr_close)
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
                        
                        current_holding = None
                        days_held = 0
                        continue
                    
                    # 2. Check Time Exit
                    days_held += 1
                    if days_held >= MAX_HOLDING_DAYS:
                        prev_close = df_hold.iloc[df_hold.index.get_loc(date)-1]['Close']
                        daily_return = (curr_close - prev_close) / prev_close
                        total_profit += daily_return
                        
                        history.append({
                            'Date': date,
                            'Action': 'TIME_EXIT',
                            'Ticker': current_holding,
                            'Return%': daily_return*100,
                            'Cumulative%': total_profit*100
                        })
                        
                        current_holding = None
                        days_held = 0
                    else:
                        # 3. Hold & Update Stop
                        prev_close = df_hold.iloc[df_hold.index.get_loc(date)-1]['Close']
                        daily_return = (curr_close - prev_close) / prev_close
                        total_profit += daily_return
                        
                        # Trailing Stop
                        new_stop = curr_close - (ATR_MULTIPLIER * curr_atr)
                        stop_price = max(stop_price, new_stop)
                        
                        history.append({
                            'Date': date,
                            'Action': 'HOLD',
                            'Ticker': current_holding,
                            'Return%': daily_return*100,
                            'Cumulative%': total_profit*100
                        })
                        continue
                
                except Exception as e:
                    print(f"Error managing trade: {e}")
                    current_holding = None
            
            # --- STAGE 2: SELECTION (If Cash) ---
            if current_holding is None:
                candidates = []
                
                for ticker in self.tickers:
                    if ticker not in all_predictions:
                        continue
                    
                    try:
                        pred_future = all_predictions[ticker].loc[date, 'PredPrice_Horizon']
                        curr_price = self.data_store[ticker].loc[date, 'Close']
                        exp_return = (pred_future - curr_price) / curr_price
                        candidates.append({'ticker': ticker, 'exp_return': exp_return})
                    except:
                        continue
                
                if not candidates:
                    history.append({
                        'Date': date,
                        'Action': 'CASH',
                        'Ticker': 'CASH',
                        'Return%': 0.0,
                        'Cumulative%': total_profit*100
                    })
                    continue
                
                # Pick Best
                candidates.sort(key=lambda x: x['exp_return'], reverse=True)
                best_candidate = candidates[0]
                
                # Entry Condition
                if best_candidate['exp_return'] > MIN_RETURN_THRESHOLD:
                    current_holding = best_candidate['ticker']
                    days_held = 0
                    
                    # Initialize Stop
                    df_new = self.data_store[current_holding]
                    curr_close = df_new.loc[date, 'Close']
                    curr_atr = df_new.loc[date, 'ATR']
                    stop_price = curr_close - (ATR_MULTIPLIER * curr_atr)
                    
                    history.append({
                        'Date': date,
                        'Action': 'BUY',
                        'Ticker': current_holding,
                        'Return%': 0.0,
                        'Cumulative%': total_profit*100
                    })
                else:
                    history.append({
                        'Date': date,
                        'Action': 'CASH',
                        'Ticker': 'CASH',
                        'Return%': 0.0,
                        'Cumulative%': total_profit*100
                    })
        
        # Summary
        hist_df = pd.DataFrame(history)
        
        if hist_df.empty:
            print("No trades executed.")
            return
        
        print("\nSimulation Results (Sample):")
        print(hist_df.tail(10))
        print(f"\nTotal Cumulative Return: {total_profit*100:.2f}%")
        print(f"Average Daily Return: {(total_profit/len(common_dates))*100:.2f}%")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(hist_df['Date'], hist_df['Cumulative%'], label='Attention-Enhanced Strategy', linewidth=2)
        plt.title(f'Portfolio Performance (Attention Model, Horizon={self.horizon}d)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('portfolio_performance.png', dpi=150)
        print("Performance plot saved to portfolio_performance.png")

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
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
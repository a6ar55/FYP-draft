import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
import tensorflow_probability as tfp
from tqdm import tqdm

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

# ==========================================
# GPU SETUP (L4 24GB Optimization)
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        
        # Enable Mixed Precision for L4 (Ampere/Ada architecture benefits)
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed precision (float16) enabled.")
    except RuntimeError as e:
        print(e)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Add, LayerNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. CONFIGURATION
# ==========================================
LOOKBACK = 60
PREDICTION_HORIZON = 1 
TEST_SPLIT = 0.2
EPOCHS = 50 
BATCH_SIZE = 128 # Reduced from 512 to prevent OOM
ROLLOUT_STEPS = 64 # Reduced from 128
MINI_BATCH_SIZE = 2048 # Process updates in chunks
GAMMA = 0.99 
ENTROPY_BETA = 0.01 
VALUE_COEF = 0.5
AUX_LOSS_COEF = 0.5 

# ==========================================
# 2. VECTORIZED ENVIRONMENT
# ==========================================
class VectorTradingEnv:
    """
    Vectorized trading environment for high GPU utilization.
    Simulates `num_envs` independent trajectories in parallel.
    """
    def __init__(self, data, lookback=60, num_envs=128):
        self.data = data # Shape (N, Features)
        self.lookback = lookback
        self.num_envs = num_envs
        self.n_samples = len(data)
        
        # Current step index for each environment
        # Initialize randomly to decorrelate samples
        self.current_steps = np.random.randint(
            lookback, self.n_samples - 1, size=num_envs
        )
        
    def reset(self):
        # Reset to random positions
        self.current_steps = np.random.randint(
            self.lookback, self.n_samples - 1, size=self.num_envs
        )
        return self._get_states()
        
    def step(self, actions):
        """
        actions: (num_envs,) array of 0 or 1
        """
        # 1. Calculate Rewards for current steps
        curr_prices = self.data[self.current_steps, 3] # Close is index 3
        next_prices = self.data[self.current_steps + 1, 3]
        
        raw_returns = next_prices - curr_prices
        rewards = np.where(actions == 1, raw_returns * 100, 0.0)
        
        # 2. Advance steps
        self.current_steps += 1
        
        # 3. Check Done & Auto-Reset
        # If an env reaches end, reset it immediately (infinite horizon approximation)
        dones = self.current_steps >= (self.n_samples - 1)
        if np.any(dones):
            # Reset done envs to random positions
            self.current_steps[dones] = np.random.randint(
                self.lookback, self.n_samples - 1, size=np.sum(dones)
            )
            
        # 4. Get Next States
        next_states = self._get_states()
        
        # Return true next prices for aux loss (handle resets carefully? 
        # For aux loss, we want prediction of *actual* next step. 
        # If reset, next_price is from new start. That's fine.)
        true_next_prices = self.data[self.current_steps, 3] # After increment, this is next
        
        return next_states, rewards, dones, true_next_prices
        
    def _get_states(self):
        # Efficiently gather slices
        # This might be slow in pure numpy loop, but let's try.
        # Ideally we'd use stride_tricks or pre-generated windows if memory allows.
        # Given 24GB VRAM, we can pre-generate all windows?
        # Data len ~2000-5000? 5000 * 60 * 16 * 4 bytes ~ 19MB. Tiny.
        # Let's pre-generate windows for speed!
        
        # NOTE: If data is huge, this is bad. But for single stock daily data, it's fine.
        # Actually, let's stick to slicing for generality, it's not THAT slow compared to model fwd.
        
        states = np.empty((self.num_envs, self.lookback, self.data.shape[1]), dtype=np.float32)
        for i in range(self.num_envs):
            idx = self.current_steps[i]
            states[i] = self.data[idx-self.lookback : idx]
        return states

# ==========================================
# 3. MODEL (ACTOR-CRITIC)
# ==========================================
def build_actor_critic(input_shape):
    inputs = Input(shape=input_shape)
    
    # Shared Backbone (xLSTM-like)
    x = LSTM(64, return_sequences=True)(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = LSTM(64, return_sequences=False)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 1. Policy Head (Actor) -> Logits for [Cash, Long]
    policy_logits = Dense(2, name='policy_logits')(x)
    
    # 2. Value Head (Critic) -> Expected Return
    value = Dense(1, name='value')(x)
    
    # 3. Auxiliary Head -> Next Price Prediction (Supervised helper)
    price_pred = Dense(1, name='price_pred')(x)
    
    model = Model(inputs=inputs, outputs=[policy_logits, value, price_pred])
    return model

# ==========================================
# 4. TRAINING LOGIC (A2C)
# ==========================================
class PortfolioManager:
    def __init__(self, tickers, lookback=60):
        self.tickers = tickers
        self.lookback = lookback
        self.models = {}
        self.scalers = {}
        self.data_store = {}
        self.optimizers = {}
        
        if not os.path.exists('models'):
            os.makedirs('models')
            
    def load_data(self, file_path, ticker):
        if not os.path.exists(file_path): return None
        df = pd.read_csv(file_path)
        df = df[df['Ticker'] == ticker].copy()
        if df.empty: return None
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        
        required_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'Sentiment', 'Subjectivity',
            'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'ATR', 'OBV', 'SMA_50', 'SMA_200'
        ]
        for col in required_cols:
            if col not in df.columns: df[col] = 0.0
        return df[required_cols]

    def train_all(self):
        print("\n" + "="*50)
        print(f"RL TRAINING PHASE (A2C) - Vectorized")
        print("="*50)
        
        self.metrics_history = {
            'reward': [], 'policy_loss': [], 'value_loss': [], 'aux_loss': []
        }
        
        for ticker in self.tickers:
            print(f"Training Agent for {ticker}...")
            
            # Data Prep
            df = self.load_data('combined_data.csv', ticker)
            if df is None: continue
            
            data = df.values
            train_len = int(len(data) * (1 - TEST_SPLIT))
            train_data = data[:train_len]
            
            if len(train_data) < self.lookback + 100:
                print(f"Skipping {ticker}: Insufficient data.")
                continue
                
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_data)
            self.scalers[ticker] = scaler
            self.data_store[ticker] = df
            
            # Initialize Vector Environment and Model
            # Use BATCH_SIZE as num_envs to fill GPU
            env = VectorTradingEnv(train_scaled, self.lookback, num_envs=BATCH_SIZE)
            model = build_actor_critic((self.lookback, train_scaled.shape[1]))
            optimizer = Adam(learning_rate=0.0005)
            self.optimizers[ticker] = optimizer
            
            # Training Loop
            ticker_rewards = []
            
            pbar = tqdm(range(EPOCHS), desc=f"Training {ticker}", unit="epoch")
            for epoch in pbar:
                # Reset envs at start of epoch? 
                # No, keep them running for continuity, just reset internal state if needed.
                # Actually, VectorEnv handles resets. We just get initial state.
                if epoch == 0:
                    states = env.reset()
                
                # Rollout buffers
                # We collect ROLLOUT_STEPS for ALL BATCH_SIZE envs
                # Shape: [ROLLOUT_STEPS, BATCH_SIZE, ...]
                mb_states = []
                mb_actions = []
                mb_rewards = []
                mb_values = []
                mb_dones = []
                mb_next_prices = []
                
                # 1. Collect Trajectories
                for _ in range(ROLLOUT_STEPS):
                    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                    logits, values, _ = model(states_tensor)
                    
                    # Sample Actions
                    action_probs = tf.nn.softmax(logits)
                    dist = tfp.distributions.Categorical(probs=action_probs)
                    actions = dist.sample().numpy() # Shape (BATCH_SIZE,)
                    
                    next_states, rewards, dones, true_next_prices = env.step(actions)
                    
                    mb_states.append(states_tensor)
                    mb_actions.append(actions)
                    mb_rewards.append(rewards)
                    mb_values.append(values[:, 0])
                    mb_dones.append(dones)
                    mb_next_prices.append(true_next_prices)
                    
                    states = next_states
                    
                # 2. Compute Returns & Advantages (GAE)
                # Bootstrap value
                last_state_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                _, last_values, _ = model(last_state_tensor)
                last_values = last_values[:, 0].numpy()
                
                mb_returns = np.zeros_like(mb_rewards)
                mb_advantages = np.zeros_like(mb_rewards)
                
                lastgaelam = 0
                for t in reversed(range(ROLLOUT_STEPS)):
                    if t == ROLLOUT_STEPS - 1:
                        nextnonterminal = 1.0 - mb_dones[t]
                        nextvalues = last_values
                    else:
                        nextnonterminal = 1.0 - mb_dones[t]
                        nextvalues = mb_values[t+1]
                        
                    delta = mb_rewards[t] + GAMMA * nextvalues * nextnonterminal - mb_values[t]
                    lastgaelam = delta + GAMMA * 0.95 * nextnonterminal * lastgaelam
                    mb_advantages[t] = lastgaelam
                    mb_returns[t] = mb_advantages[t] + mb_values[t]
                    
                # Flatten batches for training
                # [ROLLOUT_STEPS, BATCH_SIZE] -> [ROLLOUT_STEPS * BATCH_SIZE]
                flat_states = tf.concat(mb_states, axis=0) # [Steps*Batch, Lookback, Feats]
                flat_actions = np.concatenate(mb_actions)
                flat_returns = np.concatenate(mb_returns)
                flat_next_prices = np.concatenate(mb_next_prices)
                
                # Shuffle for mini-batch training
                indices = np.arange(len(flat_actions))
                np.random.shuffle(indices)
                
                epoch_policy_loss = []
                epoch_value_loss = []
                epoch_aux_loss = []
                
                # Mini-batch Update Loop
                for start_idx in range(0, len(indices), MINI_BATCH_SIZE):
                    end_idx = start_idx + MINI_BATCH_SIZE
                    batch_indices = indices[start_idx:end_idx]
                    
                    mb_states_t = tf.gather(flat_states, batch_indices)
                    mb_actions_t = tf.convert_to_tensor(flat_actions[batch_indices], dtype=tf.int32)
                    mb_returns_t = tf.convert_to_tensor(flat_returns[batch_indices], dtype=tf.float32)
                    mb_next_prices_t = tf.convert_to_tensor(flat_next_prices[batch_indices], dtype=tf.float32)
                    
                    # 3. Update Step
                    with tf.GradientTape() as tape:
                        logits, values_pred, price_preds = model(mb_states_t)
                        
                        # Cast to float32
                        logits = tf.cast(logits, tf.float32)
                        values_pred = tf.cast(values_pred, tf.float32)
                        price_preds = tf.cast(price_preds, tf.float32)
                        
                        values_pred = tf.squeeze(values_pred)
                        price_preds = tf.squeeze(price_preds)
                        
                        # Policy Loss
                        action_probs = tf.nn.softmax(logits)
                        dist = tfp.distributions.Categorical(probs=action_probs)
                        log_probs = dist.log_prob(mb_actions_t)
                        advantages = mb_returns_t - values_pred
                        policy_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
                        
                        # Value Loss
                        value_loss = tf.reduce_mean(tf.square(mb_returns_t - values_pred))
                        
                        # Entropy Loss
                        entropy_loss = -tf.reduce_mean(dist.entropy())
                        
                        # Aux Loss
                        aux_loss = tf.reduce_mean(tf.square(mb_next_prices_t - price_preds))
                        
                        total_loss = policy_loss + (VALUE_COEF * value_loss) + (ENTROPY_BETA * entropy_loss) + (AUX_LOSS_COEF * aux_loss)
                    
                    grads = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    
                    epoch_policy_loss.append(policy_loss.numpy())
                    epoch_value_loss.append(value_loss.numpy())
                    epoch_aux_loss.append(aux_loss.numpy())
                
                # Metrics
                avg_ep_reward = np.mean(mb_rewards) * ROLLOUT_STEPS # Approx episodic reward
                ticker_rewards.append(avg_ep_reward)
                self.metrics_history['reward'].append(avg_ep_reward)
                self.metrics_history['policy_loss'].append(np.mean(epoch_policy_loss))
                self.metrics_history['value_loss'].append(np.mean(epoch_value_loss))
                self.metrics_history['aux_loss'].append(np.mean(epoch_aux_loss))
                
                # Update progress bar
                avg_reward = np.mean(ticker_rewards[-5:]) if len(ticker_rewards) >= 5 else np.mean(ticker_rewards)
                pbar.set_postfix({'AvgRew': f"{avg_reward:.2f}", 'Loss': f"{np.mean(epoch_value_loss):.2f}"})
            
            self.models[ticker] = model
            model.save_weights(f'models/{ticker}_rl.weights.h5')
            print(f"✓ {ticker} Trained")
            
        self.plot_metrics()

    def plot_metrics(self):
        print("\nGenerating RL Training Plots...")
        plt.figure(figsize=(15, 10))
        
        metrics = ['reward', 'policy_loss', 'value_loss', 'aux_loss']
        titles = ['Avg Reward', 'Policy Loss', 'Value Loss', 'Aux (Price) Loss']
        
        for i, (key, title) in enumerate(zip(metrics, titles)):
            plt.subplot(2, 2, i+1)
            data = self.metrics_history[key]
            if len(data) > 100:
                kernel_size = 50
                kernel = np.ones(kernel_size) / kernel_size
                data = np.convolve(data, kernel, mode='valid')
            plt.plot(data)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('rl_training_metrics.png')
        print("Saved rl_training_metrics.png")

    def simulate_trading(self):
        print("\n" + "="*50)
        print("RL POLICY SIMULATION")
        print("="*50)
        
        total_profit = 0.0
        
        for ticker in self.tickers:
            if ticker not in self.models: continue
            
            df = self.data_store[ticker]
            data = df.values
            scaler = self.scalers[ticker]
            
            train_len = int(len(data) * (1 - TEST_SPLIT))
            test_data = data[train_len:]
            test_scaled = scaler.transform(test_data)
            
            if len(test_scaled) < self.lookback: continue
            
            model = self.models[ticker]
            
            cash = 10000.0
            holdings = 0.0
            portfolio_values = []
            
            curr_step = self.lookback
            
            while curr_step < len(test_scaled) - 1:
                state = test_scaled[curr_step-self.lookback : curr_step]
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                
                logits, _, _ = model(state_tensor)
                action_probs = tf.nn.softmax(logits).numpy()[0]
                action = np.argmax(action_probs)
                
                curr_price = test_data[curr_step, 3]
                
                if action == 1 and cash > 0: # Buy
                    holdings = cash / curr_price
                    cash = 0.0
                elif action == 0 and holdings > 0: # Sell
                    cash = holdings * curr_price
                    holdings = 0.0
                    
                curr_val = cash + (holdings * curr_price)
                portfolio_values.append(curr_val)
                
                curr_step += 1
                
            final_val = portfolio_values[-1] if portfolio_values else 10000.0
            ret = (final_val - 10000.0) / 10000.0
            total_profit += ret
            
            print(f"{ticker}: Return = {ret*100:.2f}%")
            
            plt.figure(figsize=(10, 4))
            plt.plot(portfolio_values)
            plt.title(f"{ticker} RL Strategy Performance")
            plt.savefig(f"models/{ticker}_perf.png")
            plt.close()

if __name__ == "__main__":
    try:
        df = pd.read_csv('combined_data.csv')
        tickers = df['Ticker'].unique()
        
        pm = PortfolioManager(tickers, lookback=LOOKBACK)
        pm.train_all()
        pm.simulate_trading()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


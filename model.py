import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
DATASETS = [
    'Infosys_Ltd_INFY.NS.csv',
    'ITC_Ltd_ITC.NS.csv',
    'Bharti_Airtel_Ltd_BHARTIARTL.NS.csv',
    'Tata_Consultancy...s_Ltd_TCS.NS.csv',
    'Hindustan_Unilev...INDUNILVR.NS.csv',
    'Life_Insurance_Co...India_LICI.NS.csv',
    'State_Bank_of_India_SBIN.NS.csv',
    'Reliance_Industrie...RELIANCE.NS.csv',
    'ICICI_Bank_Ltd_ICICIBANK.NS.csv',
    'HDFC_Bank_Ltd_HDFCBANK.NS.csv'
]

# Hyperparameters
LOOKBACK = 60        # Past 60 days used to predict the next day
TEST_SPLIT = 0.2     # 20% of data reserved for testing
EPOCHS = 50          # Max training iterations
BATCH_SIZE = 32      # Batch size

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_data(filename):
    """
    Loads data, ensures Date index is sorted, and handles timezone.
    """
    if not os.path.exists(filename):
        print(f"[WARNING] Skipping {filename}: File not found.")
        return None

    df = pd.read_csv(filename)

    # Parse dates with timezone handling (utc=True converts all to UTC)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # Select features (Multivariate)
    features = ['Open', 'High', 'Low', 'Volume', 'Close']
    df = df[features]

    return df

def create_sequences(dataset, lookback=60):
    """
    Converts array into X (past sequence) and y (future value)
    """
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i-lookback:i])
        y.append(dataset[i, 4])         # Index 4 is 'Close'
    return np.array(X), np.array(y)

def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculates the percentage of times the model correctly predicted
    the direction of the price movement (Up/Down) relative to the previous day.
    """
    # We compare current day (i) vs previous day (i-1)
    # Note: y_true and y_pred are aligned.
    # We need to compare predicted movement vs actual movement from the SAME previous actual point.

    correct_directions = 0
    total_comparisons = len(y_true) - 1

    for i in range(1, len(y_true)):
        # Previous day's actual price
        prev_price = y_true[i-1]

        # Current day's prices
        actual_price = y_true[i]
        predicted_price = y_pred[i]

        # Did it actually go up or down?
        actual_delta = actual_price - prev_price

        # Did the model predict it would go up or down?
        predicted_delta = predicted_price - prev_price

        # Check if signs match (both positive or both negative)
        # We use np.sign returns -1, 0, or 1
        if np.sign(actual_delta) == np.sign(predicted_delta):
            correct_directions += 1

    return (correct_directions / total_comparisons) * 100

def train_predict_evaluate(filename):
    print(f"\n{'='*60}")
    print(f"Processing Dataset: {filename}")
    print(f"{'='*60}")

    # --- A. Load Data ---
    df = load_data(filename)
    if df is None: return

    dataset = df.values

    # --- B. STRICT TIME SPLIT ---
    training_data_len = int(np.ceil(len(dataset) * (1 - TEST_SPLIT)))
    train_data = dataset[0:training_data_len, :]
    test_data = dataset[training_data_len - LOOKBACK:, :]

    # --- C. SCALING ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    # --- D. CREATE SEQUENCES ---
    X_train, y_train = create_sequences(train_scaled, LOOKBACK)
    X_test, y_test = create_sequences(test_scaled, LOOKBACK)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # --- E. BUILD MODEL (LSTM) ---
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # --- F. TRAIN ---
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # --- G. PREDICT ---
    predictions = model.predict(X_test)

    # --- H. INVERSE TRANSFORM ---
    dummy_predicted = np.zeros((len(predictions), 5))
    dummy_actual = np.zeros((len(y_test), 5))

    dummy_predicted[:, 4] = predictions.flatten()
    dummy_actual[:, 4] = y_test.flatten()

    predictions_rescaled = scaler.inverse_transform(dummy_predicted)[:, 4]
    y_test_rescaled = scaler.inverse_transform(dummy_actual)[:, 4]

    # --- I. EVALUATION METRICS ---
    rmse = math.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    mape = np.mean(np.abs((y_test_rescaled - predictions_rescaled) / y_test_rescaled)) * 100
    r2 = r2_score(y_test_rescaled, predictions_rescaled)

    # Directional Accuracy
    dir_acc = calculate_directional_accuracy(y_test_rescaled, predictions_rescaled)

    print(f"\n>>> FINAL RESULTS FOR {filename} <<<")
    print(f"RMSE (Root Mean Squared Error):  {rmse:.2f}")
    print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
    print(f"MAPE (Mean Abs % Error):         {mape:.2f}%")
    print(f"R2 Score:                        {r2:.4f}")
    print(f"Directional Accuracy:            {dir_acc:.2f}%  <-- (Higher is better)")

    # --- J. VISUALIZE ---
    plt.figure(figsize=(14, 6))

    # Plotting strictly the test range
    test_range = df.index[training_data_len:]

    plt.plot(test_range, y_test_rescaled, color='#007bff', label='Actual Price', linewidth=2)
    plt.plot(test_range, predictions_rescaled, color='#ff4500', label='Predicted Price', linewidth=2, linestyle='--')

    plt.title(f'{filename} Prediction\nDir Acc: {dir_acc:.2f}% | MAPE: {mape:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Close Price (INR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# 3. EXECUTION LOOP
# ==========================================
for file in DATASETS:
    try:
        train_predict_evaluate(file)
    except Exception as e:
        print(f"Error processing {file}: {e}")

print("\nAll processing complete.")

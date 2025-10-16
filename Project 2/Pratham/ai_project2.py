
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Set all random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

nalco = yf.Ticker("NATIONALUM.NS")
df = nalco.history(period="1y")
df

df.reset_index(inplace=True)
df.head()

df = df.sort_values('Date')
df['MA5'] = df['Close'].rolling(window=5).mean()     # ma = moving average
df['MA10'] = df['Close'].rolling(window=10).mean()
df.head(10)

# df[['MA5', 'MA10', 'MA20']] = df[['MA5', 'MA10', 'MA20']].fillna(method='bfill')
# df.head()

df['Return'] = df['Close'].pct_change()

# ----- 3️⃣ Price Range (Volatility Indicator) -----
df['High_Low_Range'] = df['High'] - df['Low']

# ----- 4️⃣ Momentum (Close - Previous Close) -----
df['Momentum'] = df['Close'] - df['Close'].shift(1)

# ----- 5️⃣ Volume Change -----
df['Volume_Change'] = df['Volume'].pct_change()

# ----- 6️⃣ Relative Strength Index (RSI) -----
window_length = 14
delta = df['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)

avg_gain = pd.Series(gain).rolling(window=window_length, min_periods=1).mean()
avg_loss = pd.Series(loss).rolling(window=window_length, min_periods=1).mean()

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# ----- 7️⃣ Exponential Moving Averages (EMA) -----
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

# ----- 8️⃣ MACD (Moving Average Convergence Divergence) -----
df['MACD'] = df['EMA12'] - df['EMA26']

# Drop NaN rows created by rolling calculations
df = df.dropna().reset_index(drop=True)

# Show final engineered dataset
print(df.head(10))

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------
# FIX: Handle Infinite Values Before Scaling
# ---------------------------------------------

# Select the columns you plan to use as features
features_to_check = ['Close', 'MA5', 'MA10', 'Return', 'Volume_Change']

# Replace positive/negative infinity with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any rows that now contain NaN (from the infinity replacement)
# This removes the problematic days where Volume was zero.
df = df.dropna(subset=features_to_check).reset_index(drop=True)

# Show final engineered dataset (Optional print check)
# print(df.head(10))

# ---------------------------------------------
# Proceed to scaling
# ---------------------------------------------
# Select features and target
features = ['Close', 'MA5', 'MA10', 'Return', 'Volume_Change']
data = df[features].values # This 'data' should now be clean of 'inf'
# ... (rest of the code)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

time_step = 10
X, y = [], []

for i in range(len(scaled_data) - time_step - 1):
    X.append(scaled_data[i:(i + time_step), :])   # previous 10 days
    y.append(scaled_data[i + time_step, 0])       # next day's Close

X = np.array(X)
y = np.array(y)

# Split into train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

ann_model = Sequential([
    Dense(64, input_shape=(X_train.shape[1]*X_train.shape[2],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Ensure deterministic behavior
tf.config.experimental.enable_op_determinism()

ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
ann_model.summary()

# Flatten input for ANN
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

history_ann = ann_model.fit(
    X_train_flat, y_train,
    validation_data=(X_test_flat, y_test),
    epochs=50, batch_size=16, verbose=1
)
ann_pred = ann_model.predict(X_test_flat)

ann_rmse = np.sqrt(mean_squared_error(y_test, ann_pred))
ann_r2 = r2_score(y_test, ann_pred)
print(f"ANN RMSE:  {ann_rmse:.4f}, R²: {ann_r2:.4f}")

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

history_lstm = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50, batch_size=16, verbose=1
)

# LSTM predictions
lstm_pred = lstm_model.predict(X_test)

lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
lstm_r2 = r2_score(y_test, lstm_pred)
print(f"LSTM RMSE: {lstm_rmse:.4f}, R²: {lstm_r2:.4f}")

plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual Prices', color='black')
plt.plot(ann_pred, label='ANN Predictions', color='orange')
plt.plot(lstm_pred, label='LSTM Predictions', color='green')
plt.title("Actual vs Predicted Stock Prices (NATIONALUM.NS)")
plt.xlabel("Time")
plt.ylabel("Normalized Price")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history_ann.history['loss'], label='ANN Train Loss')
plt.plot(history_ann.history['val_loss'], label='ANN Val Loss')
plt.plot(history_lstm.history['loss'], label='LSTM Train Loss')
plt.plot(history_lstm.history['val_loss'], label='LSTM Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
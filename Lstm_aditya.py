# Generate a fully updated Colab-ready script with:
# - Target smoothing
# - Outlier clipping
# - Log return transformation
# - Z-score normalization
# - Residual feedback
# - Error clustering
# - Rolling LSTM training

final_colab_code = """
# === SETUP ===
!pip install -q pandas numpy scikit-learn tensorflow matplotlib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from google.colab import files

# === UPLOAD FILE ===
print(\"\\n📁 Please upload your ESc1_2025.csv file\")
uploaded = files.upload()

# === READ FILE ===
df = pd.read_csv("ESc1_2025.csv")
df["date-time"] = pd.to_datetime(df["date-time"])
df.sort_values("date-time", inplace=True)

# === CREATE TARGET (log return + smoothing + clipping + z-score) ===
df["log_return_t+5"] = np.log(df["close"].shift(-5) / df["close"])
df["log_return_t+5_smooth"] = df["log_return_t+5"].rolling(window=3, center=True, min_periods=1).mean()

# Clip outliers
q_low = df["log_return_t+5_smooth"].quantile(0.01)
q_high = df["log_return_t+5_smooth"].quantile(0.99)
df["log_return_t+5_clipped"] = df["log_return_t+5_smooth"].clip(q_low, q_high)

# Z-score normalization of target
target_scaler = StandardScaler()
df["log_return_t+5_final"] = target_scaler.fit_transform(df[["log_return_t+5_clipped"]])

# === CREATE LAG FEATURES ===
max_lag = 5
lag_features = []
for lag in range(1, max_lag + 1):
    for col in ["close", "vwap", "volume"]:
        lag_col = f"{col}_lag_{lag}"
        df[lag_col] = df[col].shift(lag)
        lag_features.append(lag_col)

# === RESIDUAL FEATURES ===
df["residual"] = np.nan
df["residual_lag_1"] = 0
df["residual_avg_3"] = 0

# === DROP ROWS WITH NA ===
required_columns = lag_features + ["log_return_t+5_final"]
df.dropna(subset=required_columns, inplace=True)
df.reset_index(drop=True, inplace=True)

# === SCALE INPUT FEATURES ===
feature_cols = lag_features + ["residual_lag_1", "residual_avg_3"]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# === ERROR CLUSTERING ===
df['residual_for_cluster'] = df['residual'].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42)
df['error_cluster'] = kmeans.fit_predict(df[['residual_for_cluster']])
cluster_dummies = pd.get_dummies(df['error_cluster'], prefix='error_cluster')
df = pd.concat([df, cluster_dummies], axis=1)
cluster_features = [col for col in df.columns if col.startswith('error_cluster_')]
feature_cols += cluster_features

# === DEFINE LSTM MODEL ===
def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inp)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

# === ROLLING TRAINING ===
train_window = 3000
val_window = 20
sequence_length = 10
noise_std = 0.3
target_col = "log_return_t+5_final"

predictions, actuals = [], []

for start_idx in range(train_window, len(df) - val_window - sequence_length, val_window):
    end_idx = start_idx + val_window
    train_data = df.iloc[start_idx - train_window:start_idx]
    X_train, y_train = [], []
    for i in range(sequence_length, len(train_data)):
        seq = train_data[feature_cols].iloc[i-sequence_length:i].values
        target = train_data[target_col].iloc[i] + np.random.normal(0, noise_std)
        X_train.append(seq)
        y_train.append(target)

    val_data = df.iloc[start_idx:end_idx]
    X_val, y_val = [], []
    for i in range(sequence_length, len(val_data)):
        seq = val_data[feature_cols].iloc[i-sequence_length:i].values
        target = val_data[target_col].iloc[i]
        X_val.append(seq)
        y_val.append(target)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)

    if len(X_val) == 0 or len(X_train) == 0:
        continue

    model = build_model((sequence_length, X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred = model.predict(X_val, verbose=0).flatten()
    predictions.extend(y_pred)
    actuals.extend(y_val)

    residual_window = y_val - y_pred
    residual_idx = range(start_idx + sequence_length, end_idx)
    for i, resid in zip(residual_idx, residual_window):
        df.loc[i, "residual"] = resid
        df.loc[i, "residual_lag_1"] = df.loc[i - 1, "residual"] if i - 1 >= 0 else 0
        df.loc[i, "residual_avg_3"] = df["residual"].iloc[max(i-3, 0):i].mean()

# === FINAL EVALUATION ===
rmse = np.sqrt(mean_squared_error(actuals, predictions))
print(f"\\n✅ Final Rolling RMSE (on Z-scored target): {rmse:.4f}")

# === PLOT RESULTS ===
plt.figure(figsize=(14, 5))
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Predicted vs Actual (Z-Scored, Smoothed, Clipped Log Returns)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""

# Save the full corrected script to file
final_script_path = "/mnt/data/Colab_Rolling_LSTM_Denoised_Final.py"
with open(final_script_path, "w") as f:
    f.write(final_colab_code)

final_script_path

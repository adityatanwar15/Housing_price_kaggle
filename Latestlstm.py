# === SETUP ===
!pip install -q pandas numpy scikit-learn tensorflow matplotlib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from google.colab import files

# === UPLOAD FILE ===
print("\n📁 Please upload your ESc1_2025.csv file")
uploaded = files.upload()

# === READ & SORT DATA ===
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

# Z-score normalization of target (global; consider moving inside loop if you'd like truly in-sample scaling)
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

# === INITIALIZE RESIDUAL FEATURES ===
df["residual"] = np.nan
df["residual_lag_1"] = 0.0
df["residual_avg_3"] = 0.0

# === DROP ROWS WITH MISSING LAGS/TARGET ===
required = lag_features + ["log_return_t+5_final"]
df.dropna(subset=required, inplace=True)
df.reset_index(drop=True, inplace=True)

# === MODEL BUILDER ===
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

# === ROLLING TRAINING & VALIDATION ===
train_window    = 3000
val_window      = 20
sequence_length = 10
noise_std       = 0.3
target_col      = "log_return_t+5_final"

predictions = []
actuals     = []

for start_idx in range(train_window, len(df) - val_window):
    # --- clear TF session to avoid memory bloat ---
    K.clear_session()
    
    # --- prepare in-sample residuals ---
    df["residual_for_cluster"] = df["residual"].fillna(0.0)
    
    # --- define base feature set ---
    base_features = lag_features + ["residual_lag_1", "residual_avg_3"]
    
    # --- slice training window ---
    train_df = df.iloc[start_idx - train_window : start_idx].copy()
    
    # --- fit scaler on training features only ---
    scaler = StandardScaler().fit(train_df[base_features])
    
    # --- fit k-means on in-sample residuals ---
    kmeans = KMeans(n_clusters=3, random_state=42).fit(
        train_df[["residual_for_cluster"]]
    )
    
    # --- build training sequences ---
    X_train, y_train = [], []
    for i in range(sequence_length, len(train_df)):
        seq_df = train_df.iloc[i - sequence_length : i]
        # scale base
        seq_base = scaler.transform(seq_df[base_features])
        # cluster per timestep
        labels = kmeans.predict(seq_df[["residual_for_cluster"]])
        clust_df = pd.get_dummies(labels, prefix="err_clust")
        # ensure all 3 cluster cols exist
        for c in [f"err_clust_{j}" for j in range(3)]:
            if c not in clust_df:
                clust_df[c] = 0
        clust_arr = clust_df[[f"err_clust_{j}" for j in range(3)]].values
        # combine
        seq = np.hstack([seq_base, clust_arr])
        X_train.append(seq)
        # add noise for residual feedback
        y_train.append(train_df[target_col].iloc[i] + np.random.normal(0, noise_std))
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    # --- build validation sequences (next 20 bars) ---
    X_val, y_val = [], []
    for idx in range(start_idx, start_idx + val_window):
        seq_df = df.iloc[idx - sequence_length : idx]
        seq_base = scaler.transform(seq_df[base_features])
        labels   = kmeans.predict(seq_df[["residual_for_cluster"]])
        clust_df = pd.get_dummies(labels, prefix="err_clust")
        for c in [f"err_clust_{j}" for j in range(3)]:
            if c not in clust_df:
                clust_df[c] = 0
        clust_arr = clust_df[[f"err_clust_{j}" for j in range(3)]].values
        seq = np.hstack([seq_base, clust_arr])
        X_val.append(seq)
        y_val.append(df.loc[idx, target_col])
    
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    
    if X_train.size == 0 or X_val.size == 0:
        continue
    
    # --- train & predict ---
    model = build_model((sequence_length, X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    y_pred = model.predict(X_val, verbose=0).flatten()
    predictions.extend(y_pred.tolist())
    actuals.extend(y_val.tolist())
    
    # --- update residual feedback features ---
    for j, idx in enumerate(range(start_idx, start_idx + val_window)):
        resid = y_val[j] - y_pred[j]
        df.loc[idx, "residual"]       = resid
        df.loc[idx, "residual_lag_1"] = df.loc[idx - 1, "residual"] if idx > 0 else 0.0
        df.loc[idx, "residual_avg_3"] = df["residual"].iloc[max(idx - 3, 0) : idx].mean()

# === FINAL EVALUATION ===
rmse = np.sqrt(mean_squared_error(actuals, predictions))
print(f"\n✅ Final Rolling RMSE (on Z-scored target): {rmse:.4f}")

# === PLOT RESULTS ===
plt.figure(figsize=(14, 5))
plt.plot(actuals,   label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("Rolling Validation: 20-Bar Forecasts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

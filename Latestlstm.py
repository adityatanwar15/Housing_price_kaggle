# === COMPLETE COLAB‐READY SCRIPT WITH ROLLING‐WINDOW & OPTUNA HYPERPARAMETER TUNING ===

# === SETUP ===
!pip install -q pandas numpy scikit-learn tensorflow matplotlib plotly optuna

import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional,
    LSTM, Dropout, Dense, Attention, GlobalAveragePooling1D
)
from google.colab import files
import plotly.graph_objs as go

# === UPLOAD & READ DATA ===
print("📁 Please upload your ESc1_2025.csv file")
uploaded = files.upload()
df = pd.read_csv(next(iter(uploaded)))
df["date-time"] = pd.to_datetime(df["date-time"])
df.sort_values("date-time", inplace=True)
df.reset_index(drop=True, inplace=True)

# === TARGET CONSTRUCTION ===
df["log_ret5"] = np.log(df["close"].shift(-5) / df["close"])
df["log_ret5_smooth"] = df["log_ret5"].rolling(3, center=True, min_periods=1).mean()
q_low, q_high = df["log_ret5_smooth"].quantile([0.01, 0.99])
df["log_ret5_clip"] = df["log_ret5_smooth"].clip(q_low, q_high)
scaler_t = StandardScaler().fit(df[["log_ret5_clip"]].dropna())
df["target"] = scaler_t.transform(df[["log_ret5_clip"]])

# === FEATURE ENGINEERING ===
df["spread"]    = df["bid size"] - df["ask size"]
df["imbalance"] = (df["bid size"] - df["ask size"]) / (df["bid size"] + df["ask size"])

max_lag = 5
feature_cols = []
for lag in range(1, max_lag + 1):
    for col in ["close", "vwap", "volume", "spread", "imbalance"]:
        name = f"{col}_lag_{lag}"
        df[name] = df[col].shift(lag)
        feature_cols.append(name)

df.dropna(subset=feature_cols + ["target"], inplace=True)
df.reset_index(drop=True, inplace=True)

# === ROLLING WINDOW SETTINGS ===
train_window = 3000
val_window   = 20
seq_len      = 10

# sequence builder
def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# hypermodel builder using trial hyperparams
def build_model_hp(trial, input_shape):
    conv_filters   = trial.suggest_categorical("conv_filters", [32, 64])
    lstm_units     = trial.suggest_categorical("lstm_units", [32, 64])
    dropout_rate   = trial.suggest_float("dropout_rate", 0.1, 0.3, step=0.1)
    learning_rate  = trial.suggest_categorical("learning_rate", [1e-4, 1e-3])
    
    inp = Input(shape=input_shape)
    x = Conv1D(conv_filters, 3, padding="same", activation="relu")(inp)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True, recurrent_dropout=dropout_rate)
    )(x)
    att = Attention()([x, x])
    x   = GlobalAveragePooling1D()(att)
    x   = Dropout(dropout_rate)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    
    model = Model(inp, out)
    opt   = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=opt, loss="huber")
    return model

# objective for Optuna
def objective(trial):
    tf.keras.backend.clear_session()
    errors = []
    # walk-forward by val_window
    for start in range(train_window, len(df) - val_window, val_window):
        train_df = df.iloc[start - train_window : start]
        val_df   = df.iloc[start : start + val_window]
        
        # scale features on train
        scaler = StandardScaler().fit(train_df[feature_cols])
        X_tr_base = scaler.transform(train_df[feature_cols])
        X_val_base= scaler.transform(val_df[feature_cols])
        y_tr_base = train_df["target"].values
        y_val_base= val_df["target"].values
        
        # build sequences
        X_tr_seq, y_tr_seq = build_sequences(X_tr_base, y_tr_base, seq_len)
        X_val_seq, y_val_seq= build_sequences(X_val_base, y_val_base, seq_len)
        if len(X_val_seq) == 0:
            continue
        
        # build & train model
        model_hp = build_model_hp(trial, (seq_len, X_tr_seq.shape[2]))
        model_hp.fit(X_tr_seq, y_tr_seq, epochs=5, batch_size=32, verbose=0)
        
        # predict and inverse-transform to log-return
        y_pred_seq = model_hp.predict(X_val_seq, verbose=0).flatten()
        y_pred_clip= scaler_t.inverse_transform(y_pred_seq.reshape(-1,1)).flatten()
        y_true_clip= scaler_t.inverse_transform(y_val_seq.reshape(-1,1)).flatten()
        
        # RMSE for this window
        rmse_win = mean_squared_error(y_true_clip, y_pred_clip, squared=False)
        errors.append(rmse_win)
    
    return np.mean(errors)

# run hyperparameter tuning
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best hyperparameters:", study.best_params)

# === FINAL ROLLING PREDICTION WITH BEST HYPERPARAMS ===
best = study.best_params
def build_model_final(input_shape):
    inp = Input(shape=input_shape)
    x = Conv1D(best["conv_filters"], 3, padding="same", activation="relu")(inp)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(
        LSTM(best["lstm_units"], return_sequences=True, recurrent_dropout=best["dropout_rate"])
    )(x)
    att = Attention()([x, x])
    x   = GlobalAveragePooling1D()(att)
    x   = Dropout(best["dropout_rate"])(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    opt   = tf.keras.optimizers.Adam(learning_rate=best["learning_rate"], clipnorm=1.0)
    model.compile(optimizer=opt, loss="huber")
    return model

# perform rolling predictions on price
pred_prices, true_prices = [], []
for start in range(train_window, len(df) - val_window, val_window):
    tf.keras.backend.clear_session()
    train_df = df.iloc[start - train_window : start]
    val_df   = df.iloc[start : start + val_window]
    
    scaler = StandardScaler().fit(train_df[feature_cols])
    X_tr_base = scaler.transform(train_df[feature_cols])
    X_val_base= scaler.transform(val_df[feature_cols])
    y_tr_base = train_df["target"].values
    y_val_base= val_df["target"].values
    
    X_tr_seq, y_tr_seq = build_sequences(X_tr_base, y_tr_base, seq_len)
    X_val_seq, y_val_seq= build_sequences(X_val_base, y_val_base, seq_len)
    if len(X_val_seq) == 0:
        continue
    
    model_f = build_model_final((seq_len, X_tr_seq.shape[2]))
    model_f.fit(X_tr_seq, y_tr_seq, epochs=20, batch_size=32, verbose=0)
    
    y_pred_seq = model_f.predict(X_val_seq, verbose=0).flatten()
    y_pred_clip= scaler_t.inverse_transform(y_pred_seq.reshape(-1,1)).flatten()
    y_true_clip= scaler_t.inverse_transform(y_val_seq.reshape(-1,1)).flatten()
    
    idxs = np.arange(start + seq_len, start + seq_len + len(y_pred_clip))
    base_prices = df.loc[idxs, "close"].values
    pred_price  = base_prices * np.exp(y_pred_clip)
    true_price  = base_prices * np.exp(y_true_clip)
    
    pred_prices.extend(pred_price)
    true_prices.extend(true_price)

# metrics on price
mae_price = mean_absolute_error(true_prices, pred_prices)
print(f"\n✅ Final Rolling MAE on Price: {mae_price:.4f}")

# append & export
df["pred_price"]  = np.nan
df["actual_price"]= np.nan
pred_start = train_window + seq_len
indices = np.arange(pred_start, pred_start + len(pred_prices))
df.loc[indices, "pred_price"]   = pred_prices
df.loc[indices, "actual_price"] = true_prices

output = "ESc1_2025_price_preds_optuna.csv"
df.to_csv(output, index=False)
files.download(output)

# visualize
fig = go.Figure()
fig.add_trace(go.Scatter(y=true_prices, mode="lines", name="True Price"))
fig.add_trace(go.Scatter(y=pred_prices, mode="lines", name="Predicted Price"))
fig.update_layout(
    title="Rolling Price Predictions with Optimized CNN-BiLSTM-Attn",
    xaxis_title="Step",
    yaxis_title="Price"
)
fig.show()

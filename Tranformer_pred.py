# === COMPLETE COLAB‐READY SCRIPT WITH RAW‐PARAM MDN HEAD & ROLLING VALIDATION ===

# === SETUP ===
!pip install -q pandas numpy scikit-learn tensorflow matplotlib plotly tensorflow-probability

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K, layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objs as go
from google.colab import files

tfd = tfp.distributions

# === UPLOAD & READ DATA ===

df = pd.read_csv("ESc1_2025.csv)
df["date-time"] = pd.to_datetime(df["date-time"])
df.sort_values("date-time", inplace=True)
df.reset_index(drop=True, inplace=True)

# === TARGET: clipped & normalized log-return for t+5 ===
df["log_ret5"] = np.log(df["close"].shift(-5) / df["close"])
df["log_ret5_smooth"] = df["log_ret5"].rolling(3, center=True, min_periods=1).mean()
q_low, q_high = df["log_ret5_smooth"].quantile([0.01, 0.99])
df["log_ret5_clip"] = df["log_ret5_smooth"].clip(q_low, q_high)
scaler_t = StandardScaler().fit(df[["log_ret5_clip"]].dropna())
df["target"] = scaler_t.transform(df[["log_ret5_clip"]])

# === FEATURES: order-book + lagged raw ===
df["spread"]    = df["ask size"] - df["bid size"]
df["imbalance"] = (df["bid size"] - df["ask size"]) / (df["bid size"] + df["ask size"])
max_lag = 5
feature_cols = []
for lag in range(1, max_lag+1):
    for col in ["close","vwap","volume","spread","imbalance"]:
        name = f"{col}_lag_{lag}"
        df[name] = df[col].shift(lag)
        feature_cols.append(name)

df.dropna(subset=feature_cols + ["target"], inplace=True)
df.reset_index(drop=True, inplace=True)

# === ROLLING VALIDATION SETTINGS ===
train_window    = 3000
val_window      = 20
sequence_length = 10
num_components  = 3  # MDN components

# === SEQUENCE BUILDER ===
def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# === MDN LOSS FUNCTION ===
def mdn_loss(num_components):
    def loss(y_true, params):
        # params shape: (batch, num_components * 3)
        logits = params[..., :num_components]
        means  = params[..., num_components:2*num_components]
        sigma_raw = params[..., 2*num_components:]
        scales = tf.nn.softplus(sigma_raw) + 1e-6
        cat = tfd.Categorical(logits=logits)
        comps = tfd.Normal(loc=means, scale=scales)
        mixture = tfd.MixtureSameFamily(mixture_distribution=cat,
                                        components_distribution=comps)
        return -mixture.log_prob(y_true)
    return loss

# === MODEL BUILDER WITH RAW‐PARAM MDN HEAD ===
def build_model_mdn(input_shape):
    inp = layers.Input(shape=input_shape)
    # CNN → BiLSTM → Attention trunk
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, recurrent_dropout=0.2))(x)
    att = layers.Attention()([x, x])
    x = layers.GlobalAveragePooling1D()(att)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    # raw MDN params
    out = layers.Dense(num_components * 3)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss=mdn_loss(num_components))
    return model

# === ROLLING TRAIN & PREDICT ===
pred_price_all = []
true_price_all = []

for start in range(train_window, len(df) - val_window):
    K.clear_session()
    train_df = df.iloc[start - train_window:start]
    val_df   = df.iloc[start:start + val_window]

    scaler_f = StandardScaler().fit(train_df[feature_cols])
    X_train_base = scaler_f.transform(train_df[feature_cols])
    X_val_base   = scaler_f.transform(val_df[feature_cols])
    y_train_base = train_df["target"].values
    y_val_base   = val_df["target"].values

    X_tr_seq, y_tr_seq = build_sequences(X_train_base, y_train_base, sequence_length)
    X_va_seq, y_va_seq = build_sequences(X_val_base,   y_val_base,   sequence_length)
    if len(X_va_seq) == 0:
        continue

    # train MDN model
    model_mdn = build_model_mdn((sequence_length, len(feature_cols)))
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model_mdn.fit(X_tr_seq, y_tr_seq, epochs=10, batch_size=32, verbose=0, callbacks=[es])

    # predict params and build mixture
    params_pred = model_mdn.predict(X_va_seq, verbose=0)
    logits = params_pred[:, :num_components]
    means  = params_pred[:, num_components:2*num_components]
    sigma_raw = params_pred[:, 2*num_components:]
    scales = np.log1p(np.exp(sigma_raw)) + 1e-6

    mixture = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.Normal(loc=means, scale=scales)
    )
    y_pred_norm = mixture.mean().numpy().flatten()
    y_true_norm = y_va_seq

    # inverse transform to clipped log-return
    y_pred_clip = scaler_t.inverse_transform(y_pred_norm.reshape(-1,1)).flatten()
    y_true_clip = scaler_t.inverse_transform(y_true_norm.reshape(-1,1)).flatten()

    # convert to price
    idxs = np.arange(start + sequence_length, start + sequence_length + len(y_pred_clip))
    base_prices = df.loc[idxs, "close"].values
    pred_price  = base_prices * np.exp(y_pred_clip)
    true_price  = base_prices * np.exp(y_true_clip)

    pred_price_all.extend(pred_price)
    true_price_all.extend(true_price)

# === FINAL METRICS & EXPORT ===
mae_price = mean_absolute_error(true_price_all, pred_price_all)
print(f"\n✅ Rolling MAE on Price with MDN: {mae_price:.4f}")

# append & save
df["pred_price_mdn"] = np.nan
df["true_price_mdn"] = np.nan
indices = np.arange(train_window + sequence_length, train_window + sequence_length + len(pred_price_all))
df.loc[indices, "pred_price_mdn"] = pred_price_all
df.loc[indices, "true_price_mdn"] = true_price_all

out_path = "ESc1_2025_price_preds_mdn.csv"
df.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
try:
    files.download(out_path)
except:
    pass

# === VISUALIZE ===
fig = go.Figure()
fig.add_trace(go.Scatter(y=true_price_all, mode="lines", name="Actual"))
fig.add_trace(go.Scatter(y=pred_price_all, mode="lines", name="Predicted"))
fig.update_layout(title="Rolling Price Predictions with MDN Head",
                  xaxis_title="Step", yaxis_title="Price")
fig.show()

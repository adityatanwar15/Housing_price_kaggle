import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.base import clone
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
file_path = 'ESc1_2025.csv'
model_dir = "models_every_step"
os.makedirs(model_dir, exist_ok=True)

train_window = 3000
max_predictions = 300
base_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# === LOAD & CLEAN ===
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower()
df['date-time'] = pd.to_datetime(df['date-time'])
df = df.sort_values('date-time').reset_index(drop=True)

# === TARGETS ===
df['target_close'] = df['close'].shift(-1)
df['target_high'] = df['high'].shift(-1)
df['target_low'] = df['low'].shift(-1)

# === FEATURE ENGINEERING ===
for lag in range(1, 4):
    if lag == 1:
        span = 5
        df[f'high_lag_{lag}'] = df['high'].shift(lag)
        df[f'ewma_low_{span}'] = df['low'].ewm(span=span).mean()
        df[f'ewma_close_{span}'] = df['close'].ewm(span=span).mean()
    elif lag == 2:
        span = 10
        df[f'ewma_low_{span}'] = df['low'].ewm(span=span).mean()
        df[f'ewma_high_{span}'] = df['high'].ewm(span=span).mean()
        df[f'rolling_low_mean_{span}'] = df['low'].rolling(window=span).mean()
    elif lag == 3:
        span = 20
        df[f'rolling_high_mean_{span}'] = df['high'].rolling(window=span).mean()
        df[f'rolling_low_mean_{span}'] = df['low'].rolling(window=span).mean()
        df[f'ewma_high_{span}'] = df['high'].ewm(span=span).mean()

df['hour'] = df['date-time'].dt.hour
df['dayofweek'] = df['date-time'].dt.dayofweek
df['rel_vol_10'] = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-9)
df['bid_momentum'] = df['bid size'] / (df['bid size'].rolling(window=10).mean() + 1e-9)
df['ask_momentum'] = df['ask size'] / (df['ask size'].rolling(window=10).mean() + 1e-9)
df['bidask_volume_ratio'] = (df['bid size'] - df['ask size']) / df['volume']
df['vwap_diff'] = df['vwap'] - df['close']

df = df.dropna().reset_index(drop=True)

features = [col for col in df.columns if (
    col.startswith(('rolling', 'ewma', 'rel_vol', 'bidask_volume_ratio', 'bid_', 'ask_', 'vwap_diff')) or
    col in ['hour', 'volume', 'vwap', 'dayofweek', 'close', 'high', 'low'])]

# === ROLLING PREDICTION ===
start_date = pd.Timestamp("2025-03-01", tz="UTC")
start_idx = df[df['date-time'] >= start_date].index[0]
df_full = df.iloc[start_idx - train_window:].reset_index(drop=True)

predictions, actuals, timestamps = [], [], []

for i in range(train_window, len(df_full)):
    row = df_full.iloc[i]
    if row['hour'] < 10:
        continue

    step = i - train_window
    if step >= max_predictions:
        break

    X_train = df_full.iloc[i - train_window:i][features]
    y_close = df_full.iloc[i - train_window:i]['target_close']
    y_high = df_full.iloc[i - train_window:i]['target_high']
    y_low = df_full.iloc[i - train_window:i]['target_low']
    X_test = row[features].values.reshape(1, -1)

    # === Train & Save Model ===
    model_close = clone(base_model).fit(X_train, y_close)
    model_high = clone(base_model).fit(X_train, y_high)
    model_low = clone(base_model).fit(X_train, y_low)

    joblib.dump(model_close, os.path.join(model_dir, f"model_close_{step}.pkl"))
    joblib.dump(model_high, os.path.join(model_dir, f"model_high_{step}.pkl"))
    joblib.dump(model_low, os.path.join(model_dir, f"model_low_{step}.pkl"))

    pred_close = model_close.predict(X_test)[0]
    pred_high = model_high.predict(X_test)[0]
    pred_low = model_low.predict(X_test)[0]

    predictions.append([pred_close, pred_high, pred_low])
    actuals.append([row['target_close'], row['target_high'], row['target_low']])
    timestamps.append(row['date-time'])

# === RESULT DF & METRICS ===
def metrics(true, pred):
    return {
        "MAE": mean_absolute_error(true, pred),
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "MAPE": mean_absolute_percentage_error(true, pred)
    }

result_df = pd.DataFrame(predictions, columns=['pred_close', 'pred_high', 'pred_low'])
result_df['actual_close'] = [x[0] for x in actuals]
result_df['actual_high'] = [x[1] for x in actuals]
result_df['actual_low'] = [x[2] for x in actuals]
result_df['time'] = timestamps

print("CLOSE:", metrics(result_df['actual_close'], result_df['pred_close']))
print("HIGH:", metrics(result_df['actual_high'], result_df['pred_high']))
print("LOW:", metrics(result_df['actual_low'], result_df['pred_low']))
print(result_df.tail(10))

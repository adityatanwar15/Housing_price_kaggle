import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")


# === LOAD ===
file_path = 'ESc1_2025.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}. Please check the path or upload the correct file.")

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower()
df['date-time'] = pd.to_datetime(df['date-time'])
df = df.sort_values('date-time').reset_index(drop=True)

# === TARGETS ===
df['target_close'] = df['close'].shift(-1)
df['target_high'] = df['high'].shift(-1)
df['target_low'] = df['low'].shift(-1)

# === FEATURE ENGINEERING ===
for lag in range(1, 3):
    if lag==1:
        span=5
        df[f'high_lag_{lag}'] = df['high'].shift(lag)
        df[f'ewma_low_{span}'] = df['low'].ewm(span=span).mean()
        df[f'ewma_close_{span}'] = df['close'].ewm(span=span).mean()
    elif lag==2:
        span=10
        df[f'ewma_low_{span}'] = df['low'].ewm(span=span).mean()
        df[f'ewma_high_{span}'] = df['high'].ewm(span=span).mean()
        df[f'ewma_low_{span}'] = df['low'].ewm(span=span).mean()
        df[f'rolling_low_mean_{span}'] = df['low'].rolling(window=span).mean()
        
    elif lag==3:
        span=20
        df[f'rolling_high_mean_{span}'] = df['high'].rolling(window=span).mean()
        df[f'rolling_low_mean_{span}'] = df['low'].rolling(window=span).mean()
        df[f'ewma_high_{span}'] = df['high'].ewm(span=span).mean()
    

    # span = lag * 5
    # df[f'ewma_close_{span}'] = df['close'].ewm(span=span).mean()
    # df[f'ewma_high_{span}'] = df['high'].ewm(span=span).mean()
    # df[f'ewma_low_{span}'] = df['low'].ewm(span=span).mean()
    # df[f'rolling_high_mean_{span}'] = df['high'].rolling(window=span).mean()
    # df[f'rolling_low_mean_{span}'] = df['low'].rolling(window=span).mean()

df['hour'] = df['date-time'].dt.hour
df['dayofweek'] = df['date-time'].dt.dayofweek

# === VOLUME / ORDER BOOK FEATURES ===
df['rel_vol_10'] = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-9)
df['bid_momentum'] = df['bid size'] / (df['bid size'].rolling(window=10).mean() + 1e-9)
df['ask_momentum'] = df['bid size'] / (df['bid size'].rolling(window=10).mean() + 1e-9)
df['bidask_volume_ratio'] = (df['bid size'] - df['ask size'] ) / (df['volume'])
df['vwap_diff'] = df['vwap'] - df['close']

df = df.dropna().reset_index(drop=True)

# === FEATURES ===
features = [col for col in df.columns if (
    col.startswith(('rolling', 'ewma', 'rel_vol', 'bidask_volume_ratio', 'bid_', 'ask_', 'vwap_diff')) or
    col in ['hour', 'volume', 'vwap','dayofweek','close','high','low'])]

# === MODEL LOOP ===
total_len = len(df)
start_idx = int(total_len * 0.3)
end_idx = total_len

predictions, actuals, timestamps = [], [], []
for i in range(start_idx, end_idx):
    if len(predictions) < 10000 and 10 <= df['hour'].iloc[i] < 24  :
        
        train_start = max(0, i - int(total_len * 0.3))
        X_train = df.iloc[train_start:i][features]
        X_test = df.iloc[i][features].values.reshape(1, -1)
        print(len(predictions))

        y_close = df.iloc[train_start:i]['target_close']
        y_high = df.iloc[train_start:i]['target_high']
        y_low = df.iloc[train_start:i]['target_low']

        model_close = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model_high = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model_low = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        model_close.fit(X_train, y_close)
        model_high.fit(X_train, y_high)
        model_low.fit(X_train, y_low)

        pred_close = model_close.predict(X_test)[0]
        pred_high = model_high.predict(X_test)[0]
        pred_low = model_low.predict(X_test)[0]

        predictions.append([pred_close, pred_high, pred_low])
        actuals.append([
            df.iloc[i]['target_close'],
            df.iloc[i]['target_high'],
            df.iloc[i]['target_low']
        ])
        timestamps.append(df.iloc[i]['date-time'])

# === RESULT DF ===
result_df = pd.DataFrame(predictions, columns=['pred_close', 'pred_high', 'pred_low'])
result_df['actual_close'] = [x[0] for x in actuals]
result_df['actual_high'] = [x[1] for x in actuals]
result_df['actual_low'] = [x[2] for x in actuals]
result_df['time'] = timestamps

# === METRICS ===
def metrics(true, pred):
    return {
        "MAE": mean_absolute_error(true, pred),
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "MAPE": mean_absolute_percentage_error(true, pred)
    }

print("CLOSE:", metrics(result_df['actual_close'], result_df['pred_close']))
print("HIGH:", metrics(result_df['actual_high'], result_df['pred_high']))
print("LOW:", metrics(result_df['actual_low'], result_df['pred_low']))

# === VISUALIZATIONS ===
result_df['error_close'] = result_df['pred_close'] - result_df['actual_close']
result_df['abs_error_close'] = result_df['error_close'].abs()
result_df['hour'] = result_df['time'].dt.hour
result_df['day'] = result_df['time'].dt.day_name()

# Plotly: Actual vs Predicted Close Price
fig = go.Figure()
fig.add_trace(go.Scatter(x=result_df['time'], y=result_df['actual_close'], mode='lines+markers', name='Actual Close'))
fig.add_trace(go.Scatter(x=result_df['time'], y=result_df['pred_close'], mode='lines+markers', name='Predicted Close'))
fig.update_layout(title='Actual vs Predicted Close Price', xaxis_title='Time', yaxis_title='Price', template='plotly_dark')
fig.show()

# Error distribution with Plotly
fig = px.histogram(result_df, x='error_close', nbins=20, title="Error Distribution (Actual - Predicted Close)", template='plotly_dark')
fig.show()

# Feature importance with Plotly
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model_close.feature_importances_
}).sort_values("Importance", ascending=False)

fig = px.bar(feature_importance.head(15), x="Importance", y="Feature", orientation='h', title="Top 15 Feature Importances (Close Prediction)", template='plotly_dark')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()

# Heatmap of MAE by hour and day using seaborn
pivot = result_df.pivot_table(index='day', columns='hour', values='abs_error_close', aggfunc='mean')
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=0.5)
plt.title("Heatmap of MAE (Close Price) by Day of Week and Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.show()

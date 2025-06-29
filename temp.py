

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# === LOAD & CLEAN ===
df = pd.read_excel("ESc1_new2.xlsx")
df.columns = df.columns.str.strip().str.lower()

df['date-time'] = pd.to_datetime(df['date-time'])
df = df.sort_values('date-time').reset_index(drop=True)

df['close'] = df['close']
df['high'] = df['high']
df['low'] = df['low']

# === TARGETS ===
df['target_close'] = df['close'].shift(-1)
df['target_high'] = df['high'].shift(-1)
df['target_low'] = df['low'].shift(-1)

# === LOG RETURNS ===
df['log_return_close'] = np.log(df['close'] / df['close'].shift(1))
df['log_return_high'] = np.log(df['high'] / df['high'].shift(1))
df['log_return_low'] = np.log(df['low'] / df['low'].shift(1))

# === LAG FEATURES ===

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
    
# === TIME FEATURES ===
df['hour'] = df['date-time'].dt.hour
df['dayofweek'] = df['date-time'].dt.dayofweek


#df = df[(df['hour'] >= 8) & (df['hour'] <= 21)]

# === CLEAN ===
df = df.dropna()

# === FEATURES ===
features = [col for col in df.columns if (
    col.startswith(('high_lag_', 'rolling', 'ewma')) or 
    col in ['hour', 'volume', 'bid size',  'vwap'])]

# === ROLLING MODEL SETUP ===
total_len = len(df)
start_idx = int(total_len * 0.6)
end_idx = int(total_len * 1)

predictions, actuals = [], []
m=0
start_t=datetime.now()
print(start_t)
for i in range(start_idx, end_idx):
    #print(i,)

    if m<25 :
        if int(df['hour'].iloc[i])>=12 and int(df['hour'].iloc[i])<22:
     #       print(i)
            
            train_start = max(0, i - int(total_len * 0.6))
            train_X = df.iloc[train_start:i][features]
            test_X = df.iloc[i][features].values.reshape(1, -1)

            train_y_close = df.iloc[train_start:i]['target_close']
            train_y_high = df.iloc[train_start:i]['target_high']
            train_y_low = df.iloc[train_start:i]['target_low']
            timestamp = df.iloc[i]['date-time']
            print(i,timestamp)

            # === MODELS ===
            model_close = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model_high = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model_low = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

            model_close.fit(train_X, train_y_close)
            model_high.fit(train_X, train_y_high)
            model_low.fit(train_X, train_y_low)

            pred_close = model_close.predict(test_X)[0]
            pred_high = model_high.predict(test_X)[0]
            pred_low = model_low.predict(test_X)[0]

            actual_close = df.iloc[i]['target_close']
            actual_high = df.iloc[i]['target_high']
            actual_low = df.iloc[i]['target_low']

            predictions.append([timestamp, pred_close, pred_high, pred_low])
            actuals.append([actual_close, actual_high, actual_low])
            m+=1

end_t=datetime.now()
print(end_t-start_t)


# === RESULTS ===
result_df = pd.DataFrame(predictions, columns=["time", "pred_close", "pred_high", "pred_low"])
result_df["actual_close"] = [a[0] for a in actuals]
result_df["actual_high"] = [a[1] for a in actuals]
result_df["actual_low"] = [a[2] for a in actuals]

# === METRICS ===
def metrics(true, pred):
    return {
        "MAE": mean_absolute_error(true, pred),
        "MSE": mean_squared_error(true, pred),
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "MAPE": mean_absolute_percentage_error(true, pred)
        }

print("CLOSE METRICS:", metrics(result_df["actual_close"], result_df["pred_close"]))
print("HIGH METRICS:", metrics(result_df["actual_high"], result_df["pred_high"]))
print("LOW METRICS:", metrics(result_df["actual_low"], result_df["pred_low"]))

# === SAVE ===
#result_df.to_csv("rolling_predictionnhan11ced.csv", index=False)

# === PLOTLY VISUALS ===
def plot_predictions(df, col_actual, col_predicted, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df[col_actual], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df["time"], y=df[col_predicted], mode='lines', name='Predicted'))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price', template='plotly_dark')
    fig.show()

plot_predictions(result_df, "actual_close", "pred_close", "Actual vs Predicted Close Price")
plot_predictions(result_df, "actual_high", "pred_high", "Actual vs Predicted High Price")
plot_predictions(result_df, "actual_low", "pred_low", "Actual vs Predicted Low Price")

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    fi = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(20)

    fig = px.bar(fi, x="Importance", y="Feature", orientation='h', title=title, template='plotly_dark')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.show()

plot_feature_importance(model_close, features, "Feature Importance - Close Price Model")

# === OPTIONAL: ERROR DISTRIBUTION ===
result_df["error_close"] = result_df["actual_close"] - result_df["pred_close"]
fig = px.histogram(result_df, x="error_close", nbins=50, title="Error Distribution: Close Prediction", template='plotly_dark')
fig.show()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import xlwings as xw
import warnings
warnings.filterwarnings("ignore")

# === FUNCTION TO LOAD & CLEAN DATA ===
def load_and_clean_data(sheet):
    df = pd.DataFrame(sheet.range("A1").options(pd.DataFrame, header=1, index=False).value)  # Load data from Excel
    df.columns = df.columns.str.strip().str.lower()
    df['date-time'] = pd.to_datetime(df['date-time'])
    df = df.sort_values('date-time').reset_index(drop=True)  # Sort data by date-time
    return df

# === FUNCTION TO MAKE PREDICTIONS ===
def make_predictions(df, features):
    predictions = []
    train_data = df.iloc[-3000:]  # Take the last 3000 values for training

    # Train models
    model_close = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_high = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_low = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    # Prepare training data
    train_X = train_data[features]
    train_y_close = train_data['target_close']
    train_y_high = train_data['target_high']
    train_y_low = train_data['target_low']

    model_close.fit(train_X, train_y_close)
    model_high.fit(train_X, train_y_high)
    model_low.fit(train_X, train_y_low)

    # Use the complete dataset for predictions
    for i in range(len(df)):
        last_row = df.iloc[i]  # Get the current row for prediction
        test_X = last_row[features].values.reshape(1, -1)

        # Predict the next 5 values
        for _ in range(5):
            pred_close = model_close.predict(test_X)[0]
            pred_high = model_high.predict(test_X)[0]
            pred_low = model_low.predict(test_X)[0]

            predictions.append([last_row['date-time'], pred_close, pred_high, pred_low])

            # Update test_X for the next prediction
            new_row = np.array([pred_close, pred_high, pred_low] + list(test_X[0][3:]))  # Shift the features
            test_X = new_row.reshape(1, -1)

    return predictions

# === MAIN LOOP ===
file_path = "ESc1_15T.xlsx"  # Path to your Excel file
wb = xw.Book(file_path)  # Open the Excel workbook
sheet = wb.sheets[0]  # Select the first sheet

# Load and clean data
df = load_and_clean_data(sheet)

# Create target variables
df['target_close'] = df['close'].shift(-1)
df['target_high'] = df['high'].shift(-1)
df['target_low'] = df['low'].shift(-1)

# Calculate log returns
df['log_return_close'] = np.log(df['close'] / df['close'].shift(1))
df['log_return_high'] = np.log(df['high'] / df['high'].shift(1))
df['log_return_low'] = np.log(df['low'] / df['low'].shift(1))

# Create lag features
for lag in range(1, 6):
    df[f'close_lag_{lag}'] = df['close'].shift(lag)
    df[f'return_lag_{lag}'] = df['log_return_close'].shift(lag)
    df[f'high_lag_{lag}'] = df['high'].shift(lag)
    df[f'low_lag_{lag}'] = df['low'].shift(lag)

# Create rolling statistics
for w in [5, 10, 20]:
    df[f'rolling_close_mean_{w}'] = df['close'].rolling(window=w).mean()
    df[f'rolling_close_std_{w}'] = df['close'].rolling(window=w).std()
    df[f'rolling_high_mean_{w}'] = df['high'].rolling(window=w).mean()
    df[f'rolling_high_std_{w}'] = df['high'].rolling(window=w).std()
    df[f'rolling_low_mean_{w}'] = df['low'].rolling(window=w).mean()
    df[f'rolling_low_std_{w}'] = df['low'].rolling(window=w).std()

# Create EWMA
for span in [5, 10, 20]:
    df[f'ewma_close_{span}'] = df['close'].ewm(span=span).mean()
    df[f'ewma_high_{span}'] = df['high'].ewm(span=span).mean()
    df[f'ewma_low_{span}'] = df['low'].ewm(span=span).mean()

# Create time features
df['hour'] = df['date-time'].dt.hour
df['dayofweek'] = df['date-time'].dt.dayofweek

# Clean data
df = df.dropna()

# Define features for the model
features = [f'close_lag_{lag}' for lag in range(1, 6)] + \
           [f'return_lag_{lag}' for lag in range(1, 6)] + \
           [f'high_lag_{lag}' for lag in range(1, 6)] + \
           [f'low_lag_{lag}' for lag in range(1, 6)] + \
           [f'rolling_close_mean_{w}' for w in [5, 10, 20]] + \
           [f'rolling_high_mean_{w}' for w in [5, 10, 20]] + \
           [f'rolling_low_mean_{w}' for w in [5, 10, 20]] + \
           [f'ewma_close_{span}' for span in [5, 10, 20]] + \
           [f'ewma_high_{span}' for span in [5, 10, 20]] + \
           [f'ewma_low_{span}' for span in [5, 10, 20]] + \
           ['hour', 'dayofweek']

# Make predictions
predictions = make_predictions(df, features)

# Convert predictions to DataFrame
result_df = pd.DataFrame(predictions, columns=["time", "pred_close", "pred_high", "pred_low"])

# Print predictions
print("Predictions for the next 5 candles:")
print(result_df)

# === PLOT PREDICTIONS ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=result_df["time"], y=result_df["pred_close"], mode='lines', name='Predicted Close'))
fig.add_trace(go.Scatter(x=result_df["time"], y=result_df["pred_high"], mode='lines', name='Predicted High'))
fig.add_trace(go.Scatter(x=result_df["time"], y=result_df["pred_low"], mode='lines', name='Predicted Low'))
fig.update_layout(title='Predicted Prices for Next 5 Candles', xaxis_title='Time', yaxis_title='Price', template='plotly_dark')
fig.show()

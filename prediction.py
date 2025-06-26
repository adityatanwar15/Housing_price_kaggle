import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import xlwings as xw
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# === FUNCTION TO LOAD & CLEAN DATA ===
def load_and_clean_data(sheet):
    try:
        # Attempt to read data from the specified range
        data = sheet.used_range.options(pd.DataFrame, header=1, index=False).value
        if data is None or data.empty:
            raise ValueError("No data found in the specified range.")
        
        df = pd.DataFrame(data)  # Load data into a DataFrame
        df.columns = df.columns.str.strip().str.lower()  # Clean column names
        df['date-time'] = pd.to_datetime(df['date-time'])  # Convert date-time column
        df = df.sort_values('date-time').reset_index(drop=True)  # Sort data by date-time
        return df
    except Exception as e:
        print(f"Error reading data from Excel: {e}")
        return None

# === FUNCTION TO WAIT UNTIL NEXT QUARTER CHANGE ===
def wait_until_next_quarter():
    now = datetime.now()
    next_quarter = (now.minute // 15 + 1) * 15 % 60
    next_hour = now.hour + (now.minute // 15 + 1) // 4
    if next_hour == 24:
        next_hour = 0
    next_time = now.replace(hour=next_hour, minute=next_quarter, second=0, microsecond=0)
    if next_time < now:
        next_time += timedelta(hours=1)
    return next_time

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
            print(last_row['date-time'], pred_close, pred_high, pred_low)

            predictions.append([last_row['date-time'], pred_close, pred_high, pred_low])

            # Update test_X for the next prediction
            new_row = np.array([pred_close, pred_high, pred_low] + list(test_X[0][3:]))  # Shift the features
            test_X = new_row.reshape(1, -1)

    return predictions

# === MAIN LOOP ===
file_path = "ESc1_15T_v2.xlsx"  # Path to your Excel file
try:
    wb = xw.Book(file_path)  # Open the Excel workbook
    sheet = wb.sheets[0]  # Select the first sheet
except Exception as e:
    print(f"Error opening Excel file: {e}")
    exit()

# Main loop for predictions every 15 minutes after quarter change
while True:
    # Wait until the next quarter change
    next_time = wait_until_next_quarter()
    time_to_wait = (next_time - datetime.now()).total_seconds()

    # Countdown until the next prediction
    while time_to_wait > 0:
        current_time = datetime.now()
        print(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | Time Remaining for Next Prediction: {int(time_to_wait)} seconds", end='\r')
        time.sleep(1)  # Update every second
        time_to_wait -= 1

    # Load and clean data
    df = load_and_clean_data(sheet)
    if df is None:
        print("Failed to load data. Exiting.")
        exit()

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

    # Print the current time
    current_time = datetime.now()
    print(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

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

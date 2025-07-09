# quarter_hour_predictor_top3000.py

import time
import datetime
import pandas as pd
import numpy as np
import xlwings as xw
from sklearn.linear_model import MultiTaskLasso

# === CONFIG ===
EXCEL_FILE  = r"C:\path\to\your\ESc1_2025.xlsx"
SHEET_NAME   = "Sheet1"
MODEL_LAGS   = 5
LASSO_ALPHA  = 0.01
WINDOW_SIZE  = 3000   # take the top 3000 (newest) rows each run

# === DATA I/O ===
def read_data_top3000():
    wb  = xw.Book(EXCEL_FILE)
    sht = wb.sheets[SHEET_NAME]
    # read the full table
    df_full = sht.range("A1").options(
        pd.DataFrame, header=1, index=False, expand='table'
    ).value
    df_full['date-time'] = pd.to_datetime(df_full['date-time'])
    # take top WINDOW_SIZE rows (newest first), then sort ascending
    df_win = df_full.head(WINDOW_SIZE).copy()
    df_win.sort_values('date-time', inplace=True)
    df_win.reset_index(drop=True, inplace=True)
    return df_win

# === MODEL HELPERS ===
def create_lag_matrix(df_diff, imbalance, p):
    X, Y = [], []
    for t in range(p, len(df_diff)):
        X.append(np.concatenate([
            df_diff.values[t-p:t].flatten(),
            [imbalance.iloc[t-1]]
        ]))
        Y.append(df_diff.values[t])
    return np.array(X), np.array(Y)

def compute_lasso_var_prediction(df, p=MODEL_LAGS, alpha=LASSO_ALPHA):
    # feature engineering
    df = df.copy()
    df['imbalance'] = (df['bid size'] - df['ask size']) / (df['bid size'] + df['ask size'])
    diff_vars = ['low','close','high','volume']
    df_diff = df[diff_vars].diff().dropna().reset_index(drop=True)
    imb    = df['imbalance'].iloc[1:].reset_index(drop=True)
    ts     = df['date-time'].iloc[1:].reset_index(drop=True)
    # build training matrix
    X, Y = create_lag_matrix(df_diff, imb, p)
    model = MultiTaskLasso(alpha=alpha, max_iter=10000)
    model.fit(X, Y)
    # predict next diff
    last_block = np.concatenate([df_diff.values[-p:].flatten(), [imb.iloc[-1]]])
    d_pred     = model.predict(last_block.reshape(1,-1))[0]
    # compute forecast timestamp as last_ts + (last_ts - prev_ts)
    last_ts  = ts.iloc[-1]
    prev_ts  = ts.iloc[-2]
    pred_ts  = last_ts + (last_ts - prev_ts)
    # invert diffs to price space
    base = df.iloc[-1]
    return pred_ts, (
        base['low']   + d_pred[0],
        base['close'] + d_pred[1],
        base['high']  + d_pred[2]
    )

# === SCHEDULER ===
def seconds_until_next_quarter():
    now = datetime.datetime.now()
    next_min = (now.minute // 15 + 1) * 15
    if next_min >= 60:
        wake = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
    else:
        wake = now.replace(minute=next_min, second=0, microsecond=0)
    return (wake - now).total_seconds()

def main_loop():
    print("Starting quarter-hour predictor (top 3000 rows)...")
    while True:
        wait = seconds_until_next_quarter()
        time.sleep(wait)
        try:
            df_win = read_data_top3000()
            pred_ts, (low_p, close_p, high_p) = compute_lasso_var_prediction(df_win)
            now_str  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ts_str   = pred_ts.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now_str} → Forecast for {ts_str} | "
                  f"Low: {low_p:.4f}, Close: {close_p:.4f}, High: {high_p:.4f}")
        except Exception as e:
            print(f"Error during prediction at {datetime.datetime.now()}: {e}")

if __name__ == "__main__":
    main_loop()

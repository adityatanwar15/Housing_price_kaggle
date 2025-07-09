# Colab-ready Lasso-VAR backtest conditioned on hour > 10

# --- Install & import ---
!pip install statsmodels scikit-learn --quiet

import pandas as pd
import numpy as np
from sklearn.linear_model import MultiTaskLasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Upload / load data ---
# from google.colab import files
# uploaded = files.upload()  # uncomment to upload ESc1_2025.csv

df = pd.read_csv('ESc1_2025.csv', parse_dates=['date-time'])
data = df[['date-time','low','close','high','volume','bid size','ask size']].copy()

# --- Feature engineering ---
data['imbalance'] = (
    data['bid size'] - data['ask size']
) / (
    data['bid size'] + data['ask size']
)
diff_vars = ['low','close','high','volume']
data_diff = data[diff_vars].diff().dropna()
imbalance = data.loc[data_diff.index, 'imbalance']

def create_lag_matrix(df_diff, imbalance, p):
    X, Y = [], []
    for t in range(p, len(df_diff)):
        X.append(np.concatenate([
            df_diff.values[t-p:t].flatten(),
            [imbalance.values[t-1]]
        ]))
        Y.append(df_diff.values[t])
    return np.array(X), np.array(Y)

# --- Backtest conditional on hour > 10 ---
n_preds = 7500
p = 5
alpha = 0.01

trues_low, trues_close, trues_high = [], [], []
preds_low, preds_close, preds_high = [], [], []

i = 1
while len(trues_close) < n_preds and i <= len(data_diff):
    ts = data['date-time'].iloc[-i]
    if ts.hour > 10:
        print(i)
        # train up to index -i
        df_tr = data_diff.iloc[:-i]
        imb_tr = imbalance.iloc[:-i]
        X_tr, Y_tr = create_lag_matrix(df_tr, imb_tr, p)
        model = MultiTaskLasso(alpha=alpha, max_iter=10000)
        model.fit(X_tr, Y_tr)
        last_block = np.concatenate([
            df_tr.values[-p:].flatten(),
            [imb_tr.iloc[-1]]
        ])
        pred_diff = model.predict(last_block.reshape(1,-1))[0]

        base_idx = -i-1
        # invert diffs
        pred_l = data['low'].iloc[base_idx]   + pred_diff[0]
        pred_c = data['close'].iloc[base_idx] + pred_diff[1]
        pred_h = data['high'].iloc[base_idx]  + pred_diff[2]

        true_l = data['low'].iloc[-i]
        true_c = data['close'].iloc[-i]
        true_h = data['high'].iloc[-i]

        trues_low.append(true_l)
        trues_close.append(true_c)
        trues_high.append(true_h)
        preds_low.append(pred_l)
        preds_close.append(pred_c)
        preds_high.append(pred_h)
    i += 1

# reverse for chronological order
trues_low   = np.array(trues_low[::-1])
trues_close = np.array(trues_close[::-1])
trues_high  = np.array(trues_high[::-1])
preds_low   = np.array(preds_low[::-1])
preds_close = np.array(preds_close[::-1])
preds_high  = np.array(preds_high[::-1])

# --- Metrics for close ---
mse_c = mean_squared_error(trues_close, preds_close)
mae_c = mean_absolute_error(trues_close, preds_close)
print(f"Lasso-VAR (hour > 10) over {len(trues_close)} predictions → MSE: {mse_c:.4f}, MAE: {mae_c:.4f}\n")

# --- Results DataFrame ---
results = pd.DataFrame({
    'Actual Low':    trues_low,
    'Pred Low':      preds_low,
    'Actual Close':  trues_close,
    'Pred Close':    preds_close,
    'Actual High':   trues_high,
    'Pred High':     preds_high,
})
print("Predictions (only when hour > 10):")
print(results.to_string(index=False))

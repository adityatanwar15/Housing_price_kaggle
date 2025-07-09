import pandas as pd
import numpy as np
from sklearn.linear_model import MultiTaskLasso

# Load data
df = pd.read_csv('/mnt/data/ESc1_2025.csv', parse_dates=['date-time'])
# Use correct column names with spaces
data = df[['low','close','high','volume','bid size','ask size']].copy()

# Compute imbalance
data['imbalance'] = (data['bid size'] - data['ask size']) / (data['bid size'] + data['ask size'])

# Prepare differenced data
diff_vars = ['low','close','high','volume']
data_diff = data[diff_vars].diff().dropna()
imbalance = data['imbalance'].iloc[data_diff.index]

# Function to create lagged feature matrices
def create_lag_matrix(df_diff, imbalance, p):
    T = len(df_diff)
    X, Y = [], []
    for t in range(p, T):
        lag_vals = df_diff.values[t-p:t].flatten()
        imb_val = imbalance.values[t-1]
        X.append(np.concatenate([lag_vals, [imb_val]]))
        Y.append(df_diff.values[t])
    return np.array(X), np.array(Y)

p = 5  # lag order

# Baseline VAR backtest (OLS, no regularization)
def backtest_baseline(n_days, p):
    preds, trues = [], []
    for i in range(1, n_days+1):
        df_tr = data_diff.iloc[:-i]
        imb_tr = imbalance.iloc[:-i]
        X_tr, Y_tr = create_lag_matrix(df_tr, imb_tr, p)
        coef, _, _, _ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)
        last_lags = np.concatenate([df_tr.values[-p:].flatten(), [imb_tr.iloc[-1]]])
        y_pred_diff = last_lags.dot(coef)
        pred_price = data['close'].iloc[-i-1] + y_pred_diff[1]
        preds.append(pred_price)
        trues.append(data['close'].iloc[-i])
    return np.array(trues[::-1]), np.array(preds[::-1])

# Lasso-penalized VAR backtest
def backtest_lasso(n_days, p, alpha):
    preds, trues = [], []
    for i in range(1, n_days+1):
        df_tr = data_diff.iloc[:-i]
        imb_tr = imbalance.iloc[:-i]
        X_tr, Y_tr = create_lag_matrix(df_tr, imb_tr, p)
        model = MultiTaskLasso(alpha=alpha, max_iter=10000)
        model.fit(X_tr, Y_tr)
        last_lags = np.concatenate([df_tr.values[-p:].flatten(), [imb_tr.iloc[-1]]])
        y_pred_diff = model.predict(last_lags.reshape(1, -1))[0]
        pred_price = data['close'].iloc[-i-1] + y_pred_diff[1]
        preds.append(pred_price)
        trues.append(data['close'].iloc[-i])
    return np.array(trues[::-1]), np.array(preds[::-1])

# Run backtests for the last 20 periods
n_days = 20
true_b, pred_b = backtest_baseline(n_days, p)
true_l, pred_l = backtest_lasso(n_days, p, alpha=0.01)

mae_b = np.mean(np.abs(true_b - pred_b))
mae_l = np.mean(np.abs(true_l - pred_l))

print(f"{mae_b:.4f}")
print(f"{mae_l:.4f}")

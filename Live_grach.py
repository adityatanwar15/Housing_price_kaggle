import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from arch import arch_model  # pip install arch
import time
from datetime import datetime

# --- 1) Configuration: use the same window for recent data and training ---
WINDOW_N = 6000  # number of most-recent rows to load & train on

# --- 2) Helpers ---
def get_recent_data(path: str, n: int = WINDOW_N) -> pd.DataFrame:
    """
    Load CSV (newest-first), take the top n rows (most recent),
    sort into chronological order, and index by datetime.
    """
    df = pd.read_csv(path, parse_dates=['date-time'])
    df_recent = df.iloc[:n].sort_values('date-time').set_index('date-time')
    return df_recent

def one_step_forecast(df_recent: pd.DataFrame) -> tuple:
    """
    Given df_recent of length WINDOW_N, fit VAR(5) and AR(5)-GARCH(1,1)
    on the last WINDOW_N diffs, and map the one-step-ahead diff
    forecasts back to price levels.
    """
    df_diff = df_recent[['low','close','high']].diff().dropna()
    train   = df_diff.iloc[-WINDOW_N:]
    # VAR(5)
    var_res = VAR(train).fit(5)
    var_fc  = var_res.forecast(train.values[-5:], steps=1)[0]
    # GARCH
    garch_fc = {}
    for s in ['low','close','high']:
        am = arch_model(train[s], mean='AR', lags=5, vol='Garch', p=1, q=1)
        res = am.fit(disp='off')
        garch_fc[s] = res.forecast(horizon=1).mean.iloc[-1,0]
    # map back to price
    last = df_recent[['low','close','high']].iloc[-1]
    var_pred    = pd.Series(var_fc,     index=['low','close','high']) + last
    gar_pred    = pd.Series(garch_fc) + last
    hybrid_pred = 0.5*(pd.Series(var_fc, index=['low','close','high']) + pd.Series(garch_fc)) + last
    return last, var_pred, gar_pred, hybrid_pred

def compute_next_signal_time() -> datetime:
    """
    Compute the datetime of the next quarter-hour mark.
    """
    now = datetime.now()
    next_minute = ((now.minute // 15) + 1) * 15
    if next_minute == 60:
        # roll over to next hour
        target = now.replace(hour=(now.hour+1)%24, minute=0, second=0, microsecond=0)
    else:
        target = now.replace(minute=next_minute, second=0, microsecond=0)
    return target

# --- 3) Main live loop ---
def main():
    path = 'ESc1_2025.csv'
    print("Starting live forecasts every quarter-hour. Press Ctrl+C to stop.")
    while True:
        # load & forecast
        df_recent       = get_recent_data(path)
        last_price, var_p, gar_p, hyb_p = one_step_forecast(df_recent)

        # print last actual bar
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{ts}] Last actual prices:")
        print(last_price.to_frame(name='Actual').T)

        # print predictions
        preds = pd.DataFrame({
            'VAR':    var_p,
            'GARCH':  gar_p,
            'Hybrid': hyb_p
        })
        print("Next-bar predictions:")
        print(preds)

        # compute and print next signal time
        next_ts = compute_next_signal_time()
        print("Next signal at:", next_ts.strftime('%Y-%m-%d %H:%M:%S'))

        # wait until that time
        sleep_secs = (next_ts - datetime.now()).total_seconds()
        if sleep_secs > 0:
            time.sleep(sleep_secs)

if __name__ == '__main__':
    main()

import numpy as np
from statsmodels.tsa.api import VAR
from arch import arch_model  # install via: pip install arch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import gc
import warnings
warnings.filterwarnings("ignore")
#from ace_tools import display_dataframe_to_user
# --- Load and prep data ---
df = pd.read_csv('ESc1_2025.csv', parse_dates=['date-time'])
df.set_index('date-time', inplace=True)
df_diff = df[['low', 'close', 'high']].diff().dropna()
# Parameters
training_window = 3000   # fixed-length training window
n_preds = 7000
all_positions = list(range(len(df_diff) - n_preds, len(df_diff)))
test_positions = [pos for pos in all_positions if df_diff.index[pos].hour > 10]
# Containers
timestamps = []
true_vals = {'low': [], 'close': [], 'high': []}
pred_var = {'low': [], 'close': [], 'high': []}
pred_garch = {'low': [], 'close': [], 'high': []}
pred_hybrid = {'low': [], 'close': [], 'high': []}
# Rolling-window backtest with fixed window & hour>10 filter
for pos in test_positions:
    start = pos - training_window
    if start < 0:
        continue
    train = df_diff.iloc[start:pos]
    # 1) VAR(5) forecast
    var_mod = VAR(train)
    var_res = var_mod.fit(5)
    var_fc = var_res.forecast(train.values[-5:], steps=1)[0]
    # 2) AR(5)-GARCH(1,1) for each series
    garch_fc = {}
    for series in ['low', 'close', 'high']:
        g_mod = arch_model(train[series], mean='AR', lags=5, vol='Garch', p=1, q=1)
        g_res = g_mod.fit(disp='off')
        garch_fc[series] = g_res.forecast(horizon=1).mean.iloc[-1, 0]
        del g_mod, g_res
    # Map back to price
    idx = df_diff.index[pos]
    loc = df.index.get_loc(idx)
    prev_loc = loc - 1
    timestamps.append(idx)
    for i, series in enumerate(['low', 'close', 'high']):
        true_vals[series].append(df[series].iloc[loc])
        pred_var[series].append(df[series].iloc[prev_loc] + var_fc[i])
        pred_garch[series].append(df[series].iloc[prev_loc] + garch_fc[series])
        pred_hybrid[series].append(df[series].iloc[prev_loc] + 0.5 * (var_fc[i] + garch_fc[series]))
    # Cleanup
    del var_mod, var_res
    gc.collect()
# Compile results DataFrame
result_df = pd.DataFrame({
    'timestamp': timestamps,
    'actual_low':  true_vals['low'],
    'var_low':     pred_var['low'],
    'garch_low':   pred_garch['low'],
    'hybrid_low':  pred_hybrid['low'],
    'actual_close': true_vals['close'],
    'var_close':    pred_var['close'],
    'garch_close':  pred_garch['close'],
    'hybrid_close': pred_hybrid['close'],
    'actual_high':  true_vals['high'],
    'var_high':     pred_var['high'],
    'garch_high':   pred_garch['high'],
    'hybrid_high':  pred_hybrid['high']
})
print('pred vs Actual (hour > 10)', result_df)
# Accuracy metrics
rows = []
for series in ['low', 'close', 'high']:
    mse_v = mean_squared_error(true_vals[series], pred_var[series])
    mae_v = mean_absolute_error(true_vals[series], pred_var[series])
    mse_g = mean_squared_error(true_vals[series], pred_garch[series])
    mae_g = mean_absolute_error(true_vals[series], pred_garch[series])
    mse_h = mean_squared_error(true_vals[series], pred_hybrid[series])
    mae_h = mean_absolute_error(true_vals[series], pred_hybrid[series])
    rows.append({
        'Series': series.title(),
        'VAR MSE': mse_v, 'VAR MAE': mae_v,
        'GARCH MSE': mse_g, 'GARCH MAE': mae_g,
        'Hybrid MSE': mse_h, 'Hybrid MAE': mae_h
    })
metrics_df = pd.DataFrame(rows)
print('Accuracy Metrics (hour > 10)', metrics_df)
# Plotly visualizations for each series
for series in ['low', 'close', 'high']:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df[f'actual_{series}'],
                             mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df[f'var_{series}'],
                             mode='lines+markers', name='VAR'))
    fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df[f'garch_{series}'],
                             mode='lines+markers', name='GARCH'))
    fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df[f'hybrid_{series}'],
                             mode='lines+markers', name='Hybrid'))
    fig.update_layout(
        title=f'{series.title()} Price: Actual vs Forecast (hour > 10)',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        legend_title='Series',
        template='plotly_white'
    )
    fig.show()

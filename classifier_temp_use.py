import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import warnings
import os
from scipy.stats import linregress
warnings.filterwarnings('ignore')

# CONFIGURATION
ROLLING_WINDOW = 50
NUM_PREDICTIONS = 5
TICK_THRESHOLD = 0.0005  # Reduced threshold for more sensitivity
FEATURE_COLUMNS = []  # Populated later



def rolling_slope(series, window):
    """Calculate rolling linear regression slope"""
    slopes = np.full(len(series), np.nan)
    for i in range(window, len(series)):
        y = series.iloc[i-window:i]
        x = np.arange(window)
        slope, _, _, _, _ = linregress(x, y)
        slopes[i] = slope
    return slopes

def rolling_hurst(series, window):
    """Calculate rolling Hurst exponent"""
    hurst_vals = np.full(len(series), np.nan)
    for i in range(window, len(series)):
        window_data = series.iloc[i-window:i]
        if window_data.std() == 0:
            hurst_vals[i] = 0.5
            continue
        lags = range(2, window)
        tau = [np.std(np.subtract(window_data[lag:], window_data[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst_vals[i] = poly[0]*2.0
    return hurst_vals

def rogers_satchell_vol(high, low, close, open_, window):
    """Calculate Rogers-Satchell volatility estimator"""
    rs = np.log(high/open_) * np.log(high/close) + np.log(low/open_) * np.log(low/close)
    return rs.rolling(window).std()

def rolling_autocorr(series, window):
    """Calculate rolling autocorrelation"""
    ac = np.full(len(series), np.nan)
    for i in range(window, len(series)):
        ac[i] = series.iloc[i-window:i].autocorr(lag=1)
    return ac

def calculate_rsi(prices, window):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def process_tick_data_to_candles():
    """Convert tick data to 15-minute candles with all features"""

    dfs_temp = []
    FILE_PATTERN=r"C:\Users\aditya-tanwar\OneDrive - MMC\Documents\my_work\study_work\data\\"

    chunk_files = os.listdir(FILE_PATTERN)#['chunk_0001.parquet', 'chunk_0002.parquet', 'chunk_0003.parquet', 'chunk_0004.parquet']
    print(1)
    
    for file in chunk_files:
        try:
            df_temp = pd.read_parquet(FILE_PATTERN+file)
            # Filter for Trade data and clean
            df_temp = df_temp[df_temp['Type'] == 'Trade'].dropna(subset=['Price', 'Volume'])
            df_temp=df_temp[['Date-Time', 'GMT Offset', 'Price', 'Volume', 'Bid Price', 'Ask Price']]
            df_temp['GMT Offset'] += 7
            df_temp['Date-Time'] = pd.to_datetime(df_temp['Date-Time']) + pd.to_timedelta(df_temp['GMT Offset'], unit='h')
            df_temp['candle_time'] = df_temp['Date-Time'].dt.floor('15min')
            dfs_temp.append(df_temp[['Date-Time', 'Price', 'Volume', 'candle_time']])
            print(f"Loaded {len(df_temp)} trades from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dfs_temp:
        raise ValueError("No data loaded successfully")
    
    # Combine all data
    df = pd.concat(dfs_temp).sort_values('Date-Time')
    df['day'] = df['Date-Time'].dt.date


    # Tick logic: count a tick every time price changes
    df['price_change'] = df['Price'].diff().fillna(0) != 0
    df['tick_id'] = df.groupby('day')['price_change'].cumsum()
    print(2)
    # Aggregate to 15-min candles
    candles = df.groupby(['day', 'candle_time']).agg({
        'Price': ['first', 'max', 'min', 'last'],
        'Volume': 'sum',
        'tick_id': 'nunique'
    })
    candles.columns = ['open', 'high', 'low', 'close', 'volume', 'num_ticks']
    candles = candles.reset_index()

    # Volume bucket features
    bucket_edges = [100, 200, 500, 1000, 2500, 5000,7500,10000,15000,50000]
    bucket_names = [f'vol_{bucket_edges[i]}' for i in range(len(bucket_edges))]
    tick_names = [f'tick_{bucket_edges[i]}' for i in range(len(bucket_edges))]

    for i in range(len(bucket_edges)):
        candles[bucket_names[i]] = candles['volume']/bucket_edges[i]
        candles[tick_names[i]] = candles['num_ticks']/bucket_edges[i]
        
    print(3)

    # Add small and large volume buckets for order flow
    candles['vol_1_5'] = df.groupby(['day', 'candle_time']).apply(
        lambda g: ((g['Volume'] >= 1) & (g['Volume'] <= 5)).sum()
    ).values
    candles['vol_101_plus'] = df.groupby(['day', 'candle_time']).apply(
        lambda g: (g['Volume'] > 100).sum()
    ).values

    # VWAP for each 15-min candle
    candles['vwap'] = df.groupby(['day', 'candle_time']).apply(
        lambda g: (g['Price'] * g['Volume']).sum() / g['Volume'].sum() if g['Volume'].sum() > 0 else np.nan
    ).values
    print(4)
    # 1-min VWAP for rolling average
    one_min = df.copy()
    one_min['one_min_time'] = one_min['Date-Time'].dt.floor('1min')
    vwap_1m = one_min.groupby(['day', 'one_min_time']).apply(
        lambda g: (g['Price'] * g['Volume']).sum() / g['Volume'].sum() if g['Volume'].sum() > 0 else np.nan
    )
    vwap_1m = vwap_1m.rename('vwap_1m').reset_index()
    print(5)
    candles['vwap_1m_avg_15'] = np.nan
    for idx, row in candles.iterrows():
        day = row['day']
        candle_time = row['candle_time']
        vwap_hist = vwap_1m[(vwap_1m['day'] == day) & (vwap_1m['one_min_time'] <= candle_time)]['vwap_1m'].tail(15)
        candles.at[idx, 'vwap_1m_avg_15'] = vwap_hist.mean() if not vwap_hist.empty else np.nan

    return candles

def add_all_features(candles):
    """Add all features to candle DataFrame"""

    short_window = 5#max(3, min(5, total_candles // 20))
    med_window = 14#max(5, min(10, total_candles // 15))
    long_window = 100#max(10, min(20, total_candles // 10))

    # Basic moving averages
    candles['sma_short'] = candles['close'].rolling(short_window, min_periods=1).mean()
    candles['sma_med'] = candles['close'].rolling(med_window, min_periods=1).mean()

    candles['linear_slope_'+str(short_window)] = rolling_slope(candles['close'], short_window)
    candles['linear_slope_'+str(med_window)] = rolling_slope(candles['close'], med_window)
    candles['hurst_'+str(med_window)] = rolling_hurst(candles['close'], med_window)

    candles['vol_rogers_satchell_'+str(short_window)] = rogers_satchell_vol(
        candles['high'], candles['low'], candles['close'], candles['open'], short_window
    )

    # Quantile bins
    q0 = candles['close'].rolling(long_window, min_periods=1).quantile(0.0)
    q25 = candles['close'].rolling(long_window, min_periods=1).quantile(0.25)
    q75 = candles['close'].rolling(long_window, min_periods=1).quantile(0.75)
    q108 = candles['close'].rolling(long_window, min_periods=1).quantile(1.0)
    candles['bin_0_25'] = ((candles['close'] >= q0) & (candles['close'] < q25)).astype(int)
    candles['bin_25_75'] = ((candles['close'] >= q25) & (candles['close'] < q75)).astype(int)
    candles['bin_75_108'] = ((candles['close'] >= q75) & (candles['close'] <= q108)).astype(int)

    #candles['auto_corr_6'] = rolling_autocorr(candles['close'], 6)

    # Relative transformations
    candles['pct_linear_slope'] = (candles['linear_slope_'+str(short_window)] - candles['linear_slope_'+str(med_window)]) / candles['vol_rogers_satchell_'+str(short_window)]

    # VWAP features
    candles['vwap_ratio_short'] = candles['close'] / ((candles['vwap']+candles['vwap_1m_avg_15'])/2)
    candles['vwap_distance_short'] = (candles['close'] - ((candles['vwap']+candles['vwap_1m_avg_15'])/2)) / ((candles['vwap']+candles['vwap_1m_avg_15'])/2)

    # Order flow features
    candles['imbalance'] = (candles['vol_1_5'] - candles['vol_101_plus']) / (candles['vol_1_5'] + candles['vol_101_plus'] + 1e-8)
    candles['small_vol_ratio'] = candles['vol_1_5'] / (candles['volume'] + 1e-8)
    candles['large_vol_ratio'] = candles['vol_101_plus'] / (candles['volume'] + 1e-8)

    # RSI momentum
    candles['rsi_short'] = calculate_rsi(candles['close'],short_window)
    candles['rsi_med'] = calculate_rsi(candles['close'], med_window)

    # Trend features
    candles['sma_cross'] = candles['sma_short'] - candles['sma_med']


    # VWAP features
    candles['vwap_short'] = candles['vwap'].rolling(short_window).mean()
    candles['vwap_med'] = candles['vwap'].rolling(med_window).mean()
    candles['vwap_long_short'] = candles['vwap_1m_avg_15'].rolling(short_window).mean()
 
    
    # Volume features
    candles['volume_sma'] = candles['volume'].rolling(20).mean()
    candles['volume_ratio'] = candles['volume'] / candles['volume_sma']
    candles['volume_momentum'] = candles['volume'].pct_change(5)
    
    # Tick intensity features
    candles['tick_intensity'] = candles['num_ticks'] / candles['num_ticks'].rolling(20).mean()
    candles['tick_momentum'] = candles['num_ticks'].pct_change(3)
    
    # Price range features
    candles['high_low_ratio'] = (candles['high'] - candles['low']) / candles['close']
    candles['close_position'] = (candles['close'] - candles['low']) / (candles['high'] - candles['low'])
    
    # Advanced features
    candles['price_acceleration'] = candles['close'].pct_change().diff()
    candles['volume_price_trend'] = candles['volume'] * candles['close'].pct_change()
    
    # Rolling statistics
    
    # Enhanced tick-volume velocity
    candles['rolling_ticks'] = candles['num_ticks'].rolling(window=5, min_periods=1).sum()
    candles['rolling_volume'] = candles['volume'].rolling(window=5, min_periods=1).sum()
    candles['price_change_pct'] = candles['close'].pct_change(periods=5)
    candles['tick_volume_velocity'] = (candles['price_change_pct'] * candles['rolling_ticks']) / (candles['rolling_volume'] + 1e-8)
    
    # Define feature columns (excluding basic OHLCV)
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = [col for col in candles.columns if col not in ['candle_time', 'open', 'high', 'low', 'close', 'volume', 'num_ticks']]
    
    # Fill missing values
    for col in FEATURE_COLUMNS:
        candles[col] = candles[col].fillna(method='ffill').fillna(0)
        # Replace infinite values
        candles[col] = candles[col].replace([np.inf, -np.inf], 0)
    
    print(f"Created {len(FEATURE_COLUMNS)} features for classification")
    candles.to_csv('ESc1_featured_candles.csv')


    return candles



# STEP 3: Enhanced classification with multiple models
def run_classification_predictions(candles, window=ROLLING_WINDOW, num_preds=NUM_PREDICTIONS):
    results = []
    
    # Create target variable with multiple thresholds
    y_ret = candles['close'].pct_change().shift(-1)
    candles['y_class'] = np.select(
        [y_ret < -TICK_THRESHOLD, y_ret > TICK_THRESHOLD], 
        [-1, 1], 
        0
    )
    
    print(f"Target distribution: {candles['y_class'].value_counts().sort_index()}")
    
    # Track model performance
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(window, window + num_preds):
        print(i)
        train = candles.iloc[i-window:i].copy()
        test = candles.iloc[i:i+1]
        
        # Prepare training data
        X_train = train[FEATURE_COLUMNS].iloc[:-1]  # Remove last row to align with shifted target
        y_train = train['y_class'].shift(-1).iloc[:-1].dropna()
        
        # Ensure alignment
        min_len = min(len(X_train), len(y_train))
        X_train = X_train.iloc[:min_len]
        y_train = y_train.iloc[:min_len]
        print(len(X_train), len(y_train))
        
        if len(X_train) < 30 or len(y_train.unique()) < 2:
    
            continue
        
        #Train multiple models and ensemble
        models = {
            'gb': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                random_state=42
            )
        }
        # models = {
        #     'gb': GradientBoostingClassifier(
        #         n_estimators=100, 
        #         max_depth=6, 
        #         learning_rate=0.1,
        #         subsample=0.8,
        #         random_state=42
        #     )
        # }


        # models = {
        #     'gb': GradientBoostingClassifier(
        #         n_estimators=100, 
        #         max_depth=6, 
        #         learning_rate=0.1,
        #         subsample=0.8,
        #         random_state=42
        #     )
        # }
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            try:
                print(1)
                model.fit(X_train, y_train)
                trained_models[name] = model
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        if not trained_models:
            print(1)
            continue
        
        # Make predictions
        X_test = test[FEATURE_COLUMNS]
        predictions = {}
        probabilities = {}
        
        for name, model in trained_models.items():
            pred = model.predict(X_test)[0]
            proba = model.predict_proba(X_test)[0]
            predictions[name] = pred
            probabilities[name] = proba
        
        # Ensemble prediction (majority vote)
        pred_values = list(predictions.values())
        ensemble_pred = max(set(pred_values), key=pred_values.count)
        
        # Average probabilities
        avg_proba = np.mean([prob for prob in probabilities.values()], axis=0)
        
        
        actual = int(test['y_class'].values[0])
        
        # Track accuracy
        if ensemble_pred == actual:
            correct_predictions += 1
        total_predictions += 1
        
        results.append({
            'timestamp': test['candle_time'].values[0],
            'pred_direction': int(ensemble_pred),
            'actual_direction': actual,
            'prob_down': avg_proba[0] if len(avg_proba) > 0 else np.nan,
            'prob_flat': avg_proba[1] if len(avg_proba) > 1 else np.nan,
            'prob_up': avg_proba[2] if len(avg_proba) > 2 else np.nan,
            'actual_close': test['close'].values[0],
            'actual_volume': test['volume'].values[0],
            'gb_pred': predictions.get('gb', np.nan),
            'rf_pred': predictions.get('rf', np.nan)
        })
        
        if total_predictions % 50 == 0:
            current_acc = (correct_predictions / total_predictions) * 100
            print(f"Progress: {total_predictions}/{num_preds}, Current Accuracy: {current_acc:.2f}%")
    
    return pd.DataFrame(results)

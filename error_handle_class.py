# INTEGRATED VAR-GARCH + CLASSIFICATION PIPELINE - PRODUCTION READY
# Combines VAR-GARCH regression with enhanced classification using comprehensive features
# Based on your updated regressor code with extensive feature engineering

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from statsmodels.tsa.api import VAR
from arch import arch_model  # pip install arch
import gc
import glob

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
PRICE_THRESHOLD = 0.001      # 0.1% threshold for classification
CANDLE_FREQ = '15min'        # Candle frequency
FILE_PATTERN = "chunk_*.parquet"
HOUR_FILTER = 10             # Only predict when hour > this value (as in your code)
TRAINING_WINDOW = 5000        # Fixed training window (adaptive based on data size)
VAR_MAX_LAGS = 5            # Maximum VAR lags
GARCH_LAGS = 5              # AR lags for GARCH
MAX_PREDICTIONS = 10000      # Set to number to limit predictions (None = all)

print("="*80)
print("INTEGRATED VAR-GARCH + CLASSIFICATION PIPELINE")
print("="*80)

# =============================================================================
# 1. DATA LOADING AND CANDLE CREATION
# =============================================================================
print("\n1. Loading data and creating candles...")
FILE_PATTERN=r'C:\Users\aditya-tanwar\OneDrive - MMC\Documents\my_work\study_work\data'

# Find and load parquet files
parquet_files = glob.glob(FILE_PATTERN)
if not parquet_files:
    print(f"❌ No files found matching pattern: {FILE_PATTERN}")
    exit()

parquet_files.sort()
print(f"Found {len(parquet_files)} files")

# Load and combine data
all_data = []
total_trades = 0

for file_path in parquet_files:
    try:
        print(f"  Loading {file_path}...", end=" ")
        df = pd.read_parquet(file_path)
        df = df[df['Type'] == 'Trade'].copy()
        df = df.dropna(subset=['Price', 'Volume'])

        if len(df) > 0:
            all_data.append(df[['Date-Time', 'Price', 'Volume']])
            total_trades += len(df)
            print(f"{len(df):,} trades")
        else:
            print("no valid trades")

    except Exception as e:
        print(f"❌ Error: {e}")
        continue

if not all_data:
    print("❌ No valid data found")
    exit()

# Combine and sort
df = pd.concat(all_data, ignore_index=True)
df['Date-Time'] = pd.to_datetime(df['Date-Time'])
df = df.sort_values('Date-Time')

print(f"\nTotal trades: {len(df):,}")
print(f"Date range: {df['Date-Time'].min()} to {df['Date-Time'].max()}")

# Create 15-minute candles
df['candle_time'] = df['Date-Time'].dt.floor(CANDLE_FREQ)

# OHLCV aggregation
candles = df.groupby('candle_time').agg({
    'Price': ['first', 'max', 'min', 'last'],
    'Volume': 'sum'
}).round(4)

candles.columns = ['open', 'high', 'low', 'close', 'volume']
candles = candles.reset_index()
candles.set_index('candle_time', inplace=True)

print(f"Created {len(candles)} candles")

# Volume bucket features (from your original code)
bucket_ranges = [(1, 5), (6, 20), (21, 100), (101, float('inf'))]
bucket_names = ['vol_1_5', 'vol_6_20', 'vol_21_100', 'vol_101_plus']

for i, (lo, hi) in enumerate(bucket_ranges):
    bucket_name = bucket_names[i]
    if hi == float('inf'):
        bucket_counts = df[df['Volume'] >= lo].groupby('candle_time').size()
    else:
        bucket_counts = df[(df['Volume'] >= lo) & (df['Volume'] <= hi)].groupby('candle_time').size()
    candles[bucket_name] = candles.index.map(bucket_counts).fillna(0)

# =============================================================================
# 2. COMPREHENSIVE FEATURE ENGINEERING
# =============================================================================
print("\n2. Engineering comprehensive features...")

# Determine adaptive window sizes based on dataset
total_candles = len(candles)
short_window = max(3, total_candles // 50)
med_window = max(5, total_candles // 30)
long_window = max(10, total_candles // 20)

print(f"  Using adaptive windows: {short_window}, {med_window}, {long_window}")

# === PRICE-BASED FEATURES ===
candles['returns'] = candles['close'].pct_change()
candles['log_returns'] = np.log(candles['close'] / candles['close'].shift(1))
candles['price_range'] = (candles['high'] - candles['low']) / candles['close']
candles['body_size'] = abs(candles['close'] - candles['open']) / candles['close']
candles['upper_shadow'] = (candles['high'] - np.maximum(candles['open'], candles['close'])) / candles['close']
candles['lower_shadow'] = (np.minimum(candles['open'], candles['close']) - candles['low']) / candles['close']

# === VOLUME FEATURES ===
candles['volume_ma_short'] = candles['volume'].rolling(short_window, min_periods=1).mean()
candles['volume_ma_med'] = candles['volume'].rolling(med_window, min_periods=1).mean()
candles['volume_ma_long'] = candles['volume'].rolling(long_window, min_periods=1).mean()
candles['volume_ratio_short'] = candles['volume'] / candles['volume_ma_short']
candles['volume_ratio_med'] = candles['volume'] / candles['volume_ma_med']
candles['volume_ratio_long'] = candles['volume'] / candles['volume_ma_long']
candles['volume_std_short'] = candles['volume'].rolling(short_window, min_periods=1).std()
candles['volume_std_med'] = candles['volume'].rolling(med_window, min_periods=1).std()

# === TECHNICAL INDICATORS ===
candles['sma_short'] = candles['close'].rolling(short_window, min_periods=1).mean()
candles['sma_med'] = candles['close'].rolling(med_window, min_periods=1).mean()
candles['sma_long'] = candles['close'].rolling(long_window, min_periods=1).mean()
candles['ema_short'] = candles['close'].ewm(span=short_window).mean()
candles['ema_med'] = candles['close'].ewm(span=med_window).mean()
candles['ema_long'] = candles['close'].ewm(span=long_window).mean()

# === MOMENTUM FEATURES ===
candles['momentum_short'] = candles['close'] / candles['close'].shift(short_window) - 1
candles['momentum_med'] = candles['close'] / candles['close'].shift(med_window) - 1
candles['momentum_long'] = candles['close'] / candles['close'].shift(long_window) - 1
candles['roc_short'] = candles['close'].pct_change(short_window)
candles['roc_med'] = candles['close'].pct_change(med_window)

# === VOLATILITY MEASURES ===
candles['volatility_short'] = candles['returns'].rolling(short_window, min_periods=1).std()
candles['volatility_med'] = candles['returns'].rolling(med_window, min_periods=1).std()
candles['volatility_long'] = candles['returns'].rolling(long_window, min_periods=1).std()
candles['atr'] = (candles['high'] - candles['low']).rolling(med_window, min_periods=1).mean()

# === VWAP AND RELATED ===
candles['vwap_short'] = (candles['close'] * candles['volume']).rolling(short_window, min_periods=1).sum() / candles['volume'].rolling(short_window, min_periods=1).sum()
candles['vwap_med'] = (candles['close'] * candles['volume']).rolling(med_window, min_periods=1).sum() / candles['volume'].rolling(med_window, min_periods=1).sum()
candles['vwap_ratio_short'] = candles['close'] / candles['vwap_short']
candles['vwap_ratio_med'] = candles['close'] / candles['vwap_med']
candles['vwap_distance_short'] = (candles['close'] - candles['vwap_short']) / candles['vwap_short']
candles['vwap_distance_med'] = (candles['close'] - candles['vwap_med']) / candles['vwap_med']

# === ORDER FLOW FEATURES ===
candles['imbalance'] = (candles['vol_1_5'] - candles['vol_101_plus']) / (candles['vol_1_5'] + candles['vol_101_plus'] + 1e-8)
candles['small_vol_ratio'] = candles['vol_1_5'] / (candles['volume'] + 1e-8)
candles['large_vol_ratio'] = candles['vol_101_plus'] / (candles['volume'] + 1e-8)
candles['mid_vol_ratio'] = (candles['vol_6_20'] + candles['vol_21_100']) / (candles['volume'] + 1e-8)

# === RSI-LIKE MOMENTUM ===
def calculate_rsi(prices, window):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

candles['rsi_short'] = calculate_rsi(candles['close'], short_window)
candles['rsi_med'] = calculate_rsi(candles['close'], med_window)

# === BOLLINGER BAND-LIKE FEATURES ===
candles['bb_upper_short'] = candles['sma_short'] + 2 * candles['volatility_short']
candles['bb_lower_short'] = candles['sma_short'] - 2 * candles['volatility_short']
candles['bb_position_short'] = (candles['close'] - candles['bb_lower_short']) / (candles['bb_upper_short'] - candles['bb_lower_short'] + 1e-8)

candles['bb_upper_med'] = candles['sma_med'] + 2 * candles['volatility_med']
candles['bb_lower_med'] = candles['sma_med'] - 2 * candles['volatility_med']
candles['bb_position_med'] = (candles['close'] - candles['bb_lower_med']) / (candles['bb_upper_med'] - candles['bb_lower_med'] + 1e-8)

# === TREND FEATURES ===
candles['trend_short'] = np.where(candles['close'] > candles['sma_short'], 1, -1)
candles['trend_med'] = np.where(candles['close'] > candles['sma_med'], 1, -1)
candles['trend_strength'] = (candles['close'] - candles['sma_med']) / candles['sma_med']

# === CROSS-SECTIONAL FEATURES ===
candles['sma_cross'] = candles['sma_short'] - candles['sma_med']
candles['ema_cross'] = candles['ema_short'] - candles['ema_med']
candles['volume_trend'] = candles['volume_ma_short'] - candles['volume_ma_med']

# Fill NaN values
candles = candles.fillna(method='ffill').fillna(method='bfill').fillna(0)

# === CLASSIFICATION TARGET ===
returns = candles['close'].pct_change()
candles['target'] = np.select(
    [returns < -PRICE_THRESHOLD, returns > PRICE_THRESHOLD],
    [-1, 1], default=0
)
candles['target_mapped'] = candles['target'].map({-1: 0, 0: 1, 1: 2})

print(f"Features ready. Total candles: {len(candles)}")
print(f"Class distribution: {candles['target_mapped'].value_counts().sort_index().to_dict()}")

# === PREPARE DATA FOR VAR-GARCH ===
df_diff = candles[['low', 'close', 'high']].diff().dropna()

# === CLASSIFICATION FEATURES (COMPREHENSIVE SET) ===
classifier_features = [
    # Basic OHLCV
    'open', 'high', 'low', 'close', 'volume',
    # Price features
    'returns', 'log_returns', 'price_range', 'body_size', 'upper_shadow', 'lower_shadow',
    # Volume features
    'volume_ratio_short', 'volume_ratio_med', 'volume_ratio_long', 'volume_std_short', 'volume_std_med',
    # Technical indicators
    'sma_short', 'sma_med', 'sma_long', 'ema_short', 'ema_med', 'ema_long',
    # Momentum
    'momentum_short', 'momentum_med', 'momentum_long', 'roc_short', 'roc_med',
    # Volatility
    'volatility_short', 'volatility_med', 'volatility_long', 'atr',
    # VWAP
    'vwap_short', 'vwap_med', 'vwap_ratio_short', 'vwap_ratio_med', 'vwap_distance_short', 'vwap_distance_med',
    # Order flow
    'imbalance', 'small_vol_ratio', 'large_vol_ratio', 'mid_vol_ratio',
    # RSI
    'rsi_short', 'rsi_med',
    # Bollinger Bands
    'bb_position_short', 'bb_position_med',
    # Trend
    'trend_short', 'trend_med', 'trend_strength',
    # Cross-sectional
    'sma_cross', 'ema_cross', 'volume_trend'
] + bucket_names

print(f"Using {len(classifier_features)} features for classification")

# =============================================================================
# 3. ADAPTIVE PARAMETERS
# =============================================================================
# Adjust parameters based on dataset size
TRAINING_WINDOW = min(max(50, total_candles // 10), TRAINING_WINDOW)  # 50-500 candles
VAR_MAX_LAGS = min(max(3, total_candles // 100), 10)      # 3-10 lags
MIN_TRAIN_SIZE = max(max(20, total_candles // 20), 5000)   # 20-100 minimum

print(f"\n3. Adaptive parameters for {total_candles} candles:")
print(f"  Training window: {TRAINING_WINDOW}")
print(f"  VAR max lags: {VAR_MAX_LAGS}")
print(f"  Min train size: {MIN_TRAIN_SIZE}")

# =============================================================================
# 4. ROLLING PREDICTION LOOP (BASED ON YOUR CODE STRUCTURE)
# =============================================================================
print("\n4. Starting rolling predictions...")

# Calculate test positions (following your code logic)
n_preds = 5000#min(1000, len(df_diff) // 3)  # Adaptive number of predictions
all_positions = list(range(len(df_diff) - n_preds, len(df_diff)))
test_positions = [pos for pos in all_positions if candles.index[pos].hour > HOUR_FILTER]

print(f"Testing on {len(test_positions)} positions (hour > {HOUR_FILTER})")

# Containers for results (following your structure)
timestamps = []
true_vals = {'low': [], 'close': [], 'high': []}
pred_var = {'low': [], 'close': [], 'high': []}
pred_garch = {'low': [], 'close': [], 'high': []}
pred_hybrid = {'low': [], 'close': [], 'high': []}
pred_class = []
actual_class = []

successful_predictions = 0
total_attempts = 0

for pos in test_positions:
    start = pos - TRAINING_WINDOW
    if start < 0:
        continue

    total_attempts += 1

    # Check if we've reached max predictions
    if MAX_PREDICTIONS and successful_predictions >= MAX_PREDICTIONS:
        break

    try:
        # Training data
        train = df_diff.iloc[start:pos]
        train_features = candles[classifier_features].iloc[start:pos]
        train_targets = candles['target_mapped'].iloc[start:pos]

        # Skip if insufficient data
        if len(train) < MIN_TRAIN_SIZE or len(train_features) < MIN_TRAIN_SIZE:
            continue

        # =================================================================
        # VAR-GARCH REGRESSION (FOLLOWING YOUR CODE STRUCTURE)
        # =================================================================

        # 1) VAR forecast
        var_mod = VAR(train)
        # Determine optimal lag order
        optimal_lags = min(VAR_MAX_LAGS, len(train) // 10)
        var_res = var_mod.fit(maxlags=optimal_lags, ic='aic')
        var_fc = var_res.forecast(train.values[-var_res.k_ar:], steps=1)[0]

        # 2) AR-GARCH for each series
        garch_fc = {}
        for series in ['low', 'close', 'high']:
            try:
                # Use adaptive lag order for GARCH
                garch_lags = min(GARCH_LAGS, len(train) // 20)
                g_mod = arch_model(train[series], mean='AR', lags=garch_lags, vol='Garch', p=1, q=1)
                g_res = g_mod.fit(disp='off')
                garch_fc[series] = g_res.forecast(horizon=1).mean.iloc[-1, 0]
                del g_mod, g_res
            except:
                # Fallback to VAR if GARCH fails
                garch_fc[series] = var_fc[['low', 'close', 'high'].index(series)]

        # =================================================================
        # CLASSIFICATION PREDICTION
        # =================================================================

        # Check if we have enough classes for training
        unique_classes = train_targets.nunique()

        if len(train_features) >= MIN_TRAIN_SIZE and unique_classes >= 2:
            clf_model = GradientBoostingClassifier(
                n_estimators=min(50, len(train_features)), 
                max_depth=4, 
                learning_rate=0.1, 
                random_state=42
            )
            clf_model.fit(train_features, train_targets)

            current_features = candles[classifier_features].iloc[pos].values
            class_prediction = clf_model.predict(current_features.reshape(1, -1))[0]
        else:
            # Use majority class if insufficient training data
            class_prediction = train_targets.mode().iloc[0] if len(train_targets) > 0 else 1

        # =================================================================
        # COLLECT RESULTS (FOLLOWING YOUR CODE STRUCTURE)
        # =================================================================

        # Map back to price (following your logic)
        idx = candles.index[pos]
        loc = candles.index.get_loc(idx)
        prev_loc = loc - 1

        timestamps.append(idx)

        for i, series in enumerate(['low', 'close', 'high']):
            true_vals[series].append(candles[series].iloc[loc])
            pred_var[series].append(candles[series].iloc[prev_loc] + var_fc[i])
            pred_garch[series].append(candles[series].iloc[prev_loc] + garch_fc[series])
            pred_hybrid[series].append(candles[series].iloc[prev_loc] + 0.5 * (var_fc[i] + garch_fc[series]))

        pred_class.append(class_prediction)
        actual_class.append(candles['target_mapped'].iloc[pos])

        successful_predictions += 1

        # Progress reporting
        if successful_predictions % 100 == 0:
            print(f"  Completed {successful_predictions} predictions...")

        # Cleanup (following your code)
        del var_mod, var_res
        gc.collect()

    except Exception as e:
        # Continue processing even if individual prediction fails
        continue

print(f"\nCompleted {successful_predictions} successful predictions out of {total_attempts} attempts")

# =============================================================================
# 5. RESULTS ANALYSIS (FOLLOWING YOUR CODE STRUCTURE)
# =============================================================================
if timestamps:
    print("\n5. Analyzing results...")

    # Compile results DataFrame (following your structure)
    result_df = pd.DataFrame({
        'timestamp': timestamps,
        'actual_low': true_vals['low'],
        'var_low': pred_var['low'],
        'garch_low': pred_garch['low'],
        'hybrid_low': pred_hybrid['low'],
        'actual_close': true_vals['close'],
        'var_close': pred_var['close'],
        'garch_close': pred_garch['close'],
        'hybrid_close': pred_hybrid['close'],
        'actual_high': true_vals['high'],
        'var_high': pred_var['high'],
        'garch_high': pred_garch['high'],
        'hybrid_high': pred_hybrid['high'],
        'actual_class': actual_class,
        'pred_class': pred_class,
        'actual_class_name': [['Down', 'Flat', 'Up'][x] for x in actual_class],
        'pred_class_name': [['Down', 'Flat', 'Up'][x] for x in pred_class]
    })

    print('Prediction vs Actual (hour > 10)', result_df.head(10))

    # Accuracy metrics (following your structure)
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
    print('\nAccuracy Metrics (hour > 10)', metrics_df)

    # Classification metrics
    class_accuracy = accuracy_score(actual_class, pred_class)
    class_dist = pd.Series(actual_class).value_counts().sort_index()
    class_names = ['Down', 'Flat', 'Up']

    print(f"\n🎯 CLASSIFICATION METRICS:")
    print(f"  Accuracy: {class_accuracy*100:.2f}%")
    print(f"  Class Distribution:")
    for i, count in class_dist.items():
        pct = count / len(actual_class) * 100
        print(f"    {class_names[i]}: {count} ({pct:.1f}%)")

    print(f"\n💾 SAVING RESULTS...")
    result_df.to_csv('var_garch_classification_results.csv', index=False)
    print(f"✅ Saved {len(result_df)} predictions to 'var_garch_classification_results.csv'")

    print(f"\n🚀 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   📈 VAR predictions: Based on vector autoregression")
    print(f"   🔥 GARCH predictions: AR-GARCH volatility modeling")
    print(f"   🎯 Hybrid predictions: 50% VAR + 50% GARCH")
    print(f"   🎯 Classification predictions: Down/Flat/Up directions")
    print(f"   📊 Total predictions: {len(result_df)}")
    print(f"   ⏰ Time range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
    print(f"   💾 Results saved to CSV with all actual vs predicted values")

else:
    print("❌ No successful predictions generated. Check your data and parameters.")
    print("\nTroubleshooting tips:")
    print("- Ensure you have enough candles (>100 recommended)")
    print("- Check if hour filter is too restrictive")
    print("- Verify your parquet files contain valid trade data")
    print("- Install required library: pip install arch")

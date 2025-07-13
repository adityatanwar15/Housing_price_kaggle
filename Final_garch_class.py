import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from statsmodels.tsa.api import VAR
try:
    from arch import arch_model
    GARCH_AVAILABLE = True
    print("✅ GARCH library available")
except ImportError:
    GARCH_AVAILABLE = False
    print("⚠️ GARCH library not available - will use VAR only")
    print("   Install with: pip install arch")
import gc
import glob
import os

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
PRICE_THRESHOLD = 0.001      # 0.1% threshold for classification
CANDLE_FREQ = '15min'        # Candle frequency
FILE_PATTERN = "chunk_*.parquet"  
FILE_PATTERN=r'C:\Users\aditya-tanwar\OneDrive - MMC\Documents\my_work\study_work\data\\'# Pattern to match your files
HOUR_FILTER = 9              # Only predict when hour > this value
TRAINING_WINDOW = 5000         # Training window size (adaptive)
VAR_MAX_LAGS = 3            # Maximum VAR lags
MIN_TRAIN_SIZE = 5000         # Minimum training size
MAX_SAMPLE_SIZE = None    # Maximum trades to process (None = all)

print("="*80)
print("VAR-GARCH + CLASSIFICATION PIPELINE - PRODUCTION VERSION")
print("="*80)

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print("\\n1. Loading and preparing data...")

# Find parquet files
parquet_files = os.listdir(FILE_PATTERN)
if not parquet_files:
    print(f"❌ No files found matching pattern: {FILE_PATTERN}")
    print("Available parquet files in directory:")
    for f in os.listdir('.'):
        if f.endswith('.parquet'):
            print(f"  - {f}")
    exit()

parquet_files.sort()
print(f"Found {len(parquet_files)} files: {parquet_files}")

#Load and combine data
all_data = []
total_trades = 0

for file_path in parquet_files:
    try:
        print(f"  Loading {file_path}...", end=" ")
        df_temp = pd.read_parquet(FILE_PATTERN+file_path)
        
        # Filter for trades only
        if 'Type' in df_temp.columns:
            df_temp = df_temp[df_temp['Type'] == 'Trade'].copy()
        
        # Clean data
        df_temp = df_temp.dropna(subset=['Price', 'Volume'])
        
        if len(df_temp) > 0:
            all_data.append(df_temp[['Date-Time', 'Price', 'Volume']])
            total_trades += len(df_temp)
            print(f"{len(df_temp):,} trades")
        else:
            print("no valid trades")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        continue

if not all_data:
    print("❌ No valid data found")
    exit()

# Combine and sort all data
df = pd.concat(all_data, ignore_index=True)
df['Date-Time'] = pd.to_datetime(df['Date-Time'])
df = df.sort_values('Date-Time')

# Apply sample size limit if specified
if MAX_SAMPLE_SIZE and len(df) > MAX_SAMPLE_SIZE:
    print(f"Sampling {MAX_SAMPLE_SIZE:,} trades from {len(df):,} total")
    df = df.tail(MAX_SAMPLE_SIZE)  # Use most recent data

print(f"\\nTotal trades: {len(df):,}")
print(f"Date range: {df['Date-Time'].min()} to {df['Date-Time'].max()}")

# =============================================================================
# 2. CANDLE CREATION
# =============================================================================
print("\\n2. Creating candles...")

df['candle_time'] = df['Date-Time'].dt.floor(CANDLE_FREQ)

# Create OHLCV candles
candles = df.groupby('candle_time').agg({
    'Price': ['first', 'max', 'min', 'last'],
    'Volume': 'sum'
}).round(4)

candles.columns = ['open', 'high', 'low', 'close', 'volume']
candles = candles.reset_index()
candles.set_index('candle_time', inplace=True)

print(f"Created {len(candles)} candles")
print(f"Candle range: {candles.index.min()} to {candles.index.max()}")

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
# 3. COMPREHENSIVE FEATURE ENGINEERING
# =============================================================================
print("\\n3. Engineering features...")

# Determine adaptive window sizes based on dataset
total_candles = len(candles)
short_window = max(3, min(5, total_candles // 20))
med_window = max(5, min(10, total_candles // 15))
long_window = max(10, min(20, total_candles // 10))

print(f"  Using adaptive windows: {short_window}, {med_window}, {long_window}")

# === CORE PRICE FEATURES ===
candles['returns'] = candles['close'].pct_change()
candles['log_returns'] = np.log(candles['close'] / candles['close'].shift(1))
candles['price_range'] = (candles['high'] - candles['low']) / candles['close']
candles['body_size'] = abs(candles['close'] - candles['open']) / candles['close']
candles['upper_shadow'] = (candles['high'] - np.maximum(candles['open'], candles['close'])) / candles['close']
candles['lower_shadow'] = (np.minimum(candles['open'], candles['close']) - candles['low']) / candles['close']

# === VOLUME FEATURES ===
candles['volume_ma_short'] = candles['volume'].rolling(short_window, min_periods=1).mean()
candles['volume_ma_med'] = candles['volume'].rolling(med_window, min_periods=1).mean()
candles['volume_ratio_short'] = candles['volume'] / candles['volume_ma_short']
candles['volume_ratio_med'] = candles['volume'] / candles['volume_ma_med']
candles['volume_std_short'] = candles['volume'].rolling(short_window, min_periods=1).std()

# === TECHNICAL INDICATORS ===
candles['sma_short'] = candles['close'].rolling(short_window, min_periods=1).mean()
candles['sma_med'] = candles['close'].rolling(med_window, min_periods=1).mean()
candles['sma_long'] = candles['close'].rolling(long_window, min_periods=1).mean()
candles['ema_short'] = candles['close'].ewm(span=short_window).mean()
candles['ema_med'] = candles['close'].ewm(span=med_window).mean()

# === MOMENTUM FEATURES ===
candles['momentum_short'] = candles['close'] / candles['close'].shift(short_window) - 1
candles['momentum_med'] = candles['close'] / candles['close'].shift(med_window) - 1
candles['roc_short'] = candles['close'].pct_change(short_window)
candles['roc_med'] = candles['close'].pct_change(med_window)

# === VOLATILITY MEASURES ===
candles['volatility_short'] = candles['returns'].rolling(short_window, min_periods=1).std()
candles['volatility_med'] = candles['returns'].rolling(med_window, min_periods=1).std()
candles['volatility_long'] = candles['returns'].rolling(long_window, min_periods=1).std()

# === VWAP FEATURES ===
candles['vwap_short'] = (candles['close'] * candles['volume']).rolling(short_window, min_periods=1).sum() / candles['volume'].rolling(short_window, min_periods=1).sum()
candles['vwap_med'] = (candles['close'] * candles['volume']).rolling(med_window, min_periods=1).sum() / candles['volume'].rolling(med_window, min_periods=1).sum()
candles['vwap_ratio_short'] = candles['close'] / candles['vwap_short']
candles['vwap_distance_short'] = (candles['close'] - candles['vwap_short']) / candles['vwap_short']

# === ORDER FLOW FEATURES ===
candles['imbalance'] = (candles['vol_1_5'] - candles['vol_101_plus']) / (candles['vol_1_5'] + candles['vol_101_plus'] + 1e-8)
candles['small_vol_ratio'] = candles['vol_1_5'] / (candles['volume'] + 1e-8)
candles['large_vol_ratio'] = candles['vol_101_plus'] / (candles['volume'] + 1e-8)

# === RSI-LIKE MOMENTUM ===
def calculate_rsi(prices, window):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

candles['rsi_short'] = calculate_rsi(candles['close'], short_window)
candles['rsi_med'] = calculate_rsi(candles['close'], med_window)

# === TREND FEATURES ===
candles['trend_short'] = np.where(candles['close'] > candles['sma_short'], 1, -1)
candles['trend_med'] = np.where(candles['close'] > candles['sma_med'], 1, -1)
candles['sma_cross'] = candles['sma_short'] - candles['sma_med']

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
print(f"Differenced data shape: {df_diff.shape}")

# === CLASSIFICATION FEATURES ===
classifier_features = [
    # Basic OHLCV
    'open', 'high', 'low', 'close', 'volume',
    # Price features
    'returns', 'log_returns', 'price_range', 'body_size', 'upper_shadow', 'lower_shadow',
    # Volume features
    'volume_ratio_short', 'volume_ratio_med', 'volume_std_short',
    # Technical indicators
    'sma_short', 'sma_med', 'sma_long', 'ema_short', 'ema_med',
    # Momentum
    'momentum_short', 'momentum_med', 'roc_short', 'roc_med',
    # Volatility
    'volatility_short', 'volatility_med', 'volatility_long',
    # VWAP
    'vwap_short', 'vwap_med', 'vwap_ratio_short', 'vwap_distance_short',
    # Order flow
    'imbalance', 'small_vol_ratio', 'large_vol_ratio',
    # RSI and trend
    'rsi_short', 'rsi_med', 'trend_short', 'trend_med', 'sma_cross'
] + bucket_names

print(f"Using {len(classifier_features)} features for classification")

# =============================================================================
# 4. ADAPTIVE PARAMETERS
# =============================================================================
# Adjust parameters based on dataset size
# TRAINING_WINDOW = min(max(20, total_candles // 8), 200)
# VAR_MAX_LAGS = min(max(2, total_candles // 50), 8)
# MIN_TRAIN_SIZE = min(max(10, total_candles // 30), 50)

print(f"\\n4. Adaptive parameters for {total_candles} candles:")
print(f"  Training window: {TRAINING_WINDOW}")
print(f"  VAR max lags: {VAR_MAX_LAGS}")
print(f"  Min train size: {MIN_TRAIN_SIZE}")

# =============================================================================
# 5. ROLLING PREDICTION LOOP
# =============================================================================
print("\\n5. Starting rolling predictions...")

# Calculate test positions
n_preds = MIN_TRAIN_SIZE#min(500, len(df_diff) // 3)  # Limit predictions for reasonable runtime
all_positions = list(range(len(df_diff) - n_preds, len(df_diff)))
test_positions = [pos for pos in all_positions if candles.index[pos].hour > HOUR_FILTER]

print(f"Testing on {len(test_positions)} positions (hour > {HOUR_FILTER})")

if not test_positions:
    print("⚠️ No positions pass hour filter. Using all positions...")
    test_positions = all_positions[-min(100, len(all_positions)):]  # Use last 100

# Results containers
timestamps = []
true_vals = {'low': [], 'close': [], 'high': []}
pred_var = {'low': [], 'close': [], 'high': []}
pred_garch = {'low': [], 'close': [], 'high': []}
pred_hybrid = {'low': [], 'close': [], 'high': []}
pred_class = []
actual_class = []

successful_predictions = 0
total_attempts = 0

# Main prediction loop
for pos in test_positions:
    start = pos - TRAINING_WINDOW
    if start < 0:
        continue
    
    total_attempts += 1
    
    try:
        # Training data
        train_diff = df_diff.iloc[start:pos]
        train_features = candles[classifier_features].iloc[start:pos]
        train_targets = candles['target_mapped'].iloc[start:pos]
        
        # Skip if insufficient data
        if len(train_diff) < MIN_TRAIN_SIZE:
            continue
        
        # =================================================================
        # VAR-GARCH REGRESSION
        # =================================================================
        
        # 1) VAR forecast
        max_lags = min(VAR_MAX_LAGS, len(train_diff) // 8)
        if max_lags < 1:
            max_lags = 1
            
        var_mod = VAR(train_diff)
        var_res = var_mod.fit(maxlags=max_lags, ic='aic')
        var_order = var_res.k_ar
        
        if len(train_diff) >= var_order:
            var_fc = var_res.forecast(train_diff.values[-var_order:], steps=1)[0]
        else:
            continue
        
        # 2) GARCH forecast (if available)
        if GARCH_AVAILABLE:
            garch_fc = {}
            for series in ['low', 'close', 'high']:
                try:
                    garch_lags = min(3, len(train_diff) // 10)
                    if garch_lags < 1:
                        garch_lags = 1
                    g_mod = arch_model(train_diff[series], mean='AR', lags=garch_lags, vol='Garch', p=1, q=1)
                    g_res = g_mod.fit(disp='off')
                    garch_fc[series] = g_res.forecast(horizon=1).mean.iloc[-1, 0]
                    del g_mod, g_res
                except:
                    # Fallback to VAR
                    garch_fc[series] = var_fc[['low', 'close', 'high'].index(series)]
        else:
            # Use VAR for all if GARCH not available
            garch_fc = {series: var_fc[i] for i, series in enumerate(['low', 'close', 'high'])}
        
        # =================================================================
        # CLASSIFICATION
        # =================================================================
        
        unique_classes = train_targets.nunique()
        
        if unique_classes >= 2 and len(train_features) >= MIN_TRAIN_SIZE:
            # Adaptive model parameters
            n_estimators = min(20, max(5, len(train_features) // 3))
            clf_model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=3,
                learning_rate=0.2,
                random_state=42
            )
            clf_model.fit(train_features, train_targets)
            
            current_features = candles[classifier_features].iloc[pos].values
            class_prediction = clf_model.predict(current_features.reshape(1, -1))[0]
        else:
            # Use majority class
            class_prediction = train_targets.mode().iloc[0] if len(train_targets) > 0 else 1
        
        # =================================================================
        # COLLECT RESULTS
        # =================================================================
        
        idx = candles.index[pos]
        loc = candles.index.get_loc(idx)
        prev_loc = loc - 1
        
        timestamps.append(idx)
        
        # Store predictions and actuals
        for i, series in enumerate(['low', 'close', 'high']):
            true_vals[series].append(candles[series].iloc[loc])
            pred_var[series].append(candles[series].iloc[prev_loc] + var_fc[i])
            pred_garch[series].append(candles[series].iloc[prev_loc] + garch_fc[series])
            pred_hybrid[series].append(candles[series].iloc[prev_loc] + 0.5 * (var_fc[i] + garch_fc[series]))
        
        pred_class.append(class_prediction)
        actual_class.append(candles['target_mapped'].iloc[pos])
        
        successful_predictions += 1
        
        # Progress reporting
        if successful_predictions % 50 == 0:
            print(f"  Completed {successful_predictions} predictions...")
        
        # Cleanup
        del var_mod, var_res
        gc.collect()
        
    except Exception as e:
        # Continue processing even if individual prediction fails
        continue

print(f"\\nCompleted {successful_predictions} successful predictions out of {total_attempts} attempts")

# =============================================================================
# 6. RESULTS ANALYSIS AND EXPORT
# =============================================================================
if timestamps:
    print("\\n6. Analyzing and exporting results...")
    
    # Create comprehensive results DataFrame
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
    
    # Calculate comprehensive metrics
    print("\\n📊 REGRESSION METRICS:")
    metrics_rows = []
    for series in ['low', 'close', 'high']:
        mse_v = mean_squared_error(true_vals[series], pred_var[series])
        mae_v = mean_absolute_error(true_vals[series], pred_var[series])
        mse_g = mean_squared_error(true_vals[series], pred_garch[series])
        mae_g = mean_absolute_error(true_vals[series], pred_garch[series])
        mse_h = mean_squared_error(true_vals[series], pred_hybrid[series])
        mae_h = mean_absolute_error(true_vals[series], pred_hybrid[series])
        
        metrics_rows.append({
            'Series': series.title(),
            'VAR_MSE': mse_v, 'VAR_MAE': mae_v,
            'GARCH_MSE': mse_g, 'GARCH_MAE': mae_g,
            'Hybrid_MSE': mse_h, 'Hybrid_MAE': mae_h
        })
        
        print(f"  {series.upper()}: VAR MSE={mse_v:.4f}, GARCH MSE={mse_g:.4f}, Hybrid MSE={mse_h:.4f}")
    
    # Classification metrics
    class_accuracy = accuracy_score(actual_class, pred_class)
    class_dist = pd.Series(actual_class).value_counts().sort_index()
    class_names = ['Down', 'Flat', 'Up']
    
    print(f"\\n🎯 CLASSIFICATION METRICS:")
    print(f"  Accuracy: {class_accuracy*100:.2f}%")
    print(f"  Class Distribution:")
    for i, count in class_dist.items():
        pct = count / len(actual_class) * 100
        print(f"    {class_names[i]}: {count} ({pct:.1f}%)")
    
    # Export results
    output_file = 'var_garch_classification_results.csv'
    result_df.to_csv(output_file, index=False)
    
    # Export metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv('prediction_metrics.csv', index=False)
    
    print(f"\\n💾 RESULTS EXPORTED:")
    print(f"  📈 Predictions: {output_file} ({len(result_df)} rows)")
    print(f"  📊 Metrics: prediction_metrics.csv")
    
    # Display sample results
    print(f"\\n📋 SAMPLE RESULTS:")
    display_cols = ['timestamp', 'actual_close', 'var_close', 'garch_close', 'hybrid_close', 'actual_class_name', 'pred_class_name']
    print(result_df[display_cols].head(10).to_string(index=False))
    
    print(f"\\n🚀 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   📈 VAR predictions: Vector autoregression forecasts")
    print(f"   🔥 GARCH predictions: AR-GARCH volatility modeling")
    print(f"   🎯 Hybrid predictions: Combined VAR + GARCH")
    print(f"   🎯 Classification: Down/Flat/Up direction predictions")

# Create the final optimized code file
code_content = '''# COMPLETE 3-CLASS PRICE PREDICTION PIPELINE - PRODUCTION VERSION
# Optimized for multiple parquet files with robust error handling

import pandas as pd
import numpy as np
import warnings
import os
import glob
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Try to import optional libraries
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
    print("✓ XGBoost available")
except ImportError:
    HAVE_XGB = False
    print("⚠ XGBoost not available, using sklearn models only")

try:
    import ta
    HAVE_TA = True
    print("✓ TA library available")
except ImportError:
    HAVE_TA = False
    print("⚠ TA library not available, using basic features only")
    print("  Install with: pip install ta")

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
PRICE_THRESHOLD = 0.001      # 0.1% threshold for up/down classification
CANDLE_FREQ = '15min'        # Candle frequency
N_SPLITS = 5                 # Cross-validation folds
MAX_FILES = None             # Set to number to limit files (None = all files)
SAMPLE_RATIO = None          # Set to 0.1 for 10% sample (None = all data)

# File pattern - adjust this to match your files
FILE_PATTERN = "chunk_*.parquet"  # or specify exact list

print("="*70)
print("3-CLASS PRICE PREDICTION PIPELINE - PRODUCTION VERSION")
print("="*70)

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================
print("\\n1. Loading data...")

# Find parquet files
parquet_files = glob.glob(FILE_PATTERN)
if not parquet_files:
    print(f"❌ No files found matching pattern: {FILE_PATTERN}")
    print("Please check your file pattern or current directory")
    exit()

parquet_files.sort()
if MAX_FILES:
    parquet_files = parquet_files[:MAX_FILES]

print(f"Found {len(parquet_files)} parquet files")

# Load and combine data
all_data = []
total_rows = 0

for i, file_path in enumerate(parquet_files):
    try:
        print(f"  Loading {file_path}... ", end="")
        df = pd.read_parquet(file_path)
        
        # Filter for trades and clean
        df = df[df['Type'] == 'Trade'].copy()
        df = df.dropna(subset=['Price', 'Volume'])
        
        if len(df) > 0:
            all_data.append(df[['Date-Time', 'Price', 'Volume']])
            total_rows += len(df)
            print(f"{len(df):,} trades")
        else:
            print("no valid trades")
            
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        continue

if not all_data:
    print("❌ No valid data found in any files")
    exit()

# Combine all data
print(f"\\nCombining data from {len(all_data)} files...")
df = pd.concat(all_data, ignore_index=True)
df['Date-Time'] = pd.to_datetime(df['Date-Time'])
df = df.sort_values('Date-Time')

print(f"Total records: {len(df):,}")
print(f"Date range: {df['Date-Time'].min()} to {df['Date-Time'].max()}")

# Optional sampling for faster processing
if SAMPLE_RATIO:
    original_size = len(df)
    df = df.sample(frac=SAMPLE_RATIO, random_state=42).sort_values('Date-Time')
    print(f"Sampled {SAMPLE_RATIO*100}% of data: {len(df):,} records")

# =============================================================================
# 2. CANDLE CREATION
# =============================================================================
print(f"\\n2. Creating {CANDLE_FREQ} candles...")

# Group by time intervals
df['candle_time'] = df['Date-Time'].dt.floor(CANDLE_FREQ)

# Create OHLCV data
candles = df.groupby('candle_time').agg({
    'Price': ['first', 'max', 'min', 'last'],
    'Volume': 'sum'
}).round(4)

# Flatten column names
candles.columns = ['open', 'high', 'low', 'close', 'volume']
candles = candles.reset_index()

print(f"Created {len(candles)} candles")

# =============================================================================
# 3. VOLUME BUCKET FEATURES
# =============================================================================
print("\\n3. Adding volume bucket features...")

# Define volume buckets
bucket_ranges = [(1, 5), (6, 20), (21, 100), (101, float('inf'))]
bucket_names = ['vol_1_5', 'vol_6_20', 'vol_21_100', 'vol_101_plus']

# Count ticks in each volume bucket for each candle
for i, (lo, hi) in enumerate(bucket_ranges):
    bucket_name = bucket_names[i]
    
    if hi == float('inf'):
        bucket_counts = df[df['Volume'] >= lo].groupby('candle_time').size()
    else:
        bucket_counts = df[(df['Volume'] >= lo) & (df['Volume'] <= hi)].groupby('candle_time').size()
    
    candles[bucket_name] = candles['candle_time'].map(bucket_counts).fillna(0)

print(f"✓ Added {len(bucket_names)} volume bucket features")

# =============================================================================
# 4. TECHNICAL FEATURES
# =============================================================================
print("\\n4. Adding technical features...")

# Basic features
candles['vwap'] = (candles['close'] * candles['volume']).rolling(10).sum() / candles['volume'].rolling(10).sum()
candles['price_range'] = (candles['high'] - candles['low']) / candles['close']
candles['vwap_ratio'] = candles['close'] / candles['vwap']

# Moving averages
candles['sma_10'] = candles['close'].rolling(10).mean()
candles['sma_30'] = candles['close'].rolling(30).mean()
candles['sma_ratio'] = candles['sma_10'] / candles['sma_30']

# Momentum features
candles['momentum_5'] = candles['close'] / candles['close'].shift(5) - 1
candles['momentum_10'] = candles['close'] / candles['close'].shift(10) - 1

# Volume features
candles['volume_sma'] = candles['volume'].rolling(10).mean()
candles['volume_ratio'] = candles['volume'] / candles['volume_sma']

# Volatility
candles['volatility'] = candles['close'].pct_change().rolling(10).std()

# Price position features
candles['high_low_ratio'] = candles['high'] / candles['low']
candles['close_position'] = (candles['close'] - candles['low']) / (candles['high'] - candles['low'])

basic_feature_count = 12
print(f"✓ Added {basic_feature_count} basic technical features")

# Advanced features (if TA library available)
advanced_feature_count = 0
if HAVE_TA:
    print("  Adding advanced TA features...")
    
    try:
        # RSI
        candles['rsi'] = ta.momentum.rsi(candles['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(candles['close'])
        candles['macd'] = macd.macd()
        candles['macd_signal'] = macd.macd_signal()
        candles['macd_diff'] = candles['macd'] - candles['macd_signal']
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(candles['close'])
        candles['bb_upper'] = bb.bollinger_hband()
        candles['bb_lower'] = bb.bollinger_lband()
        candles['bb_ratio'] = (candles['close'] - candles['bb_lower']) / (candles['bb_upper'] - candles['bb_lower'])
        
        # Average True Range
        candles['atr'] = ta.volatility.average_true_range(candles['high'], candles['low'], candles['close'])
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(candles['high'], candles['low'], candles['close'])
        candles['stoch_k'] = stoch.stoch()
        candles['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        candles['williams_r'] = ta.momentum.williams_r(candles['high'], candles['low'], candles['close'])
        
        advanced_feature_count = 10
        print(f"  ✓ Added {advanced_feature_count} advanced TA features")
        
    except Exception as e:
        print(f"  ⚠ Error adding some advanced features: {e}")
        advanced_feature_count = 0

# =============================================================================
# 5. TARGET VARIABLE
# =============================================================================
print("\\n5. Creating target variable...")

returns = candles['close'].pct_change()
candles['target'] = np.select(
    [returns < -PRICE_THRESHOLD, returns > PRICE_THRESHOLD],
    [-1, 1], 
    default=0
)

# =============================================================================
# 6. FEATURE PREPARATION
# =============================================================================
print("\\n6. Preparing features for modeling...")

# Define feature sets
basic_features = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'price_range', 'vwap_ratio',
                 'sma_10', 'sma_30', 'sma_ratio', 'momentum_5', 'momentum_10', 'volume_ratio', 
                 'volatility', 'high_low_ratio', 'close_position']

volume_features = bucket_names

advanced_features = []
if HAVE_TA and advanced_feature_count > 0:
    advanced_features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_ratio', 'atr', 
                        'stoch_k', 'stoch_d', 'williams_r']

# Combine all features
feature_cols = basic_features + volume_features + advanced_features

# Remove features that don't exist
existing_features = [col for col in feature_cols if col in candles.columns]
missing_features = [col for col in feature_cols if col not in candles.columns]

if missing_features:
    print(f"⚠ Missing features: {missing_features}")

print(f"Using {len(existing_features)} features")

# Clean data and prepare for modeling
candles_clean = candles.dropna()
X = candles_clean[existing_features]
y = candles_clean['target']

# Map target to 0, 1, 2 for sklearn
y_mapped = y.map({-1: 0, 0: 1, 1: 2})

print(f"Final dataset shape: {X.shape}")
print("Class distribution:")
class_counts = y.value_counts().sort_index()
class_pcts = y.value_counts(normalize=True).sort_index() * 100
for label, name in zip([-1, 0, 1], ['Down', 'Flat', 'Up']):
    if label in class_counts:
        print(f"  {name}: {class_counts[label]} ({class_pcts[label]:.1f}%)")

# Check if we have enough data
if len(X) < 100:
    print("⚠ Warning: Very small dataset. Consider using more data or reducing sample ratio.")
    N_SPLITS = min(3, len(X) // 20)

# =============================================================================
# 7. MODEL TRAINING AND EVALUATION
# =============================================================================
print(f"\\n7. Running {N_SPLITS}-fold time series cross-validation...")

# Define models with optimized parameters
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", 
        min_samples_split=5, min_samples_leaf=2, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, 
        subsample=0.8, random_state=42)
}

if HAVE_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", eval_metric="mlogloss", 
        random_state=42, verbosity=0)

# Cross-validation
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
results = {}

for name, clf in models.items():
    print(f"\\nTraining {name}...")
    fold_accuracies = []
    fold_reports = []
    fold_conf_matrices = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_mapped.iloc[train_idx], y_mapped.iloc[test_idx]
        
        # Train model
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
        
        # Classification report
        report = classification_report(y_test, y_pred, labels=[0, 1, 2], 
                                     target_names=['Down', 'Flat', 'Up'], 
                                     output_dict=True, zero_division=0)
        fold_reports.append(report)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        fold_conf_matrices.append(conf_matrix)
        
        print(f"  Fold {fold+1}: {accuracy*100:.1f}% accuracy")
    
    # Store results
    results[name] = {
        'accuracy': np.mean(fold_accuracies),
        'std': np.std(fold_accuracies),
        'reports': fold_reports,
        'conf_matrices': fold_conf_matrices
    }

# =============================================================================
# 8. RESULTS ANALYSIS
# =============================================================================
print("\\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

best_model = None
best_accuracy = 0

for name, res in results.items():
    accuracy = res['accuracy']
    std = res['std']
    print(f"\\n{name}:")
    print(f"  Average Accuracy: {accuracy*100:.2f}% (±{std*100:.2f}%)")
    
    # Average classification report
    avg_report = {}
    for class_idx, class_name in enumerate(['Down', 'Flat', 'Up']):
        metrics = {'precision': [], 'recall': [], 'f1-score': []}
        for report in res['reports']:
            if str(class_idx) in report:
                for metric in metrics:
                    metrics[metric].append(report[str(class_idx)][metric])
        
        avg_metrics = {metric: np.mean(values) if values else 0.0 
                      for metric, values in metrics.items()}
        avg_report[class_name] = avg_metrics
    
    print("  Average Classification Report:")
    for class_name, metrics in avg_report.items():
        print(f"    {class_name:>4}: Precision={metrics['precision']:.3f} "
              f"Recall={metrics['recall']:.3f} F1={metrics['f1-score']:.3f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name

print(f"\\n🏆 Best Model: {best_model} with {best_accuracy*100:.2f}% accuracy")

# =============================================================================
# 9. FEATURE IMPORTANCE
# =============================================================================
if best_model and best_model in models:
    print(f"\\n" + "="*50)
    print(f"FEATURE IMPORTANCE ({best_model})")
    print("="*50)
    
    # Train final model on all data
    final_model = models[best_model]
    final_model.fit(X, y_mapped)
    
    # Get feature importance
    if hasattr(final_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': existing_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\\nTop 15 most important features:")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
            print(f"{i+1:2d}. {row['feature']:>20}: {row['importance']:.4f}")

# =============================================================================
# 10. SUMMARY
# =============================================================================
print(f"\\n" + "="*70)
print("PIPELINE SUMMARY")
print("="*70)
print(f"✓ Processed {len(parquet_files)} files with {total_rows:,} total trades")
print(f"✓ Created {len(candles_clean)} clean {CANDLE_FREQ} candles")
print(f"✓ Used {len(existing_features)} features:")
print(f"  - {len(basic_features)} basic technical features")
print(f"  - {len(volume_features)} volume bucket features")
print(f"  - {advanced_feature_count} advanced TA features")
print(f"✓ Best model: {best_model} ({best_accuracy*100:.1f}% accuracy)")
print(f"✓ Price threshold: ±{PRICE_THRESHOLD*100:.1f}%")

print("\\n🚀 Pipeline completed successfully!")
print("\\nTo customize:")
print("- Adjust PRICE_THRESHOLD for different sensitivity")
print("- Change CANDLE_FREQ for different timeframes")
print("- Modify FILE_PATTERN to match your files")
print("- Set SAMPLE_RATIO for faster testing")
'''

# Save to file
with open('price_prediction_pipeline.py', 'w') as f:
    f.write(code_content)

print("✅ Created 'price_prediction_pipeline.py'")
print("\\nThis optimized version includes:")
print("- Robust file handling for multiple parquet files")
print("- Optional sampling for faster testing")
print("- Better error handling and progress reporting")
print("- More technical features")
print("- Detailed results analysis")
print("- Easy configuration parameters at the top")

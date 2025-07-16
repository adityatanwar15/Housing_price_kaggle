import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
ROLLING_WINDOW = 100
NUM_PREDICTIONS = 200
TICK_THRESHOLD = 0.0005  # Reduced threshold for more sensitivity
FEATURE_COLUMNS = []  # Populated later

# STEP 1: Load and process uploaded parquet files
def load_candles():
    # Load all uploaded chunks
    chunk_files = ['chunk_0000.parquet', 'chunk_0001.parquet', 'chunk_0002.parquet', 'chunk_0003.parquet']
    dfs = []
    
    for file in chunk_files:
        try:
            df = pd.read_parquet(file)
            # Filter for Trade data and clean
            df = df[df['Type'] == 'Trade'].dropna(subset=['Price', 'Volume'])
            df['Date-Time'] = pd.to_datetime(df['Date-Time'])
            df['candle_time'] = df['Date-Time'].dt.floor('15min')
            dfs.append(df[['Date-Time', 'Price', 'Volume', 'candle_time']])
            print(f"Loaded {len(df)} trades from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dfs:
        raise ValueError("No data loaded successfully")
    
    # Combine all data
    trades = pd.concat(dfs).sort_values('Date-Time')
    print(f"Total trades loaded: {len(trades)}")
    
    # Create 15-minute candles
    grouped = trades.groupby('candle_time').agg(
        open=('Price', 'first'),
        high=('Price', 'max'),
        low=('Price', 'min'),
        close=('Price', 'last'),
        volume=('Volume', 'sum'),
        num_ticks=('Price', 'count'),
        total_turnover=('Price', lambda x: (x * trades.loc[x.index, 'Volume']).sum())
    ).reset_index()
    
    grouped['vwap'] = grouped['total_turnover'] / grouped['volume']
    grouped.dropna(inplace=True)
    print(f"Created {len(grouped)} 15-minute candles")
    
    return grouped

# STEP 2: Enhanced feature engineering for classification
def add_classification_features(c):
    # Basic price features
    c['vwap'] = c['vwap'].fillna(c['close'])
    
    # Multiple timeframe moving averages
    for period in [5, 10, 15, 20, 30]:
        c[f'sma_{period}'] = c['close'].rolling(period).mean()
        c[f'price_vs_sma_{period}'] = (c['close'] - c[f'sma_{period}']) / c[f'sma_{period}']
    
    # VWAP features
    c['vwap_short'] = c['vwap'].rolling(5).mean()
    c['vwap_med'] = c['vwap'].rolling(15).mean()
    c['vwap_long'] = c['vwap'].rolling(30).mean()
    c['price_vs_vwap'] = (c['close'] - c['vwap']) / c['vwap']
    c['vwap_trend'] = c['vwap'].pct_change(5)
    
    # Price momentum and volatility
    for period in [3, 5, 10]:
        c[f'momentum_{period}'] = c['close'].pct_change(period)
        c[f'volatility_{period}'] = c['close'].rolling(period).std() / c['close'].rolling(period).mean()
    
    # Volume features
    c['volume_sma'] = c['volume'].rolling(20).mean()
    c['volume_ratio'] = c['volume'] / c['volume_sma']
    c['volume_momentum'] = c['volume'].pct_change(5)
    
    # Tick intensity features
    c['tick_intensity'] = c['num_ticks'] / c['num_ticks'].rolling(20).mean()
    c['tick_momentum'] = c['num_ticks'].pct_change(3)
    
    # Price range features
    c['high_low_ratio'] = (c['high'] - c['low']) / c['close']
    c['close_position'] = (c['close'] - c['low']) / (c['high'] - c['low'])
    
    # Advanced features
    c['price_acceleration'] = c['close'].pct_change().diff()
    c['volume_price_trend'] = c['volume'] * c['close'].pct_change()
    
    # Rolling statistics
    c['price_zscore'] = (c['close'] - c['close'].rolling(20).mean()) / c['close'].rolling(20).std()
    c['volume_zscore'] = (c['volume'] - c['volume'].rolling(20).mean()) / c['volume'].rolling(20).std()
    
    # Trend strength
    c['trend_strength'] = abs(c['close'].rolling(10).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1]))
    
    # Enhanced tick-volume velocity
    c['rolling_ticks'] = c['num_ticks'].rolling(window=5, min_periods=1).sum()
    c['rolling_volume'] = c['volume'].rolling(window=5, min_periods=1).sum()
    c['price_change_pct'] = c['close'].pct_change(periods=5)
    c['tick_volume_velocity'] = (c['price_change_pct'] * c['rolling_ticks']) / (c['rolling_volume'] + 1e-8)
    
    # Define feature columns (excluding basic OHLCV)
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = [col for col in c.columns if col not in ['candle_time', 'open', 'high', 'low', 'close', 'volume', 'num_ticks', 'total_turnover']]
    
    # Fill missing values
    for col in FEATURE_COLUMNS:
        c[col] = c[col].fillna(method='ffill').fillna(0)
        # Replace infinite values
        c[col] = c[col].replace([np.inf, -np.inf], 0)
    
    print(f"Created {len(FEATURE_COLUMNS)} features for classification")
    return c

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
    
    for i in range(window, min(window + num_preds, len(candles) - 1)):
        train = candles.iloc[i-window:i].copy()
        test = candles.iloc[i:i+1]
        
        # Prepare training data
        X_train = train[FEATURE_COLUMNS].iloc[:-1]  # Remove last row to align with shifted target
        y_train = train['y_class'].shift(-1).iloc[:-1].dropna()
        
        # Ensure alignment
        min_len = min(len(X_train), len(y_train))
        X_train = X_train.iloc[:min_len]
        y_train = y_train.iloc[:min_len]
        
        if len(X_train) < 30 or len(y_train.unique()) < 2:
            continue
        
        # Train multiple models and ensemble
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
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        if not trained_models:
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

# MAIN EXECUTION
if __name__ == "__main__":
    print("📥 Loading data from uploaded files...")
    candles = load_candles()
    
    print("🧠 Adding enhanced classification features...")
    candles = add_classification_features(candles)
    
    print("🚀 Running classification predictions...")
    results_df = run_classification_predictions(candles)
    
    if len(results_df) > 0:
        # Calculate accuracy
        acc = (results_df['pred_direction'] == results_df['actual_direction']).mean() * 100
        print(f"\n🎯 Final Classification Accuracy: {acc:.2f}%")
        
        # Detailed performance analysis
        print("\n📊 Performance Analysis:")
        print("Confusion Matrix:")
        print(confusion_matrix(results_df['actual_direction'], results_df['pred_direction']))
        
        print("\nClassification Report:")
        print(classification_report(results_df['actual_direction'], results_df['pred_direction']))
        
        # Direction-wise accuracy
        for direction in [-1, 0, 1]:
            subset = results_df[results_df['actual_direction'] == direction]
            if len(subset) > 0:
                dir_acc = (subset['pred_direction'] == direction).mean() * 100
                direction_name = {-1: 'DOWN', 0: 'FLAT', 1: 'UP'}[direction]
                print(f"{direction_name} accuracy: {dir_acc:.2f}% ({len(subset)} samples)")
        
        print(f"\n✅ Results ready! Total predictions: {len(results_df)}")
    else:
        print("❌ No predictions generated. Check data and parameters.")

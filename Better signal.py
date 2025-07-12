import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the enhanced 15-minute candles data generated earlier
candles = pd.read_csv('enhanced_15min_candles.csv', index_col=0, parse_dates=True)

# ------------------------------------------------------------------
# TARGET DEFINITION
# ------------------------------------------------------------------
# "Packet" (interpreted here as percentage) change in Close price.
# We'll classify each candle as:
#   1 = Bullish  (Close higher than previous Close by > 0.10%)
#   0 = Sideways (|ΔClose| ≤ 0.10%)
#  -1 = Bearish  (Close lower than previous Close by > 0.10%)

candles = candles.sort_index()
# Compute % change vs previous candle
candles['pct_change_close'] = candles['Close'].pct_change()*100  # in %

threshold = 0.10  # 0.10% threshold (\~1 tick on futures)

def label_row(p):
    if p > threshold:  # bullish
        return 1
    elif p < -threshold:  # bearish
        return -1
    else:
        return 0

candles['target'] = candles['pct_change_close'].apply(label_row)

# Drop first row with NaN % change
candles = candles.dropna(subset=['pct_change_close'])

# ------------------------------------------------------------------
# FEATURE SET
# ------------------------------------------------------------------
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'vwap', 'vwap_150', 'total_ticks'
] + [col for col in candles.columns if col.startswith('ticks_vol_')]

X = candles[feature_cols]
y = candles['target']

# Replace any remaining NaNs (possible in vwap_150 early rows)
X = X.fillna(X.median())

# ------------------------------------------------------------------
# TRAIN-TEST SPLIT & MODEL PIPELINE
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=False)  # time-series => no shuffle

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced'
    ))
])

pipeline.fit(X_train, y_train)

# ------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------
print("Classification report on held-out test set:\n")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))

# Quick accuracy metric
acc = (y_pred == y_test).mean()*100
print(f"\nOverall Accuracy: {acc:.2f}%")

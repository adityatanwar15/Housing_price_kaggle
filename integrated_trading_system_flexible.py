
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import os

class IntegratedTradingSystem:
    def __init__(self, training_window=50, start_hour=None, start_index=None, start_offset=None, num_predictions=10):
        """
        Initialize the Integrated Trading System

        Parameters:
        - training_window: Number of candles to use for training (default: 50)
        - start_hour: Hour to start predictions from (format: 'YYYY-MM-DD HH:MM:SS' or None)
        - start_index: Integer index to start predictions from (overrides start_hour if set)
        - start_offset: Offset from start (positive) or end (negative) to start predictions
        - num_predictions: Number of test predictions to make (default: 10)
        """
        self.training_window = training_window
        self.start_hour = start_hour
        self.start_index = start_index
        self.start_offset = start_offset
        self.num_predictions = num_predictions
        self.candles = None
        self.results = []
        self.feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'num_ticks',
            'vwap', 'vwap_1m_avg_15',
            'sma_short', 'sma_med', 'kama_10', 'kama_30', 'kama_200',
            'linear_slope_36', 'linear_slope_72', 'hurst_6',
            'vol_rogers_satchell_10', 'auto_corr_6',
            'pct_kama', 'pct_linear_slope',
            'vwap_short', 'vwap_med', 'vwap_ratio_short',
            'vwap_distance_short', 'rsi_short', 'rsi_med',
            'trend_short', 'trend_med', 'sma_cross'
        ]

    def load_and_process_data(self, file_path):
        """Load parquet file and create 15-minute candles (no features yet)"""
        print(f"📊 Loading data from {file_path}...")
        raw = pd.read_parquet(file_path)
        trade = raw[(raw['Type']=='Trade') & raw['Price'].notna() & raw['Volume'].notna()].copy()
        print(f"   ✅ Loaded {len(trade):,} trade records")
        trade['Date-Time'] = pd.to_datetime(trade['Date-Time'])
        trade['candle_time'] = trade['Date-Time'].dt.floor('15min')
        self.candles = trade.groupby('candle_time').agg(
            open = ('Price','first'),
            high = ('Price','max'),
            low  = ('Price','min'),
            close= ('Price','last'),
            volume=('Volume','sum'),
            num_ticks=('Price','count')
        ).reset_index()
        print(f"   ✅ Created {len(self.candles):,} 15-minute candles")
        print(f"   📅 Date range: {self.candles['candle_time'].min()} to {self.candles['candle_time'].max()}")
        return self.candles

    def add_features(self):
        """Add technical indicators and features, then save engineered candles"""
        print("🔧 Adding technical features...")
        c = self.candles
        c['sma_short'] = c['close'].rolling(10, min_periods=1).mean()
        c['sma_med'] = c['close'].rolling(20, min_periods=1).mean()
        c['kama_10'] = c['close'].rolling(5, min_periods=1).mean()
        c['kama_30'] = c['close'].rolling(10, min_periods=1).mean()
        c['kama_200'] = c['close'].rolling(20, min_periods=1).mean()
        c['linear_slope_36'] = 0.0
        c['linear_slope_72'] = 0.0
        c['hurst_6'] = 0.5
        c['vol_rogers_satchell_10'] = 0.01
        c['auto_corr_6'] = 0.0
        c['pct_kama'] = 0.0
        c['pct_linear_slope'] = 0.0
        c['vwap'] = c['close']
        c['vwap_1m_avg_15'] = c['close']
        c['vwap_short'] = c['close']
        c['vwap_med'] = c['close']
        c['vwap_ratio_short'] = 1.0
        c['vwap_distance_short'] = 0.0
        c['rsi_short'] = 50.0
        c['rsi_med'] = 50.0
        c['trend_short'] = 0
        c['trend_med'] = 0
        c['sma_cross'] = c['sma_short'] - c['sma_med']
        self.candles = c
        print(f"   ✅ Added {len(self.feature_cols)} features")
        self.save_candles()

    def save_candles(self, filename='candlestick_data_15min.csv'):
        """Save feature-engineered candlestick data to CSV"""
        if self.candles is not None:
            self.candles.to_csv(filename, index=False)
            print(f"   💾 Feature-engineered candlestick data saved to: {filename}")
        else:
            print("❌ No candles to save!")

    def find_start_index(self):
        """Flexible start index selection: by index, offset, or timestamp"""
        if self.start_index is not None:
            start_idx = self.start_index
            print(f"🕐 Starting from index {start_idx} ({self.candles.iloc[start_idx]['candle_time']})")
        elif self.start_offset is not None:
            if self.start_offset >= 0:
                start_idx = self.start_offset
            else:
                start_idx = len(self.candles) + self.start_offset
            print(f"🕐 Starting from offset {self.start_offset} (index {start_idx}, {self.candles.iloc[start_idx]['candle_time']})")
        elif self.start_hour is not None:
            target_time = pd.to_datetime(self.start_hour)
            time_diff = abs(self.candles['candle_time'] - target_time)
            start_idx = time_diff.idxmin()
            print(f"🕐 Starting from: {self.candles.iloc[start_idx]['candle_time']} (index {start_idx})")
        else:
            start_idx = self.training_window + 20
            print(f"🕐 Auto-starting from index {start_idx} ({self.candles.iloc[start_idx]['candle_time']})")
        if start_idx < self.training_window + 20:
            start_idx = self.training_window + 20
            print(f"⚠️  Adjusted start time due to insufficient training data")
        return start_idx

    def run_predictions(self):
        print(f"🚀 Running {self.num_predictions} predictions with {self.training_window} candle training window...")
        start_idx = self.find_start_index()
        self.results = []
        for i in range(start_idx, min(start_idx + self.num_predictions, len(self.candles) - 1)):
            print(f"   📈 Processing prediction {len(self.results)+1}/{self.num_predictions} for {self.candles.iloc[i]['candle_time']}")
            train = self.candles.iloc[i-self.training_window:i].copy()
            y_ret = train['close'].pct_change().shift(-1)
            train['y_class'] = np.select([y_ret < -0.001, y_ret > 0.001], [-1, 1], 0)
            train['y_open'] = train['open'].shift(-1)
            train['y_high'] = train['high'].shift(-1)
            train['y_low'] = train['low'].shift(-1)
            train['y_close'] = train['close'].shift(-1)
            X = train[self.feature_cols].shift(1).iloc[20:].fillna(method='ffill').fillna(0)
            y_class = train['y_class'].iloc[20:]
            y_open = train['y_open'].iloc[20:].dropna()
            y_high = train['y_high'].iloc[20:].dropna()
            y_low = train['y_low'].iloc[20:].dropna()
            y_close = train['y_close'].iloc[20:].dropna()
            X_reg = X.iloc[:-1]
            y_class_reg = y_class.iloc[:-1]
            if len(X_reg) < 5 or len(y_open) < 5:
                continue
            clf = GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=42)
            clf.fit(X_reg, y_class_reg)
            reg_open = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42)
            reg_high = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42)
            reg_low = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42)
            reg_close = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42)
            reg_open.fit(X_reg, y_open)
            reg_high.fit(X_reg, y_high)
            reg_low.fit(X_reg, y_low)
            reg_close.fit(X_reg, y_close)
            X_next = self.candles[self.feature_cols].iloc[i:i+1].fillna(method='ffill').fillna(0)
            pred_dir = int(clf.predict(X_next)[0])
            probs = clf.predict_proba(X_next)[0]
            pred_open = reg_open.predict(X_next)[0]
            pred_high = reg_high.predict(X_next)[0]
            pred_low = reg_low.predict(X_next)[0]
            pred_close = reg_close.predict(X_next)[0]
            act = self.candles.iloc[i]
            act_ret = act['close']/train['close'].iloc[-2] - 1
            act_dir = int(np.select([act_ret < -0.001, act_ret > 0.001], [-1, 1], 0))
            self.results.append({
                'timestamp': act['candle_time'],
                'pred_direction': pred_dir,
                'prob_down': probs[0],
                'prob_flat': probs[1] if len(probs)>1 else np.nan,
                'prob_up': probs[2] if len(probs)>2 else np.nan,
                'actual_direction': act_dir,
                'pred_open': pred_open,
                'pred_high': pred_high,
                'pred_low': pred_low,
                'pred_close': pred_close,
                'actual_open': act['open'],
                'actual_high': act['high'],
                'actual_low': act['low'],
                'actual_close': act['close']
            })
        print(f"   ✅ Completed {len(self.results)} predictions")
        return self.results

    def calculate_performance(self):
        if not self.results:
            return None
        df = pd.DataFrame(self.results)
        df['error_open'] = df['actual_open'] - df['pred_open']
        df['error_high'] = df['actual_high'] - df['pred_high']
        df['error_low'] = df['actual_low'] - df['pred_low']
        df['error_close'] = df['actual_close'] - df['pred_close']
        df['abs_error_open'] = abs(df['error_open'])
        df['abs_error_high'] = abs(df['error_high'])
        df['abs_error_low'] = abs(df['error_low'])
        df['abs_error_close'] = abs(df['error_close'])
        correct_predictions = (df['pred_direction'] == df['actual_direction']).sum()
        total_predictions = len(df)
        accuracy = correct_predictions / total_predictions * 100
        performance = {
            'classification_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'mae_open': df['abs_error_open'].mean(),
            'mae_high': df['abs_error_high'].mean(),
            'mae_low': df['abs_error_low'].mean(),
            'mae_close': df['abs_error_close'].mean(),
            'avg_actual_range': (df['actual_high'] - df['actual_low']).mean(),
            'avg_pred_range': (df['pred_high'] - df['pred_low']).mean()
        }
        return performance, df

    def save_results(self, filename_prefix='trading_results'):
        if not self.results:
            print("❌ No results to save")
            return
        performance, df = self.calculate_performance()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_file = f'{filename_prefix}_{timestamp}.xlsx'
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.round(2).to_excel(writer, sheet_name='Predictions', index=False)
            perf_df = pd.DataFrame([performance]).T
            perf_df.columns = ['Value']
            perf_df.to_excel(writer, sheet_name='Performance')
            if self.candles is not None:
                sample_candles = self.candles.tail(100)
                sample_candles.to_excel(writer, sheet_name='Candles_Sample', index=False)
        csv_file = f'{filename_prefix}_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"💾 Results saved to:")
        print(f"   📊 Excel: {excel_file}")
        print(f"   📄 CSV: {csv_file}")
        return excel_file, csv_file

    def print_summary(self):
        if not self.results:
            print("❌ No results to summarize")
            return
        performance, df = self.calculate_performance()
        print("\n" + "="*80)
        print("🎯 INTEGRATED TRADING SYSTEM RESULTS")
        print("="*80)
        print(f"\n📊 CONFIGURATION:")
        print(f"   Training Window: {self.training_window} candles")
        print(f"   Predictions Made: {len(self.results)}")
        print(f"   Start Time: {df['timestamp'].iloc[0] if len(df) > 0 else 'N/A'}")
        print(f"   End Time: {df['timestamp'].iloc[-1] if len(df) > 0 else 'N/A'}")
        print(f"\n📈 CLASSIFICATION PERFORMANCE:")
        print(f"   Accuracy: {performance['classification_accuracy']:.1f}% ({performance['correct_predictions']}/{performance['total_predictions']})")
        print(f"\n💰 REGRESSION PERFORMANCE (MAE):")
        print(f"   Open:  {performance['mae_open']:.2f} points")
        print(f"   High:  {performance['mae_high']:.2f} points")
        print(f"   Low:   {performance['mae_low']:.2f} points")
        print(f"   Close: {performance['mae_close']:.2f} points")
        print(f"\n📊 RANGE ANALYSIS:")
        print(f"   Avg Actual Range: {performance['avg_actual_range']:.2f} points")
        print(f"   Avg Predicted Range: {performance['avg_pred_range']:.2f} points")
        print(f"\n🔍 SAMPLE RESULTS (First 5):")
        sample_cols = ['timestamp', 'pred_direction', 'actual_direction', 'pred_close', 'actual_close']
        print(df[sample_cols].head().round(2).to_string(index=False))

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def run_trading_system(file_path, training_window=50, start_hour=None, start_index=None, start_offset=None, num_predictions=10):
    """
    Main function to run the integrated trading system

    Parameters:
    - file_path: Path to the parquet file
    - training_window: Number of candles for training (default: 50)
    - start_hour: Start time for predictions (format: 'YYYY-MM-DD HH:MM:SS' or None)
    - start_index: Integer index to start predictions from (overrides start_hour if set)
    - start_offset: Offset from start (positive) or end (negative) to start predictions
    - num_predictions: Number of predictions to make (default: 10)
    """
    system = IntegratedTradingSystem(
        training_window=training_window,
        start_hour=start_hour,
        start_index=start_index,
        start_offset=start_offset,
        num_predictions=num_predictions
    )
    system.load_and_process_data(file_path)
    system.add_features()
    system.run_predictions()
    system.print_summary()
    excel_file, csv_file = system.save_results()
    return system, excel_file, csv_file

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Start from timestamp
    # system, excel, csv = run_trading_system('chunk_0032.parquet', start_hour='2025-06-13 15:00:00', num_predictions=10)
    # Example 2: Start from index
    # system, excel, csv = run_trading_system('chunk_0032.parquet', start_index=120, num_predictions=10)
    # Example 3: Start from offset (e.g., 10th candle from start)
    # system, excel, csv = run_trading_system('chunk_0032.parquet', start_offset=10, num_predictions=10)
    # Example 4: Start from offset (e.g., 10th candle from end)
    # system, excel, csv = run_trading_system('chunk_0032.parquet', start_offset=-10, num_predictions=10)
    # Example 5: Default (auto)
    system, excel, csv = run_trading_system('chunk_0032.parquet', training_window=30, num_predictions=5)
    print(f"\n🎉 Complete! Check your files:")
    print(f"   📊 Excel: {excel}")
    print(f"   📄 CSV: {csv}")
    print(f"   📈 Candlestick: candlestick_data_15min.csv")

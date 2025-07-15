
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
            'vwap', 'vwap_1m_avg_15', 'vwap_short', 'vwap_med', 
            'vwap_ratio_short', 'vwap_distance_short',
            'sma_short', 'sma_med', 'kama_10', 'kama_30', 'kama_200',
            'linear_slope_36', 'linear_slope_72', 'hurst_6',
            'vol_rogers_satchell_10', 'auto_corr_6',
            'pct_kama', 'pct_linear_slope',
            'rsi_short', 'rsi_med', 'trend_short', 'trend_med', 'sma_cross',
            'volume_sma', 'volume_ratio', 'tick_intensity', 'price_velocity'
        ]

    def load_and_process_data(self, file_path):
        """Load parquet file and create 15-minute candles with proper volume/tick accumulation"""
        print(f"📊 Loading data from {file_path}...")
        FILE_PATTERN=r'C:\Users\aditya-tanwar\OneDrive - MMC\Documents\my_work\study_work\data\\'
        MAX_FILES=None
        #Find parquet files
        parquet_files = os.listdir(FILE_PATTERN)
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

        for  file_path in parquet_files:
            try:
                print(f"  Loading {file_path}... ", end="")
                df = pd.read_parquet(FILE_PATTERN+file_path)
                
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
        trade = pd.concat(all_data, ignore_index=True)

        trade['Date-Time'] = pd.to_datetime(trade['Date-Time'])
        trade['candle_time'] = trade['Date-Time'].dt.floor('15min')

        # Proper OHLCV aggregation with volume-weighted calculations
        self.candles = trade.groupby('candle_time').agg(
            open = ('Price', 'first'),
            high = ('Price', 'max'),
            low  = ('Price', 'min'),
            close= ('Price', 'last'),
            volume=('Volume', 'sum'),  # Total volume
            num_ticks=('Price', 'count'),  # Number of ticks/trades
            volume_weighted_price=('Price', lambda x: np.average(x, weights=trade.loc[x.index, 'Volume'])),  # VWAP for this candle
            total_turnover=('Volume', lambda x: (trade.loc[x.index, 'Price'] * trade.loc[x.index, 'Volume']).sum())  # Price * Volume sum
        ).reset_index()

        # Calculate proper VWAP (Volume Weighted Average Price)
        self.candles['vwap'] = self.candles['total_turnover'] / self.candles['volume']
        self.candles['vwap'] = self.candles['vwap'].fillna(self.candles['close'])  # Fallback to close if no volume

        print(f"   ✅ Created {len(self.candles):,} 15-minute candles")
        print(f"   📅 Date range: {self.candles['candle_time'].min()} to {self.candles['candle_time'].max()}")
        print(f"   📊 Avg volume per candle: {self.candles['volume'].mean():.0f}")
        print(f"   📈 Avg ticks per candle: {self.candles['num_ticks'].mean():.0f}")

        return self.candles

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_linear_slope(self, prices, period):
        """Calculate linear regression slope over period"""
        slopes = []
        for i in range(len(prices)):
            if i < period - 1:
                slopes.append(0)
            else:
                y = prices.iloc[i-period+1:i+1].values
                x = np.arange(len(y))
                if len(y) > 1 and np.std(y) > 0:
                    slope, _, _, _, _ = linregress(x, y)
                    slopes.append(slope)
                else:
                    slopes.append(0)
        return pd.Series(slopes, index=prices.index)

    def calculate_hurst_exponent(self, prices, period=6):
        """Calculate Hurst exponent (simplified version)"""
        hurst_values = []
        for i in range(len(prices)):
            if i < period:
                hurst_values.append(0.5)
            else:
                ts = prices.iloc[i-period:i].values
                if len(ts) > 2:
                    # Simplified Hurst calculation
                    lags = range(2, min(period//2 + 1, len(ts)))
                    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                    if len(tau) > 1 and all(t > 0 for t in tau):
                        poly = np.polyfit(np.log(lags), np.log(tau), 1)
                        hurst_values.append(max(0, min(1, poly[0])))
                    else:
                        hurst_values.append(0.5)
                else:
                    hurst_values.append(0.5)
        return pd.Series(hurst_values, index=prices.index)

    def calculate_rogers_satchell_volatility(self, high, low, open_price, close, period=10):
        """Calculate Rogers-Satchell volatility estimator"""
        rs = np.log(high/close) * np.log(high/open_price) + np.log(low/close) * np.log(low/open_price)
        return rs.rolling(window=period, min_periods=1).mean().fillna(0.01)

    def calculate_autocorrelation(self, prices, lag=6):
        """Calculate autocorrelation at given lag"""
        autocorr_values = []
        for i in range(len(prices)):
            if i < lag * 2:
                autocorr_values.append(0)
            else:
                series = prices.iloc[i-lag*2:i]
                if len(series) > lag:
                    corr = series.autocorr(lag=lag)
                    autocorr_values.append(corr if not np.isnan(corr) else 0)
                else:
                    autocorr_values.append(0)
        return pd.Series(autocorr_values, index=prices.index)

    def add_features(self):
        """Add comprehensive technical indicators and features"""
        print("🔧 Adding comprehensive technical features...")
        c = self.candles

        # Basic moving averages
        c['sma_short'] = c['close'].rolling(10, min_periods=1).mean()
        c['sma_med'] = c['close'].rolling(20, min_periods=1).mean()
        c['kama_10'] = c['close'].rolling(5, min_periods=1).mean()
        c['kama_30'] = c['close'].rolling(10, min_periods=1).mean()
        c['kama_200'] = c['close'].rolling(20, min_periods=1).mean()

        # VWAP indicators (already calculated basic VWAP)
        c['vwap_1m_avg_15'] = c['vwap'].rolling(15, min_periods=1).mean()
        c['vwap_short'] = c['vwap'].rolling(5, min_periods=1).mean()
        c['vwap_med'] = c['vwap'].rolling(10, min_periods=1).mean()
        c['vwap_ratio_short'] = c['close'] / c['vwap_short']
        c['vwap_distance_short'] = c['close'] - c['vwap_short']

        # RSI calculations
        print("   📊 Calculating RSI...")
        c['rsi_short'] = self.calculate_rsi(c['close'], 14)
        c['rsi_med'] = self.calculate_rsi(c['close'], 21)

        # Linear slopes
        print("   📈 Calculating linear slopes...")
        c['linear_slope_36'] = self.calculate_linear_slope(c['close'], 36)
        c['linear_slope_72'] = self.calculate_linear_slope(c['close'], 72)

        # Hurst exponent
        print("   🌊 Calculating Hurst exponent...")
        c['hurst_6'] = self.calculate_hurst_exponent(c['close'], 6)

        # Rogers-Satchell volatility
        print("   📊 Calculating volatility...")
        c['vol_rogers_satchell_10'] = self.calculate_rogers_satchell_volatility(
            c['high'], c['low'], c['open'], c['close'], 10
        )

        # Autocorrelation
        print("   🔄 Calculating autocorrelation...")
        c['auto_corr_6'] = self.calculate_autocorrelation(c['close'], 6)

        # Percentage changes and ratios
        c['pct_kama'] = (c['close'] - c['kama_10']) / c['kama_10'] * 100
        c['pct_linear_slope'] = c['linear_slope_36'] / c['close'] * 100

        # Trend indicators
        c['trend_short'] = np.where(c['sma_short'] > c['sma_short'].shift(1), 1, 
                                   np.where(c['sma_short'] < c['sma_short'].shift(1), -1, 0))
        c['trend_med'] = np.where(c['sma_med'] > c['sma_med'].shift(1), 1,
                                 np.where(c['sma_med'] < c['sma_med'].shift(1), -1, 0))

        # SMA cross
        c['sma_cross'] = c['sma_short'] - c['sma_med']

        # Volume indicators
        print("   📊 Calculating volume indicators...")
        c['volume_sma'] = c['volume'].rolling(20, min_periods=1).mean()
        c['volume_ratio'] = c['volume'] / c['volume_sma']
        c['tick_intensity'] = c['num_ticks'] / c['num_ticks'].rolling(20, min_periods=1).mean()

        # Price velocity (rate of change)
        c['price_velocity'] = c['close'].pct_change(5) * 100

        # Fill any remaining NaN values
        for col in self.feature_cols:
            if col in c.columns:
                c[col] = c[col].fillna(method='ffill').fillna(0)

        self.candles = c
        print(f"   ✅ Added {len(self.feature_cols)} comprehensive features")
        self.save_candles()

    def save_candles(self, filename='candlestick_data_15min_engineered.csv'):
        """Save feature-engineered candlestick data to CSV"""
        if self.candles is not None:
            self.candles.to_csv(filename, index=False)
            print(f"   💾 Feature-engineered candlestick data saved to: {filename}")
            print(f"   📊 Saved {len(self.candles)} candles with {len(self.candles.columns)} columns")
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
            start_idx = max(self.training_window + 80, 100)  # Ensure enough data for all indicators
            print(f"🕐 Auto-starting from index {start_idx} ({self.candles.iloc[start_idx]['candle_time']})")

        # Ensure we have enough training data for all indicators
        min_required = max(self.training_window + 80, 100)
        if start_idx < min_required:
            start_idx = min_required
            print(f"⚠️  Adjusted start time to index {start_idx} due to indicator requirements")

        return start_idx

    def run_predictions(self):
        """Run the integrated classification + regression predictions"""
        print(f"🚀 Running {self.num_predictions} predictions with {self.training_window} candle training window...")
        start_idx = self.find_start_index()
        self.results = []

        for i in range(start_idx, min(start_idx + self.num_predictions, len(self.candles) - 1)):
            print(f"   📈 Processing prediction {len(self.results)+1}/{self.num_predictions} for {self.candles.iloc[i]['candle_time']}")

            # Get training data
            train = self.candles.iloc[i-self.training_window:i].copy()

            # --- CLASSIFICATION TARGET ---
            y_ret = train['close'].pct_change().shift(-1)
            train['y_class'] = np.select([y_ret < -0.001, y_ret > 0.001], [-1, 1], 0)

            # --- REGRESSION TARGETS (next candle's OHLC) ---
            train['y_open'] = train['open'].shift(-1)
            train['y_high'] = train['high'].shift(-1)
            train['y_low'] = train['low'].shift(-1)
            train['y_close'] = train['close'].shift(-1)

            # Prepare features (shift by 1 to avoid forward-looking bias)
            X = train[self.feature_cols].shift(1).iloc[20:].fillna(method='ffill').fillna(0)

            # Classification targets
            y_class = train['y_class'].iloc[20:]

            # Regression targets - drop NaN values
            y_open = train['y_open'].iloc[20:].dropna()
            y_high = train['y_high'].iloc[20:].dropna()
            y_low = train['y_low'].iloc[20:].dropna()
            y_close = train['y_close'].iloc[20:].dropna()

            # Align X with valid y indices (remove last row since we shifted -1)
            X_reg = X.iloc[:-1]
            y_class_reg = y_class.iloc[:-1]

            # Make sure we have enough data
            if len(X_reg) < 10 or len(y_open) < 10:
                continue

            # Train classification model
            clf = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
            clf.fit(X_reg, y_class_reg)

            # Train regression models
            reg_open = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
            reg_high = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
            reg_low = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
            reg_close = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)

            reg_open.fit(X_reg, y_open)
            reg_high.fit(X_reg, y_high)
            reg_low.fit(X_reg, y_low)
            reg_close.fit(X_reg, y_close)

            # Make predictions for current candle
            X_next = self.candles[self.feature_cols].iloc[i:i+1].fillna(method='ffill').fillna(0)

            # Classification predictions
            pred_dir = int(clf.predict(X_next)[0])
            probs = clf.predict_proba(X_next)[0]

            # Regression predictions
            pred_open = reg_open.predict(X_next)[0]
            pred_high = reg_high.predict(X_next)[0]
            pred_low = reg_low.predict(X_next)[0]
            pred_close = reg_close.predict(X_next)[0]

            # Actual values
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
                'actual_close': act['close'],
                'actual_volume': act['volume'],
                'actual_num_ticks': act['num_ticks'],
                'actual_vwap': act['vwap']
            })

        print(f"   ✅ Completed {len(self.results)} predictions")
        return self.results

    def calculate_performance(self):
        """Calculate comprehensive performance metrics"""
        if not self.results:
            return None

        df = pd.DataFrame(self.results)

        # Calculate errors
        df['error_open'] = df['actual_open'] - df['pred_open']
        df['error_high'] = df['actual_high'] - df['pred_high']
        df['error_low'] = df['actual_low'] - df['pred_low']
        df['error_close'] = df['actual_close'] - df['pred_close']

        df['abs_error_open'] = abs(df['error_open'])
        df['abs_error_high'] = abs(df['error_high'])
        df['abs_error_low'] = abs(df['error_low'])
        df['abs_error_close'] = abs(df['error_close'])

        # Percentage errors
        df['pct_error_close'] = abs(df['error_close']) / df['actual_close'] * 100

        # Classification accuracy
        correct_predictions = (df['pred_direction'] == df['actual_direction']).sum()
        total_predictions = len(df)
        accuracy = correct_predictions / total_predictions * 100

        # Direction-specific accuracy
        up_correct = ((df['pred_direction'] == 1) & (df['actual_direction'] == 1)).sum()
        down_correct = ((df['pred_direction'] == -1) & (df['actual_direction'] == -1)).sum()
        flat_correct = ((df['pred_direction'] == 0) & (df['actual_direction'] == 0)).sum()

        performance = {
            'classification_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'up_correct': up_correct,
            'down_correct': down_correct,
            'flat_correct': flat_correct,
            'mae_open': df['abs_error_open'].mean(),
            'mae_high': df['abs_error_high'].mean(),
            'mae_low': df['abs_error_low'].mean(),
            'mae_close': df['abs_error_close'].mean(),
            'mape_close': df['pct_error_close'].mean(),
            'avg_actual_range': (df['actual_high'] - df['actual_low']).mean(),
            'avg_pred_range': (df['pred_high'] - df['pred_low']).mean(),
            'avg_volume': df['actual_volume'].mean(),
            'avg_ticks': df['actual_num_ticks'].mean()
        }

        return performance, df

    def save_results(self, filename_prefix='trading_results'):
        """Save comprehensive results to Excel and CSV files"""
        if not self.results:
            print("❌ No results to save")
            return

        performance, df = self.calculate_performance()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save to Excel with multiple sheets
        excel_file = f'{filename_prefix}_str{timestamp}.xlsx'
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Main results
            df['timestamp'] = df['timestamp'].astype(str)
            candles['candle_time']=candles['candle_time'].astype('str')
            df.round(2).to_excel(writer, sheet_name='Predictions', index=False)

            # Performance summary
            perf_df = pd.DataFrame([performance]).T
            perf_df.columns = ['Value']
            perf_df.to_excel(writer, sheet_name='Performance')

            # Feature importance (if available)
            if hasattr(self, 'feature_importance'):
                self.feature_importance.to_excel(writer, sheet_name='Feature_Importance')

            # Full candlestick data with features
            if self.candles is not None:
                self.candles.to_excel(writer, sheet_name='Full_Candles', index=False)

        # Save to CSV
        csv_file = f'{filename_prefix}_{timestamp}.csv'
        df.to_csv(csv_file, index=False)

        print(f"💾 Results saved to:")
        print(f"   📊 Excel: {excel_file}")
        print(f"   📄 CSV: {csv_file}")

        return excel_file, csv_file

    def print_summary(self):
        """Print comprehensive performance summary"""
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
        print(f"   Features Used: {len(self.feature_cols)}")
        print(f"   Start Time: {df['timestamp'].iloc[0] if len(df) > 0 else 'N/A'}")
        print(f"   End Time: {df['timestamp'].iloc[-1] if len(df) > 0 else 'N/A'}")

        print(f"\n📈 CLASSIFICATION PERFORMANCE:")
        print(f"   Overall Accuracy: {performance['classification_accuracy']:.1f}% ({performance['correct_predictions']}/{performance['total_predictions']})")
        print(f"   Up Predictions Correct: {performance['up_correct']}")
        print(f"   Down Predictions Correct: {performance['down_correct']}")
        print(f"   Flat Predictions Correct: {performance['flat_correct']}")

        print(f"\n💰 REGRESSION PERFORMANCE:")
        print(f"   MAE Open:  {performance['mae_open']:.2f} points")
        print(f"   MAE High:  {performance['mae_high']:.2f} points")
        print(f"   MAE Low:   {performance['mae_low']:.2f} points")
        print(f"   MAE Close: {performance['mae_close']:.2f} points")
        print(f"   MAPE Close: {performance['mape_close']:.2f}%")

        print(f"\n📊 MARKET DATA ANALYSIS:")
        print(f"   Avg Actual Range: {performance['avg_actual_range']:.2f} points")
        print(f"   Avg Predicted Range: {performance['avg_pred_range']:.2f} points")
        print(f"   Avg Volume per Candle: {performance['avg_volume']:.0f}")
        print(f"   Avg Ticks per Candle: {performance['avg_ticks']:.0f}")

        print(f"\n🔍 SAMPLE RESULTS (First 5):")
        sample_cols = ['timestamp', 'pred_direction', 'actual_direction', 'pred_close', 'actual_close', 'actual_volume']
        print(df[sample_cols].head().round(2).to_string(index=False))

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_trading_system(file_path, training_window=50, start_hour=None, start_index=None, start_offset=None, num_predictions=10):
    """
    Main function to run the comprehensive integrated trading system

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

    # Load and process data
    system.load_and_process_data(file_path)

    # Add comprehensive features
    system.add_features()

    # Run predictions
    system.run_predictions()

    # Print summary
    system.print_summary()

    # Save results
    excel_file, csv_file = system.save_results()

    return system, excel_file, csv_file

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    FILE_PATTERN=r'C:\Users\aditya-tanwar\OneDrive - MMC\Documents\my_work\study_work\data\\'
        
    # Example usage with comprehensive features
    system, excel, csv = run_trading_system(
        file_path=FILE_PATTERN,
        training_window=5000,
        start_hour=9,  # Auto start
        num_predictions=5000
    )

    print(f"\n🎉 Complete! Check your files:")
    print(f"   📊 Excel: {excel}")
    print(f"   📄 CSV: {csv}")
    print(f"   📈 Candlestick: candlestick_data_15min_engineered.csv")

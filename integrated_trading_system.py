
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced forecasting (install if needed)
try:
    from statsmodels.tsa.api import VAR
    from arch import arch_model
    ADVANCED_FORECASTING = True
except ImportError:
    print("Warning: statsmodels and/or arch not available. Using simplified forecasting.")
    ADVANCED_FORECASTING = False

import gc

class IntegratedTradingSystem:
    def __init__(self, training_window=1000, retrain_frequency=50, forecast_window=3000):
        """
        Integrated trading system with classification and price forecasting

        training_window: Number of candles for classification model
        retrain_frequency: Retrain classification model every N candles
        forecast_window: Number of candles for VAR/GARCH forecasting
        """
        # Classification model parameters
        self.training_window = training_window
        self.retrain_frequency = retrain_frequency
        self.candles_since_retrain = 0
        self.classification_model = None

        # Forecasting parameters
        self.forecast_window = forecast_window
        self.price_history = pd.DataFrame()

        # Window sizes for features
        self.short_window = 10
        self.med_window = 20
        self.long_window = 50

        self.feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'num_ticks',
            'vol_100_199', 'vol_200_499', 'vol_500_999', 'vol_1000_1999', 
            'vol_2000_4999', 'vol_5000_5499', 'vol_5500_5999', 'vol_6000_7499', 
            'vol_7500_8999', 'vol_9000_9999', 'vol_10000_plus',
            'vwap', 'vwap_1m_avg_15',
            'kama_10', 'kama_30', 'kama_200',
            'linear_slope_36', 'linear_slope_72', 'hurst_6',
            'vol_rogers_satchell_10', 'bin_0_25', 'bin_25_75', 'bin_75_108',
            'auto_corr_6', 'pct_kama', 'pct_linear_slope',
            'sma_short', 'sma_med', 'vwap_short', 'vwap_med',
            'vwap_ratio_short', 'vwap_distance_short',
            'imbalance', 'small_vol_ratio', 'large_vol_ratio',
            'rsi_short', 'rsi_med', 'trend_short', 'trend_med', 'sma_cross'
        ]

        self.historical_candles = pd.DataFrame()

        # Forecasting results storage
        self.forecast_results = {
            'timestamps': [],
            'actual': {'low': [], 'close': [], 'high': []},
            'var_forecast': {'low': [], 'close': [], 'high': []},
            'garch_forecast': {'low': [], 'close': [], 'high': []},
            'hybrid_forecast': {'low': [], 'close': [], 'high': []}
        }

    def rolling_slope(self, series, window):
        """Calculate rolling linear regression slope"""
        slopes = np.full(len(series), np.nan)
        for i in range(window, len(series)):
            y = series.iloc[i-window:i]
            x = np.arange(window)
            slope, _, _, _, _ = linregress(x, y)
            slopes[i] = slope
        return slopes

    def rolling_hurst(self, series, window):
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

    def rogers_satchell_vol(self, high, low, close, open_, window):
        """Calculate Rogers-Satchell volatility estimator"""
        rs = np.log(high/open_) * np.log(high/close) + np.log(low/open_) * np.log(low/close)
        return rs.rolling(window).std()

    def rolling_autocorr(self, series, window):
        """Calculate rolling autocorrelation"""
        ac = np.full(len(series), np.nan)
        for i in range(window, len(series)):
            ac[i] = series.iloc[i-window:i].autocorr(lag=1)
        return ac

    def calculate_rsi(self, prices, window):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def process_tick_data_to_candles(self, tick_data):
        """Convert tick data to 15-minute candles with all features"""
        df = tick_data.copy()
        df['Date-Time'] = pd.to_datetime(df['Date-Time'])
        df = df.sort_values('Date-Time')
        df['day'] = df['Date-Time'].dt.date

        # Create 15-min candle time
        df['candle_time'] = df['Date-Time'].dt.floor('15min')

        # Tick logic: count a tick every time price changes
        df['price_change'] = df['Price'].diff().fillna(0) != 0
        df['tick_id'] = df.groupby('day')['price_change'].cumsum()

        # Aggregate to 15-min candles
        candles = df.groupby(['day', 'candle_time']).agg({
            'Price': ['first', 'max', 'min', 'last'],
            'Volume': 'sum',
            'tick_id': 'nunique'
        })
        candles.columns = ['open', 'high', 'low', 'close', 'volume', 'num_ticks']
        candles = candles.reset_index()

        # Volume bucket features
        bucket_edges = [100, 200, 500, 1000, 2000, 5000, 5500, 6000, 7500, 9000, 10000]
        bucket_names = [f'vol_{bucket_edges[i]}_{bucket_edges[i+1]-1}' for i in range(len(bucket_edges)-1)]
        bucket_names.append(f'vol_{bucket_edges[-1]}_plus')

        for i in range(len(bucket_edges)):
            lo = bucket_edges[i]
            hi = bucket_edges[i+1] if i+1 < len(bucket_edges) else np.inf
            candles[bucket_names[i]] = df.groupby(['day', 'candle_time']).apply(
                lambda g: ((g['Volume'] >= lo) & (g['Volume'] < hi)).sum()
            ).values

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

        # 1-min VWAP for rolling average
        one_min = df.copy()
        one_min['one_min_time'] = one_min['Date-Time'].dt.floor('1min')
        vwap_1m = one_min.groupby(['day', 'one_min_time']).apply(
            lambda g: (g['Price'] * g['Volume']).sum() / g['Volume'].sum() if g['Volume'].sum() > 0 else np.nan
        )
        vwap_1m = vwap_1m.rename('vwap_1m').reset_index()

        candles['vwap_1m_avg_15'] = np.nan
        for idx, row in candles.iterrows():
            day = row['day']
            candle_time = row['candle_time']
            vwap_hist = vwap_1m[(vwap_1m['day'] == day) & (vwap_1m['one_min_time'] <= candle_time)]['vwap_1m'].tail(15)
            candles.at[idx, 'vwap_1m_avg_15'] = vwap_hist.mean() if not vwap_hist.empty else np.nan

        return candles

    def add_all_features(self, candles):
        """Add all features to candle DataFrame"""
        # Basic moving averages
        candles['sma_short'] = candles['close'].rolling(self.short_window, min_periods=1).mean()
        candles['sma_med'] = candles['close'].rolling(self.med_window, min_periods=1).mean()

        # Original features
        candles['kama_10'] = candles['close'].rolling(5, min_periods=1).mean()
        candles['kama_30'] = candles['close'].rolling(10, min_periods=1).mean()
        candles['kama_200'] = candles['close'].rolling(20, min_periods=1).mean()

        candles['linear_slope_36'] = self.rolling_slope(candles['close'], 6)
        candles['linear_slope_72'] = self.rolling_slope(candles['close'], 12)
        candles['hurst_6'] = self.rolling_hurst(candles['close'], 6)

        candles['vol_rogers_satchell_10'] = self.rogers_satchell_vol(
            candles['high'], candles['low'], candles['close'], candles['open'], 10
        )

        # Quantile bins
        q0 = candles['close'].rolling(20, min_periods=1).quantile(0.0)
        q25 = candles['close'].rolling(20, min_periods=1).quantile(0.25)
        q75 = candles['close'].rolling(20, min_periods=1).quantile(0.75)
        q108 = candles['close'].rolling(20, min_periods=1).quantile(1.0)
        candles['bin_0_25'] = ((candles['close'] >= q0) & (candles['close'] < q25)).astype(int)
        candles['bin_25_75'] = ((candles['close'] >= q25) & (candles['close'] < q75)).astype(int)
        candles['bin_75_108'] = ((candles['close'] >= q75) & (candles['close'] <= q108)).astype(int)

        candles['auto_corr_6'] = self.rolling_autocorr(candles['close'], 6)

        # Relative transformations
        candles['pct_kama'] = ((candles['kama_30'] - candles['kama_200']) / candles['kama_200']) / candles['vol_rogers_satchell_10']
        candles['pct_linear_slope'] = (candles['linear_slope_36'] - candles['linear_slope_72']) / candles['vol_rogers_satchell_10']

        # VWAP features
        candles['vwap_short'] = (candles['close'] * candles['volume']).rolling(self.short_window, min_periods=1).sum() / candles['volume'].rolling(self.short_window, min_periods=1).sum()
        candles['vwap_med'] = (candles['close'] * candles['volume']).rolling(self.med_window, min_periods=1).sum() / candles['volume'].rolling(self.med_window, min_periods=1).sum()
        candles['vwap_ratio_short'] = candles['close'] / candles['vwap_short']
        candles['vwap_distance_short'] = (candles['close'] - candles['vwap_short']) / candles['vwap_short']

        # Order flow features
        candles['imbalance'] = (candles['vol_1_5'] - candles['vol_101_plus']) / (candles['vol_1_5'] + candles['vol_101_plus'] + 1e-8)
        candles['small_vol_ratio'] = candles['vol_1_5'] / (candles['volume'] + 1e-8)
        candles['large_vol_ratio'] = candles['vol_101_plus'] / (candles['volume'] + 1e-8)

        # RSI momentum
        candles['rsi_short'] = self.calculate_rsi(candles['close'], self.short_window)
        candles['rsi_med'] = self.calculate_rsi(candles['close'], self.med_window)

        # Trend features
        candles['trend_short'] = np.where(candles['close'] > candles['sma_short'], 1, -1)
        candles['trend_med'] = np.where(candles['close'] > candles['sma_med'], 1, -1)
        candles['sma_cross'] = candles['sma_short'] - candles['sma_med']

        return candles

    def train_classification_model(self, candles_data):
        """Train classification model on rolling window"""
        if len(candles_data) > self.training_window:
            train_data = candles_data.tail(self.training_window).copy()
        else:
            train_data = candles_data.copy()

        # Create target (next candle's return classification)
        target_ret = train_data['close'].pct_change().shift(-1)
        train_data['target'] = np.select(
            [target_ret < -0.001, target_ret > 0.001],
            [-1, 1], default=0
        )

        # Prepare features (shift by 1 to avoid forward-looking bias)
        X = train_data[self.feature_cols].shift(1).iloc[20:]
        Y = train_data['target'].iloc[20:]

        # Fill missing values
        X = X.fillna(method='ffill').fillna(0)
        Y = Y[X.index]

        if len(X) > 50:
            self.classification_model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
            self.classification_model.fit(X, Y)
            self.candles_since_retrain = 0
            print(f"Classification model retrained on {len(X)} samples")
            return True
        return False

    def forecast_prices(self, price_data, current_timestamp):
        """
        Forecast next period prices using VAR/GARCH models

        price_data: DataFrame with OHLC data indexed by timestamp
        current_timestamp: Current timestamp for prediction
        Returns: dict with forecasts
        """
        if not ADVANCED_FORECASTING:
            # Simple forecast using last price + random walk
            last_prices = price_data[['low', 'close', 'high']].iloc[-1]
            return {
                'var_forecast': last_prices.to_dict(),
                'garch_forecast': last_prices.to_dict(),
                'hybrid_forecast': last_prices.to_dict(),
                'timestamp': current_timestamp
            }

        # Filter for trading hours (hour > 10) to match original logic
        if current_timestamp.hour <= 10:
            return None

        # Use only the most recent forecast_window data points
        if len(price_data) > self.forecast_window:
            forecast_data = price_data.tail(self.forecast_window).copy()
        else:
            forecast_data = price_data.copy()

        if len(forecast_data) < 100:  # Minimum data requirement
            return None

        # Calculate price differences for VAR/GARCH models
        price_diff = forecast_data[['low', 'close', 'high']].diff().dropna()

        if len(price_diff) < 50:
            return None

        try:
            # VAR(5) forecast
            var_model = VAR(price_diff)
            var_result = var_model.fit(5)
            var_forecast = var_result.forecast(price_diff.values[-5:], steps=1)[0]

            # GARCH forecasts for each series
            garch_forecasts = {}
            for series in ['low', 'close', 'high']:
                try:
                    garch_model = arch_model(price_diff[series], mean='AR', lags=5, vol='Garch', p=1, q=1)
                    garch_result = garch_model.fit(disp='off')
                    garch_forecasts[series] = garch_result.forecast(horizon=1).mean.iloc[-1, 0]
                    del garch_model, garch_result
                except:
                    garch_forecasts[series] = 0.0

            # Convert differences back to price levels
            last_prices = forecast_data[['low', 'close', 'high']].iloc[-1]

            forecasts = {
                'var_forecast': {
                    'low': last_prices['low'] + var_forecast[0],
                    'close': last_prices['close'] + var_forecast[1],
                    'high': last_prices['high'] + var_forecast[2]
                },
                'garch_forecast': {
                    'low': last_prices['low'] + garch_forecasts['low'],
                    'close': last_prices['close'] + garch_forecasts['close'],
                    'high': last_prices['high'] + garch_forecasts['high']
                },
                'timestamp': current_timestamp
            }

            # Hybrid forecast (50/50 combination)
            forecasts['hybrid_forecast'] = {
                'low': 0.5 * (forecasts['var_forecast']['low'] + forecasts['garch_forecast']['low']),
                'close': 0.5 * (forecasts['var_forecast']['close'] + forecasts['garch_forecast']['close']),
                'high': 0.5 * (forecasts['var_forecast']['high'] + forecasts['garch_forecast']['high'])
            }

            # Cleanup
            del var_model, var_result
            gc.collect()

            return forecasts

        except Exception as e:
            print(f"Forecasting error: {e}")
            return None

    def update_and_predict(self, new_tick_data):
        """
        Main update function: processes new data and makes both classification and price predictions

        new_tick_data: DataFrame with new tick data
        Returns: dict with both classification and price forecasts
        """
        # Process new tick data to candles
        new_candles = self.process_tick_data_to_candles(new_tick_data)
        new_candles = self.add_all_features(new_candles)

        # Update historical candles buffer
        if self.historical_candles.empty:
            self.historical_candles = new_candles
        else:
            self.historical_candles = pd.concat([self.historical_candles, new_candles]).drop_duplicates(
                subset=['candle_time'], keep='last'
            ).sort_values('candle_time').reset_index(drop=True)

        # Keep only recent data for memory management
        if len(self.historical_candles) > max(self.training_window, self.forecast_window) + 100:
            self.historical_candles = self.historical_candles.tail(max(self.training_window, self.forecast_window) + 100)

        self.candles_since_retrain += len(new_candles)

        # Retrain classification model if needed
        if (self.classification_model is None or 
            self.candles_since_retrain >= self.retrain_frequency):
            self.train_classification_model(self.historical_candles)

        # Prepare price data for forecasting (set candle_time as index)
        price_data = self.historical_candles.set_index('candle_time')[['open', 'high', 'low', 'close', 'volume']].copy()

        results = {}

        # Classification prediction
        if self.classification_model is not None and len(self.historical_candles) > 0:
            latest_features = self.historical_candles[self.feature_cols].iloc[-1:].fillna(method='ffill').fillna(0)

            prediction = self.classification_model.predict(latest_features)[0]
            probabilities = self.classification_model.predict_proba(latest_features)[0]

            class_names = {-1: 'Down', 0: 'Flat', 1: 'Up'}

            results['classification'] = {
                'prediction': prediction,
                'prediction_class': class_names[prediction],
                'probabilities': {
                    'Down': probabilities[0] if len(probabilities) > 0 else 0,
                    'Flat': probabilities[1] if len(probabilities) > 1 else 0,
                    'Up': probabilities[2] if len(probabilities) > 2 else 0
                }
            }

        # Price forecasting
        if len(price_data) > 0:
            current_timestamp = price_data.index[-1]
            price_forecast = self.forecast_prices(price_data, current_timestamp)
            if price_forecast:
                results['price_forecast'] = price_forecast

        results['timestamp'] = datetime.now()
        results['candles_in_buffer'] = len(self.historical_candles)

        return results

    def get_forecast_performance(self):
        """Calculate and return forecasting performance metrics"""
        if not self.forecast_results['timestamps']:
            return None

        metrics = {}
        for series in ['low', 'close', 'high']:
            if self.forecast_results['actual'][series]:
                actual = self.forecast_results['actual'][series]

                for model in ['var_forecast', 'garch_forecast', 'hybrid_forecast']:
                    if self.forecast_results[model][series]:
                        predicted = self.forecast_results[model][series]
                        mse = mean_squared_error(actual, predicted)
                        mae = mean_absolute_error(actual, predicted)

                        metrics[f'{model}_{series}_mse'] = mse
                        metrics[f'{model}_{series}_mae'] = mae

        return metrics

    def save_system(self, filepath):
        """Save the entire system state"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"System saved to {filepath}")

    @classmethod
    def load_system(cls, filepath):
        """Load system state"""
        with open(filepath, 'rb') as f:
            system = pickle.load(f)
        print(f"System loaded from {filepath}")
        return system


# Example usage for integrated live trading:
if __name__ == "__main__":
    # Initialize integrated system
    trading_system = IntegratedTradingSystem(
        training_window=1000,      # Classification model window
        retrain_frequency=50,      # Retrain every 50 candles
        forecast_window=3000       # Price forecasting window
    )

    # Live trading loop example:
    # while True:
    #     # Get new tick data
    #     new_ticks = get_latest_tick_data()
    #     
    #     # Get both classification and price predictions
    #     results = trading_system.update_and_predict(new_ticks)
    #     
    #     if 'classification' in results:
    #         print(f"Direction: {results['classification']['prediction_class']}")
    #         print(f"Probabilities: {results['classification']['probabilities']}")
    #     
    #     if 'price_forecast' in results:
    #         print(f"Price forecasts: {results['price_forecast']}")
    #     
    #     # Execute trading strategy based on both signals
    #     # execute_integrated_strategy(results)
    #     
    #     time.sleep(60)  # Wait for next update

    print("Integrated trading system with classification and price forecasting ready!")
    print("Features:")
    print("- Rolling classification model (direction prediction)")
    print("- VAR/GARCH price forecasting")
    print("- No forward-looking bias in either model")
    print("- Automatic model retraining")
    print("- Memory management for 24/7 operation")

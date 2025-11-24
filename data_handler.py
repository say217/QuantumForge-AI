# data_handler.py
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DataHandler:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()
        self.dates = None
        self.n_features = None

    def download_data(self):
        cache_file = f'data_{self.config.ticker}.pkl'
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.dates = data.index
                print(f"Loaded cached data for {self.config.ticker}")
                return data
            except Exception:
                print(f"Cache file corrupted, re-downloading data for {self.config.ticker}")
        try:
            print(f"Downloading data for {self.config.ticker}...")
            data = yf.download(
                self.config.ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d",
                progress=False
            )
            if data.empty:
                raise ValueError(f"No data retrieved for ticker {self.config.ticker}")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            required_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
            available_cols = [col for col in required_cols if col in data.columns]
            if 'Close' not in available_cols:
                raise ValueError(f"Critical error: 'Close' price not available for {self.config.ticker}")
            self.config.features = [col for col in self.config.features if col in available_cols]
            if len(data) < self.config.window_size:
                raise ValueError(f"Insufficient data: {len(data)} samples, need at least {self.config.window_size}")
            data = data[self.config.features].copy()
            self.dates = data.index
            data = self._add_technical_indicators(data)
            data = self._handle_missing_values(data)
            print(f"Data shape after preprocessing: {data.shape}")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return data
        except Exception as e:
            print(f"Error downloading data for {self.config.ticker}: {e}")
            return None

    def _add_technical_indicators(self, data: pd.DataFrame):
        try:
            data['SMA_10'] = data['Close'].rolling(window=10, min_periods=1).mean()
            data['RSI'] = self._compute_rsi(data['Close'])
            data['MACD'] = self._compute_macd(data['Close'])
            data['BB_Upper'], data['BB_Lower'] = self._compute_bollinger_bands(data['Close'])
            data['ATR'] = self._compute_atr(data)
            data['Price_Change'] = data['Close'].pct_change()
            data['Log_Close'] = np.log(data['Close'] + 1e-8)
            data['Volatility_10'] = data['Close'].pct_change().rolling(10, min_periods=1).std()
            data['Momentum_5'] = data['Close'] - data['Close'].shift(5)
            data['Momentum_10'] = data['Close'] - data['Close'].shift(10)
            data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
            data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
            data['Volatility_20'] = data['Close'].pct_change().rolling(20, min_periods=1).std()
            data['Volatility_50'] = data['Close'].pct_change().rolling(50, min_periods=1).std()
        except Exception as e:
            print(f"Error computing technical indicators: {e}")
            for col in self.config.technical_indicators:
                if col not in data.columns:
                    data[col] = data['Close']
        return data

    def _handle_missing_values(self, data: pd.DataFrame):
        data = data.fillna(method='ffill')
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
        data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())
        data = data.fillna(0)
        return data

    def _compute_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _compute_macd(self, prices, slow=26, fast=12, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)

    def _compute_bollinger_bands(self, prices, window=20, num_std=2):
        sma = prices.rolling(window=window, min_periods=1).mean()
        std = prices.rolling(window=window, min_periods=1).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper.fillna(sma), lower.fillna(sma)

    def _compute_atr(self, data, period=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr.fillna(atr.mean())

    def prepare_data(self, data: pd.DataFrame):
        feature_list = list(dict.fromkeys(self.config.features + self.config.technical_indicators))
        available_features = [f for f in feature_list if f in data.columns]
        print(f"Using features: {available_features}")
        self.n_features = len(available_features)
        features_vals = data[available_features].values
        scaled_features = self.scaler.fit_transform(features_vals)
        X, y = [], []
        W = self.config.window_size
        for i in range(len(scaled_features) - W):
            X.append(scaled_features[i:i + W])
            y.append(scaled_features[i + W, available_features.index('Close')])
        X, y = np.array(X), np.array(y)
        train_size = int(len(X) * self.config.train_split)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        return (X_train, y_train), (X_test, y_test), scaled_features

    def inverse_target_transform(self, scaled_data):
        scaled = np.asarray(scaled_data).reshape(-1, 1)
        dummy = np.zeros((len(scaled), self.n_features))
        dummy[:, 0] = scaled.flatten()
        return self.scaler.inverse_transform(dummy)[:, 0]
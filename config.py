# config.py
import random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    def __init__(self, ticker_symbol: str):
        self.ticker = ticker_symbol.upper()
        self.start_date = "2022-01-01"
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.features = ['Close', 'High', 'Low']
        self.technical_indicators = [
            'SMA_10', 'RSI', 'MACD',
            'BB_Upper', 'BB_Lower', 'ATR',
            'Price_Change', 'Log_Close', 'Volatility_10',
            'Momentum_5', 'Momentum_10', 'EMA_10', 'EMA_20', 'Volatility_20', 'Volatility_50'
        ]
        self.window_size = 40
        self.train_split = 0.8
        self.forecast_days = 4
        self.past_days_plot = 14
        self.batch_size = 64
        self.epochs = 50
        self.learning_rate = 5e-4
        self.lstm_units = 192
        self.num_layers = 3
        self.dropout = 0.25
        self.loss = 'huber'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_features = None
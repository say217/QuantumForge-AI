# utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
import torch 
def plot_results(dates, actuals, preds, title, config):
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6), facecolor='#1c2526')
    plt.plot(dates, actuals, label='Actual', color='#00ff00', linewidth=2)
    plt.plot(dates, preds, '--', label='Predicted', color='#00f7ff', linewidth=2)
    plt.title(f"{title} - {config.ticker}", fontsize=14, color='white')
    plt.xlabel('Date', fontsize=12, color='white')
    plt.ylabel('Price (USD)', fontsize=12, color='white')
    plt.legend(fontsize=10, facecolor='#1c2526', edgecolor='white', labelcolor='white')
    plt.grid(True, alpha=0.3, color='#cccccc')
    plt.gcf().autofmt_xdate()
    plt.tick_params(colors='white')
    plt.tight_layout()
    plt.show()

def plot_forecast(dates, prices, forecast_dates, forecast_prices, std, config):
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 8), facecolor='#1c2526')
    past_days = config.past_days_plot
    historical_dates = dates[-past_days:]
    historical_prices = prices[-past_days:]
    plt.plot(historical_dates, historical_prices, 'o-', label='Historical (Past 60 Days)', color='#ff00ff', linewidth=2)
    plt.plot(forecast_dates, forecast_prices, 'o-', label=f'Forecast (Next {config.forecast_days} Days)', color='#ffff00', linewidth=2, markersize=6)
    plt.fill_between(forecast_dates, forecast_prices - std, forecast_prices + std, alpha=0.2, color='#ffff00', label='Confidence Interval')
    plt.axvline(x=dates[-1], color='#cccccc', linestyle='--', alpha=0.7, label='Today')
    plt.title(f"{config.ticker} - Past {past_days} Days & {config.forecast_days}-Day Forecast", fontsize=14, color='white')
    plt.xlabel('Date', fontsize=12, color='white')
    plt.ylabel('Price (USD)', fontsize=12, color='white')
    plt.legend(fontsize=10, facecolor='#1c2526', edgecolor='white', labelcolor='white')
    plt.grid(True, alpha=0.3, color='#cccccc')
    plt.gcf().autofmt_xdate()
    plt.tick_params(colors='white')
    plt.tight_layout()
    print(f"\n{config.ticker} - {config.forecast_days} Day Forecast:")
    print("-" * 50)
    for date, price in zip(forecast_dates, forecast_prices):
        print(f"{date.strftime('%Y-%m-%d (%A)')}: ${price:.2f}")
    plt.show()

def predict_future(model, last_window, num_days, data_handler, config):
    model.eval()
    predictions_scaled = []
    current_window = last_window.copy()
    for _ in range(num_days):
        input_tensor = torch.FloatTensor(current_window).unsqueeze(0).to(config.device)
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().item()
        predictions_scaled.append(pred_scaled)
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1, 0] = pred_scaled
    return data_handler.inverse_target_transform(np.array(predictions_scaled))

def plot_frequency_and_heatmap(data: pd.DataFrame, ticker: str, features: list, technical_indicators: list):
    plt.style.use('dark_background')
    all_features = list(dict.fromkeys(features + technical_indicators))
    available_features = [f for f in all_features if f in data.columns]
    hist_features = ['Close', 'RSI', 'SMA_10', 'BB_Upper', 'BB_Lower', 'Volatility_10', 'Volatility_20', 'Price_Change', 'MACD']
    hist_features = [f for f in hist_features if f in available_features]
    if not hist_features:
        print("Warning: No valid features for histogram. Skipping frequency plot.")
        return
   
    plt.figure(figsize=(15, 10), facecolor='#1c2526')
    colors = ['#00ffab', '#ff6f61', '#ffd700', '#6ab04c', '#ff85ff', '#00b7eb', '#ff9f43', '#5c5c8a', '#ff4f81']
   
    for i, feature in enumerate(hist_features, 1):
        plt.subplot(3, 3, i)
        data_clean = data[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if data_clean.empty:
            print(f"Warning: No valid data for {feature}. Skipping histogram.")
            continue
        sns.histplot(data_clean, bins=20, kde=True, color=colors[i-1], edgecolor='white', alpha=0.7)
        plt.title(f'{feature} Distribution', fontsize=10, color='white')
        plt.xlabel(feature, fontsize=8, color='white')
        plt.ylabel('Frequency', fontsize=8, color='white')
        plt.grid(True, alpha=0.3, color='gray')
        plt.tick_params(colors='white')
   
    plt.suptitle(f'{ticker} Feature Distributions', fontsize=14, color='white')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
   
    plt.figure(figsize=(12, 10), facecolor='#1c2526')
    correlation_matrix = data[available_features].corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='Spectral',
        center=0,
        vmin=-1,
        vmax=1,
        fmt='.2f',
        square=True,
        cbar_kws={'label': 'Correlation', 'shrink': 0.8},
        annot_kws={'size': 8, 'color': 'white'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f'{ticker} Feature Correlation Heatmap', fontsize=14, color='white')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(rotation=0, color='white')
    plt.tight_layout()
    plt.show()
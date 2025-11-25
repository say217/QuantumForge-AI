# QuantumForge AI: The Ultimate Neural Trading Oracle + Research Agent + Market Price Prediction Analyser AI
<img width="824" height="297" alt="Gemini_Generated_Image_zwndozwndozwndoz" src="https://github.com/user-attachments/assets/ca00b741-2c9f-4a97-88ee-be69b7d267b1" />

# Overview

**(Agents Intensive - Capstone Project)**
QuantumForge AI, An enhanced Stock Predictor, It is a comprehensive Python-based system designed for stock price forecasting, technical analysis, and generative AI-driven insights. It leverages deep learning (BiGRU with Attention and Conv1D layers), technical indicators, and integrates with Google's Gemini AI for automated investment reports and a conversational research agent. The system supports real-time data fetching via yfinance, model training, evaluation, multi-day forecasting, and interactive querying for stocks, market news, or deep research topics.

Built for quantitative analysts, traders, and enthusiasts, it provides:

- **Predictive Modeling:** Uses historical data to forecast stock prices with uncertainty intervals.
- **Visualization:** Dark-themed plots for distributions, correlations, test results, and forecasts.
- **AI Augmentation:** Generates professional reports and handles natural language queries.
- **Extensibility:** Modular design for easy fine-tuning or adding new indicators/models.

The project emphasizes reproducibility (seeded randomness), efficiency (mixed-precision training, caching), and usability (conversational CLI loop). **As of the current date (November 25, 2025), it uses up-to-date libraries and assumes access to a GPU for optimal performance.**

## Live Stock Price & Forecast Access
At any time during the session, you can request instant stock quotes, 10-day price history tables, or full forecasting pipelines. For example:
“stock GOOGL” → shows current price, volume, and recent performance table.
“predict Amazon” → trains model, displays actual vs predicted chart, 4-day forecast with confidence interval, and saves a complete AI-generated report.

<img width="929" height="590" alt="Screenshot 2025-11-25 015154" src="https://github.com/user-attachments/assets/45bf3e0b-d516-4503-bb4b-04bae2d3947c" />



The project emphasizes reproducibility (seeded randomness), efficiency (mixed-precision training, caching), and usability (conversational CLI loop). As of the current date (November 25, 2025), it uses up-to-date libraries and assumes access to a GPU for optimal performance.


| **Category**                 | **Feature Description**                                                                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Pipeline**            | Downloads & preprocesses stock data with **15+ technical indicators** (RSI, MACD, Bollinger Bands, ATR, EMAs, Momentum, Volatility, etc.) |
| **Advanced Model**           | Bi-directional **GRU** with **multi-scale Conv1D** feature extraction + **self-attention** for deep temporal pattern learning             |
| **Training Enhancements**    | Early stopping, learning rate scheduling, gradient clipping, and **mixed precision (AMP)** for faster & stable convergence                |
| **Evaluation Metrics**       | RMSE, MAE, R², directional accuracy + **backtesting** on holdout data                                                                     |
| **Forecasting**              | Recursive multi-step predictions (**default: 4 days**) with confidence intervals                                                          |
| **Visualizations**           | Feature histograms, correlation heatmaps, actual vs predicted plots, forecast charts                                                      |
| **AI-Generated Reports**     | **Gemini-powered** Markdown reports with executive summaries, recommendations (e.g., *Strong Buy*), and key technical levels              |
| **Conversational Interface** | Commands like *"predict Apple"*, *"deep research on AI stocks"*, or general queries like *"latest on tech stocks?"*                       |
| **Caching & Persistence**    | Automatically saves data & trained models; supports **resume training**                                                                   |
| **Research Agent**           | Deep research using DuckDuckGo/Wikipedia with fact-checking & structured Markdown output                                                  |

## Data Handling (Paragraph Version)

The DataHandler(Config) class manages the complete data pipeline, beginning with downloading OHLCV price data from yfinance (default range: January 1, 2022 to the current date). It computes a rich set of technical indicators, including SMA-10, RSI-14, MACD-12/26, Bollinger Bands-20, ATR-14, price change, log-close, volatility windows, momentum shifts, and multiple EMAs. After indicator generation, the handler cleans the dataset by resolving NaN/inf values using forward-fill, mean imputation, and zero-filling. All features are scaled using MinMaxScaler, and the system generates sliding-window sequences (default: 40-day windows). The dataset is then split into an 80/20 train-test ratio and cached as pickles for faster reuse. The class outputs scaled training and testing sequences—(X_train, y_train) and (X_test, y_test)—along with the fully scaled dataset, and also provides an inverse transformation utility to convert predictions back into real USD prices.
<img width="899" height="409" alt="Screenshot 2025-11-25 015452" src="https://github.com/user-attachments/assets/6ab1cafc-f680-44e9-9a5b-fc6863dc4a32" />

## Model Architecture (Paragraph Version)

The EnhancedGRU(nn.Module) model is designed for deep temporal forecasting using sequence inputs of shape (batch, 40, ~18), consisting of OHLC features and technical indicators. It begins with multi-kernel Conv1D layers (kernels 3 and 5) that extract localized patterns and concatenate their outputs to expand the feature dimension. These enriched sequences pass into a bidirectional GRU (1 layer, 128 hidden units per direction, 256 total) that captures both forward and backward temporal dependencies. A scaled dot-product attention mechanism then weights GRU outputs to emphasize the most informative time steps. The regression head consists of a Linear layer (256 → 1), LayerNorm, and Dropout(0.15). The full forward pass follows Conv → GRU → Attention → Linear, using ReLU activations. The model is trained with Huber (SmoothL1) loss for robustness to outliers and automatically selects CUDA when available, otherwise defaults to CPU.

## Training and Evaluation (Paragraph Version)

The Trainer(Config) class handles the full training workflow using the Adam optimizer (learning rate 5e-4) along with a ReduceLROnPlateau scheduler set with a patience of 10 epochs. Training is performed with a batch size of 64 for up to 50 epochs, using an early-stopping mechanism with a patience of 15 to prevent overfitting. Mixed-precision training (AMP) with GradScaler is enabled to improve both speed and memory efficiency. Throughout training, the system monitors training and validation loss, automatically saving the best-performing model to models/{ticker}_model.pth.

For evaluation, the trainer computes RMSE, MAE, and R² using inverse-transformed predictions to return real USD price metrics, along with Directional Accuracy, which measures the percentage of correctly predicted upward or downward price movements. Visual evaluation includes a dark-themed line plot comparing actual vs. predicted prices, using green and cyan lines for clear contrast and readability.



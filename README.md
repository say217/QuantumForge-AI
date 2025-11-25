# QuantumForge AI: The Ultimate Stock & Economic Assistant, Market Price Prediction To Deep Research.
<span style="color:#ff2d55;">November 25, 2025 — Kolkata, West Bengal, India — Created by: Sayak Samanta || </span>
<span style="color:#ff2d55;">This Project is made for Agents Intensive - Capstone Project, Agents Intensive Course By Google </span>



<img width="1024" height="402" alt="image" src="https://github.com/user-attachments/assets/5bed300d-6a8e-4539-b8fc-efe15cd48654" />

_AI Generated Image_

# Overview

**QuantumForge AI** is an enhanced stock prediction system— a comprehensive Python-based framework designed for `stock price forecasting`, `technical analysis`, and `generative AI–driven insights`, `Deep Research Agent`. It leverages deep learning (BiGRU with Attention and Conv1D layers), technical indicators, and integrates with Google's Gemini AI for automated investment reports and a conversational research agent.

The system supports real-time data fetching via `yfinance`, model training, evaluation, multi-day forecasting, and interactive querying for stocks, market news, or deep research topics.

Built for quantitative analysts, traders, and market enthusiasts, it provides:

- **Predictive Modeling:** Utilizes historical data to forecast stock prices with uncertainty intervals.
- **Visualization:** Dark-themed plots for distributions, correlations, test results, and forecasts.
- **AI Augmentation:** Generates professional-grade reports and handles natural language financial queries.
- **Extensibility:** Modular design enables easy fine-tuning or the addition of new indicators and models.

The project emphasizes reproducibility (seeded randomness), efficiency (mixed-precision training, caching), and usability (a conversational CLI loop).  
**As of November 25, 2025, the project uses up-to-date libraries and assumes access to a GPU for optimal performance.**


## Live Stock Price & Forecast Access
At any time during the session, you can request instant stock quotes, 10-day price history tables, or full forecasting pipelines. For example:
“stock GOOGL” → shows current price, volume, and recent performance table.
“predict Amazon” → trains model, displays actual vs predicted chart, 4-day forecast with confidence interval, and saves a complete AI-generated report.

<img width="729" height="450" alt="Screenshot 2025-11-25 015154" src="https://github.com/user-attachments/assets/45bf3e0b-d516-4503-bb4b-04bae2d3947c" />

_Jupyter Notebook Project Output Image_



## Model Architecture (Paragraph Version)

The EnhancedGRU(nn.Module) model is designed for deep temporal forecasting using sequence inputs of shape (batch, 40, ~18), consisting of OHLC features and technical indicators. It begins with multi-kernel Conv1D layers (kernels 3 and 5) that extract localized patterns and concatenate their outputs to expand the feature dimension. These enriched sequences pass into a bidirectional GRU (1 layer, 128 hidden units per direction, 256 total) that captures both forward and backward temporal dependencies. A scaled dot-product attention mechanism then weights GRU outputs to emphasize the most informative time steps. The regression head consists of a Linear layer (256 → 1), LayerNorm, and Dropout(0.15). The full forward pass follows Conv → GRU → Attention → Linear, using ReLU activations. The model is trained with Huber (SmoothL1) loss for robustness to outliers and automatically selects CUDA when available, otherwise defaults to CPU.

## Training and Evaluation (Paragraph Version)

The Trainer(Config) class handles the full training workflow using the Adam optimizer (learning rate 5e-4) along with a ReduceLROnPlateau scheduler set with a patience of 10 epochs. Training is performed with a batch size of 64 for up to 50 epochs, using an early-stopping mechanism with a patience of 15 to prevent overfitting. Mixed-precision training (AMP) with GradScaler is enabled to improve both speed and memory efficiency. Throughout training, the system monitors training and validation loss, automatically saving the best-performing model to models/{ticker}_model.pth.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/e6077c15-c6a9-4a5c-85ae-628aa829fd43" width="500"></td>
    <td><img src="https://github.com/user-attachments/assets/e35176ed-98f7-43f7-8e68-0e31fc3f1509" width="500"></td>
  </tr>

</table>

_Jupyter Notebook Project Output Image_


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

## AI Integration:
The system uses Gemini with the configuration gemini-2.0-flash (tokens=1600, temp=0.7) and supports a report generator that produces structured Markdown outputs containing sections such as Summary, Performance, Forecast, Risk, Recommendation, and Targets. Each report is saved following the pattern reports/{ticker}_Gemini_Report_{timestamp}.md. For general interactions, the assistant switches to a "Goldman Sachs strategist" persona to handle non-stock queries in a more conversational and analytical tone.

## Research Agent:
The research agent provides multi-depth analysis through functions like deep_research(topic, depth='normal') and agent(user_input), drawing from DuckDuckGo's JSON API (10 results) and Wikipedia summaries, with ranking based on snippet length. It aggregates scraped content and sends it through a Gemini prompt to produce a structured ten-section research report with citations, followed by a secondary Gemini fact-checking pass. Users can control depth using commands such as "research X" for normal depth, "quick research X" for shallow output, and "deep research X" for a more detailed, expert-level analysis.

For evaluation, the trainer computes RMSE, MAE, and R² using inverse-transformed predictions to return real USD price metrics, along with Directional Accuracy, which measures the percentage of correctly predicted upward or downward price movements. Visual evaluation includes a dark-themed line plot comparing actual vs. predicted prices, using green and cyan lines for clear contrast and readability.


<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/521e3c01-4245-431d-af2e-48177ce2cee4" width="500"></td>
    <td><img src="https://github.com/user-attachments/assets/3aebdc7d-c5a8-4260-be3e-7562aea87955" width="500"></td>
  </tr>

</table>


_Jupyter Notebook Project Output Image_

## Installation and Setup
```bash
Clone/Setup Environment:textgit clone https://github.com/say217/QuantumForge-AI.git
cd enhanced-stock-predictor
python -m venv env
source env/bin/activate  # Linux/Mac
# or env\Scripts\activate  # Windows
```

## Install Dependencies (see below): 
```bash
pip install -r requirements.txt.
```
## API Keys:
Go to AI Studio
Open your browser and visit the AI Studio website. Log in / Sign up
Use your email or GitHub/Google account to sign in. Create a Free-Tier API Key
Find the API Keys section (usually under Settings or Developer menu).
Click Create New API Key.
Select Free Tier and confirm.
Copy the API Key
Once the key is created, click Copy to copy it to your clipboard.
```
Set GOOGLE_API_KEY env var (for Gemini): export GOOGLE_API_KEY="your_key".
yfinance handles Yahoo Finance implicitly (no key needed)
```

The image below shows a heat map of the correlation between the feature data, 
Price-based features such as Close, High, Low, SMA, EMA, and Log-Close show extremely high correlations (0.97–1.00), which is expected because they are all derived from the same underlying price series. Bollinger Bands (Upper and Lower) also correlate strongly with price (0.95–0.99), indicating that they largely track price movements and may offer limited additional information unless paired with volatility or trend indicators. In contrast, RSI, MACD, and Momentum features display only moderate correlations (generally 0.17–0.70), suggesting they capture shifts in trend and momentum that could add predictive value. Volatility features (10, 20, 50-period) are among the least correlated, making them more independent and potentially useful for diversifying the feature set.
Also the below image of the LLM model’s output, demonstrating that the language model (Gemini-2.0-Flash) behaves like an economic specialist and provides research-based insights. However, users should always double-check the information and avoid making decisions solely based on AI. This is only a prototype disclaimer.”
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a1c8d6cb-7d28-458d-9685-7fd19c5b0242" width="500"></td>
    <td><img src="https://github.com/user-attachments/assets/9de2cc17-a7ef-4fad-8d3a-10417a9d77e4" width="500"></td>
  </tr>

</table>

_Jupyter Notebook Project Output Image_


# QUANTUMFORGE AI –  DISCLAIMER  
<span style="color:#ff2d55;">(November 25, 2025)</span>

<span style="color:#ff3b30; font-weight:bold;">NOT PROFESSIONAL FINANCIAL ADVICE</span>  
<span style="color:#ff9500;">QuantumForge AI</span> is an **experimental, educational, and research-oriented tool** created for entertainment, learning, and quantitative exploration purposes **only**.  
The predictions, forecasts, investment reports, recommendations <span style="color:#ff2d55;">(Strong Buy / Buy / Hold / Sell / Strong Sell)</span>, price targets, and any opinions generated by the system — whether from the neural network model or Google’s Gemini AI — **do NOT constitute financial, investment, or trading advice**.

<span style="color:#ff3b30; font-weight:bold;">USE AT YOUR OWN RISK</span>  
You alone are responsible for any trading, investment, or financial decisions you make.  
The authors, contributors, and distributors of <span style="color:#ff9500;">QuantumForge AI</span> accept **zero liability** for any financial losses, missed gains, emotional distress, or any other consequences arising from using this software.

<span style="color:#ff3b30; font-weight:bold;">REGULATORY NOTICE</span><br>
This tool is <strong>not registered</strong> with the SEC, FINRA, FCA, or any other financial regulatory authority.<br><br>
<span style="color:#ff2d55;">By using QuantumForge AI, you acknowledge and accept this disclaimer in full.</span><br><br>
<span style="color:#ff2d55;">
This is a student-based project created to demonstrate my skills and ideas. It is only a prototype and should not be used for real financial decision-making.
</span>

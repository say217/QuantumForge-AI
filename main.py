
# main.py
import os
import torch
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
import numpy as np
from config import Config, set_seed
from data_handler import DataHandler
from dataset import StockDataset
from model import EnhancedGRU
from trainer import Trainer
from utils import plot_results, plot_forecast, predict_future, plot_frequency_and_heatmap



import yfinance as yf
import pandas as pd
import google.generativeai as genai
import requests
import re
from functools import lru_cache
from IPython.display import Markdown, display

# Map common stock names to tickers (expand as needed)
TICKER_MAP = {
    "_apple": "AAPL",
    "_tesla": "TSLA",
    "_microsoft": "MSFT",
    "_nvidia": "NVDA",
    "_google": "GOOGL",
    "_amazon": "AMZN",
    "_meta": "META",
    "_netflix": "NFLX",
    "_amd": "AMD",
    "_spy": "SPY",
    "_qqq": "QQQ"
}
genai.configure(api_key="AIzaSyAXCcmFLgObbzbgd-7QqyEG9jP0iW8h2aA") 
session = requests.Session()
@lru_cache(maxsize=64)
def duckduckgo_search(query, max_results=8):
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "no_redirect": 1,
    }
    results = []
    try:
        r = session.get(url, params=params, timeout=8)
        data = r.json()
        # Abstract
        if data.get("AbstractText"):
            results.append({
                "title": "DuckDuckGo Abstract",
                "link": data.get("AbstractURL", ""),
                "snippet": data.get("AbstractText", "")
            })
        # Related topics
        for item in data.get("RelatedTopics", []):
            if isinstance(item, dict) and "FirstURL" in item:
                results.append({
                    "title": item.get("Text", ""),
                    "link": item.get("FirstURL", ""),
                    "snippet": item.get("Text", "")
                })
            if len(results) >= max_results:
                break
    except Exception as e:
        results.append({"title": "Error", "link": "", "snippet": str(e)})
    return results
def get_stock_history(ticker: str, days: int = 10):
    """
    Fetch last `days` of historical stock data using yfinance.
    Returns a formatted Markdown table and summary.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{days}d")
        if df.empty:
            return f"# No Stock Data\n\nNo stock data found for ticker '{ticker}'."
       
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = df.index.strftime("%Y-%m-%d")
       
        # Format table as Markdown
        table_rows = df.to_markdown(index=True, tablefmt="pipe")
       
        latest = df.iloc[-1]
        summary = (
            f"**Latest Price:** ${latest['Close']:.2f}\n"
            f"**Day High:** ${latest['High']:.2f}\n"
            f"**Day Low:** ${latest['Low']:.2f}\n"
            f"**Volume:** {int(latest['Volume']):,}"
        )
       
        return f"# Stock History for {ticker.upper()} (Last {days} Days)\n\n" \
               f"{table_rows}\n\n" \
               f"## Summary (Most Recent Day)\n\n{summary}"
    except Exception as e:
        return f"# Error\n\nError fetching stock data: {str(e)}"
@lru_cache(maxsize=64)
def wikipedia_search(query, max_results=1):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    results = []
    try:
        r = session.get(url, timeout=8)
        data = r.json()
        if "extract" in data:
            results.append({
                "title": data.get("title", "Wikipedia"),
                "link": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "snippet": data.get("extract", "")
            })
    except:
        pass
    return results
def rank_sources(sources):
    """Rank by snippet length + minimal heuristics."""
    return sorted(
        sources,
        key=lambda s: len(s.get("snippet", "")),
        reverse=True
    )
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={
        "max_output_tokens": 2000,
        "temperature": 0.45,
        "top_p": 0.92,
    }
)
def deep_research(topic: str, depth="normal"):
    print(f"\n[Research Agent] Searching for: {topic}")
    ddg = duckduckgo_search(topic, max_results=10)
    wiki = wikipedia_search(topic)
    sources = rank_sources(ddg + wiki)
    combined_text = "\n".join(
        f"[{i+1}] {s['title']}\n{s['link']}\n{s['snippet']}\n"
        for i, s in enumerate(sources)
    )
    depth_map = {
        "shallow": "Write a concise research note (3-5 paragraphs max). Use Markdown headers and bullet points for clarity.",
        "normal": "Write a full research report with structured Markdown sections (H1, H2, lists).",
        "deep": "Write an extremely detailed multi-section expert research report with advanced insights. Use Markdown extensively: headers, tables, bold/italics, code blocks where relevant."
    }
    depth_instruction = depth_map.get(depth, depth_map["normal"])
    # ------------------------------------------------
    # Stage 2: Main Research Pass
    # ------------------------------------------------
    prompt = f"""
You are a world-class research LLM. Always respond in clean Markdown format.
Below is raw data scraped from the web:
==============================
 SOURCES
==============================
{combined_text}
==============================
Your tasks:
1. Extract factual information.
2. Resolve contradictions across sources.
3. Add missing context.
4. Expand with expert-level knowledge.
5. Produce a deep structured report in Markdown (use # for H1, ## for H2, - for bullets, | for tables).
6. Cite using [1], [2], etc. at the end of relevant sentences.
Depth requirement:
{depth_instruction}
Topic: "{topic}"
Required sections (use as H1/H2 headers):
1. # Executive Summary
2. ## Background & History
3. ## Key Concepts & Subfields
4. ## Current State of Research
5. ## Leading Experts & Institutions
6. ## Major Papers & Findings
7. ## Open Problems & Controversies
8. ## Forecasts & Future Outlook
9. ## Actionable Insights
10. ## Source Citations
"""
    main_report = model.generate_content(prompt).text
    # ------------------------------------------------
    # Stage 3: Fact Checking
    # ------------------------------------------------
    verify_prompt = f"""
You are a fact-checking model. Always respond in clean Markdown format, preserving structure.
Review the following research report:
{main_report}
Fix:
- Factual errors
- Contradictions
- Unclear statements
- Unsupported claims
Rewrite ONLY the corrected version. Keep the exact same Markdown structure and sections.
"""
    checked_report = model.generate_content(verify_prompt).text
    return f"# Deep Research Report: {topic}\n\n{checked_report}"
def agent(user_input: str):
    text = user_input.strip()
    # --- deep research ---
    if m := re.match(r"deep research (.+)", text, re.I):
        output = deep_research(m.group(1), "deep")
        display(Markdown(output))
        return
    # --- quick research ---
    if m := re.match(r"quick research (.+)", text, re.I):
        output = deep_research(m.group(1), "shallow")
        display(Markdown(output))
        return
    # --- normal research ---
    if m := re.match(r"research (.+)", text, re.I):
        output = deep_research(m.group(1), "normal")
        display(Markdown(output))
        return
    # --- stock lookup ---
    if m := re.match(r"stock (.+)", text, re.I):
        ticker = m.group(1).strip().upper()
        output = get_stock_history(ticker, days=10)
        display(Markdown(output))
        return
    # --- default LLM response ---
    response = model.generate_content(text).text
    display(Markdown(f"\n\n{response}"))
# ===================================================================
# TALKATIVE AI STOCK ASSISTANT (Conversational Loop)
# ===================================================================
print("\n" + "="*70)
print(" xAI QUANT DESK — YOUR PERSONAL STOCK ADVISOR")
print(" GRU Model + Gemini Analysis + Deep Research Agent")
print("="*70)
print("Hello! I'm your AI Economic Expert and Stock Market Assistant.")
print("I can predict stock prices, generate reports, discuss markets, share current news, and perform deep research.")
print("Just say something like: 'predict Apple', 'what about Tesla', 'run NVDA forecast', 'deep research on AI stocks', 'stock AAPL'.")
print("For general questions: 'What's the latest on tech stocks?' or 'exit' to quit.\n")
def parse_user_input(user_input: str) -> str:
    """Parse natural language to ticker symbol."""
    user_input_lower = user_input.lower().strip()
    if "predict " in user_input_lower:
        query = user_input_lower.replace("predict ", "").strip()
    elif "run " in user_input_lower:
        query = user_input_lower.replace("run ", "").strip()
    elif "what about " in user_input_lower:
        query = user_input_lower.replace("what about ", "").strip()
    elif "forecast " in user_input_lower:
        query = user_input_lower.replace("forecast ", "").strip()
    else:
        query = user_input_lower
  
    # Direct ticker match
    if query.upper() in ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "NFLX", "AMD", "SPY", "QQQ"]:
        return query.upper()
  
    # Name to ticker map
    for name, ticker in TICKER_MAP.items():
        if name in query:
            print(f"Recognized '{query}' as {ticker}. Running analysis...\n")
            return ticker
  
    # Fallback: no ticker detected
    return None
def handle_general_chat(user_input: str):
    """Handle non-prediction queries using Gemini as economics/stock expert."""
    general_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.GenerationConfig(
            max_output_tokens=800,
            temperature=0.7,
            top_p=0.95,
        )
    )
   
    chat_prompt = f"""
You are a world-class economics professor and stock market strategist with 30+ years at Goldman Sachs.
Respond professionally, insightfully, and concisely to the user's query about economics, markets, or stocks.
Include relevant current news, trends, or analysis based on your knowledge (as of {datetime.now().strftime('%Y-%m-%d')}).
Keep it engaging but institutional-grade. If needed, suggest a prediction run.
User query: {user_input}
"""
    try:
        response = general_model.generate_content(chat_prompt)
        display(Markdown(response.text)) # <-- PROPER MARKDOWN RENDERING
    except Exception as e:
        print("AI: Apologies, a brief technical glitch. Ask me about market trends or try 'predict AAPL'.")
def main(ticker="AAPL"):
    print("=" * 60)
    print("Enhanced Stock Price Prediction System (GRU + Attention + Gemini AI Report)")
    print("=" * 60)
 
    config = Config(ticker)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True) # ← New: for AI reports
 
    data_handler = DataHandler(config)
    data = data_handler.download_data()
    if data is None or data.empty:
        raise SystemExit(f"Failed to download data for {ticker}.")
    plot_frequency_and_heatmap(data, ticker, config.features, config.technical_indicators)
 
    (X_train, y_train), (X_test, y_test), scaled_data = data_handler.prepare_data(data)
    config.n_features = data_handler.n_features
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    print(f"\nInitializing GRU model on {config.device}")
    model = EnhancedGRU(config).to(config.device)
    trainer = Trainer(config)
    print("Starting training...")
    train_losses, test_losses = trainer.train(model, train_loader, test_loader)
    # Load best model
    try:
        model.load_state_dict(torch.load(f'models/{ticker}_model.pth', map_location=config.device))
        print(f"Loaded best checkpoint for {ticker}")
    except FileNotFoundError:
        print("No saved model found — using final trained state")
    # Evaluation
    test_preds, test_actuals = trainer.evaluate(model, test_loader, data_handler)
 
    if len(test_actuals) > 0:
        rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
        mae = mean_absolute_error(test_actuals, test_preds)
        r2 = r2_score(test_actuals, test_preds)
        directional_acc = trainer.directional_accuracy(test_actuals, test_preds)
     
        print(f"\n{ticker} Model Performance:")
        print("-" * 40)
        print(f"RMSE: ${rmse:.2f} | MAE: ${mae:.2f} | R²: {r2:.4f} | Dir. Acc: {directional_acc:.2f}%")
     
        test_dates = data_handler.dates[len(X_train) + config.window_size:]
        plot_results(test_dates[:len(test_actuals)], test_actuals, test_preds, f"{ticker} Test Set", config)
    # Forecast future
    print(f"\nGenerating {config.forecast_days}-day forecast...")
    last_window = scaled_data[-config.window_size:]
    future_prices = predict_future(model, last_window, config.forecast_days, data_handler, config)
 
    start_date = datetime.now() + timedelta(days=1)
    future_dates = [start_date + timedelta(days=i) for i in range(config.forecast_days)]
 
    historical_prices = data_handler.inverse_target_transform(scaled_data[:, 0])
    plot_forecast(data_handler.dates, historical_prices, future_dates, future_prices,
                  np.std(test_preds - test_actuals) if len(test_actuals)>0 else 1.0, config)
    # ———————— GEMINI LLM REPORT GENERATION ————————
    print(f"\nGenerating institutional-grade AI report using Gemini...")
    generate_comprehensive_report(
        ticker=ticker,
        historical_prices=historical_prices,
        future_prices=future_prices,
        future_dates=future_dates,
        test_actuals=test_actuals,
        test_preds=test_preds,
        rmse=rmse if 'rmse' in locals() else 0,
        mae=mae if 'mae' in locals() else 0,
        r2=r2 if 'r2' in locals() else 0,
        directional_acc=directional_acc if 'directional_acc' in locals() else 0,
        config=config
    )
    print("\nFull analysis completed!")
    print("=" * 60)
def generate_comprehensive_report(ticker, historical_prices, future_prices, future_dates,
                                  test_actuals, test_preds, rmse, mae, r2, directional_acc, config):
 
    # Choose model: fast + cheap OR deeper analysis
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash", # Updated to stable model; change if needed
        generation_config=genai.GenerationConfig(
            max_output_tokens=1600,
            temperature=0.7,
            top_p=0.95,
        )
    )
    prompt = f"""
You are a senior quantitative analyst at a $10B hedge fund.
Write a professional, institutional-grade investment report for {ticker.upper()}.
Current date: {datetime.now().strftime('%B %d, %Y')}
Model: GRU + Technical Indicators + Multi-step forecasting
Test Metrics → RMSE: ${rmse:.2f} | MAE: ${mae:.2f} | R²: {r2:.4f} | Directional Accuracy: {directional_acc:.1f}%
Last Close: ${historical_prices[-1]:.2f}
10-Day AI Forecast → ${future_prices[-1]:.2f} ({(future_prices[-1]/historical_prices[-1]-1)*100:+.2f}%)
Next 10 trading day predictions:
{chr(10).join([f"{d.strftime('%b %d (%a)')}: ${p:.2f}" for d, p in zip(future_dates, future_prices)])}
Write a full Markdown report with these sections:
1. Executive Summary (one-sentence conviction view)
2. Model Performance & Reliability
3. Forecast Analysis
4. Risk & Confidence Assessment
5. Trading Recommendation (Strong Buy / Buy / Hold / Sell / Strong Sell)
6. Key Levels & Targets
Use professional tone. Be direct and decisive.
"""
    try:
        response = model.generate_content(prompt)
     
        print("\n" + "="*80)
        print(f" COMPREHENSIVE AI INVESTMENT REPORT FOR {ticker.upper()}")
        print("="*80 + "\n")
     
        # This works in Jupyter, Colab, or VS Code notebooks
        try:
            from IPython.display import Markdown, display
            display(Markdown(response.text))
        except ImportError:
            print(response.text) # Fallback for plain terminal
     
        # Also save to file
        os.makedirs("reports", exist_ok=True)
        with open(f"reports/{ticker}_Gemini_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md", "w") as f:
            f.write(f"# {ticker.upper()} - AI Investment Report (Gemini)\n\n")
            f.write(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n\n")
            f.write(response.text)
         
        print(f"\nReport saved to reports/{ticker}_Gemini_Report_*.md")
     
    except Exception as e:
        print(f"Gemini API Error: {e}")
        print("Check your API key and billing status at https://aistudio.google.com/app/apikey")
if __name__ == "__main__":
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("AI: Thank you for trading with me. Stay sharp in the markets!")
                break
            elif not user_input:
                print("AI: How can I assist with stocks today? E.g., 'predict Apple' or ask about market news.")
                continue
            # Check for research or stock commands first (from deep agent)
            agent(user_input)
            # If not handled by agent, proceed to stock/general
            ticker = parse_user_input(user_input)
            if ticker:
                print(f"AI: Understood! Processing '{user_input}' → {ticker}...")
                main(ticker=ticker)
                print("\nAI: Analysis complete. What next? (e.g., 'predict Tesla' or ask about economy)")
            else:
                # General chat mode
                handle_general_chat(user_input)
        except KeyboardInterrupt:
            print("\nAI: Session ended. Happy investing!")
            break
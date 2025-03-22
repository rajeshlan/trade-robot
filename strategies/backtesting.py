# python strategies\backtesting.py

import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import ccxt
import logging
import random
import time
import torch
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables (called once at the top)
load_dotenv(dotenv_path=r'D:\\RAJESH FOLDER\\PROJECTS\\trade-robot\\config\\API.env')

# Load tokenizer and model directly
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Sentiment Analysis Function
def get_sentiment_signal(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    label = np.argmax(probabilities, axis=1)[0]
    sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
    return 1 if sentiment == "POSITIVE" else -1

# --- Data Fetching and Processing --- #
def fetch_data(symbol, timeframe='1d', limit=200):
    """
    Fetch historical OHLCV data from Binance for a given symbol.
    """
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return None

# --- Technical Indicators --- #
def add_technical_indicators(df):
    """
    Add common technical indicators using pandas_ta library.
    """
    df['SMA'] = ta.sma(df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['EMA'] = ta.ema(df['close'], length=14)
    return df.dropna()

# --- Data Preparation for Model --- #
def prepare_data(df, sequence_length):
    """
    Prepare data for model input, scaling the 'close' prices.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['close']])

    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df_scaled[i-sequence_length:i, 0])
        y.append(df_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# --- Risk Management and Backtesting --- #
def apply_slippage(price, slippage_percentage=0.1):
    """
    Apply slippage to price, to simulate real market conditions.
    """
    slippage = random.uniform(-slippage_percentage, slippage_percentage)
    return price * (1 + slippage)

def backtest_with_slippage(df, model, X, y, scaler, transaction_cost=0.001, stop_loss=0.02, take_profit=0.05, risk_factor=1, slippage=0.01):
    """
    Run a backtest with slippage, transaction cost, stop loss, and take profit.
    """
    predictions = predict(model, X, scaler)
    df['predictions'] = np.nan  # Initialize the predictions column with NaN values

    # Only update the predictions column for valid rows (after sequence_length)
    df.iloc[len(df) - len(predictions):, df.columns.get_loc('predictions')] = predictions

    initial_balance = 10000
    balance = initial_balance
    position = 0
    entry_price = 0

    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    current_risk_factor = df['ATR'].iloc[-1] * risk_factor

    for i in range(len(df)):
        current_price = apply_slippage(df['close'].iloc[i], slippage)

        position_size = (balance * 0.02) / (df['ATR'].iloc[i] * current_risk_factor)

        if df['predictions'].iloc[i] > current_price * (1 + take_profit):
            if position > 0:
                balance += position * current_price * (1 - transaction_cost)
                position = 0
        elif df['predictions'].iloc[i] < current_price * (1 - stop_loss):
            if position > 0:
                balance += position * current_price * (1 - transaction_cost)
                position = 0
        elif df['predictions'].iloc[i] > current_price * (1 + transaction_cost):
            if position == 0:
                position = position_size
                balance -= position * current_price
        elif df['predictions'].iloc[i] < current_price * (1 - transaction_cost):
            if position > 0:
                balance += position * current_price * (1 - transaction_cost)
                position = 0

    if position > 0:
        balance += position * current_price * (1 - transaction_cost)

    logger.info(f"Final balance after backtest with slippage: {balance:.2f}")
    return balance

def adaptive_risk_management(df, atr_period=14):
    """
    Adaptive risk management using Average True Range (ATR).
    """
    df['ATR'] = df['close'].rolling(window=atr_period).std()
    risk_factor = df['ATR'].iloc[-1]
    return risk_factor

# --- Model Training and Prediction --- #
def train_model(data, sequence_length=60):
    """
    Train an XGBoost model with the given data.
    """
    X, y, scaler = prepare_data(data, sequence_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Model Mean Squared Error: {mse:.4f}")

    return model, scaler

def predict(model, X, scaler):
    """
    Predict future values using the trained model.
    """
    predictions = model.predict(X)

    # Ensure predictions are reshaped properly for inverse transformation
    predictions = predictions.reshape(-1, 1)
    
    # Inverse transform to get the actual price predictions
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()

# --- Trading and Order Placement --- #
def place_order(exchange, symbol, side, amount):
    """
    Place a market order on the Bybit exchange.
    """
    if side == "buy":
        order = exchange.create_market_buy_order(symbol, amount)
    elif side == "sell":
        order = exchange.create_market_sell_order(symbol, amount)
    else:
        raise ValueError("Order side must be 'buy' or 'sell'")

    return order

def main():
    # Symbol for Bybit market
    symbol = 'BTC/USDT'
    
    # Fetching data (same as before)
    data = fetch_data(symbol)
    
    if data is not None:
        data = add_technical_indicators(data)

        sentiment_text = "The market is looking great today"
        sentiment_signal = get_sentiment_signal(sentiment_text)
        logger.info(f"Sentiment signal: {sentiment_signal}")

        model, scaler = train_model(data)

        X, y, scaler = prepare_data(data, 60)
        final_balance = backtest_with_slippage(data, model, X, y, scaler)

        logger.info(f"Final balance after backtest: {final_balance:.2f}")

        risk_factor = adaptive_risk_management(data)
        logger.info(f"Risk factor based on ATR: {risk_factor:.4f}")

    # Initialize Bybit exchange using the correct API keys (already loaded via load_dotenv)
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")

    exchange = ccxt.bybit({'apiKey': api_key, 'secret': api_secret})

    # Example of placing an order (you can replace 'buy' or 'sell' with your preferred action)
    side = "buy"
    amount = 0.001

    try:
        order = place_order(exchange, symbol, side, amount)
        logger.info(f"Order placed: {order}")
    except ccxt.InsufficientFunds as e:
        logger.error("Insufficient balance to place the order. Please add funds.")
    except Exception as e:
        logger.error(f"Error placing order: {e}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")

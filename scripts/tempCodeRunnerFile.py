#python scripts\tempCodeRunnerFile.py

import logging
import ccxt
import pandas as pd
import os
import time
import ntplib
import joblib
import pandas_ta as ta
from dotenv import load_dotenv
from datetime import datetime, timezone

# Setup logging with enhanced structure
LOG_FILE = "trade_bot.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv(dotenv_path=r'D:\RAJESH FOLDER\PROJECTS\trade-robot\config\API.env')

# Bybit API credentials
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')

if not API_KEY or not API_SECRET:
    logging.error("API credentials are missing. Check your .env file.")
    exit(1)

def synchronize_system_time(retries=3):
    """
    Synchronize system time with an NTP server, with retries and alternate servers.
    Returns the time offset in milliseconds.
    """
    ntp_servers = ['pool.ntp.org', 'time.google.com', 'time.windows.com']
    logging.info("Starting system time synchronization...")

    for attempt in range(retries):
        logging.info(f"Attempt {attempt + 1} of {retries}.")
        for server in ntp_servers:
            try:
                response = ntplib.NTPClient().request(server, timeout=5)
                offset = int(response.offset * 1000)  # Use NTP response offset directly
                logging.info(f"System time synchronized with offset: {offset} ms using server {server}")
                return offset
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for server {server}: {e}")

    logging.error("All attempts to synchronize time failed.")
    return 0  # Return zero offset if synchronization fails

def verify_exchange_time(exchange, time_offset):
    server_time = exchange.fetch_time()
    local_time = int((time.time() * 1000) + time_offset)
    logging.info(f"Server time: {server_time}, Local time with offset: {local_time}")
    if abs(server_time - local_time) > 1000:  # 1-second tolerance
        logging.warning("Time mismatch detected!")
    else:
        logging.info("Time synchronization successful.")

def initialize_exchange(api_key, api_secret):
    """
    Initialize the Bybit exchange.
    """
    try:
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        logging.info("Initialized Bybit exchange")
        return exchange
    except ccxt.BaseError as e:
        logging.error("Failed to initialize exchange: %s", e)
        raise e

def fetch_ohlcv(exchange, symbol, timeframe='1h', limit=100, time_offset=0):
    """
    Fetch OHLCV data from the exchange.
    """
    try:
        params = {
            'recvWindow': 100000,  # Set a large enough recv_window
            'timestamp': int(time.time() * 1000) + time_offset
        }
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Fetched OHLCV data for {symbol}")
        return df
    except ccxt.BaseError as e:
        logging.error("Error fetching OHLCV data: %s", e)
        raise e

def calculate_indicators(df):
    """
    Calculate technical indicators for the DataFrame.
    """
    try:
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)
        df['EMA_12'] = ta.ema(df['close'], length=12)
        df['EMA_26'] = ta.ema(df['close'], length=26)
        macd = ta.macd(df['close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['RSI'] = ta.rsi(df['close'], length=14)

        df.bfill(inplace=True)
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        logging.info("Calculated technical indicators")
    except Exception as e:
        logging.error("Error calculating indicators: %s", e)
        raise e
    return df

def trading_strategy(df, sma_short=50, sma_long=200):
    """
    Define the trading strategy based on SMA crossover.
    """
    try:
        signals = ['hold']  # Initialize with 'hold' for the first entry
        for i in range(1, len(df)):
            if (
                df[f'SMA_{sma_short}'].iloc[i] > df[f'SMA_{sma_long}'].iloc[i]
                and df[f'SMA_{sma_short}'].iloc[i - 1] <= df[f'SMA_{sma_long}'].iloc[i - 1]
            ):
                signals.append('buy')
            elif (
                df[f'SMA_{sma_short}'].iloc[i] < df[f'SMA_{sma_long}'].iloc[i]
                and df[f'SMA_{sma_short}'].iloc[i - 1] >= df[f'SMA_{sma_long}'].iloc[i - 1]
            ):
                signals.append('sell')
            else:
                signals.append('hold')
        df['signal'] = signals
        logging.info("Defined trading strategy")
    except Exception as e:
        logging.error("Error defining trading strategy: %s", e)
        raise e
    return df

def fetch_balance(exchange):
    """
    Fetch the account balance.
    """
    try:
        balance = exchange.fetch_balance()
        logging.info(f"Fetched account balance: {balance}")
        return balance
    except ccxt.BaseError as e:
        logging.error("Error fetching balance: %s", e)
        raise e

def execute_trade(exchange, symbol, signal, available_usd, btc_price):
    """
    Execute trades based on the signal.
    """
    try:
        if signal == 'buy' and available_usd > 0:
            amount = available_usd / btc_price
            exchange.create_market_buy_order(symbol, amount)
            available_usd -= amount * btc_price
            logging.info(f"Executed Buy Order for {amount} BTC.")
        elif signal == 'sell':
            exchange.create_market_sell_order(symbol, available_usd / btc_price)
            available_usd = 0  # Assume all sold for simplicity
            logging.info(f"Executed Sell Order for {available_usd / btc_price} BTC.")

    except ccxt.BaseError as e:
        logging.error(f"Error executing {signal} order: {e}")
        raise e

def load_trained_model(model_path):
    """
    Load pre-trained ML model for price prediction or trade decision-making.
    """
    try:
        model = joblib.load(model_path)
        logging.info("Loaded trained model successfully")
        return model
    except Exception as e:
        logging.error("Error loading model: %s", e)
        raise e

def predict_signal(model, df):
    """
    Use the pre-trained model to predict the next signal (buy/sell/hold).
    """
    try:
        features = df[['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal']].values
        predictions = model.predict(features)
        df['predicted_signal'] = predictions
        logging.info("Predicted trading signals using AI/ML model")
    except Exception as e:
        logging.error("Error predicting signals: %s", e)
        raise e
    return df

def summarize_account_balance(account_data):
    """
    Summarize account balance for reporting.
    """
    coins = account_data['result']['list'][0]['coin']
    summary = {
        coin['coin']: {
            'equity': coin['equity'],
            'wallet_balance': coin['walletBalance'],
            'usd_value': coin['usdValue']
        } for coin in coins
    }
    return summary

def main():
    """
    Main execution logic for the trading bot.
    """
    try:
        # Synchronize time with NTP server
        time_offset = synchronize_system_time()
        logging.info("Time synchronized with offset: %d", time_offset)

        # Initialize exchange
        exchange = initialize_exchange(API_KEY, API_SECRET)

        # Fetch OHLCV data
        df = fetch_ohlcv(exchange, 'BTCUSDT', time_offset=time_offset)

        # Calculate technical indicators
        df = calculate_indicators(df)

        # Define trading strategy
        df = trading_strategy(df)

        # Fetch account balance
        balance = fetch_balance(exchange)
        available_usd = balance['total'].get('USDT', 0)
        if available_usd == 0:
            logging.error("No USDT available for trading.")
            return
        btc_price = exchange.fetch_ticker('BTCUSDT')['last']

        # Execute trades based on signals
        for _, row in df.iterrows():
            execute_trade(exchange, 'BTCUSDT', row['signal'], available_usd, btc_price)

        # Output the resulting DataFrame
        print("Execution completed. Here are the last few rows of the DataFrame:")
        print(df.tail())

    except Exception as e:
        logging.error("An error occurred during execution: %s", e)

if __name__ == "__main__":
    main()

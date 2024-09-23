from multiprocessing import Value
import numpy as np
from textblob import TextBlob
import tweepy
import os
import ccxt
import pandas as pd
import pandas_ta as ta
import logging
import time
from datetime import datetime
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def synchronize_time_with_exchange(exchange: ccxt.Exchange) -> int:
    """
    Synchronize the local system time with the exchange server time.
    
    Args:
    - exchange: ccxt.Exchange object
    
    Returns:
    - time_offset: Time offset in milliseconds
    """
    try:
        server_time = exchange.milliseconds()
        local_time = int(datetime.now().timestamp() * 1000)
        time_offset = server_time - local_time
        logging.info("Time synchronized with exchange. Offset: %d milliseconds", time_offset)
        return time_offset
    except ccxt.BaseError as sync_error:
        logging.error("Failed to synchronize time with exchange: %s", sync_error)
        raise sync_error

# fetch_data.py

import pandas as pd

def get_historical_data(file_path):
    """
    Load historical data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise IOError(f"Error loading historical data from {file_path}: {e}")

# Add any other functions like get_tweets, analyze_sentiment, etc.


def get_tweets(api_key, api_secret, query):
    auth = tweepy.AppAuthHandler(api_key, api_secret)
    api = tweepy.API(auth)
    tweets = api.search(q=query, count=100, lang='en')
    return [tweet.text for tweet in tweets]

def analyze_sentiment(tweets):
    sentiment_scores = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
    return np.mean(sentiment_scores)

def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given symbol and timeframe from the exchange.
    
    Args:
    - exchange: ccxt.Exchange object
    - symbol: Trading pair symbol (e.g., 'BTCUSDT')
    - timeframe: Timeframe for OHLCV data (default: '1h')
    - limit: Number of data points to fetch (default: 100)
    
    Returns:
    - df: DataFrame containing OHLCV data
    """
    try:
        # Synchronize time with exchange
        time_offset = synchronize_time_with_exchange(exchange)
        
        # Fetch OHLCV data
        params = {
            'param1': Value,
            'param2': Value,
            'recvWindow': 10000,  # Adjust recvWindow as needed
            'timestamp': exchange.milliseconds() + time_offset
        }
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info("Fetched OHLCV data for %s", symbol)
        
        if df.isnull().values.any():
            raise ValueError("Fetched data contains null values")
        
        return df
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BaseError) as e:
        logging.error("Error fetching OHLCV data: %s", e)
        raise e
        
        
    except ccxt.NetworkError as net_error:
        logging.error("Network error while fetching OHLCV data: %s", net_error)
        raise net_error
    except ccxt.ExchangeError as exchange_error:
        logging.error("Exchange error while fetching OHLCV data: %s", exchange_error)
        raise exchange_error
    except ccxt.BaseError as base_error:
        logging.error("Unexpected error while fetching OHLCV data: %s", base_error)
        raise base_error

def perform_technical_analysis(df: pd.DataFrame, sma_lengths: Tuple[int, int] = (20, 50), rsi_length: int = 14, macd_params: Tuple[int, int, int] = (12, 26, 9)) -> pd.DataFrame:
    """
    Perform technical analysis on the OHLCV data DataFrame.
    
    Args:
    - df: DataFrame containing OHLCV data
    - sma_lengths: Tuple containing lengths for SMAs (default: (20, 50))
    - rsi_length: Length for RSI (default: 14)
    - macd_params: Tuple containing MACD parameters (fast, slow, signal) (default: (12, 26, 9))
    
    Returns:
    - df: DataFrame with added technical indicators
    """
    try:
        # Adding technical indicators
        sma_short, sma_long = sma_lengths
        macd_fast, macd_slow, macd_signal = macd_params
        
        # Calculate SMAs
        df[f'SMA_{sma_short}'] = ta.sma(df['close'], length=sma_short)
        df[f'SMA_{sma_long}'] = ta.sma(df['close'], length=sma_long)
        
        # Calculate RSI
        df['RSI'] = ta.rsi(df['close'], length=rsi_length)
        
        # Calculate MACD
        macd_data = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        
        # Assign MACD values
        df['MACD'] = macd_data['MACD'] if 'MACD' in macd_data else None
        df['MACD_signal'] = macd_data['MACD_signal'] if 'MACD_signal' in macd_data else None
        
        # Log detected patterns
        logging.info("Calculated SMA (%d, %d), RSI (%d), and MACD (%d, %d, %d) indicators", sma_short, sma_long, rsi_length, macd_fast, macd_slow, macd_signal)
        
        detect_signals(df, sma_lengths)
        
        return df
    
    except Exception as e:
        logging.error("An error occurred during technical analysis: %s", e)
        raise e


def detect_signals(df: pd.DataFrame, sma_lengths: Tuple[int, int]) -> None:
    """
    Detect bullish or bearish signals in the OHLCV data.
    
    Args:
    - df: DataFrame containing OHLCV data
    - sma_lengths: Tuple containing lengths for SMAs
    """
    try:
        sma_short, sma_long = sma_lengths
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Example signal detection for SMA crossover
        if pd.notna(previous[f'SMA_{sma_short}']) and pd.notna(previous[f'SMA_{sma_long}']) and pd.notna(latest[f'SMA_{sma_short}']) and pd.notna(latest[f'SMA_{sma_long}']):
            if previous[f'SMA_{sma_short}'] < previous[f'SMA_{sma_long}'] and latest[f'SMA_{sma_short}'] > latest[f'SMA_{sma_long}']:
                logging.info("Bullish SMA crossover detected")
            elif previous[f'SMA_{sma_short}'] > previous[f'SMA_{sma_long}'] and latest[f'SMA_{sma_short}'] < latest[f'SMA_{sma_long}']:
                logging.info("Bearish SMA crossover detected")
        
        # Example signal detection for RSI conditions (using correct column name)
        if pd.notna(latest['RSI']):
            if latest['RSI'] > 70:
                logging.info("RSI indicates overbought conditions")
            elif latest['RSI'] < 30:
                logging.info("RSI indicates oversold conditions")
        
        # Example signal detection for MACD crossover (using correct column name)
        if pd.notna(previous['MACD']) and pd.notna(previous['MACD_signal']) and pd.notna(latest['MACD']) and pd.notna(latest['MACD_signal']):
            if previous['MACD'] < previous['MACD_signal'] and latest['MACD'] > latest['MACD_signal']:
                logging.info("Bullish MACD crossover detected")
            elif previous['MACD'] > previous['MACD_signal'] and latest['MACD'] < latest['MACD_signal']:
                logging.info("Bearish MACD crossover detected")
        
    except Exception as e:
        logging.error("An error occurred during signal detection: %s", e)
        raise e
    
def fetch_historical_data(exchange, symbol, timeframe, limit=100, params=None):
    """
    Fetch historical OHLCV data from the specified exchange.
    
    :param exchange: The exchange object initialized with API keys.
    :param symbol: The trading pair symbol (e.g., 'BTCUSDT').
    :param timeframe: The timeframe for the data (e.g., '1d' for daily data).
    :param limit: The number of data points to retrieve (default is 100).
    :param params: Additional parameters for the API call (default is None).
    :return: A list of OHLCV data.
    """
    if params is None:
        params = {}
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, params=params)
        return ohlcv
    except ccxt.BaseError as e:
        print(f"An error occurred: {e}")
        return []

def fetch_real_time_data(exchange: ccxt.Exchange, symbol: str, timeframe: str = '1m', limit: int = 100):
    """
    Fetch real-time OHLCV data for a symbol from the exchange.
    Continuously fetches new data points.
    
    Args:
    - exchange: ccxt.Exchange object
    - symbol: Trading pair symbol (e.g., 'BTCUSDT')
    - timeframe: Timeframe for OHLCV data (default: '1m')
    - limit: Number of data points to initially fetch (default: 100)
    """
    try:
        while True:
            # Fetch new data points
            new_df = fetch_ohlcv(exchange, symbol, timeframe, limit)
            
            # Perform technical analysis on new data
            new_df = perform_technical_analysis(new_df)
            
            # Print or process new data as needed
            print(new_df.tail())  # Example: print last few rows of new data
            
            # Sleep for a minute (or desired interval)
            time.sleep(60)  # Sleep for 60 seconds (1 minute)
            
    except ccxt.NetworkError as net_error:
        logging.error("A network error occurred: %s", net_error)
        # Retry or handle the error as needed
    except ccxt.ExchangeError as exchange_error:
        logging.error("An exchange error occurred: %s", exchange_error)
        # Handle the exchange-specific error
    except Exception as error:
        logging.error("An unexpected error occurred: %s", error)
        # Handle any other unexpected errors

def fetch_data(self, symbol, timeframe='1h', limit=100):
    try:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Fetched real-time OHLCV data for {symbol}")
        return df
    except ccxt.BaseError as e:
        logging.error("Error fetching real-time OHLCV data: %s", e)
        raise e

def main():
    try:
        # Retrieve API keys and secrets from environment variables
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')

        if not api_key or not api_secret:
            raise ValueError("BYBIT_API_KEY or BYBIT_API_SECRET environment variables are not set.")

        # Initialize the Bybit exchange
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # This helps to avoid rate limit errors
        })

        # Start fetching real-time data
        symbol = 'BTCUSDT'
        fetch_real_time_data(exchange, symbol)

    except ccxt.NetworkError as net_error:
        logging.error("A network error occurred: %s", net_error)
        # Retry or handle the error as needed
    except ccxt.ExchangeError as exchange_error:
        logging.error("An exchange error occurred: %s", exchange_error)
        # Handle the exchange-specific error
    except ValueError as value_error:
        logging.error("Value error occurred: %s", value_error)
        # Handle missing environment variables or other value-related errors
    except Exception as error:
        logging.error("An unexpected error occurred: %s", error)
        # Handle any other unexpected errors

# tradingbot/fetch_data.py

def fetch_historical_data(exchange, symbol, timeframe='1h', limit=100, params=None):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)


if __name__ == "__main__":
    main()


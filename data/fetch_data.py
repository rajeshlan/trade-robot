#RUN ON DIFFECT SYSTEM AS YOUR PC HAS ISSUES

from multiprocessing import Value
import numpy as np
from textblob import TextBlob
import tweepy
import os
import ccxt
import pandas as pd
import pandas_ta as ta
import pandas as pd
import logging
import time
import ccxt
import logging
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Self, Tuple
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='F:\\trading\\improvised-code-of-the-pdf-GPT-main\\config\\API.env')


bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def synchronize_time_with_exchange(exchange):
    try:
        exchange_time = exchange.fetch_time()
        logging.info(f"Exchange time: {exchange_time}")
        # Add your logic for time synchronization here.
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BaseError) as e:
        logging.error(f"Error synchronizing time with exchange: {type(e).__name__}, {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return None

# fetch_data.py
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
def get_tweets(bearer_token, query):
    client = tweepy.Client(bearer_token=bearer_token)
    try:
        response = client.search_recent_tweets(query=query, max_results=100, tweet_fields=["lang"])
        tweets = response.data
        return [tweet.text for tweet in tweets if tweet.lang == 'en'] if tweets else []
    except Exception as e:
        logging.error(f"Error fetching tweets: {e}")
        return []

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    logging.info(f"Sentiment analysis: {sentiment}")
    return sentiment

def fetch_ohlcv(exchange, symbol, timeframe='1m', limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        logging.info(f"Fetched {len(ohlcv)} data points for {symbol} on {timeframe} timeframe.")
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except ccxt.BaseError as e:
        logging.error(f"Error fetching OHLCV data: {type(e).__name__} - {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching OHLCV data: {str(e)}")
        return None


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
            print(new_df)
            
            # Sleep for a defined period before fetching new data
            time.sleep(60)  # Adjust sleep duration as needed
    except Exception as e:
        logging.error(f"Error fetching real-time data: {e}")

# Define the FetchData class
class FetchData:
    def __init__(self, bearer_token: str, api_key: str, api_secret: str, query: str, symbol: str, timeframe: str = '1m'):
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.api_secret = api_secret
        self.query = query
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Properly initialize the Bybit exchange with authentication
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use 'spot' if not trading futures
            }
        })

    def get_tweets(self):
        """
        Fetch tweets based on the specified query.
        """
        client = tweepy.Client(bearer_token=self.bearer_token)
        try:
            response = client.search_recent_tweets(query=self.query, max_results=100, tweet_fields=["lang"])
            tweets = response.data
            return [tweet.text for tweet in tweets if tweet.lang == 'en'] if tweets else []
        except Exception as e:
            logging.error(f"Error fetching tweets: {e}")
            return []


    
    def get_tweets(self):
        """
        Fetch tweets based on the specified query.
        """
        try:
            # Log that the function is attempting to use the bearer token
            logging.info(f"Using bearer token: {self.bearer_token[:10]}...")  # Mask the token for security

            # Initialize the OAuth2 handler with the bearer token
            auth = tweepy.OAuth2BearerHandler(self.bearer_token)
            api = tweepy.API(auth)

            # Fetch tweets (make sure the 'search_tweets' method parameters are valid)
            tweets = api.search_tweets(q=self.query, count=100, lang='en')

            # Return only the text content of the tweets
            return [tweet.text for tweet in tweets]

        except Exception as e:
            # Log the error if fetching tweets fails
            logging.error(f"Error fetching tweets: {e}")
            return []



    def analyze_sentiment(self, text):
        """
        Analyze sentiment of the provided text using TextBlob.
        """
        blob = TextBlob(text)
        return blob.sentiment

    def fetch_and_analyze_data(self):
        """
        Fetch real-time data and analyze sentiment.
        """
        while True:
            try:
                # Fetch real-time OHLCV data
                ohlcv = fetch_ohlcv(self.exchange, self.symbol, self.timeframe)
                logging.info(f"Fetched {len(ohlcv)} data points for {self.symbol}")

                # Perform sentiment analysis on the fetched tweets
                tweets = self.get_tweets()
                for tweet in tweets:
                    sentiment = self.analyze_sentiment(tweet)
                    logging.info(f"Tweet: {tweet}, Sentiment: {sentiment}")

                # Sleep for a defined period before fetching new data
                time.sleep(60)  # Adjust sleep duration as needed
            except Exception as e:
                logging.error(f"Error in fetch_and_analyze_data: {e}")

# Usage example
if __name__ == "__main__":
    # Initialize FetchData object
    fetcher = FetchData(api_key='your_api_key', 
                    api_secret='your_api_secret', 
                    bearer_token=bearer_token, 
                    query='Bitcoin', 
                    symbol='BTC/USDT')

    # Fetch tweets
    tweets = fetcher.get_tweets()

    # Fetch and analyze data
    fetcher.fetch_and_analyze_data()


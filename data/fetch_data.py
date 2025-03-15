# python data\fetch_data.py (2024-12-28 13:21:44,920 - ERROR - Error fetching tweets: 429 Too Many Requests)

import os
import logging
from datetime import datetime
import ccxt
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv(dotenv_path='D:\\RAJESH FOLDER\\PROJECTS\\trade-robot\\config\\API.env')

# Twitter Bearer Token
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

# Bybit API credentials
bybit_api_key = os.getenv("BYBIT_API_KEY")
bybit_api_secret = os.getenv("BYBIT_API_SECRET")

def fetch_data_from_bybit(symbol='BTC/USDT:USDT', timeframe='1m', limit=500) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit and return a DataFrame.

    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT:USDT').
        timeframe (str): Timeframe for OHLCV data.
        limit (int): Number of data points to fetch.

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data.
    """
    df = pd.DataFrame()  # ✅ Ensure df is always initialized

    try:
        # Initialize the Bybit exchange instance
        exchange = ccxt.bybit({
            'apiKey': bybit_api_key,
            'secret': bybit_api_secret,
            'enableRateLimit': True,
        })

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        # Convert to pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # ✅ Now it's safe to print df.columns
        print("Fetched DataFrame columns:", df.columns)

        logging.info(f"Successfully fetched {len(df)} OHLCV data points from Bybit.")

    except Exception as e:
        logging.error(f"Error fetching OHLCV data: {e}")

    return df  # ✅ Always return a DataFrame, even if empty

def perform_technical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform technical analysis by adding SMA, RSI, and MACD to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data.

    Returns:
        pd.DataFrame: DataFrame with added indicators.
    """
    try:
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)

        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']

        logging.info("Technical analysis completed. Indicators added.")
        return df
    except Exception as e:
        logging.error(f"Error performing technical analysis: {e}")
        return df

def get_tweets(bearer_token: str, query: str) -> list:
    """
    Fetch recent tweets using the Twitter API.

    Args:
        bearer_token (str): Twitter API bearer token.
        query (str): Query string to search for tweets.

    Returns:
        list: List of tweets in English.
    """
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        response = client.search_recent_tweets(query=query, max_results=10, tweet_fields=["lang"])
        
        if response.errors:
            logging.error(f"Twitter API Error: {response.errors}")
            return []

        tweets = response.data
        if tweets:
            english_tweets = [tweet.text for tweet in tweets if hasattr(tweet, "lang") and tweet.lang == "en"]
            logging.info(f"Fetched {len(english_tweets)} English tweets for query '{query}'.")
            return english_tweets
        else:
            logging.info("No tweets found for the query.")
            return []
    except tweepy.TweepyException as e:
        logging.error(f"Error fetching tweets: {e}")
        return []

def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a text using VaderSentiment.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: Sentiment analysis scores.
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        logging.info(f"Sentiment analysis result: {sentiment}")
        return sentiment
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return {}

def main():
    """
    Main function to execute the workflow.
    """
    # Fetch data from Bybit
    logging.info("Fetching OHLCV data from Bybit...")
    ohlcv_data = fetch_data_from_bybit(limit=500)

    if not ohlcv_data.empty:
        # Perform technical analysis
        logging.info("Performing technical analysis...")
        analyzed_data = perform_technical_analysis(ohlcv_data)
        logging.info(f"Analyzed data preview:\n{analyzed_data[['timestamp', 'close', 'SMA_20', 'RSI']].head()}")

    # Fetch and analyze tweets
    logging.info("Fetching tweets for sentiment analysis...")
    tweets = get_tweets(bearer_token, query="Bitcoin")

    if tweets:
        for tweet in tweets[:5]:  # Analyze first 5 tweets for brevity
            analyze_sentiment(tweet)

if __name__ == "__main__":
    main()

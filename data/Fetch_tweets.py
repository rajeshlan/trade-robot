# python data\Fetch_tweets.py (enhanced with better rate limit handling)

import os
import random
import time
import requests
import json
from transformers import pipeline  # Correct import from HuggingFace transformers
import tweepy
import sqlite3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('./config/API.env')

SEARCH_TERMS = ["Bitcoin", "BTC", "Ethereum", "ETH", "cryptocurrency", "crypto"]
QUERY = " OR ".join(SEARCH_TERMS)
MAX_TWEETS = 10  # Reduced to 50 to minimize request size and reduce rate limit hits

# Rotate between multiple API keys if available
TWITTER_API_KEYS = [
    os.getenv("TWITTER_BEARER_TOKEN_1"),
]
BEARER_TOKEN_INDEX = 0  # Start with the first API key

# Print tokens to debug if they are loaded correctly (REMOVE this after debugging)
# Print tokens to debug if they are loaded correctly (with error handling)
for i, token in enumerate(TWITTER_API_KEYS):
    if token:
        print(f"Token {i+1}: {token[:10]}...")  # Print the first 10 characters for debugging
    else:
        print(f"Error: No bearer token found for TWITTER_BEARER_TOKEN_{i+1}. Please check your .env file.")

def rotate_bearer_token():
    """ Rotate the bearer token to the next available key. """
    global BEARER_TOKEN_INDEX
    BEARER_TOKEN_INDEX = (BEARER_TOKEN_INDEX + 1) % len(TWITTER_API_KEYS)
    print(f"Rotating to next API key (Index: {BEARER_TOKEN_INDEX}).")
    return TWITTER_API_KEYS[BEARER_TOKEN_INDEX]


# Load the sentiment-analysis pipeline from HuggingFace (distilBERT-based)
nlp = pipeline("sentiment-analysis")

def label_sentiment_advanced(tweet_text):
    """
    Label sentiment using a pre-trained HuggingFace sentiment analysis pipeline.

    Parameters:
    tweet_text (str): The tweet text to analyze.

    Returns:
    str: Sentiment label ('positive', 'neutral', or 'negative').
    """
    sentiment_result = nlp(tweet_text)[0]
    if sentiment_result['label'] == 'POSITIVE':
        return 'positive'
    elif sentiment_result['label'] == 'NEGATIVE':
        return 'negative'
    else:
        return 'neutral'


def bearer_oauth(headers):
    token = TWITTER_API_KEYS[BEARER_TOKEN_INDEX]
    if not token:
        print(f"Error: No bearer token found at index {BEARER_TOKEN_INDEX}")
    else:
        print(f"Using token (Index {BEARER_TOKEN_INDEX}): {token[:10]}...")

    headers["Authorization"] = f"Bearer {token}"
    headers["User-Agent"] = "v2TweetLookupPython"
    return headers

def exponential_backoff_with_jitter(base=3, attempt=1, cap=300):
    """
    Implements exponential backoff with jitter to avoid retry collisions.
    Parameters:
        base: Base wait time (in seconds)
        attempt: Retry attempt number
        cap: Maximum wait time cap
    """
    return min(base * (2 ** (attempt - 1)) + random.uniform(0, 1), cap)


def fetch_tweets_v2(max_retries=5):
    """
    Fetches tweets using Twitter API v2 endpoint with proper header handling.
    Implements rate limit handling, retries, and API key rotation.
    """
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": QUERY,
        "tweet.fields": "text,lang,created_at,public_metrics,author_id",
        "max_results": min(MAX_TWEETS, 100)
    }

    headers = bearer_oauth({})  # Pass an empty dictionary and modify it

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    return [
                        {
                            "text": tweet['text'],
                            "label": label_sentiment_advanced(tweet['text'])
                        }
                        for tweet in data['data']
                    ], None
                else:
                    return None, "No data found in response."

            elif response.status_code == 429:  # Rate limit hit
                wait_time = exponential_backoff_with_jitter(attempt=attempt+1)
                print(f"Rate limit hit. Retrying after {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                headers = bearer_oauth({})  # Refresh token after retry
            else:
                return None, f"Error {response.status_code}: {response.text}"

        except Exception as e:
            return None, f"Exception occurred: {str(e)}"

    return None, f"Max retries ({max_retries}) exceeded. Failed to fetch tweets."


def fetch_tweets_v1():
    """
    Fetches tweets using Twitter API v1.1 endpoint via Tweepy with backoff.
    """
    client = tweepy.Client(
    consumer_key=os.getenv("TWITTER_API_KEY"),
    consumer_secret=os.getenv("TWITTER_API_SECRET"),
    access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
    access_token_secret=os.getenv("TWITTER_ACCESS_SECRET"),
    bearer_token=TWITTER_API_KEYS[BEARER_TOKEN_INDEX]  # Use the rotated bearer token
)


    for attempt in range(5):  # Retry up to 5 times
        try:
            response = client.search_recent_tweets(
                query=QUERY,
                max_results=MAX_TWEETS,
                tweet_fields=["text", "lang", "created_at", "public_metrics", "author_id"]
            )
            if response.data:
                return [
                    {
                        "text": tweet.text,
                        "label": label_sentiment_advanced(tweet.text)
                    }
                    for tweet in response.data
                ], None
            else:
                return None, "No tweets found."

        except tweepy.TooManyRequests:
            wait_time = exponential_backoff_with_jitter(attempt=attempt+1)
            print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            rotate_bearer_token()  # Rotate API key after each retry
        except tweepy.TweepyException as e:
            return None, f"Tweepy error: {str(e)}"
        except Exception as e:
            return None, f"Exception occurred: {str(e)}"

    return None, "Max retries exceeded."

def save_tweets_to_db(tweets):
    """
    Save tweets and their sentiment labels to an SQLite database.
    
    Parameters:
    tweets (list): List of tweet dictionaries containing text and label.
    """
    conn = sqlite3.connect('tweets.db')  # Create SQLite database (or connect if exists)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tweets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        label TEXT NOT NULL
    )
    ''')

    # Insert each tweet into the database
    for tweet in tweets:
        cursor.execute('INSERT INTO tweets (text, label) VALUES (?, ?)', (tweet['text'], tweet['label']))

    conn.commit()
    conn.close()
    print("Tweets saved to database.")

def fetch_tweets():
    """
    Attempts to fetch tweets using v2 first, then v1 if v2 fails.
    """
    print("Fetching tweets using Twitter API v2...")
    tweets, error = fetch_tweets_v2()
    if tweets:
        print("Successfully fetched tweets using v2.")
        save_tweets_to_db(tweets)  # Save to database

        return

    print(f"v2 Error: {error}")

    print("\nFetching tweets using Twitter API v1.1...")
    tweets, error = fetch_tweets_v1()
    if tweets:
        print("Successfully fetched tweets using v1.")
        save_tweets_to_db(tweets)  # Save to database

    else:
        print(f"v1 Error: {error}")


if __name__ == "__main__":
    fetch_tweets()

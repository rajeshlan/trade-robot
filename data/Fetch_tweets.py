# python data\Fetch_tweets.py (working now)

import os
import random
import time
import requests
import json
import tweepy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('./config/API.env')

SEARCH_TERMS = ["Bitcoin", "BTC", "Ethereum", "ETH", "cryptocurrency", "crypto"]
QUERY = " OR ".join(SEARCH_TERMS)
MAX_TWEETS = 100
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")


def label_sentiment(tweet_text):
    """
    Labels the sentiment of a tweet based on keywords.
    """
    positive_keywords = ['happy', 'bullish', 'up', 'profit', 'win', 'great']
    negative_keywords = ['sad', 'bearish', 'down', 'loss', 'fail', 'bad']
    
    text_lower = tweet_text.lower()
    if any(keyword in text_lower for keyword in positive_keywords):
        return 'positive'
    elif any(keyword in text_lower for keyword in negative_keywords):
        return 'negative'
    else:
        return 'neutral'


def bearer_oauth(headers):
    """
    Method required by bearer token authentication.
    This function modifies a headers dictionary directly.
    """
    headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    headers["User-Agent"] = "v2TweetLookupPython"
    return headers  # Return the modified headers dictionary


def fetch_tweets_v2(max_retries=5):
    """
    Fetches tweets using Twitter API v2 endpoint with proper header handling.
    Implements rate limit handling and retries up to max_retries.
    """
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": QUERY,
        "tweet.fields": "text,lang,created_at,public_metrics,author_id",
        "max_results": min(MAX_TWEETS, 100)
    }

    headers = bearer_oauth({})  # Pass an empty dictionary and modify it

    for attempt in range(max_retries):  # Limit retries
        try:
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    return [
                        {
                            "text": tweet['text'],
                            "label": label_sentiment(tweet['text'])
                        }
                        for tweet in data['data']
                    ], None
                else:
                    return None, "No data found in response."

            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 3))
                print(f"Rate limit hit. Retrying after {retry_after} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_after)  # Wait and retry
            else:
                return None, f"Error {response.status_code}: {response.text}"

        except Exception as e:
            return None, f"Exception occurred: {str(e)}"

    return None, f"Max retries ({max_retries}) exceeded. Failed to fetch tweets."


def fetch_tweets_v1():
    """
    Fetches tweets using Twitter API v1.1 endpoint via Tweepy with backoff.
    """
    client = tweepy.Client(bearer_token=BEARER_TOKEN)

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
                        "label": label_sentiment(tweet.text)
                    }
                    for tweet in response.data
                ], None
            else:
                return None, "No tweets found."
        except tweepy.TooManyRequests:
            wait_time = (2 ** attempt) + random.uniform(0, 2)  # Exponential backoff
            print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        except tweepy.TweepyException as e:
            return None, f"Tweepy error: {str(e)}"
        except Exception as e:
            return None, f"Exception occurred: {str(e)}"

    return None, "Max retries exceeded."

def fetch_tweets():
    """
    Attempts to fetch tweets using v2 first, then v1 if v2 fails.
    """
    print("Fetching tweets using Twitter API v2...")
    tweets, error = fetch_tweets_v2()
    if tweets:
        print("Successfully fetched tweets using v2.")
        print(json.dumps(tweets, indent=4))
        return

    print(f"v2 Error: {error}")

    print("\nFetching tweets using Twitter API v1.1...")
    tweets, error = fetch_tweets_v1()
    if tweets:
        print("Successfully fetched tweets using v1.")
        print(json.dumps(tweets, indent=4))
    else:
        print(f"v1 Error: {error}")


if __name__ == "__main__":
    fetch_tweets()

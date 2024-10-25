import os
import tweepy
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('./config/API.env')

search_terms = ["Bitcoin", "BTC", "Ethereum", "ETH", "cryptocurrency", "crypto"]
query = " OR ".join(search_terms)
max_tweets = 100

def label_sentiment(tweet_text):
    positive_keywords = ['happy', 'bullish', 'up', 'profit', 'win', 'great']
    negative_keywords = ['sad', 'bearish', 'down', 'loss', 'fail', 'bad']

    text_lower = tweet_text.lower()
    if any(keyword in text_lower for keyword in positive_keywords):
        return 'positive'
    elif any(keyword in text_lower for keyword in negative_keywords):
        return 'negative'
    else:
        return 'neutral'

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r

def fetch_v2_tweets():
    # Using v2 endpoint
    tweet_fields = "tweet.fields=text,lang,created_at,public_metrics,author_id"
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&{tweet_fields}&max_results={min(max_tweets, 100)}"

    try:
        response = requests.get(url, auth=bearer_oauth)
        if response.status_code == 200:
            data = response.json()
            labeled_tweets = []
            if 'data' in data:
                for tweet in data['data']:
                    labeled_tweets.append({
                        'text': tweet['text'],
                        'label': label_sentiment(tweet['text'])
                    })
            return labeled_tweets, None
        else:
            return None, f"Error with v2 endpoint: {response.status_code} {response.text}"
    except Exception as e:
        return None, f"Exception with v2 endpoint: {str(e)}"

def fetch_v1_tweets():
    # Initialize Tweepy client using bearer token for v1.1 endpoint
    client = tweepy.Client(bearer_token=os.getenv('TWITTER_BEARER_TOKEN'))
    
    try:
        response = client.search_recent_tweets(query=query, max_results=max_tweets, 
                                               tweet_fields=["text", "lang", "created_at", "public_metrics", "author_id"],
                                               expansions=["author_id"], 
                                               user_fields=["username", "verified", "profile_image_url"])
        
        labeled_tweets = []
        if response.data:
            for tweet in response.data:
                labeled_tweets.append({
                    'text': tweet.text,
                    'label': label_sentiment(tweet.text)
                })
            return labeled_tweets, None
        else:
            return None, "No tweets found on v1 endpoint."
    
    except tweepy.TweepyException as error:
        return None, f"Error with v1 endpoint: {error}"

def fetch_tweets():
    print("Attempting to fetch tweets using v2...")
    tweets, error = fetch_v2_tweets()
    if tweets:
        print("Successfully fetched tweets using v2 endpoint.")
        print("Labeled Tweets:", json.dumps(tweets, indent=4))
        return
    else:
        print(error)

    print("\nAttempting to fetch tweets using v1...")
    tweets, error = fetch_v1_tweets()
    if tweets:
        print("Successfully fetched tweets using v1 endpoint.")
        print("Labeled Tweets:", json.dumps(tweets, indent=4))
    else:
        print(error)

if __name__ == "__main__":
    fetch_tweets()

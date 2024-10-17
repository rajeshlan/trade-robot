import os
import tweepy
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

def fetch_tweets():
    # Initialize Tweepy client using bearer token
    client = tweepy.Client(bearer_token=os.getenv('TWITTER_BEARER_TOKEN'))

    try:
        # Search recent tweets using Twitter API v2
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

            print("Labeled Tweets:", labeled_tweets)

    except tweepy.TweepyException as error:
        print("An error occurred:", error)

fetch_tweets()

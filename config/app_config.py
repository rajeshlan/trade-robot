import os
import logging
from dotenv import load_dotenv

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Specify the full path to the .env file
dotenv_path = r'F:\trading\improvised-code-of-the-pdf-GPT-main\config\API.env'

# Load environment variables from the specified .env file
if load_dotenv(dotenv_path):
    logging.info(f".env file loaded successfully from {dotenv_path}")
else:
    logging.error(f".env file not found or not loaded from {dotenv_path}")

# Fetch API keys and credentials
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
DB_FILE = 'trading_bot.db'
TRAILING_STOP_PERCENT = 0.02
RISK_REWARD_RATIO = 2.0
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = os.getenv('SMTP_PORT')
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
SENTIMENT_API_KEY = os.getenv('SENTIMENT_API_KEY', 'default_api_key')
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', 'default_bybit_api_key')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', 'default_bybit_api_secret')
BYBIT_API_KEY_1 = os.getenv('BYBIT_API_KEY_1', 'default_bybit_api_key_1')
BYBIT_API_SECRET_1 = os.getenv('BYBIT_API_SECRET_1', 'default_bybit_api_secret_1')
BYBIT_API_KEY_2 = os.getenv('BYBIT_API_KEY_2', 'default_bybit_api_key_2')
BYBIT_API_SECRET_2 = os.getenv('BYBIT_API_SECRET_2', 'default_bybit_api_secret_2')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', 'default_twitter_bearer_token')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', 'default_twitter_api_key')
TWITTER_API_SECRET_KEY = os.getenv('TWITTER_API_SECRET_KEY', 'default_twitter_api_secret_key')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', 'default_twitter_access_token')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', 'default_twitter_access_token_secret')

# Log the output of each loaded environment variable
logging.info(f"API_KEY: {'Loaded' if API_KEY else 'Not Loaded'}")
logging.info(f"API_SECRET: {'Loaded' if API_SECRET else 'Not Loaded'}")
logging.info(f"Database file: {DB_FILE}")
logging.info(f"Trailing Stop Percent: {TRAILING_STOP_PERCENT}")
logging.info(f"Risk Reward Ratio: {RISK_REWARD_RATIO}")
logging.info(f"SMTP Server: {'Loaded' if SMTP_SERVER else 'Not Loaded'}")
logging.info(f"SMTP Port: {'Loaded' if SMTP_PORT else 'Not Loaded'}")
logging.info(f"Email User: {'Loaded' if EMAIL_USER else 'Not Loaded'}")
logging.info(f"Email Pass: {'Loaded' if EMAIL_PASS else 'Not Loaded'}")
logging.info(f"Sentiment API Key: {SENTIMENT_API_KEY}")
logging.info(f"Bybit API Key: {'Loaded' if BYBIT_API_KEY else 'Not Loaded'}")
logging.info(f"Bybit API Secret: {'Loaded' if BYBIT_API_SECRET else 'Not Loaded'}")
logging.info(f"Bybit API Key 1: {'Loaded' if BYBIT_API_KEY_1 else 'Not Loaded'}")
logging.info(f"Bybit API Secret 1: {'Loaded' if BYBIT_API_SECRET_1 else 'Not Loaded'}")
logging.info(f"Bybit API Key 2: {'Loaded' if BYBIT_API_KEY_2 else 'Not Loaded'}")
logging.info(f"Bybit API Secret 2: {'Loaded' if BYBIT_API_SECRET_2 else 'Not Loaded'}")
logging.info(f"Twitter Bearer Token: {'Loaded' if TWITTER_BEARER_TOKEN else 'Not Loaded'}")
logging.info(f"Twitter API Key: {'Loaded' if TWITTER_API_KEY else 'Not Loaded'}")
logging.info(f"Twitter API Secret Key: {'Loaded' if TWITTER_API_SECRET_KEY else 'Not Loaded'}")
logging.info(f"Twitter Access Token: {'Loaded' if TWITTER_ACCESS_TOKEN else 'Not Loaded'}")
logging.info(f"Twitter Access Token Secret: {'Loaded' if TWITTER_ACCESS_TOKEN_SECRET else 'Not Loaded'}")

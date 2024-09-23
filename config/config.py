import os
from dotenv import load_dotenv

load_dotenv()

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

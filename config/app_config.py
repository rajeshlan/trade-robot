#python config\app_config.py

import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define and Load .env file manually
dotenv_path = os.path.join(os.path.dirname(__file__), "API.env")

if not os.path.exists(dotenv_path):
    logging.error(f".env file not found at {dotenv_path}")
    exit(1)

# Explicitly load .env file and override existing variables
if not load_dotenv(dotenv_path, override=True):
    logging.error(f"Failed to load .env file from {dotenv_path}")
    exit(1)

# Debugging: Print all loaded environment variables
logging.info(f"Loaded .env file from {dotenv_path}")

def mask_sensitive(value):
    """Mask sensitive values, showing only first and last characters."""
    if value and len(value) > 4:
        return value[0] + "*" * (len(value) - 2) + value[-1]
    return value

def safe_cast(value, to_type, default=None, key=None):
    """Safely cast value with validation."""
    try:
        result = to_type(value)
        if key == 'TRAILING_STOP_PERCENT' and not (0 <= result <= 1):
            raise ValueError("Value must be between 0 and 1")
        return result
    except (ValueError, TypeError) as e:
        logging.warning(f"Invalid {key}: {value}. Error: {e}. Using default: {default}")
        return default

def get_env_var(key, default=None, required=False):
    """Fetch environment variable, enforcing required fields."""
    value = os.getenv(key, default)
    if required and not value:
        logging.error(f"Critical error: {key} is not set.")
        exit(1)
    return value

# API Configuration
API_KEY = get_env_var('API_KEY', required=True)
API_SECRET = get_env_var('API_SECRET', required=True)

# Trading Parameters
TRAILING_STOP_PERCENT = safe_cast(get_env_var('TRAILING_STOP_PERCENT', 0.02), float, 0.02, 'TRAILING_STOP_PERCENT')
RISK_REWARD_RATIO = safe_cast(get_env_var('RISK_REWARD_RATIO', 2.0), float, 2.0)

# Email Configuration
SMTP_SERVER = get_env_var('SMTP_SERVER', required=True)
SMTP_PORT = safe_cast(get_env_var('SMTP_PORT', required=True), int)
EMAIL_USER = get_env_var('EMAIL_USER', required=True)
EMAIL_PASS = get_env_var('EMAIL_PASS', required=True)

# Bybit API Keys
BYBIT_API_KEYS = {
    f"key_set_{i}": {
        "key": get_env_var(f"BYBIT_API_KEY_{i}", required=True),
        "secret": get_env_var(f"BYBIT_API_SECRET_{i}", required=True)
    } for i in range(1, 3)
}

# Twitter Credentials
TWITTER_CREDENTIALS = {
    "bearer_token": get_env_var('TWITTER_BEARER_TOKEN', required=True),
    "api_key": get_env_var('TWITTER_API_KEY', required=True),
    "api_secret_key": get_env_var('TWITTER_API_SECRET_KEY', required=True),
    "access_token": get_env_var('TWITTER_ACCESS_TOKEN', required=True),
    "access_token_secret": get_env_var('TWITTER_ACCESS_TOKEN_SECRET', required=True)
}

# Log Configuration Summary
logging.info("Configuration Summary:")
logging.info(f"API_KEY: {mask_sensitive(API_KEY)}")
logging.info(f"SMTP_SERVER: {SMTP_SERVER}")
logging.info(f"Bybit API Keys: {BYBIT_API_KEYS}")
logging.info(f"Twitter Credentials: {mask_sensitive(str(TWITTER_CREDENTIALS))}") 

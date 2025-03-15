# python exchanges\APIs.py

import ccxt
import logging
import time
import os
from dotenv import load_dotenv
import ntplib
from datetime import datetime, timezone

# Setup logging
log_file_path = r'D:\\RAJESH FOLDER\\PROJECTS\\trade-robot\\logs\\APIs.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),  # Log to a file
        logging.StreamHandler()             # Log to the console
    ]
)

# Specify the full path to the .env file
dotenv_path = r'D:\\RAJESH FOLDER\\PROJECTS\\trade-robot\\config\\API.env'

def synchronize_system_time(retries=3):
    """Synchronize system time with an NTP server, with retries and alternate servers."""
    ntp_servers = ['pool.ntp.org', 'time.google.com', 'time.windows.com']
    logging.info("Starting system time synchronization...")

    for attempt in range(retries):
        logging.info(f"Attempt {attempt + 1} of {retries}.")
        for server in ntp_servers:
            try:
                response = ntplib.NTPClient().request(server, timeout=5)
                current_time = datetime.fromtimestamp(response.tx_time, tz=timezone.utc)
                local_time = datetime.now(timezone.utc)
                offset = (response.tx_time - local_time.timestamp()) * 1000  # Convert to milliseconds
                logging.info(f"System time synchronized: {current_time} using server {server}")
                return offset
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for server {server}: {e}")

    logging.error("All attempts to synchronize time failed. Using zero offset as fallback.")
    return 0  # Return zero offset if synchronization fails

def load_api_credentials():
    """Load API credentials from the specified .env file."""
    if load_dotenv(dotenv_path):
        logging.info(f".env file loaded successfully from {dotenv_path}")
    else:
        logging.error(f".env file not found or not loaded from {dotenv_path}")

    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    if not api_key or not api_secret:
        logging.error("Error: API credentials not found in .env file.")
        return None, None
    else:
        logging.info("API credentials loaded successfully.")
        return api_key, api_secret

def create_exchange_instance(api_key, api_secret):
    """Create an instance of the Bybit exchange for perpetual trading."""
    logging.info("Creating exchange instance...")
    
    # Synchronize system time
    offset = synchronize_system_time()

    exchange = ccxt.bybit({
        'apiKey': api_key,
        'secret': api_secret,
        'options': {
            'defaultType': 'future',
            'unifiedMargin': True,
            'adjustForTimeDifference': True,
            'recvWindow': 120000,  # Explicitly set recvWindow to 120 seconds
        },
    })

    # Enable verbose logging for debugging API requests and responses
    exchange.verbose = True

    try:
        # Fetch server time and local time difference
        server_time = exchange.fetch_time()
        local_time = int(exchange.milliseconds()) + offset
        time_diff = server_time - local_time

        logging.info(f"Server time: {server_time}, Local time: {local_time}, Difference: {time_diff} ms")

        # Ensure `req_timestamp` matches `server_time`
        exchange.options['timestamp'] = server_time

        logging.info("Exchange instance created.")
        return exchange, server_time
    except Exception as e:
        logging.error(f"Failed to create exchange instance: {e}")
        raise

def fetch_and_log_markets_with_retry(exchange, retries=3):
    """Fetch markets with retries in case of InvalidNonce error."""
    for attempt in range(retries):
        try:
            logging.info(f"Fetching markets (attempt {attempt + 1})...")
            markets = exchange.load_markets()
            logging.info("Markets fetched successfully.")
            return markets
        except ccxt.base.errors.InvalidNonce as e:
            logging.error(f"InvalidNonce error: {e}. Retrying with adjusted recv_window...")
            exchange.options['recvWindow'] += 5000
            logging.info(f"Adjusted recvWindow to: {exchange.options['recvWindow']} ms")
            time.sleep(2)
        except Exception as e:
            logging.error(f"Unexpected error during market fetch: {e}")
            time.sleep(2)
    logging.error("Failed to fetch markets after retries.")
    return None

def fetch_and_log_markets(exchange):
    """Fetch markets and log available symbols."""
    logging.info(f"Fetching markets...")
    try:
        markets = exchange.load_markets()
        logging.info("Markets fetched successfully.")
        symbols = [symbol for symbol, data in markets.items() if data.get('linear') and '/USDT' in symbol]
        logging.info(f"Available symbols for linear derivatives: {symbols}")
        return symbols
    except Exception as e:
        logging.error(f"An error occurred while fetching markets: {e}")
        return []

def set_leverage(exchange, symbol, leverage):
    """Set the leverage for a given trading symbol."""
    try:
        logging.info(f"Setting leverage for {symbol} to {leverage}...")
        exchange.set_leverage(leverage, symbol)
        logging.info(f"Leverage for {symbol} set to {leverage}.")
    except Exception as e:
        logging.error(f"An error occurred while setting leverage: {e}")

# Example usage
if __name__ == "__main__":
    api_key, api_secret = load_api_credentials()

    if api_key and api_secret:
        try:
            exchange, adjusted_server_time = create_exchange_instance(api_key, api_secret)
            available_symbols = fetch_and_log_markets_with_retry(exchange)
            if available_symbols:
                symbol = 'BTC/USDT:USDT'
                leverage = 100
                if symbol in available_symbols:
                    set_leverage(exchange, symbol, leverage)
                else:
                    logging.error(f"Symbol '{symbol}' is not available for trading. Check available symbols.")
            else:
                logging.error("Failed to fetch symbols.")
        except Exception as e:
            logging.error(f"Unexpected error in main execution: {e}")

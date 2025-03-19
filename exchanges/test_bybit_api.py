# python exchanges\test_bybit_api.py

import ccxt
import logging
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv  # To load environment variables from .env file

# Load environment variables from the specific .env file path
load_dotenv(dotenv_path=r'D:\\RAJESH FOLDER\\PROJECTS\\trade-robot\\config\\API.env')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure that the API keys are loaded from environment variables
api_key = os.getenv('BYBIT_API_KEY')
api_secret = os.getenv('BYBIT_API_SECRET')
api_key_1 = os.getenv('BYBIT_API_KEY_1')
api_secret_1 = os.getenv('BYBIT_API_SECRET_1')
api_key_2 = os.getenv('BYBIT_API_KEY_2')
api_secret_2 = os.getenv('BYBIT_API_SECRET_2')

# Verify that API keys are available
if not api_key or not api_secret:
    logging.error("API key and secret must be set as environment variables or provided in the script.")
    exit(1)

logging.info("API keys loaded successfully.")

def fetch_historical_data(exchange, symbol, timeframe='1d', limit=100):
    """
    Fetch historical OHLCV data for a given symbol and timeframe.
    """
    logging.info(f"Fetching historical data for {symbol} with timeframe {timeframe}")
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def predict_price(df):
    """
    Predict future prices based on historical data using a simple linear regression model.
    """
    df['time'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    X = df[['time']]
    y = df['close']

    model = LinearRegression()
    model.fit(X, y)

    # Predict the next price point
    next_time = pd.DataFrame([[df['time'].max() + 3600]], columns=['time'])  # Ensure valid feature name
    predicted_price = model.predict(next_time)
    logging.info(f"Predicted price for next hour: {predicted_price[0]}")
    return predicted_price[0]

def test_api_credentials(api_key, api_secret):
    """
    Test API credentials by fetching account balance, open orders, and placing a test order.
    """
    try:
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

        # Fetch server time
        server_time = exchange.fetch_time()
        logging.info("Server time fetched: %s", server_time)

        # Fetch account balance
        balance = exchange.fetch_balance()
        logging.info("Successfully fetched balance: %s", balance)

        # Fetch historical data and make a price prediction
        historical_data = fetch_historical_data(exchange, 'BTC/USDT', limit=50)
        predicted_price = predict_price(historical_data)

        # Check if the predicted price is realistic
        if predicted_price > 10000:  # Example threshold to prevent placing unreasonable orders
            logging.warning("Predicted price is too high for this test, skipping order.")
            return

        # Example: Place a test order with adjusted parameters
        order_size = 0.0001  # Example order size
        if balance['total']['USDT'] >= predicted_price * order_size:
            test_order = exchange.create_limit_buy_order('BTC/USDT', order_size, predicted_price)
            logging.info("Test order placed successfully at predicted price: %s", test_order)
        else:
            logging.warning("Insufficient balance for placing the order.")

        return balance

    except ccxt.AuthenticationError as e:
        logging.error("Authentication error: %s", e)
        raise e
    except ccxt.NetworkError as e:
        logging.error("Network error: %s", e)
        raise e
    except ccxt.ExchangeError as e:
        logging.error("Exchange error: %s", e)
        raise e
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        raise e

if __name__ == "__main__":
    # Check if API keys are available
    if not api_key or not api_secret:
        logging.error("API key and secret must be set as environment variables or provided in the script.")
        exit(1)

    # Loop through each API key and secret combination
    api_keys = [api_key, api_key_1, api_key_2]
    api_secrets = [api_secret, api_secret_1, api_secret_2]

    for api_key, api_secret in zip(api_keys, api_secrets):
        if not api_key or not api_secret:
            logging.error("API key and secret must be set as environment variables or provided in the script.")
        else:
            try:
                logging.info(f"Testing API credentials for API key: {api_key[:5]}****")
                test_api_credentials(api_key, api_secret)
            except Exception as e:
                logging.error(f"Failed to test API credentials for {api_key[:5]}****: %s", e)

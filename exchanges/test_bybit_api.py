import ccxt
import logging
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    next_time = [[df['time'].max() + 3600]]  # Predict 1 hour ahead
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

        # Fetch open orders
        orders = exchange.fetch_open_orders()
        logging.info("Successfully fetched open orders: %s", orders)

        # Fetch historical data and make a price prediction
        historical_data = fetch_historical_data(exchange, 'BTC/USDT', limit=50)
        predicted_price = predict_price(historical_data)

        # Example: Place a test order with adjusted parameters
        test_order = exchange.create_limit_buy_order('BTC/USDT', 0.0001, predicted_price)
        logging.info("Test order placed successfully at predicted price: %s", test_order)

        return balance, orders

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
    api_key = os.getenv('BYBIT_API_KEY', 'YOUR_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET', 'YOUR_API_SECRET')

    if not api_key or not api_secret:
        logging.error("API key and secret must be set as environment variables or provided in the script.")
    else:
        try:
            test_api_credentials(api_key, api_secret)
        except Exception as e:
            logging.error("Failed to test API credentials: %s", e)

import ccxt
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        # Fetch account balance
        balance = exchange.fetch_balance()
        logging.info("Successfully fetched balance: %s", balance)

        # Fetch open orders
        orders = exchange.fetch_open_orders()
        logging.info("Successfully fetched open orders: %s", orders)

        # Example: Place a test order with adjusted parameters
        test_order = exchange.create_limit_buy_order('BTCUSDT', 0.0001, 10000)
        logging.info("Test order placed successfully: %s", test_order)

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

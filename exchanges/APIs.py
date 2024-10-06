import ccxt
from dotenv import load_dotenv
import os

def load_api_credentials():
    """Load API credentials from .env file."""
    load_dotenv()
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    if api_key is None or api_secret is None:
        print("Error: API credentials not found in .env file.")
    else:
        print("API credentials loaded successfully.")
    
    return api_key, api_secret

def create_exchange_instance(api_key, api_secret):
    """Create an instance of the Bybit exchange."""
    print("Creating exchange instance...")
    exchange = ccxt.bybit({
        'apiKey': api_key,
        'secret': api_secret,
    })
    print("Exchange instance created.")
    return exchange

def set_leverage(exchange, symbol, leverage):
    """Set the leverage for a given trading symbol."""
    print(f"Loading markets for exchange...")
    markets = exchange.load_markets()
    print("Markets loaded successfully.")

    if symbol in markets:
        market = markets[symbol]
        print(f"Setting leverage for {symbol} to {leverage}...")
        exchange.fapiPrivate_post_leverage({
            'symbol': market['id'],
            'leverage': leverage,
        })
        print(f"Leverage for {symbol} set to {leverage}.")
    else:
        print(f"Error: Symbol '{symbol}' not found in markets.")

# Example usage
if __name__ == "__main__":
    # Load API credentials
    api_key, api_secret = load_api_credentials()

    # Create exchange instance
    exchange = create_exchange_instance(api_key, api_secret)

    # Set leverage (Example symbol and leverage)
    symbol = 'BTC/USDT'  # Replace with your desired trading pair
    leverage = 10  # Replace with your desired leverage
    set_leverage(exchange, symbol, leverage)

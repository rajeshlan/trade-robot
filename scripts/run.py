## python scripts\run.py (need checking as it shows TradingBot.__init__() missing 1 required positional argument: 'api_secret')

import os
import sys
import time
from dotenv import load_dotenv

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import using absolute imports
from exchanges.exchanges import initialize_exchange  # Ensure you use the correct module path
from trading.tradingbot import TradingBot

# Load environment variables from .env file
load_dotenv(dotenv_path='F:/trading/improvised-code-of-the-pdf-GPT-main/API.env')

def main():
    # Retrieve API credentials from environment variables
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("API key and secret must be set in the environment variables.")

    # Initialize the exchange
    exchange = initialize_exchange(api_key, api_secret)
    print("Exchange initialized:", exchange)

    # Run the bot continuously
    while True:
        try:
            # Initialize and run the trading bot with both exchange and API secret
            bot = TradingBot(api_key, api_secret)  # ✅ Corrected
            bot.run()  # Assuming bot.run() starts the trading logic
        except Exception as e:
            print(f"Error occurred: {e}")
            with open('error_log.txt', 'a') as f:
                f.write(f"Error: {e}\n")
            time.sleep(10)


if __name__ == "__main__":
    main()

import os
import time
from dotenv import load_dotenv
from exchanges import initialize_exchange
from trading.tradingbot import TradingBot

# Load environment variables from .env file
load_dotenv(dotenv_path='C:/Users/amrita/Desktop/improvised-code-of-the-pdf-GPT-main/API.env')

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
            # Initialize and run the trading bot
            bot = TradingBot(exchange)  # Passing exchange if needed in bot initialization
            bot.run()  # Assuming bot.run() starts the trading logic
        except Exception as e:
            print(f"Error occurred: {e}")
            # Optionally log the error to a file or monitoring system
            with open('error_log.txt', 'a') as f:
                f.write(f"Error: {e}\n")
            # Add a retry delay
            time.sleep(10)

if __name__ == "__main__":
    main()

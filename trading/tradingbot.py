# python -m trading.tradingbot

from datetime import datetime
import time
import logging
import pandas as pd
import pandas_ta as ta
import ntplib
import ccxt
import os
from exchanges.APIs import load_api_credentials
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in os.sys.path:
    os.sys.path.append(project_root)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TradingBot:
    def __init__(self, api_key, api_secret, ntp_server="time.google.com", max_retries=3, backoff_factor=1):
        self.api_key = api_key
        self.api_secret = api_secret
        self.ntp_server = ntp_server
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.exchange = None
        self.time_offset = 0  # Store time offset
        self.initialize_exchange()  # ✅ Automatically initialize the exchange

    def run(self, symbol="BTCUSDT", timeframe="1h", trade_amount=0.001, interval=60):
        """
        Runs the trading bot in a continuous loop, fetching data, analyzing it, and executing trades.

        Parameters:
        - symbol (str): Trading pair (default: 'BTCUSDT')
        - timeframe (str): Timeframe for OHLCV data (default: '1h')
        - trade_amount (float): Amount to trade per order (default: 0.001 BTC)
        - interval (int): Time in seconds between data fetches (default: 60 seconds)
        """
        logging.info("Starting trading bot...")
        self.synchronize_time()
        self.initialize_exchange()

        while True:
            try:
                logging.info(f"Fetching latest market data for {symbol}...")
                df = self.fetch_data(symbol=symbol, timeframe=timeframe, limit=200)
            
                if df.empty:
                    logging.warning("Received empty dataframe. Skipping this cycle.")
                    time.sleep(interval)
                    continue

                # Calculate technical indicators
                df = self.calculate_indicators(df)
            
                # Detect buy/sell signals
                df = self.detect_signals(df)
                latest_signal = df.iloc[-1]  # Get the most recent signal

                logging.info(f"Latest signals - Buy: {latest_signal['Buy_Signal']}, Sell: {latest_signal['Sell_Signal']}")
            
                # Decision logic based on signals
                if latest_signal["Buy_Signal"]:
                    logging.info(f"BUY signal detected for {symbol} at price {latest_signal['close']}")
                    self.execute_trading_decision("buy", symbol, trade_amount)
                elif latest_signal["Sell_Signal"]:
                    logging.info(f"SELL signal detected for {symbol} at price {latest_signal['close']}")
                    self.execute_trading_decision("sell", symbol, trade_amount)
                else:
                    logging.info(f"No trade action taken for {symbol}. Holding position.")

                # Wait before the next iteration
                logging.info(f"Sleeping for {interval} seconds...")
                time.sleep(interval)

            except ccxt.NetworkError as e:
                logging.error(f"Network error occurred: {e}. Retrying in {interval} seconds...")
                time.sleep(interval)
            except ccxt.ExchangeError as e:
                logging.error(f"Exchange error occurred: {e}. Retrying in {interval} seconds...")
                time.sleep(interval)
            except KeyboardInterrupt:
                logging.info("Trading bot stopped by user.")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {e}. Retrying in {interval} seconds...")
                time.sleep(interval)



    def synchronize_time(self):
        try:
            client = ntplib.NTPClient()
            for retries in range(self.max_retries):
                try:
                    response = client.request(self.ntp_server)
                    self.time_offset = response.offset
                    logging.info(f"Time synchronized with {self.ntp_server}. Offset: {self.time_offset} seconds")
                    return self.time_offset
                except ntplib.NTPException as e:
                    logging.warning(f"Time sync failed (attempt {retries + 1}): {e}")
                    time.sleep(self.backoff_factor * (retries + 1))
            logging.error(f"Failed to sync time after {self.max_retries} retries.")
            self.time_offset = 0
            return 0
        except Exception as e:
            logging.error(f"Unexpected error during time sync: {e}")
            self.time_offset = 0
            return 0

    def initialize_exchange(self):
        try:
            self.exchange = ccxt.bybit({
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {
                    "recv_window": 10000,  # Set recv_window to 10 seconds
                },
            })
            logging.info("Exchange initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize exchange: {e}")
            raise e

    def fetch_data(self, symbol, timeframe="1h", limit=100):
        try:
            self.exchange.nonce = lambda: int((time.time() + self.time_offset) * 1000)
            logging.debug(f"Request timestamp: {self.exchange.nonce()}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            logging.info(f"Fetched OHLCV data for {symbol}")
            return df
        except ccxt.BaseError as e:
            logging.error(f"Error fetching OHLCV data: {e}")
            raise e

    def calculate_indicators(self, df):
        try:
            df["SMA_20"] = df["close"].rolling(window=20).mean()
            df["SMA_50"] = df["close"].rolling(window=50).mean()
            df["SMA_200"] = df["close"].rolling(window=200).mean()

            df["EMA_9"] = ta.ema(df["close"], length=9)
            df["RSI"] = ta.rsi(df["close"], length=14)
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            df["MACD"], df["MACD_signal"] = macd["MACD_12_26_9"], macd["MACDs_12_26_9"]
            bbands = ta.bbands(df["close"], length=20, std=2)
            df["BB_upper"], df["BB_middle"], df["BB_lower"] = bbands["BBU_20_2.0"], bbands["BBM_20_2.0"], bbands["BBL_20_2.0"]
            df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)

            logging.info("Indicators calculated successfully.")
            return df
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            raise e

    def detect_signals(self, df):
        try:
            df["Buy_Signal"] = (df["close"] > df["SMA_50"]) & (df["SMA_50"] > df["SMA_200"]) & (df["MACD"] > df["MACD_signal"]) & (df["RSI"] < 70)
            df["Sell_Signal"] = (df["close"] < df["SMA_50"]) & (df["SMA_50"] < df["SMA_200"]) & (df["MACD"] < df["MACD_signal"]) & (df["RSI"] > 30)

            # ✅ ADD THIS DEBUG LOGGING
            latest_signal = df.iloc[-1]  # Get latest row
            logging.info(f"Latest Signals - Buy: {latest_signal['Buy_Signal']}, Sell: {latest_signal['Sell_Signal']}")
        
            return df
        except Exception as e:
            logging.error(f"Error generating signals: {e}")
            raise e


    def place_order(self, action, price, symbol, amount):
        try:
            order_type = "buy" if action.lower() == "buy" else "sell"
            logging.info(f"Placing {order_type} order for {symbol} at {price}...")
            if action.lower() == "buy":
                order = self.exchange.create_market_buy_order(symbol, amount)
            elif action.lower() == "sell":
                order = self.exchange.create_market_sell_order(symbol, amount)
            logging.info(f"Order placed successfully: {order}")
            return order
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            raise e
        
    def execute_trading_decision(self, decision, symbol="BTCUSDT", amount=0.001):
        """
        Executes a trade based on the provided decision.
    
        Parameters:
        - decision (str): The trading decision ('buy', 'sell', or 'hold')
        - symbol (str): The trading pair (default is 'BTCUSDT')
        - amount (float): The amount to trade (default is 0.001 BTC)
        """
        try:
            if decision == "buy":
                logging.info(f"Executing Buy Order for {symbol}...")
                self.exchange.create_market_buy_order(symbol, amount)
            elif decision == "sell":
                logging.info(f"Executing Sell Order for {symbol}...")
                self.exchange.create_market_sell_order(symbol, amount)
            else:
                logging.info(f"Holding position. No trade executed for {symbol}.")
        except Exception as e:
            logging.error(f"Error executing trade decision ({decision}): {e}")


    @staticmethod
    def main():
        try:
            api_key, api_secret = load_api_credentials()
            bot = TradingBot(api_key, api_secret)
            bot.synchronize_time()
            bot.initialize_exchange()
            bot.execute_trading_decision("buy", "BTCUSDT", 0.001)

            data = bot.fetch_data(symbol="BTCUSDT", timeframe="1h", limit=100)
            data = bot.calculate_indicators(data)
            signals = bot.detect_signals(data)
            logging.info(signals.tail())
        except Exception as e:
            logging.error(f"Error running TradingBot: {e}")


if __name__ == "__main__":
    TradingBot.main()

# Add this at the bottom of tradingbot.py
def execute_trading_decision_global(decision, symbol="BTCUSDT", amount=0.001):
    bot = TradingBot(api_key=os.getenv('BYBIT_API_KEY'), api_secret=os.getenv('BYBIT_API_SECRET'))
    bot.initialize_exchange()
    bot.execute_trading_decision(decision, symbol, amount)

# Allow external imports
__all__ = ["TradingBot", "execute_trading_decision_global"]


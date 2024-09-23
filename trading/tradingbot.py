from ctypes import c_short
from datetime import datetime
import time
import logging
import pandas as pd
import pandas_ta as ta
import ntplib
import ccxt
from synchronize_exchange_time import synchronize_system_time
from APIs import create_exchange_instance, load_api_credentials
from fetch_data import fetch_ohlcv
from risk_management import calculate_stop_loss, calculate_take_profit, calculate_position_size
from database import create_db_connection, store_data_to_db, fetch_historical_data
from sentiment_analysis import fetch_real_time_sentiment
#from technical_indicators import calculate_sma, calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_atr
from config import API_KEY, API_SECRET, DB_FILE, TRAILING_STOP_PERCENT, RISK_REWARD_RATIO

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingBot:
    def __init__(self, api_key, api_secret, ntp_server='time.google.com', max_retries=3, backoff_factor=1):
        self.api_key = api_key
        self.api_secret = api_secret
        self.ntp_server = ntp_server
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.exchange = None

    # Time Synchronization
    def synchronize_time(self):
        try:
            client = ntplib.NTPClient()
            retries = 0
            while retries < self.max_retries:
                try:
                    response = client.request(self.ntp_server)
                    offset = response.offset
                    logging.info(f"Time synchronized with {self.ntp_server}. Offset: {offset} seconds")
                    return offset
                except ntplib.NTPException as e:
                    logging.warning(f"Failed to synchronize time on attempt {retries + 1} with {self.ntp_server}: {e}")
                    retries += 1
                    time.sleep(self.backoff_factor * retries)  # Exponential backoff
            logging.error(f"Max retries ({self.max_retries}) reached. Unable to synchronize time with {self.ntp_server}.")
            return 0  # Return 0 offset if synchronization fails
        except Exception as e:
            logging.error(f"An unexpected error occurred during time synchronization: {e}")
            return 0

    # Initialize Exchange
    def initialize_exchange(self):
        try:
            self.exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
            })
            logging.info("Initialized Bybit exchange")
        except Exception as e:
            logging.error("Failed to initialize exchange: %s", e)
            raise e

    # Fetch Data
    def fetch_data(self, symbol, timeframe='1h', limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.exchange, symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logging.info(f"Fetched OHLCV data for {symbol}")
            return df
        except ccxt.BaseError as e:
            logging.error("Error fetching OHLCV data: %s", e)
            raise e
        
    # Calculate Technical Indicators
    def calculate_indicators(self, df, sma_short=20, sma_long=50, sma_longest=200, ema_short=9, ema_mid=12, ema_long=26, rsi_length=14, macd_fast=12, macd_slow=26, macd_signal=9):
        try:
            # Calculate SMAs
            df['SMA_20'] = self.calculate_sma(df['close'], sma_short)
            df['SMA_50'] = self.calculate_sma(df['close'], sma_long)
            df['SMA_200'] = self.calculate_sma(df['close'], sma_longest)
        
            # Calculate EMAs
            df['EMA_9'] = ta.ema(df['close'], length=ema_short)
            df['EMA_12'] = ta.ema(df['close'], length=ema_mid)
            df['EMA_26'] = ta.ema(df['close'], length=ema_long)
        
            # Calculate RSI
            df['RSI'] = self.calculate_rsi(df['close'], length=rsi_length)
        
            # Calculate MACD
            df['MACD'], df['MACD_signal'] = self.calculate_macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        
            # Calculate Bollinger Bands
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(df['close'], length=20, std=2)
        
            # Calculate ATR (Average True Range)
            df['ATR'] = self.calculate_atr(df['high'], df['low'], df['close'], length=14)

            logging.info("Calculated SMA, EMA, RSI, MACD, Bollinger Bands, and ATR indicators")
            return df
        except Exception as e:
            logging.error("Error during technical analysis: %s", e)
            raise e

    def calculate_sma(self, series, window):
        return series.rolling(window=window).mean()
  
    def calculate_rsi(self, series, length=14):
        return ta.rsi(series, length=length)
  
    def calculate_macd(self, series, fast=12, slow=26, signal=9):
        macd = ta.macd(series, fast=fast, slow=slow, signal=signal)
        return macd['MACD_12_26_9'], macd['MACDs_12_26_9']

    def calculate_bollinger_bands(self, series, length=20, std=2):
        bbands = ta.bbands(series, length=length, std=std)
        return bbands['BBU_20_2.0'], bbands['BBM_20_2.0'], bbands['BBL_20_2.0']

    def calculate_atr(self, high, low, close, length=14):
        return ta.atr(high, low, close, length=length)

    # Detect Trading Signals
    def detect_signals(self, df):
        try:
            df['Buy_Signal'] = (df['close'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200']) & (df['MACD'] > df['MACD_signal']) & (df['RSI'] < 70)
            df['Sell_Signal'] = (df['close'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_200']) & (df['MACD'] < df['MACD_signal']) & (df['RSI'] > 30)
            logging.info("Generated buy and sell signals")
            return df
        except Exception as e:
            logging.error(f"An error occurred while generating signals: {e}")
            raise e

    # Trading Strategy
    def trading_strategy(self, df):
        try:
            signals = ['hold']
            for i in range(1, len(df)):
                if pd.notna(df['SMA_50'][i]) and pd.notna(df['SMA_200'][i]) and pd.notna(df['SMA_50'][i-1]) and pd.notna(df['SMA_200'][i-1]):
                    if df['SMA_50'][i] > df['SMA_200'][i] and df['SMA_50'][i-1] <= df['SMA_200'][i-1]:
                        signals.append('buy')
                    elif df['SMA_50'][i] < df['SMA_200'][i] and df['SMA_50'][i-1] >= df['SMA_200'][i-1]:
                        signals.append('sell')
                    else:
                        signals.append('hold')
                else:
                    signals.append('hold')  # Handle cases where SMA values are None

            df['signal'] = signals
            logging.info("Generated trading signals")
            return df
        except Exception as e:
            logging.error("An error occurred during trading strategy execution: %s", e)
            raise e

    # Simulate Trading
    def simulate_trading(self, df, amount=0.001):
        try:
            for i in range(len(df)):
                if df['Buy_Signal'].iloc[i]:
                    self.execute_trade('buy', df.iloc[i], amount)
                elif df['Sell_Signal'].iloc[i]:
                    self.execute_trade('sell', df.iloc[i], amount)
            logging.info("Simulated trading completed")
        except Exception as e:
            logging.error(f"An error occurred during simulated trading: {e}")
            raise e

    # Execute Trade
    def execute_trade(self, signal, data_row, amount, balance=10, risk_percentage=1, risk_reward_ratio=2):
        try:
            entry_price = data_row['close']
            stop_loss = calculate_stop_loss(entry_price, risk_percentage, 10)  # Assuming leverage is always 10
            take_profit = calculate_take_profit(entry_price, risk_reward_ratio, stop_loss)
            position_size = calculate_position_size(balance, risk_percentage, entry_price, stop_loss)
            
            if signal == 'buy':
                self.exchange.create_market_buy_order('BTCUSDT', position_size)
                logging.info(f"Buy order executed at {entry_price}, stop loss at {stop_loss}, take profit at {take_profit}")
            elif signal == 'sell':
                self.exchange.create_market_sell_order('BTCUSDT', position_size)
                logging.info(f"Sell order executed at {entry_price}, stop loss at {stop_loss}, take profit at {take_profit}")
            else:
                logging.info("No trade executed")
        except Exception as e:
            logging.error(f"Error executing trade: {e}")
            raise e

    # Place Order
    def place_order(self, action, price, symbol, amount):
        try:
            if action == 'buy':
                logging.info("Executing Buy Order")
                order = self.exchange.create_market_buy_order(symbol, amount)
            elif action == 'sell':
                logging.info("Executing Sell Order")
                order = self.exchange.create_market_sell_order(symbol, amount)
            else:
                logging.info("No trade action needed (hold signal)")
                return
            
            if 'error' in order:
                logging.error("Failed to execute order: %s", order['error'])
            else:
                logging.info("Order executed successfully: %s", order)
                return order
        except ccxt.BaseError as e:
            logging.error("An error occurred during trade execution: %s", e)
            raise e

    @staticmethod
    def main():
        try:
            # Load API credentials
            api_key, api_secret = load_api_credentials()

            bot = TradingBot(api_key, api_secret)

            # Synchronize system time
            time_offset = bot.synchronize_time()
            logging.info("System time synchronized with offset: %d ms", time_offset)

            # Initialize exchange and fetch historical data
            bot.initialize_exchange()
            historical_data = bot.fetch_data(symbol='BTCUSDT', timeframe='1d', limit=365)

            # Calculate indicators, generate signals, and simulate trading
            df_with_indicators = bot.calculate_indicators(historical_data)
            signals_df = bot.detect_signals(df_with_indicators)
            bot.simulate_trading(signals_df)

            logging.info("Trading bot executed successfully.")
        except ccxt.AuthenticationError as e:
            logging.error("Authentication error: %s. Please check your API key and secret.", e)
        except ccxt.NetworkError as e:
            logging.error("Network error: %s. Please check your internet connection.", e)
        except ccxt.ExchangeError as e:
            logging.error("Exchange error: %s. Please check the exchange status or API documentation.", e)
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)

if __name__ == "__main__":
    TradingBot.main()

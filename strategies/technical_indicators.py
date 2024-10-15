import time
import logging
import ccxt
import pandas as pd
import pandas_ta as ta
from synchronize_exchange_time import synchronize_system_time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def prepare_data(self, df):
        # Add your features and labels here
        df['returns'] = df['close'].pct_change()
        df['label'] = df['returns'].shift(-1).apply(lambda x: 1 if x > 0 else 0)
        df.dropna(inplace=True)

        features = df[['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']]
        labels = df['label']

        return features, labels

    def train_model(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions))

    def predict(self, features):
        return self.model.predict(features)

class TradingBot:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = None
        self.ml_model = MLModel()


    def initialize_exchange(self):
        """Initialize the trading exchange."""
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

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100, time_offset=0):
        """Fetch OHLCV data from the exchange."""
        params = {
            'recvWindow': 10000,
            'timestamp': int(time.time() * 1000 + time_offset)
        }
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logging.info(f"Fetched OHLCV data for {symbol}")
            return df
        except ccxt.BaseError as e:
            logging.error("Error fetching OHLCV data: %s", e)
            raise e

    @staticmethod
    def calculate_sma(series, window):
        """Calculate Simple Moving Average (SMA)."""
        return series.rolling(window=window).mean()

    @staticmethod
    def calculate_rsi(series, period=14):
        """Calculate Relative Strength Index (RSI)."""
        return ta.rsi(series, length=period)

    @staticmethod
    def calculate_macd(series, fast=12, slow=26, signal=9):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        macd = ta.macd(series, fast=fast, slow=slow, signal=signal)
        return macd['MACD_12_26_9'], macd['MACDs_12_26_9']

    @staticmethod
    def calculate_bollinger_bands(series, length=20, std=2):
        """Calculate Bollinger Bands."""
        bbands = ta.bbands(series, length=length, std=std)
        return bbands['BBU_20_2.0'], bbands['BBM_20_2.0'], bbands['BBL_20_2.0']

    @staticmethod
    def calculate_atr(high, low, close, length=14):
        """Calculate Average True Range (ATR)."""
        return ta.atr(high, low, close, length=length)

    def calculate_indicators(self, df, sma_short=20, sma_long=50, sma_longest=200,
                              ema_short=9, ema_mid=12, ema_long=26,
                              rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9):
        """Calculate technical indicators."""
        try:
            # Calculate SMAs
            df['SMA_20'] = ta.sma(df['close'], length=sma_short)
            df['SMA_50'] = ta.sma(df['close'], length=sma_long)
            df['SMA_200'] = ta.sma(df['close'], length=sma_longest)

            # Calculate EMAs
            df['EMA_9'] = ta.ema(df['close'], length=ema_short)
            df['EMA_12'] = ta.ema(df['close'], length=ema_mid)
            df['EMA_26'] = ta.ema(df['close'], length=ema_long)

            # Calculate RSI
            df['RSI'] = self.calculate_rsi(df['close'], period=rsi_period)

            # Calculate MACD
            df['MACD'], df['MACD_signal'] = self.calculate_macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)

            # Calculate Bollinger Bands
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(df['close'], length=20, std=2)

            # Calculate ATR (Average True Range)
            df['ATR'] = self.calculate_atr(df['high'], df['low'], df['close'], length=14)

            logging.info("Calculated technical indicators successfully")
            return df
        except Exception as e:
            logging.error("Error during technical analysis: %s", e)
            raise e

    def trading_strategy(self, df):
        """Implement the trading strategy and generate buy/sell signals."""
        try:
            signals = ['hold']
            for i in range(1, len(df)):
                if pd.notna(df['SMA_50'][i]) and pd.notna(df['SMA_200'][i]) and \
                   pd.notna(df['SMA_50'][i-1]) and pd.notna(df['SMA_200'][i-1]):
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

    def execute_trade(self, symbol, signal, amount=0.001):
        """Execute the trade based on the generated signal."""
        try:
            if signal == 'buy':
                logging.info("Executing Buy Order")
                order = self.exchange.create_market_buy_order(symbol, amount)
            elif signal == 'sell':
                logging.info("Executing Sell Order")
                order = self.exchange.create_market_sell_order(symbol, amount)
            else:
                logging.info("No trade action needed (hold signal)")
                return

            if 'error' in order:
                logging.error("Failed to execute order: %s", order['error'])
            else:
                logging.info("Order executed successfully: %s", order)
                
        except ccxt.BaseError as e:
            logging.error("An error occurred during trade execution: %s", e)
            raise e
        
# After fetching and calculating indicators
features, labels = bot.ml_model.prepare_data(df)
bot.ml_model.train_model(features, labels)
predictions = bot.ml_model.predict(features)

def main():
    """Main function to run the trading bot."""
    try:
        # Replace with your actual API key and secret
        API_KEY = 'LzvSGu2mYFi2L6VtBL'
        API_SECRET = 'KA3wvyIvMCJjGZEB0KVjH9WJSi30iwc9pIiG'

        time_offset = synchronize_system_time()
        logging.info("System time synchronized with offset: %d ms", time_offset)
        
        bot = TradingBot(API_KEY, API_SECRET)
        bot.initialize_exchange()
        
        # Fetch historical OHLCV data
        df = bot.fetch_ohlcv('BTCUSDT', time_offset=time_offset)
        
        # Calculate indicators and generate trading signals
        df = bot.calculate_indicators(df)
        df = bot.trading_strategy(df)
        
        # Execute trades based on the signals
        df.apply(lambda row: bot.execute_trade('BTCUSDT', row['signal']), axis=1)
        
        # Display the latest data for debugging
        print(df.tail())
        
    except ccxt.AuthenticationError as e:
        logging.error("Authentication error: %s. Please check your API key and secret.", e)
    except ccxt.NetworkError as e:
        logging.error("Network error: %s. Please check your internet connection.", e)
    except ccxt.ExchangeError as e:
        logging.error("Exchange error: %s. Please check the exchange status or API documentation.", e)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)

if __name__ == "__main__":
    main()

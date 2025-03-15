#python -m strategies.technical_indicators (RUN WITH THIS)

import time
import logging
# Note: The ccxt module is unavailable in this environment.
import pandas as pd
import numpy as np  # Using numpy for RSI calculations as pandas_ta is unavailable
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from exchanges.synchronize_exchange_time import synchronize_system_time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dotenv import load_dotenv
import os

# Load API keys from the .env file
dotenv_path = r'D:\\RAJESH FOLDER\\PROJECTS\\trade-robot\\config\\API.env'
load_dotenv(dotenv_path)

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLModel:
    def __init__(self):
        """Initialize the MLModel with a RandomForestClassifier."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def prepare_data(self, df):
        """Prepare features and labels for model training."""
        try:
            df['returns'] = df['close'].pct_change()
            df['label'] = df['returns'].shift(-1).apply(lambda x: 1 if x > 0 else 0)

            # Ensure indicators and labels are calculated without dropping too many rows
            required_cols = ['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'label']
            df.dropna(subset=required_cols, inplace=True)

            features = df[['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']]
            labels = df['label']

            return features, labels
        except Exception as e:
            logging.error("Error preparing data: %s", e)
            raise e

    def train_model(self, features, labels):
        """Train the machine learning model."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)

            predictions = self.model.predict(X_test)
            logging.info("Classification Report:\n%s", classification_report(y_test, predictions))
        except Exception as e:
            logging.error("Error during model training: %s", e)
            raise e

    def predict(self, features):
        """Predict trading signals based on the trained model."""
        try:
            return self.model.predict(features)
        except Exception as e:
            logging.error("Error during prediction: %s", e)
            raise e

class TradingBot:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = None  # Placeholder, as ccxt is unavailable
        self.ml_model = MLModel()

    def initialize_exchange(self):
        
        """Initialize the trading exchange."""
        try:
            # Placeholder as ccxt is not available
            logging.info("Initialized Bybit exchange (simulated)")
        except Exception as e:
            logging.error("Failed to initialize exchange: %s", e)
            raise e

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100, time_offset=0):
        """Simulate fetching OHLCV data (as ccxt is unavailable)."""
        try:
            logging.info("Simulating OHLCV data fetch for symbol %s", symbol)
            # Simulated dataframe for testing purposes
            data = {
                'timestamp': pd.date_range(start='2023-01-01', periods=limit, freq='h'),
                'open': pd.Series(range(limit)),
                'high': pd.Series(range(1, limit + 1)),
                'low': pd.Series(range(limit)),
                'close': pd.Series(range(1, limit + 1)),
                'volume': pd.Series(range(limit))
            }
            df = pd.DataFrame(data)
            logging.info(f"Fetched OHLCV data for {symbol} (simulated)")
            return df
        except Exception as e:
            logging.error("Error fetching OHLCV data: %s", e)
            raise e

    @staticmethod
    def calculate_sma(series, window):
        """Calculate Simple Moving Average (SMA)."""
        return series.rolling(window=window).mean()

    @staticmethod
    def calculate_rsi(series, length=14):
        """Calculate Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=length).mean()
        avg_loss = loss.rolling(window=length).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(series, fast=12, slow=26, signal=9):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_bollinger_bands(series, length=20, std=2):
        """Calculate Bollinger Bands."""
        rolling_mean = series.rolling(window=length).mean()
        rolling_std = series.rolling(window=length).std()
        upper_band = rolling_mean + (std * rolling_std)
        lower_band = rolling_mean - (std * rolling_std)
        return upper_band, rolling_mean, lower_band

    @staticmethod
    def calculate_atr(high, low, close, length=14):
        """Calculate Average True Range (ATR)."""
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=length).mean()
        return atr

    def calculate_indicators(self, df):
        """Calculate technical indicators."""
        try:
            # Calculate technical indicators
            df['SMA_20'] = self.calculate_sma(df['close'], window=20)
            df['SMA_50'] = self.calculate_sma(df['close'], window=50)
            df['SMA_200'] = self.calculate_sma(df['close'], window=200)
            df['RSI'] = self.calculate_rsi(df['close'], length=14)
            macd, signal_line = self.calculate_macd(df['close'], fast=12, slow=26, signal=9)
            df['MACD'] = macd
            df['MACD_signal'] = signal_line
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(df['close'], length=20, std=2)
            df['BB_upper'] = upper_band
            df['BB_middle'] = middle_band
            df['BB_lower'] = lower_band
            df['ATR'] = self.calculate_atr(df['high'], df['low'], df['close'], length=14)
            logging.info("Calculated technical indicators successfully")
            return df
        except Exception as e:
            logging.error("Error during technical analysis: %s", e)
            raise e

def generate_technical_correlation_heatmap(data, columns=None):
    """
    Generate a heatmap showing the correlation between different technical indicators.

    :param data: DataFrame containing the data.
    :param columns: Specific columns to include in the heatmap (optional).
    """
    try:
        # Select numeric columns only
        if columns:
            data = data[columns]
        else:
            data = data.select_dtypes(include=[np.number])

        # Calculate correlation
        correlation_matrix = data.corr()

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Technical Indicators Correlation Heatmap")
        plt.savefig('strategies/correlation_heatmap.png')
    except Exception as e:
        logging.error("An error occurred while generating the heatmap: %s", e)
        raise e

def trading_strategy(df):
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

def execute_trade(exchange, symbol, signal, amount=0.001, time_offset=0):
    """Simulate executing the trade based on the generated signal."""
    try:
        logging.info(f"Simulated trade execution: Signal={signal}, Symbol={symbol}, Amount={amount}")
    except Exception as e:
        logging.error("An error occurred during trade execution: %s", e)
        raise e

def main():
    try:
        # Ensure the API keys are loaded
        if not API_KEY or not API_SECRET:
            logging.error("API keys are missing. Please check the .env file.")
            return

        time_offset = synchronize_system_time()
        logging.info("System time synchronized with offset: %d ms", time_offset)

        # Instantiate the TradingBot class
        bot = TradingBot(API_KEY, API_SECRET)
        bot.initialize_exchange()

        # Fetch historical OHLCV data
        df = bot.fetch_ohlcv('BTCUSDT', time_offset=time_offset, limit=1000)

        # Calculate indicators and generate trading signals
        df = bot.calculate_indicators(df)
        df = trading_strategy(df)

        # Check if data is sufficient for training
        if df.shape[0] < 50:
            logging.error("Insufficient data for training. At least 50 rows are required.")
            return

        # Prepare the data for training the model
        features, labels = bot.ml_model.prepare_data(df)
        if features.empty or labels.empty:
            logging.error("Prepared features or labels are empty. Skipping training.")
            return

        bot.ml_model.train_model(features, labels)

        # Make a prediction for the next move
        latest_data = df.tail(1)[['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']]
        if latest_data.isna().any(axis=1).iloc[0]:
            logging.warning("Latest data contains NaN values. Skipping prediction.")
            return

        prediction = bot.ml_model.predict(latest_data)

        # Execute the trade based on the prediction
        signal = 'buy' if prediction == 1 else 'sell'
        execute_trade(bot.exchange, 'BTCUSDT', signal, time_offset=time_offset)

        # Generate and display correlation heatmap
        generate_technical_correlation_heatmap(df)

    except Exception as e:
        logging.error("An error occurred: %s", e)
        raise e

if __name__ == '__main__':
    main()

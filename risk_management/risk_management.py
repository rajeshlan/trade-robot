import ccxt
import logging
import pandas as pd
from datetime import date, datetime, timedelta, timezone
import os
import ntplib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def synchronize_system_time():
    """Synchronize system time with an NTP server."""
    try:
        response = ntplib.NTPClient().request('pool.ntp.org', timeout=10)
        current_time = datetime.fromtimestamp(response.tx_time)
        logging.info(f"System time synchronized: {current_time}")
        return current_time
    except Exception as e:
        logging.error("Time synchronization failed: %s", e)
        return datetime.now()

def initialize_exchange(api_key, api_secret):
    """Initialize the exchange with API keys."""
    try:
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'recvWindow': 10000}
        })
        logging.info("Initialized Bybit exchange")
        return exchange
    except Exception as e:
        logging.error("Failed to initialize exchange: %s", e)
        raise e

def fetch_historical_data(exchange, symbol, timeframe='1h', limit=100):
    """Fetch historical data from the exchange."""
    try:
        # Use datetime.now(timezone.utc) instead of datetime.utcnow()
        since = exchange.parse8601(exchange.iso8601(datetime.now(timezone.utc) - timedelta(days=limit)))
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        logging.info(f"Fetched historical data for {symbol}")
        return data
    except Exception as e:
        logging.error("Failed to fetch historical data: %s", e)
        raise e

def calculate_technical_indicators(data, sma_periods=(50, 200), ema_periods=(12, 26), rsi_period=14):
    """Calculate technical indicators for trading signals."""
    try:
        data['SMA_50'] = data['close'].rolling(window=sma_periods[0]).mean()
        data['SMA_200'] = data['close'].rolling(window=sma_periods[1]).mean()
        data['EMA_12'] = data['close'].ewm(span=ema_periods[0], adjust=False).mean()
        data['EMA_26'] = data['close'].ewm(span=ema_periods[1], adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['RSI'] = calculate_rsi(data['close'], rsi_period)
        logging.info("Calculated technical indicators")
        return data
    except Exception as e:
        logging.error("Failed to calculate technical indicators: %s", e)
        raise e

def calculate_rsi(series, period):
    """Calculate Relative Strength Index (RSI)."""
    try:
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logging.error("Failed to calculate RSI: %s", e)
        raise e

def detect_patterns(data):
    """Detect trading patterns in the data."""
    try:
        data['HeadAndShoulders'] = detect_head_and_shoulders(data)
        data['DoubleTop'] = detect_double_top(data)
        logging.info("Detected patterns")
        return data
    except Exception as e:
        logging.error("Failed to detect patterns: %s", e)
        raise e

def detect_head_and_shoulders(data):
    """Detect Head and Shoulders pattern."""
    try:
        pattern = [0] * len(data)
        for i in range(2, len(data) - 1):
            if (data['high'][i - 2] < data['high'][i - 1] > data['high'][i] and
                data['high'][i - 1] > data['high'][i + 1] and
                data['low'][i - 2] > data['low'][i - 1] < data['low'][i] and
                data['low'][i - 1] < data['low'][i + 1]):
                pattern[i] = 1
        return pattern
    except Exception as e:
        logging.error("Failed to detect Head and Shoulders pattern: %s", e)
        raise e

def detect_double_top(data):
    """Detect Double Top pattern."""
    try:
        pattern = [0] * len(data)
        for i in range(1, len(data) - 1):
            if (data['high'][i - 1] < data['high'][i] > data['high'][i + 1] and
                data['high'][i] == data['high'][i + 1]):
                pattern[i] = 1
        return pattern
    except Exception as e:
        logging.error("Failed to detect Double Top pattern: %s", e)
        raise e

def calculate_atr(data, period=14):
    """Calculate Average True Range (ATR)."""
    high_low_range = data['high'] - data['low']
    high_close_range = abs(data['high'] - data['close'].shift())
    low_close_range = abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low_range, high_close_range, low_close_range], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_stop_loss(entry_price, atr_multiplier, data):
    """Calculate Stop Loss based on ATR."""
    atr = calculate_atr(data)
    stop_loss = entry_price - atr_multiplier * atr.iloc[-1]
    return stop_loss

def calculate_take_profit(entry_price, risk_reward_ratio, stop_loss):
    """Calculate Take Profit based on risk-reward ratio."""
    take_profit_distance = abs(entry_price - stop_loss) * risk_reward_ratio
    take_profit = entry_price + take_profit_distance if entry_price > stop_loss else entry_price - take_profit_distance
    return take_profit

def place_order_with_risk_management(exchange, symbol, side, amount, stop_loss=None, take_profit=None):
    """Place order with risk management (stop loss and take profit)."""
    try:
        order = exchange.create_order(symbol, 'market', side, amount)
        logging.info(f"Market order placed: {order}")

        order_price = order.get('price')
        if order_price:
            if not stop_loss:
                stop_loss = calculate_stop_loss(order_price, 1.5, data)  # Adjust multiplier as per your strategy
            if not take_profit:
                take_profit = calculate_take_profit(order_price, 2.0, stop_loss)

            logging.info(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")

            if side == 'buy':
                exchange.create_order(symbol, 'stop', 'sell', amount, stop_loss)
                exchange.create_order(symbol, 'limit', 'sell', amount, take_profit)
            else:
                exchange.create_order(symbol, 'stop', 'buy', amount, stop_loss)
                exchange.create_order(symbol, 'limit', 'buy', amount, take_profit)

        else:
            logging.warning("Order price not available for stop-loss and take-profit orders.")
    except Exception as e:
        logging.error(f"Order placement failed: {e}")

def prepare_data_for_training(data):
    """Prepare data for training the model."""
    data = data.dropna()  # Remove NaN values first
    X = data[['open', 'high', 'low', 'close', 'volume', 'SMA_50', 'SMA_200', 'RSI']]  # Features
    y = data['close'].shift(-1).dropna()  # Shift close price to create target
    X = X[:-1]  # Adjust features to match target length

    logging.info(f"Data shape for features: {X.shape}, Target shape: {y.shape}")

    if X.shape[0] == 0 or y.shape[0] == 0:
        logging.error("Feature or target dataset is empty. Check your data preparation steps.")
        raise ValueError("Feature or target dataset is empty.")

    return X, y

def train_model(X, y):
    """Train a Random Forest model on the historical data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    logging.info(f"Model trained. MSE: {mse}")

    return model

if __name__ == "__main__":
    # Initialize exchange and fetch data
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    exchange = initialize_exchange(api_key, api_secret)

    # Define symbol and fetch historical data
    symbol = 'BTC/USDT'
    data = fetch_historical_data(exchange, symbol)

    # Calculate technical indicators and detect patterns
    data = calculate_technical_indicators(data)
    data = detect_patterns(data)

    # Prepare data for training and train the model
    X, y = prepare_data_for_training(data)
    model = train_model(X, y)

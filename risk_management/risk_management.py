# python risk_management\risk_management.py

import os
import sys
import logging
from datetime import datetime, timezone
import time
import pandas as pd
import numpy as np

# Ensure required libraries are installed
try:
    import ccxt
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from ta import add_all_ta_features
except ImportError as e:
    logging.error(f"Missing required module: {e}. Please install all dependencies.")
    raise

# Add the parent directory to sys.path for custom module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom time synchronization module
from exchanges.synchronize_exchange_time import synchronize_system_time

# Setup logging for consistent output and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_exchange(api_key, api_secret):
    """
    Initialize the exchange with API keys and dynamically adjust timestamps for Bybit.
    Handles API configuration, time synchronization, and market loading.
    """
    try:
        # Step 1: Synchronize system time to minimize API errors
        logging.info("Starting system time synchronization...")
        time_offset = synchronize_system_time()
        logging.info(f"Time offset from system time synchronization: {time_offset:.2f} ms")

        # Step 2: Initialize the exchange instance with API credentials
        logging.info("Initializing exchange instance with API credentials...")
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # Ensure rate limits are respected
            'timeout': 30000,  # 30 seconds timeout for requests
            'options': {
                'defaultType': 'future',  # Use futures trading by default
                'adjustForTimeDifference': True,  # Enable time difference adjustment
                'recvWindow': 100000,  # Set request timeout window
            },
        })

        # Log available API endpoints for debugging
        logging.debug(f"Exchange URLs after initialization: {exchange.urls}")

        # Step 3: Load markets to access trading pairs and other market data
        markets = exchange.load_markets()
        
        # Example: Fetch account balance for futures
        balance = exchange.fetch_balance()
        print(balance)

        # Step 4: Fetch server time and calculate time difference
        logging.info("Fetching server time...")
        server_time = exchange.fetch_time()
        if not isinstance(server_time, (int, float)):
            raise ValueError(f"fetch_time returned non-numeric value: {server_time}")

        local_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        initial_time_difference = server_time - local_time
        logging.info(f"Bybit server time: {server_time}, Local time: {local_time}, Initial time difference: {initial_time_difference} ms")

        # Step 5: Apply the initial time difference to exchange options
        exchange.options['timeDifference'] = initial_time_difference

        # Step 6: Replace the request method with a custom synchronized request method
        original_request = exchange.request

        def synchronized_request(path, api='public', method='GET', params=None, headers=None, body=None, config=None):
            """Custom request method with dynamic time synchronization."""
            try:
                logging.debug(f"Executing request: path={path}, api={api}, method={method}")
        
                # Use original request for time fetch to avoid recursion
                if path == 'time':
                    return original_request(path, api, method, params, headers, body, config)

                # Sync time only if necessary, not for every request to prevent recursion
                if not hasattr(exchange, 'last_time_sync') or time.time() - exchange.last_time_sync > 60:  # Sync every 60 seconds
                    try:
                        # Temporarily revert to original request for time synchronization
                        exchange.request = original_request
                        server_time = exchange.fetch_time()
                        local_time = int(datetime.now(timezone.utc).timestamp() * 1000)
                        adjusted_timestamp = local_time + (server_time - local_time)
                        exchange.options['adjustedTimestamp'] = adjusted_timestamp
                        exchange.last_time_sync = time.time()  # Mark the time we last synchronized
                    except Exception as sync_error:
                        logging.warning(f"Time synchronization failed with error: {sync_error}. Continuing with last known sync.")
                    finally:
                        # Restore the custom request method
                        exchange.request = synchronized_request

                # Execute the original request with updated time sync
                return original_request(path, api, method, params, headers, body, config)
            except Exception as e:
                logging.error(f"Error in synchronized_request: {e}")
                raise

        exchange.request = synchronized_request  # Apply the custom request method
    
        # Step 7: Load available markets for trading
        logging.info("Loading markets...")
        exchange.load_markets()
        logging.info("Initialized Bybit exchange successfully with dynamic timestamp synchronization")
        return exchange

    except ccxt.BaseError as e:
        # Handle CCXT-specific errors and log them
        logging.error(f"Failed to initialize exchange: {str(e)}")
        raise
    except Exception as e:
        # Handle general exceptions and log detailed errors
        logging.error(f"Unexpected error initializing exchange: {e}")
        raise

# Rest of your code remains unchanged as per your request

def fetch_historical_data(exchange, symbol, timeframe='1h', limit=500):
    """
    Fetch historical OHLCV data from the exchange for a given symbol and timeframe.
    Returns the data as a Pandas DataFrame.
    """
    try:
        logging.info(f"Fetching historical data for symbol={symbol}, timeframe={timeframe}, limit={limit}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)  # Fetch OHLCV data
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')  # Convert timestamps to datetime
        logging.info(f"Fetched {len(data)} data points for {symbol}")
        return data
    except Exception as e:
        logging.error("Failed to fetch historical data: %s", e)
        raise e

def calculate_technical_indicators(data, sma_periods=(50, 200), ema_periods=(12, 26), rsi_period=14):
    """
    Calculate technical indicators like SMA, EMA, MACD, and RSI for trading signals.
    Returns the DataFrame with new indicator columns.
    """
    try:
        logging.info("Calculating technical indicators...")
        # Simple Moving Averages (SMA)
        data['SMA_50'] = data['close'].rolling(window=sma_periods[0]).mean()
        data['SMA_200'] = data['close'].rolling(window=sma_periods[1]).mean()

        # Exponential Moving Averages (EMA)
        data['EMA_12'] = data['close'].ewm(span=ema_periods[0], adjust=False).mean()
        data['EMA_26'] = data['close'].ewm(span=ema_periods[1], adjust=False).mean()

        # MACD and Signal Line
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Relative Strength Index (RSI)
        data['RSI'] = calculate_rsi(data['close'], rsi_period)

        logging.info("Technical indicators calculated successfully.")
        return data
    except Exception as e:
        logging.error("Failed to calculate technical indicators: %s", e)
        raise e

def calculate_rsi(series, period):
    """
    Calculate Relative Strength Index (RSI) using rolling averages.
    """
    try:
        logging.debug(f"Calculating RSI with period={period}...")
        delta = series.diff(1)  # Difference between consecutive values
        gain = delta.where(delta > 0, 0)  # Positive changes only
        loss = -delta.where(delta < 0, 0)  # Negative changes only
        avg_gain = gain.rolling(window=period).mean()  # Rolling average of gains
        avg_loss = loss.rolling(window=period).mean()  # Rolling average of losses
        rs = avg_gain / avg_loss  # Relative Strength
        rsi = 100 - (100 / (1 + rs))  # RSI formula
        logging.debug("RSI calculation completed.")
        return rsi
    except Exception as e:
        logging.error("Failed to calculate RSI: %s", e)
        raise e

def prepare_data_for_training(data):
    """
    Prepare historical data for training a machine learning model.
    Ensures data is cleaned and features/targets are aligned.
    """
    try:
        logging.info("Preparing data for training...")
        # Forward fill missing data to avoid null values
        data.ffill(inplace=True)

        if len(data) < 100:
            logging.error("Insufficient data for training.")
            raise ValueError("Insufficient data for training. Ensure you have enough valid historical data.")

        # Extract features and target variable
        X = data[['open', 'high', 'low', 'close', 'volume']]
        y = data['close'].shift(-1).dropna()  # Predicting next close price
        X = X[:-1]  # Align feature rows with target rows

        # Validate dataset sizes
        if X.shape[0] == 0 or y.shape[0] == 0:
            logging.error("Feature or target dataset is empty.")
            raise ValueError("Feature or target dataset is empty.")

        logging.info(f"Prepared data with {X.shape[0]} samples for training.")
        return X, y
    except Exception as e:
        logging.error("Failed to prepare data for training: %s", e)
        raise e

def train_model(X, y):
    """
    Train a Random Forest model on the prepared data.
    Splits the data into training and testing sets for evaluation.
    """
    try:
        logging.info("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)  # Random Forest model
        model.fit(X_train, y_train)  # Train the model

        # Evaluate the model using Mean Squared Error (MSE)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Model training completed. MSE: {mse}")

        return model
    except Exception as e:
        logging.error("Error during model training: %s", e)
        raise e

def calculate_stop_loss(entry_price, risk_percentage, leverage=1):
    """Calculate stop-loss price based on risk percentage and leverage."""
    try:
        entry_price = float(entry_price)  # Ensure entry_price is a valid number
        logging.debug(f"Calculating stop-loss for entry_price={entry_price}, risk_percentage={risk_percentage}, leverage={leverage}...")
        stop_loss = entry_price * (1 - (risk_percentage / 100) / leverage)  # Stop-loss formula
        logging.info(f"Calculated stop-loss: {stop_loss}")
        return stop_loss
    except Exception as e:
        logging.error(f"Error calculating stop-loss: {e}")
        raise e



def calculate_take_profit(entry_price, risk_reward_ratio, stop_loss):
    """Calculate take-profit price based on risk-reward ratio and stop-loss."""
    try:
        risk_amount = entry_price - stop_loss
        take_profit = entry_price + (risk_amount * risk_reward_ratio)
        logging.info(f"Calculated take-profit: {take_profit}")
        return take_profit
    except Exception as e:
        logging.error(f"Error calculating take-profit: {e}")
        raise

def calculate_position_size(balance, risk_percentage, entry_price, stop_loss):
    """Calculate the position size based on balance, risk percentage, and stop-loss."""
    try:
        logging.debug(f"Calculating position size for balance={balance}, risk_percentage={risk_percentage}, entry_price={entry_price}, stop_loss={stop_loss}...")
        risk_amount = balance * (risk_percentage / 100)  # Total risk allowed
        risk_per_unit = entry_price - stop_loss  # Risk per unit of trade
        if risk_per_unit <= 0:
            logging.error("Stop-loss must be below entry price for a buy order.")
            raise ValueError("Stop-loss must be below entry price for a buy order")
        position_size = risk_amount / risk_per_unit  # Position size formula
        logging.info(f"Calculated position size: {position_size}")
        return position_size
    except Exception as e:
        logging.error(f"Error calculating position size: {e}")
        raise e

if __name__ == "__main__":
    # Load environment variables for API credentials
    logging.info("Loading environment variables for API credentials...")
    from config.app_config import API_KEY, API_SECRET, get_env_var

    # Fetch Bybit API credentials dynamically
    api_key = get_env_var('BYBIT_API_KEY', None)
    api_secret = get_env_var('BYBIT_API_SECRET', None)

    if not api_key or not api_secret:
        logging.error("Bybit API credentials are missing. Ensure they are set in the .env file.")
        exit()

    # Initialize exchange with provided API credentials
    try:
        logging.info("Initializing exchange...")
        exchange = initialize_exchange(api_key, api_secret)
    except Exception as e:
        logging.error(f"Error initializing exchange: {e}")
        exit()

    # Fetch historical OHLCV data for a specific trading pair
    symbol = 'BTC/USDT'
    try:
        logging.info("Fetching historical OHLCV data...")
        data = fetch_historical_data(exchange, symbol, limit=500)
        logging.info("Historical data fetched successfully")
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        exit()

    # Calculate technical indicators from historical data
    try:
        logging.info("Calculating technical indicators...")
        data = add_all_ta_features(data, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        logging.info("Technical indicators calculated successfully")
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        exit()

    # Prepare data for machine learning training and train the model
    try:
        logging.info("Preparing data for machine learning training...")
        X, y = prepare_data_for_training(data)
        logging.info("Data preparation complete. Training model...")
        model = train_model(X, y)
        logging.info("Model training completed successfully")
    except Exception as e:
        logging.error(f"Error during data preparation or model training: {e}")
        exit()

# python scripts\example_usage.py

import logging
import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for general logs; use DEBUG only when necessary
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output logs to terminal
        logging.FileHandler("trading_bot.log", mode='w')  # Save logs to a file, overwrite each run
    ]
)

# Initialize model components
scaler = StandardScaler()
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Function to fetch historical data, train the model, and backtest

def fetch_and_prepare_data(exchange, symbol):
    """
    Fetch historical OHLCV data and prepare features for model training.
    """
    try:
        logging.info("Fetching historical data for %s...", symbol)
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=200)
        logging.debug("Fetched OHLCV data: %s", ohlcv)

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['returns'] = df['close'].pct_change()
        df['moving_average_5'] = df['close'].rolling(window=5).mean()
        df['moving_average_20'] = df['close'].rolling(window=20).mean()
        df['volatility'] = df['close'].rolling(window=20).std()
        df.dropna(inplace=True)

        logging.info("Feature engineering completed. Prepared data size: %d", len(df))
        return df
    except Exception as e:
        logging.error("Error while fetching or preparing data: %s", e)
        raise

def train_model(df):
    """
    Train the model using prepared historical data.
    """
    try:
        X = df[['moving_average_5', 'moving_average_20', 'volatility']].values
        y = df['returns'].values

        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        model.fit(X_train, y_train)
        logging.info("Model trained with %d samples.", len(X_train))

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info("Model evaluation completed. MSE: %f", mse)
    except Exception as e:
        logging.error("Error during model training: %s", e)
        raise

def make_prediction(moving_average_5, moving_average_20, volatility):
    """
    Make a prediction based on current market indicators.
    """
    try:
        X_new = scaler.transform([[moving_average_5, moving_average_20, volatility]])
        prediction = model.predict(X_new)[0]
        logging.debug("Prediction result: %f", prediction)
        return prediction
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        raise

def backtest_strategy(df):
    """
    Simulate the trading strategy on historical data for backtesting.
    """
    try:
        logging.info("Starting backtest...")
        df['predicted_returns'] = model.predict(scaler.transform(df[['moving_average_5', 'moving_average_20', 'volatility']].values))
        df['strategy_returns'] = np.where(df['predicted_returns'] > 0, df['returns'], 0)

        cumulative_strategy_returns = (1 + df['strategy_returns']).cumprod()
        cumulative_market_returns = (1 + df['returns']).cumprod()

        logging.info("Backtest completed. Final strategy returns: %f", cumulative_strategy_returns.iloc[-1])
        logging.info("Final market returns: %f", cumulative_market_returns.iloc[-1])
    except Exception as e:
        logging.error("Error during backtesting: %s", e)
        raise

def example_usage(exchange):
    """
    Example function to demonstrate the usage of the trading bot.
    """
    try:
        symbol = 'BTC/USDT'
        df = fetch_and_prepare_data(exchange, symbol)
        train_model(df)
        backtest_strategy(df)

        ticker = exchange.fetch_ticker('BTC/USDT')
        current_price = ticker['last']
        logging.info("Current price: %f", current_price)    
        moving_average_5 = df['moving_average_5'].iloc[-1]
        moving_average_20 = df['moving_average_20'].iloc[-1]
        volatility = df['volatility'].iloc[-1]

        predicted_returns = make_prediction(moving_average_5, moving_average_20, volatility)
        logging.info("Predicted returns: %f", predicted_returns)

        if predicted_returns > 0.01:
            logging.info("Placing a mock buy order...")
            order = exchange.create_order('BTC/USDT', 'limit', 'buy', 0.0001, current_price)
            logging.info("Order response: %s", order)
        else:
            logging.info("No favorable trade signal detected.")
    except Exception as e:
        logging.error("Error in example usage: %s", e)
        raise

# Main script execution
if __name__ == "__main__":
    try:
        logging.info("Starting script execution...")
        exchange = ccxt.bybit({'enableRateLimit': True})
        example_usage(exchange)
    except Exception as e:
        logging.error("Fatal error: %s", e)

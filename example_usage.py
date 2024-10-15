import logging
import ccxt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from exchanges import send_notification

# Initialize the model (you can load a pre-trained model instead)
model = LinearRegression()

# Function to fetch historical data and train the model
def train_model(exchange, symbol):
    logging.info("Fetching historical data for %s...", symbol)
    ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=100)  # Daily data for the last 100 days
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Feature engineering
    df['returns'] = df['close'].pct_change()
    df['moving_average'] = df['close'].rolling(window=5).mean()
    df.dropna(inplace=True)

    X = df[['moving_average']].values  # Features
    y = df['returns'].values  # Target

    model.fit(X, y)
    logging.info("Model trained with %d samples.", len(X))

def make_prediction(current_price):
    # Reshape the current price for the model
    X_new = np.array([[current_price]])
    prediction = model.predict(X_new)
    return prediction

def example_usage(exchanges):
    for exchange in exchanges:
        try:
            logging.info("Fetching ticker data from Bybit...")
            ticker = exchange.fetch_ticker('BTCUSDT')
            current_price = ticker['last']
            logging.info("Ticker data: %s", ticker)

            # Predict price movement
            predicted_returns = make_prediction(current_price)
            logging.info("Predicted returns: %s", predicted_returns)

            # Decision making based on prediction
            if predicted_returns > 0.01:  # Example threshold
                logging.info("Placing a mock order on Bybit...")
                order = exchange.create_order('BTCUSDT', 'limit', 'buy', 0.0001, current_price)
                logging.info("Order response: %s", order)

                logging.info("Fetching account balance...")
                balance = exchange.fetch_balance()
                logging.info("Account balance: %s", balance)

                send_notification(f"Placed a mock order and fetched balance: {balance}")

            else:
                logging.info("No order placed. Prediction not favorable.")

            # Train the model with new data periodically (could be based on time or trades)
            train_model(exchange, 'BTC/USDT')

        except ccxt.NetworkError as net_error:
            logging.error("A network error occurred with Bybit: %s", net_error)
        except ccxt.ExchangeError as exchange_error:
            logging.error("An exchange error occurred with Bybit: %s", exchange_error)
        except ccxt.BaseError as base_error:
            logging.error("An unexpected error occurred with Bybit: %s", base_error)


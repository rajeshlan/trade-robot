import sqlite3
import ccxt
import pandas as pd
import logging
import matplotlib.pyplot as plt
from ccxt import bybit 
import unittest
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import sys
import os
import tradingbot


# Add the root directory of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config')))

# Now you can import app_config
from config.app_config import DB_FILE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_db_connection(db_file):
    """Create and return a database connection."""
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"Connected to the database {db_file} successfully.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database: {e}")
        return None

def store_data_to_db(conn, df, table_name):
    """Store a DataFrame into a database table."""
    try:
        with conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logging.info(f"Data stored in table {table_name} successfully.")
    except Exception as e:
        logging.error(f"Error storing data to table {table_name}: {e}")

def fetch_historical_data_from_exchange(symbol, timeframe='1h', limit=100, exchange_name='bybit'):
    """Fetch historical data (OHLCV) for a given symbol from the exchange."""
    try:
        exchange = getattr(ccxt, exchange_name)()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        # Create a DataFrame from the OHLCV data
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        logging.info(f"Historical data for {symbol} fetched successfully.")
        return df
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logging.error(f"Network or Exchange error fetching historical data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error fetching historical data: {e}")
        return pd.DataFrame()

def fetch_historical_data(conn, table_name):
    """Fetch historical data from a database table into a DataFrame."""
    try:
        query = f"SELECT * FROM {table_name} ORDER BY Date DESC"  # Fetch the most recent data
        df = pd.read_sql(query, conn)
        logging.info(f"Data fetched from table {table_name} successfully.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data from table {table_name}: {e}")
        return pd.DataFrame()

def fetch_realtime_data(symbol, exchange_name='bybit'):
    """Fetch real-time data for a given symbol from an exchange using CCXT."""
    try:
        exchange = getattr(ccxt, exchange_name)()
        ticker = exchange.fetch_ticker(symbol)
        data = {
            'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Symbol': [symbol],
            'Open': [ticker['open']],
            'High': [ticker['high']],
            'Low': [ticker['low']],
            'Close': [ticker['close']],
            'Volume': [ticker['baseVolume']]
        }
        df = pd.DataFrame(data)
        logging.info(f"Real-time data for {symbol} fetched successfully.")
        return df
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logging.error(f"Network or Exchange error fetching real-time data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error fetching real-time data: {e}")
        return pd.DataFrame()

def close_db_connection(conn):
    """Close the database connection."""
    if conn:
        conn.close()
        logging.info("Database connection closed.")

def predict_price(historical_data):
    # Log data fetched from historical_data
    logging.info(f"Data fetched from table historical_data successfully.\n{historical_data}")
    
    # Using 'Open' and 'Volume' as features for prediction
    X = historical_data[['Open', 'Volume']].values
    y = historical_data['Close'].values

    # Log the shapes of the features and target arrays
    logging.info(f"X shape: {X.shape}")
    logging.info(f"y shape: {y.shape}")

    # Split the data into training and testing sets
    X_train, y_train = X[:-1], y[:-1]  # Train on all but the last row
    X_test, y_test = X[-1:], y[-1:]    # Test on the last row

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the future price using the last row of features
    predicted_price = model.predict(X_test)
    logging.info(f"Predicted future price: {predicted_price[0]}")

    # Predict on the training set and calculate MSE
    y_pred_train = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred_train)
    logging.info(f"Mean Squared Error: {mse}")

    # Return predicted price, training actuals, and predictions
    return predicted_price[0], y_train, y_pred_train

def plot_predictions(actual_data, predicted_data):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_data, label='Actual Prices', color='blue')
    plt.plot(predicted_data, label='Predicted Prices', color='orange')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Actual vs. Predicted Prices")
    plt.show()

# Main logic
if not tradingbot.db.empty:
    future_price, actual_train, predicted_train = predict_price(tradingbot.db)
    if future_price is not None:
        logging.info(f'Predicted future price: {future_price}')
        # Plot the actual vs. predicted values
        plot_predictions(actual_train, predicted_train)

    # Create a database connection
    conn = create_db_connection(DB_FILE)

    if conn:
        # Fetch real-time data from Bybit and store in the database
        realtime_data = fetch_realtime_data('BTC/USDT')
        if not realtime_data.empty:
            store_data_to_db(conn, realtime_data, "realtime_data")

        # Fetch historical data from the exchange and store in the database
        historical_data = fetch_historical_data_from_exchange('BTC/USDT', timeframe='1h', limit=100)
        if not historical_data.empty:
            store_data_to_db(conn, historical_data, "historical_data")

        # Fetch and display the updated historical data from the database
        historical_data_db = fetch_historical_data(conn, "historical_data")
        print(historical_data_db)

        # Predict future price from historical data
        future_price, _, _ = predict_price(historical_data_db)
        if future_price is not None:
            logging.info(f'Predicted future price: {future_price}')

        # Close the database connection
        close_db_connection(conn)

# Set up Bybit exchange using ccxt
exchange = bybit()

class TestTradingBot(unittest.TestCase):

    def test_db_connection(self):
        """Test if the database connection is successful."""
        conn = create_db_connection('test.db')
        self.assertIsNotNone(conn)
        close_db_connection(conn)

    def test_store_and_fetch_real_data(self):
        """Test storing and fetching real BTC/USDT data from the database."""
        symbol = 'BTC/USDT'
        since = exchange.parse8601(f"{datetime.now().strftime('%Y-%m-%d')}T00:00:00Z")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', since=since, limit=5)
        
        test_data = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        test_data['Date'] = pd.to_datetime(test_data['Timestamp'], unit='ms')
        test_data = test_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        conn = create_db_connection('test.db')
        store_data_to_db(conn, test_data, 'real_data')
        fetched_data = fetch_historical_data(conn, 'real_data')
        self.assertGreater(len(fetched_data), 0)
        close_db_connection(conn)

    def test_predict_price_with_real_data(self):
        """Test the price prediction functionality using real BTC/USDT data."""
        symbol = 'BTC/USDT'
        since = exchange.parse8601(f"{datetime.now().strftime('%Y-%m-%d')}T00:00:00Z")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', since=since, limit=5)
        
        test_data = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        test_data['Date'] = pd.to_datetime(test_data['Timestamp'], unit='ms')
        test_data = test_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        predicted_price, _, _ = predict_price(test_data)
        self.assertIsNotNone(predicted_price)
        self.assertGreater(predicted_price, 0)

if __name__ == '__main__':
    unittest.main()

# python data\database.py

import sqlite3
import pandas as pd
import ccxt
import logging
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Switch to non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect

# Load environment variables
load_dotenv()

db_path = os.getenv("DB_PATH", "trading_bot.db")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database Operations
def create_db_connection():
    try:
        logging.info("Attempting to connect to the database.")
        conn = sqlite3.connect(db_path)
        logging.info("Database connection established.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise

def store_data_to_db(conn, table_name, df):
    try:
        logging.info(f"Storing data to table '{table_name}'.")
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        logging.info(f"Data stored in table '{table_name}'.")
    except sqlite3.Error as e:
        logging.error(f"Error storing data to table '{table_name}': {e}")
        raise



def fetch_historical_data(table_name="historical_data"):
    try:
        db_url = "sqlite:///D:/RAJESH FOLDER/PROJECTS/trade-robot/trading_bot.db"
        engine = create_engine(db_url)

        logging.info(f"Checking if table '{table_name}' exists before fetching data.")
        
        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
            raise ValueError(f"Table '{table_name}' does not exist in the database.")

        logging.info(f"Fetching data from table '{table_name}' using SQLAlchemy.")
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)

        if df.empty:
            raise ValueError(f"No data found in {table_name} table.")

        # Ensure correct column names for processing
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logging.info(f"Successfully fetched {len(df)} rows from table '{table_name}'.")
        return df

    except SQLAlchemyError as e:
        logging.error(f"SQLAlchemy error fetching data from '{table_name}': {e}")
        return pd.DataFrame()
    except ValueError as e:
        logging.error(e)
        return pd.DataFrame()

# Data Fetching
def fetch_historical_data_from_exchange(exchange, symbol, timeframe, limit):
    try:
        logging.info(f"Fetching historical data for {symbol} with timeframe {timeframe} and limit {limit}.")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Successfully fetched historical data for {symbol}.")
        return df
    except ccxt.BaseError as e:
        logging.error(f"Error fetching historical data from exchange: {e}")
        raise

def fetch_realtime_data(exchange, symbol, timeframe):
    try:
        logging.info(f"Fetching real-time data for {symbol} with timeframe {timeframe}.")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Successfully fetched real-time data for {symbol}.")
        return df
    except ccxt.BaseError as e:
        logging.error(f"Error fetching real-time data from exchange: {e}")
        raise

# Data Analysis and Predictions
def analyze_trend(df):
    logging.info("Analyzing trend based on closing prices.")
    if df['close'].iloc[-1] > df['close'].iloc[0]:
        logging.info("Trend detected: Uptrend")
        return 'Uptrend'
    elif df['close'].iloc[-1] < df['close'].iloc[0]:
        logging.info("Trend detected: Downtrend")
        return 'Downtrend'
    else:
        logging.info("Trend detected: Sideways")
        return 'Sideways'

def estimate_time_to_reach(model, df, target_price):
    try:
        logging.info(f"Estimating time to reach target price: {target_price}.")
        X = df.index.values.reshape(-1, 1)
        predicted_index = (target_price - model.intercept_) / model.coef_
        predicted_time = pd.Timestamp(df['timestamp'].iloc[0]) + pd.to_timedelta(predicted_index, unit='s')
        logging.info(f"Estimated time to reach target price: {predicted_time}.")
        return predicted_time
    except Exception as e:
        logging.error(f"Error estimating time to reach price: {e}")
        raise

def predict_price(df):
    try:
        logging.info("Starting price prediction using Linear Regression.")
        X = df.index.values.reshape(-1, 1)
        y = df['close'].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        trend = analyze_trend(df)
        logging.info(f"Prediction complete. Trend: {trend}")
        return predictions, model
    except Exception as e:
        logging.error(f"Error in price prediction: {e}")
        raise

# Visualization
def plot_predictions(df, predictions):
    try:
        logging.info("Plotting actual vs predicted prices.")
        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['close'], label='Actual Price')
        plt.plot(df['timestamp'], predictions, label='Predicted Price', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Actual vs Predicted Prices')
        plt.legend()
        plt.grid()
        plt.savefig('predictions_plot.png')  # Save plot to file instead of showing it
        logging.info("Successfully saved predictions plot as 'predictions_plot.png'.")
    except Exception as e:
        logging.error(f"Error in plotting predictions: {e}")
        raise

# Main Execution
def main():
    try:
        logging.info("Starting main execution.")
        conn = create_db_connection()

        # Configure exchange
        logging.info("Configuring exchange for Binance.")
        exchange = ccxt.binance({"rateLimit": 1200, "enableRateLimit": True})
        symbol = "BTCUSDT"
        timeframe = "1h"
        limit = 100

        # Fetch historical data
        logging.info("Fetching historical data from exchange.")
        historical_data = fetch_historical_data_from_exchange(exchange, symbol, timeframe, limit)

        # Store to DB
        logging.info("Storing historical data to database.")
        store_data_to_db(conn, "historical_data", historical_data)

        # Fetch data from DB
        logging.info("Fetching data back from database for analysis.")
        df = fetch_historical_data(conn, "historical_data")

        # Predict prices
        logging.info("Performing price prediction.")
        predictions, model = predict_price(df)

        # Plot predictions
        logging.info("Plotting predictions.")
        plot_predictions(df, predictions)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()

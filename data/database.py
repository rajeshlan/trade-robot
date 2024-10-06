import sqlite3
import pandas as pd
import logging
import ccxt
from datetime import datetime

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

def fetch_realtime_data(symbol, exchange='bybit'):
    """Fetch real-time data for a given symbol from Bybit using CCXT."""
    try:
        exchange = getattr(ccxt, exchange)()
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
    except Exception as e:
        logging.error(f"Error fetching real-time data for {symbol}: {e}")
        return pd.DataFrame()

def close_db_connection(conn):
    """Close the database connection."""
    if conn:
        conn.close()
        logging.info("Database connection closed.")

# Example usage
if __name__ == "__main__":
    DB_FILE = 'trading_bot.db'
    TABLE_NAME = 'historical_data'
    SYMBOL = 'BTC/USDT'  # Example for Bitcoin against USDT
    
    # Create a database connection
    conn = create_db_connection(DB_FILE)
    
    if conn:
        # Fetch real-time data from Bybit
        realtime_data = fetch_realtime_data(SYMBOL)
        
        if not realtime_data.empty:
            # Store real-time data to the database
            store_data_to_db(conn, realtime_data, TABLE_NAME)
        
        # Fetch and display the most recent data from the database
        historical_data = fetch_historical_data(conn, TABLE_NAME)
        print(historical_data)
        
        # Close the database connection
        close_db_connection(conn)

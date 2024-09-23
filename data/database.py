import sqlite3
import pandas as pd
import logging

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
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        logging.info(f"Data fetched from table {table_name} successfully.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data from table {table_name}: {e}")
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
    
    # Create a database connection
    conn = create_db_connection(DB_FILE)
    
    if conn:
        # Example DataFrame
        data = {
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Open': [100, 105, 102],
            'Close': [105, 102, 108]
        }
        df = pd.DataFrame(data)
        
        # Store data to the database
        store_data_to_db(conn, df, TABLE_NAME)
        
        # Fetch data from the database
        historical_data = fetch_historical_data(conn, TABLE_NAME)
        print(historical_data)
        
        # Close the database connection
        close_db_connection(conn)

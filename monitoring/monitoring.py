#python -m monitoring.monitoring

import os
import logging
import pandas as pd
from datetime import datetime
from strategies.visualizations import generate_heatmap, plot_moving_averages

# Set up logging
LOG_FILE = "monitoring.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_data():
    """
    Mock function to simulate fetching data from a database.
    Replace with the actual database fetching logic.
    """
    # Example DataFrame for testing
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='D'),
        'close': pd.Series(range(200)).apply(lambda x: 100 + x * 0.5)
    }
    return pd.DataFrame(data)

def create_output_dir():
    """
    Create the output directory if it does not exist.
    """
    output_dir = "static"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
    return output_dir

def track_performance_metrics(df):
    """
    Track and log performance metrics of the fetched data.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing price data.
    """
    # Check if the DataFrame is valid
    if df is None or df.empty:
        logging.warning("Invalid or empty data provided for performance tracking.")
        return

    # Check for required columns
    required_columns = ['close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return

    # Log basic statistics
    logging.info(f"Data Summary:")
    logging.info(f"Mean Close: {df['close'].mean():.2f}")
    logging.info(f"Median Close: {df['close'].median():.2f}")
    logging.info(f"Standard Deviation: {df['close'].std():.2f}")
    logging.info(f"Min Close: {df['close'].min():.2f}")
    logging.info(f"Max Close: {df['close'].max():.2f}")

    # Normalize only numeric columns (excluding 'timestamp')
    df_numeric = df.select_dtypes(include=[float, int])
    normalized_df = (df_numeric - df_numeric.mean()) / df_numeric.std()
    
    # If you still want to work with the 'timestamp', you can handle it separately
    logging.info(f"Normalized Data Preview: \n{normalized_df.head()}")


def monitor_data():
    """
    Main monitoring function that fetches data, generates visualizations,
    and logs key metrics.
    """
    try:
        logging.info("Starting data monitoring process...")

        # Fetch data
        df = fetch_data()
        if df.empty:
            logging.warning("Fetched data is empty. Aborting visualization.")
            return

        logging.info("Data fetched successfully. Generating visualizations...")

        # Track performance metrics
        track_performance_metrics(df)

        # Generate a heatmap for correlations
        try:
            generate_heatmap(df, normalize=True, title="Price Correlation Heatmap")
            logging.info("Correlation heatmap generated successfully.")
        except Exception as e:
            logging.error(f"Error generating correlation heatmap: {e}")

        # Plot moving averages
        try:
            output_dir = create_output_dir()
            plot_moving_averages(df)
            logging.info(f"Moving averages plot saved to {output_dir}.")
        except Exception as e:
            logging.error(f"Error generating moving averages plot: {e}")

    except Exception as e:
        logging.error(f"Unexpected error during monitoring: {e}")
    finally:
        logging.info("Data monitoring process completed.")

if __name__ == "__main__":
    monitor_data()

#python -m strategies.visualizations

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
import sys
import ccxt

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Add project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data.database import (
    create_db_connection,
    fetch_historical_data,
    fetch_historical_data_from_exchange,
)
from data.fetch_data import fetch_data_from_bybit


def generate_heatmap(data, title="Correlation Heatmap", columns=None, normalize=False, annot=True, cmap="coolwarm"):
    """
    Generate a heatmap for visualizing correlations with enhanced features.

    :param data: DataFrame containing data to visualize.
    :param title: Title of the heatmap.
    :param columns: Specific columns to include (optional).
    :param normalize: If True, normalize the data before computing the correlation.
    :param annot: Whether to display annotations in the heatmap.
    :param cmap: Colormap to use for the heatmap (default is "coolwarm").
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    logging.info("Generating heatmap...")
    logging.debug(f"Data preview:\n{data.head()}")

    # Drop or fill NaNs to prevent issues with correlation calculations
    data = data.dropna()

    # If specific columns are provided, use only those for the heatmap
    if columns:
        data = data[columns]

    # Exclude non-numeric columns
    data_numeric = data.select_dtypes(include=[float, int])

    # Normalize data if required
    if normalize:
        data_numeric = (data_numeric - data_numeric.mean()) / data_numeric.std()

    # Calculate the correlation matrix
    correlation_matrix = data_numeric.corr()
    logging.debug(f"Correlation matrix:\n{correlation_matrix}")

    # Plot the heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        correlation_matrix,
        annot=annot,
        cmap=cmap,
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        center=0,
        fmt=".2f",
        cbar_kws={"shrink": 0.8, "aspect": 20},
    )
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('strategies/performance_heatmap.png')

def generate_performance_heatmap(data, title="Performance Heatmap", metrics=None, annot=True, cmap="viridis"):
    """
    Generate a heatmap for visualizing performance metrics of different assets.

    :param data: DataFrame containing performance metrics to visualize.
    :param title: Title of the heatmap.
    :param metrics: Specific metrics to include in the heatmap (optional).
    :param annot: Whether to display annotations in the heatmap.
    :param cmap: Colormap to use for the heatmap (default is "viridis").
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    logging.info("Generating performance heatmap...")
    logging.debug(f"Data preview:\n{data.head()}")

    # Drop or fill NaNs to prevent issues with the heatmap
    data = data.dropna()

    # If specific metrics are provided, use only those columns
    if metrics:
        data = data[metrics]

    # Exclude non-numeric columns
    data_numeric = data.select_dtypes(include=[float, int])

    if data_numeric.empty:
        raise ValueError("No numeric data available to generate the heatmap.")

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data_numeric,
        annot=annot,
        cmap=cmap,
        linewidths=0.5,
        fmt=".2f",
        cbar_kws={"shrink": 0.8, "aspect": 20},
    )
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_moving_averages(df, save_path="static/moving_averages.png"):
    """
    Plot 50-period and 200-period moving averages for the closing price.

    :param df: DataFrame containing historical price data with 'close' and 'timestamp' columns.
    :param save_path: Path to save the plot image.
    """
    if "close" not in df or "timestamp" not in df:
        raise ValueError("DataFrame must contain 'close' and 'timestamp' columns.")

    logging.info("Calculating moving averages...")
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["MA200"] = df["close"].rolling(window=200).mean()

    # Plot the moving averages
    logging.info("Plotting moving averages...")
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["close"], label="Close Price", alpha=0.5)
    plt.plot(df["timestamp"], df["MA50"], label="50-period Moving Average", alpha=0.75)
    plt.plot(df["timestamp"], df["MA200"], label="200-period Moving Average", alpha=0.75)
    plt.legend()
    plt.title("Moving Averages")
    plt.grid()

    # Handle absolute path with Flask static folder
    abs_save_path = os.path.abspath(save_path)
    logging.info(f"Saving plot to {abs_save_path}...")  # Debug log before saving
    plt.savefig(abs_save_path)
    plt.close()
    logging.info(f"Moving averages plot successfully saved to {abs_save_path}")

if __name__ == "__main__":
    try:
        # Example: Fetch data from Bybit using fetch_data_from_bybit
        logging.info("Fetching data from Bybit...")
        df = fetch_data_from_bybit(symbol="BTC/USDT:USDT", timeframe="1h", limit=500)

        if df.empty:
            logging.error("No data fetched. Exiting visualization script.")
            sys.exit(1)

        # Generate a correlation heatmap
        generate_heatmap(df, normalize=True)

        # Generate a performance heatmap (example use case)
        performance_metrics = ["open", "high", "low", "close", "volume"]  # Adjust as needed
        generate_performance_heatmap(df, metrics=performance_metrics)

        # Plot moving averages
        plot_moving_averages(df)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

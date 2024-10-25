# strategies/visualizations.py

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
    # Check if data is a valid DataFrame and print the first few rows for debugging
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    print("Data Preview:")
    print(data.head())

    # Drop or fill NaNs to prevent issues with correlation calculations
    data = data.dropna()  # Or use data.fillna(0) if you prefer to fill missing values

    # If specific columns are provided, use only those for the heatmap
    if columns:
        data = data[columns]

    # Normalize data if required
    if normalize:
        data = (data - data.mean()) / data.std()  # Z-score normalization

    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Debugging: Print the correlation matrix to verify
    print("Correlation Matrix:")
    print(correlation_matrix)
    
    # Create the heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, linewidths=0.5, 
                vmin=-1, vmax=1, center=0, fmt=".2f", cbar_kws={'shrink': 0.8, 'aspect': 20})
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Ensure the plot is displayed
    plt.tight_layout()  # Adjust layout for better fitting
    plt.show()

def generate_performance_heatmap(data, metrics=None):
    """
    Generate a heatmap showing performance metrics across different assets or strategies.
    
    :param data: DataFrame containing performance data.
    :param metrics: Specific performance metrics to include (optional).
    """
    if metrics:
        data = data[metrics]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap="viridis", linewidths=0.5)
    plt.title("Performance Heatmap")
    plt.show()

def plot_moving_averages(df):
    """
    Plot 10-period and 50-period moving averages for the closing price.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing historical price data.
    """
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='Close Price', alpha=0.5)
    plt.plot(df['timestamp'], df['MA10'], label='10-period MA', alpha=0.75)
    plt.plot(df['timestamp'], df['MA50'], label='50-period MA', alpha=0.75)
    plt.legend()
    plt.title('Moving Averages')
    plt.show()

# Example usage for testing
if __name__ == "__main__":
    import sys
    import os
    # Add the root directory of the project to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from data.database import fetch_data  # Now it should be able to locate the module

    # Fetch real data from the database
    df = fetch_data()
    
    # Generate heatmap using the fetched data
    generate_heatmap(df, normalize=True)
    
    # Example of generating performance heatmap and moving averages
    # Assuming df contains the necessary columns
    # generate_performance_heatmap(df, metrics=['metric1', 'metric2'])  # Replace with actual metrics
    # plot_moving_averages(df)  # Ensure df has 'close' and 'timestamp' columns

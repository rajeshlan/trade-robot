from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import smtplib
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def track_performance_metrics(df):
    """
    Track and log basic performance metrics of the provided DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing historical price data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    if df.empty:
        logging.warning("DataFrame is empty. No metrics to track.")
        return
    
    # Calculate basic metrics
    mean_close = df['close'].mean()
    std_close = df['close'].std()
    moving_average_10 = df['close'].rolling(window=10).mean()
    moving_average_50 = df['close'].rolling(window=50).mean()
    
    # Log the metrics
    logging.info(f"Mean Close Price: {mean_close}")
    logging.info(f"Standard Deviation of Close Price: {std_close}")
    logging.info("10-period Moving Average of Close Price:")
    logging.info(moving_average_10.tail())
    logging.info("50-period Moving Average of Close Price:")
    logging.info(moving_average_50.tail())
    
    # Check for crossing moving averages (simple example of a trading signal)
    if moving_average_10.iloc[-1] > moving_average_50.iloc[-1]:
        send_notification("10-period moving average crossed above 50-period moving average.")
    elif moving_average_10.iloc[-1] < moving_average_50.iloc[-1]:
        send_notification("10-period moving average crossed below 50-period moving average.")

    # Predict future prices
    predict_future_prices(df['close'], periods=5)

def predict_future_prices(close_prices, periods):
    """
    Predict future closing prices using linear regression and determine trend duration.
    
    Parameters:
    - close_prices (pd.Series): Series of closing prices.
    - periods (int): Number of future periods to predict.
    """
    # Prepare data for linear regression
    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = close_prices.values.reshape(-1, 1)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_X = np.arange(len(close_prices), len(close_prices) + periods).reshape(-1, 1)
    predicted_prices = model.predict(future_X)

    # Calculate trend direction
    trend_direction = "unknown"
    if predicted_prices[-1] > predicted_prices[0]:
        trend_direction = "upward"
    elif predicted_prices[-1] < predicted_prices[0]:
        trend_direction = "downward"

    # Calculate mean predicted price
    predicted_mean = np.mean(predicted_prices)
    
    logging.info(f"Predicted Mean Closing Price: {predicted_mean:.2f}")
    logging.info(f"Predicted price indicates a {trend_direction} trend.")

    # Log detailed predictions
    logging.info("Predicted future prices:")
    for idx, price in enumerate(predicted_prices):
        logging.info(f"Period {idx + 1}: {price[0]:.2f}")

    # Optionally, you can send notifications based on the trend direction
    if trend_direction == "upward":
        send_notification("Predicted trend is upward.")
    elif trend_direction == "downward":
        send_notification("Predicted trend is downward.")

def send_notification(message):
    """
    Send a notification with the provided message via email and Slack.
    
    Parameters:
    - message (str): The notification message to be sent.
    """
    # Email configuration
    sender_email = "your_email@example.com"
    receiver_email = "receiver_email@example.com"
    smtp_server = "smtp.example.com"
    smtp_port = 587
    smtp_user = "your_email@example.com"
    smtp_password = "your_password"

    # Slack webhook URL
    slack_webhook_url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Trading Bot Notification"
    msg.attach(MIMEText(message, 'plain'))

    try:
        # Connect to the server and send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.close()
        
        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error("Failed to send email: %s", str(e))

    slack_message = {
        "text": message
    }

    try:
        response = requests.post(slack_webhook_url, json=slack_message)
        response.raise_for_status()
        
        logging.info("Slack notification sent successfully.")
    except requests.exceptions.RequestException as e:
        logging.error("Failed to send Slack notification: %s", str(e))
    
    # Log the message as well
    logging.info(message)

# Example usage
if __name__ == "__main__":
    # Sample DataFrame for demonstration purposes
    data = {
        'timestamp': pd.date_range(start='2021-01-01', periods=100, freq='h'),
        'open': pd.Series(range(100)),
        'high': pd.Series(range(1, 101)),
        'low': pd.Series(range(100)),
        'close': pd.Series(range(1, 101)),
        'volume': pd.Series(range(100, 200))
    }
    df = pd.DataFrame(data)
    
    track_performance_metrics(df)

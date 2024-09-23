from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import smtplib
import pandas as pd
import requests

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

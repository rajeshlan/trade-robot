import logging
import os
import sys
import smtplib
from email.mime.text import MIMEText
import pandas as pd
import ccxt
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv(dotenv_path=r'F:\trading\improvised-code-of-the-pdf-GPT-main\API.env')

# Debugging output
logging.info("Loaded API_KEY_1: %s", os.getenv('BYBIT_API_KEY_1'))
logging.info("Loaded API_KEY_2: %s", os.getenv('BYBIT_API_KEY_2'))
logging.info("SMTP_SERVER: %s", os.getenv('SMTP_SERVER'))
logging.info("SMTP_PORT: %s", os.getenv('SMTP_PORT'))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def send_email_notification(subject, message):
    """
    Send an email notification.
    """
    sender = os.getenv('EMAIL_USER')  # Fetch from environment
    receiver = os.getenv('EMAIL_USER')  # Send to yourself for testing
    password = os.getenv('EMAIL_PASS')  # Fetch from environment

    if not all([sender, receiver, password]):
        logging.error("Email configuration is incomplete. Please check your environment variables.")
        return

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP(os.getenv('SMTP_SERVER'), int(os.getenv('SMTP_PORT'))) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
            logging.info("Email notification sent successfully")
    except Exception as e:
        logging.error("Failed to send email notification: %s", e)



def send_notification(message):
    """
    Log a message and send an email notification.
    """
    logging.info(message)
    send_email_notification("Trading Bot Notification", message)


def load_api_credentials(index):
    """
    Load API credentials based on the given index from environment variables.
    """
    api_key = os.getenv(f'BYBIT_API_KEY_{index}')
    api_secret = os.getenv(f'BYBIT_API_SECRET_{index}')
    
    if api_key is not None and api_secret is not None:
        logging.info(f"Loaded API credentials for set {index}: API_KEY={api_key[:4]}****, API_SECRET={api_secret[:4]}****")
    else:
        logging.warning(f"API credentials for set {index} are missing or incomplete.")
    
    return api_key, api_secret


def initialize_exchange(api_key, api_secret):
    """
    Initialize a Bybit exchange using API credentials.
    """
    if not api_key or not api_secret:
        error_message = "API key or secret is missing."
        logging.error(error_message)
        send_notification(error_message)
        return None
    
    try:
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # This helps to avoid rate limit errors
        })
        logging.info("Initialized Bybit exchange")
        send_notification(f"Initialized Bybit exchange with API key: {api_key[:4]}****")
        return exchange
    except (ccxt.AuthenticationError, ccxt.ExchangeError, ccxt.NetworkError) as error:
        error_message = f"An error occurred with Bybit: {error}"
        logging.error(error_message)
        send_notification(error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)
        send_notification(error_message)
        raise e
    return None


def initialize_multiple_exchanges():
    """
    Initialize multiple Bybit exchanges using different API keys and secrets.
    """
    exchanges = []
    for i in range(1, 3):  # Assuming two sets of API keys and secrets
        api_key, api_secret = load_api_credentials(i)
        if api_key and api_secret:
            exchange = initialize_exchange(api_key, api_secret)
            if exchange:
                exchanges.append(exchange)
        else:
            error_message = f"API key or secret is missing for set {i}."
            logging.error(error_message)
            send_notification(error_message)
    return exchanges


def track_performance_metrics(df):
    """
    Track performance metrics from a given DataFrame containing trading data.
    """
    if df.empty:
        logging.warning("DataFrame is empty. No metrics to track.")
        return
    
    mean_close = df['close'].mean()
    std_close = df['close'].std()
    moving_average_10 = df['close'].rolling(window=10).mean()
    moving_average_50 = df['close'].rolling(window=50).mean()
    
    logging.info(f"Mean Close Price: {mean_close}")
    logging.info(f"Standard Deviation of Close Price: {std_close}")
    logging.info("10-period Moving Average of Close Price:")
    logging.info(moving_average_10.tail())
    logging.info("50-period Moving Average of Close Price:")
    logging.info(moving_average_50.tail())
    
    if moving_average_10.iloc[-1] > moving_average_50.iloc[-1]:
        send_notification("10-period moving average crossed above 50-period moving average.")
    elif moving_average_10.iloc[-1] < moving_average_50.iloc[-1]:
        send_notification("10-period moving average crossed below 50-period moving average.")


if __name__ == "__main__":
    exchanges = initialize_multiple_exchanges()
    if exchanges:
        success_message = "Successfully initialized all exchanges."
        logging.info(success_message)
        send_notification(success_message)
    else:
        error_message = "Failed to initialize exchanges."
        logging.error(error_message)
        send_notification(error_message)

    # Example data for tracking performance metrics
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

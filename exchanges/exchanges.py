import logging
import os
import ccxt
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=r'C:\Users\rajes\Desktop\improvised-code-of-the-pdf-GPT-main\API.env')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def send_email_notification(subject, message):
    import smtplib
    from email.mime.text import MIMEText

    sender = "your_email@example.com"
    receiver = "receiver_email@example.com"
    password = "your_email_password"

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP("smtp.example.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
            logging.info("Email notification sent successfully")
    except Exception as e:
        logging.error("Failed to send email notification: %s", e)


def send_notification(message):
    logging.info(message)
    send_email_notification("Trading Bot Notification", message)

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
    except ccxt.AuthenticationError as auth_error:
        error_message = f"Authentication failed with Bybit: {auth_error}"
        logging.error(error_message)
        send_notification(error_message)
    except ccxt.ExchangeError as exchange_error:
        error_message = f"Exchange error with Bybit: {exchange_error}"
        logging.error(error_message)
        send_notification(error_message)
    except ccxt.NetworkError as net_error:
        error_message = f"A network error occurred with Bybit: {net_error}"
        logging.error(error_message)
        send_notification(error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)
        send_notification(error_message)
        raise e
    return None

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

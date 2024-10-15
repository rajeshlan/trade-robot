import ccxt
import pandas as pd
import logging
import os
import time
import ntplib
import ta
import joblib
import pandas_ta as ta  # Correct import for pandas_ta
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables for API credentials
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')

def synchronize_system_time():
    """
    Synchronize system time with an NTP server.
    """
    try:
        response = ntplib.NTPClient().request('pool.ntp.org')
        current_time = datetime.fromtimestamp(response.tx_time)
        logging.info(f"System time synchronized: {current_time}")
        return int((current_time - datetime.utcnow()).total_seconds() * 1000)  # Return time offset in milliseconds
    except Exception as e:
        logging.error("Time synchronization failed: %s", e)
        return 0  # Return zero offset in case of failure

def initialize_exchange(api_key, api_secret):
    """
    Initialize the Bybit exchange.
    """
    try:
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        logging.info("Initialized Bybit exchange")
        return exchange
    except ccxt.BaseError as e:
        logging.error("Failed to initialize exchange: %s", e)
        raise e

def fetch_ohlcv(exchange, symbol, timeframe='1h', limit=100, time_offset=0):
    """
    Fetch OHLCV data from the exchange.
    """
    params = {
        'recvWindow': 10000,
        'timestamp': int(time.time() * 1000 + time_offset)
    }
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Fetched OHLCV data for {symbol}")
        return df
    except ccxt.BaseError as e:
        logging.error("Error fetching OHLCV data: %s", e)
        raise e

def calculate_indicators(df):
    """
    Calculate technical indicators using pandas_ta library.
    """
    try:
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)
        df['EMA_12'] = ta.ema(df['close'], length=12)
        df['EMA_26'] = ta.ema(df['close'], length=26)
        macd = ta.macd(df['close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['RSI'] = ta.rsi(df['close'], length=14)
        df.fillna(0, inplace=True)  # Fill NaN values with 0
        logging.info("Calculated technical indicators")
    except Exception as e:
        logging.error("Error calculating indicators: %s", e)
        raise e
    return df

def trading_strategy(df, sma_short=50, sma_long=200):
    """
    Define the trading strategy based on SMA crossover.
    """
    signals = ['hold']  # Initialize with 'hold' for the first entry
    for i in range(1, len(df)):
        if (df['SMA_' + str(sma_short)][i] > df['SMA_' + str(sma_long)][i]) and (df['SMA_' + str(sma_short)][i-1] <= df['SMA_' + str(sma_long)][i-1]):
            signals.append('buy')
        elif (df['SMA_' + str(sma_short)][i] < df['SMA_' + str(sma_long)][i]) and (df['SMA_' + str(sma_short)][i-1] >= df['SMA_' + str(sma_long)][i-1]):
            signals.append('sell')
        else:
            signals.append('hold')
    df['signal'] = signals
    logging.info("Defined trading strategy")
    return df

def fetch_balance(exchange):
    """
    Fetch the account balance.
    """
    try:
        balance = exchange.fetch_balance()
        logging.info(f"Fetched account balance: {balance}")
        return balance
    except ccxt.BaseError as e:
        logging.error("Error fetching balance: %s", e)
        raise e

def execute_trade(exchange, symbol, signal, available_usd, btc_price):
    """
    Execute trades based on signals.
    """
    try:
        if signal == 'buy':
            logging.info("Executing Buy Order")
            amount = available_usd / btc_price  # Calculate amount of BTC to buy with available USD
            exchange.create_market_buy_order(symbol, amount)
        elif signal == 'sell':
            logging.info("Executing Sell Order")
            exchange.create_market_sell_order(symbol, available_usd / btc_price)  # Use the same amount logic for sell
    except ccxt.BaseError as e:
        logging.error(f"Error executing {signal} order: {e}")
        raise e
    
def load_trained_model(model_path):
    """
    Load pre-trained ML model for price prediction or trade decision-making.
    """
    try:
        model = joblib.load(model_path)
        logging.info("Loaded trained model successfully")
        return model
    except Exception as e:
        logging.error("Error loading model: %s", e)
        raise e

def predict_signal(model, df):
    """
    Use the pre-trained model to predict the next signal (buy/sell/hold).
    """
    try:
        features = df[['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal']].values
        predictions = model.predict(features)
        df['predicted_signal'] = predictions
        logging.info("Predicted trading signals using AI/ML model")
    except Exception as e:
        logging.error("Error predicting signals: %s", e)
        raise e
    return df


def main():
    try:
        # Synchronize time with NTP server
        time_offset = synchronize_system_time()
        logging.info("Time synchronized with offset: %d", time_offset)
        
        # Initialize exchange
        exchange = initialize_exchange(API_KEY, API_SECRET)
        
        # Fetch OHLCV data
        df = fetch_ohlcv(exchange, 'BTCUSDT', time_offset=time_offset)
        
        # Calculate technical indicators
        df = calculate_indicators(df)
        
        # Define trading strategy
        df = trading_strategy(df)
        
        # Fetch account balance
        balance = fetch_balance(exchange)
        available_usd = balance['total']['USDT']  # Adjust based on your currency and balance structure
        btc_price = exchange.fetch_ticker('BTCUSDT')['last']
        
        # Execute trades based on signals
        for _, row in df.iterrows():
            execute_trade(exchange, 'BTCUSDT', row['signal'], available_usd, btc_price)
        
        # Output the resulting DataFrame
        print("Execution completed. Here are the last few rows of the DataFrame:")
        print(df.tail())
        
    except Exception as e:
        logging.error("An error occurred during the main execution: %s", e)

if __name__ == "__main__":
    main()
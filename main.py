#python main.py | Tee-Object -Append -FilePath trading_bot.log 

import logging
import sys
import os
import traceback
import ccxt
import numpy as np
from retrying import retry
import pandas as pd
import time
from datetime import datetime
import ta
from data.database import create_db_connection, fetch_historical_data, store_data_to_db
import exchanges
from data.fetch_data import fetch_data_from_bybit, get_tweets, analyze_sentiment
from risk_management.risk_management import (
    calculate_technical_indicators,
    calculate_rsi,
    prepare_data_for_training,
    train_model,
    calculate_stop_loss,
    calculate_take_profit,
    calculate_position_size
)
from strategies.trading_strategy import build_and_train_model, predict_prices, train_rl_model, rl_trading_decision, execute_trade
from trading.portfolio_management import calculate_returns, optimize_portfolio
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading'))



# Set up logging for debugging and tracking
def setup_logging():
    log_filename = "trading_bot.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w'),  # 'w' mode overwrites the file
            logging.StreamHandler(sys.stdout)   # Logs to terminal
        ]
    )
    sys.stdout.flush()
    logging.info(f"Logging started. Writing logs to {log_filename}")



# Load API credentials from environment variables
def load_api_credentials():
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    if not api_key or not api_secret:
        raise ValueError("BYBIT_API_KEY or BYBIT_API_SECRET environment variables are not set.")
    return api_key, api_secret

# Initialize the Bybit exchange using ccxt
def initialize_exchange(api_key, api_secret):
    exchange = ccxt.bybit({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
    })
    logging.info("Initialized Bybit exchange")
    return exchange

# Fetch OHLCV data with retry logic for reliability
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_ohlcv_with_retry(exchange, symbol, timeframe='1h', limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Fetched OHLCV data for {symbol}")
        return df
    except ccxt.BaseError as e:
        logging.error("Error fetching OHLCV data: %s", e)
        raise e

# Calculate technical indicators using the ta library
def calculate_indicators(df):
    """
    Calculate key technical indicators for trading.

    Parameters:
    df (DataFrame): Input dataframe with 'close' price data.

    Returns:
    DataFrame: Updated dataframe with calculated indicators.
    """
    try:
        df = df.copy()  # Prevent modifying the original DataFrame

        # Debug: Print initial columns
        logging.info(f"Initial DataFrame columns: {df.columns}")

        # Drop 'timestamp' column if it exists
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
            logging.info("Dropped 'timestamp' column.")

        # Ensure the index is reset to avoid DatetimeIndex issues
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=True)
            logging.info("Reset index to remove DatetimeIndex.")

        # Debug: Print columns after cleanup
        logging.info(f"Columns after preprocessing: {df.columns}")

        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Debug: Check if any non-numeric values remain
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            logging.warning(f"Non-numeric columns found after conversion: {non_numeric_cols}")

        # Calculate Moving Averages
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)

        # Compute MACD and RSI
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)

        logging.info("Technical indicators calculated successfully.")
        return df

    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        raise

# Define a simple moving average crossover trading strategy
def trading_strategy(df, sma_short=50, sma_long=200):
    try:
        signals = ['hold']
        for i in range(1, len(df)):
            if df['SMA_' + str(sma_short)].iloc[i] > df['SMA_' + str(sma_long)].iloc[i] and df['SMA_' + str(sma_short)].iloc[i-1] <= df['SMA_' + str(sma_long)].iloc[i-1]:
                signals.append('buy')
            elif df['SMA_' + str(sma_short)].iloc[i] < df['SMA_' + str(sma_long)].iloc[i] and df['SMA_' + str(sma_short)].iloc[i-1] >= df['SMA_' + str(sma_long)].iloc[i-1]:
                signals.append('sell')
            else:
                signals.append('hold')
        df['signal'] = signals
        logging.info("Defined trading strategy")
        return df
    except KeyError as e:
        logging.error("Error detecting signals: %s", e)
        raise e

# Execute trades based on signals using ccxt
def execute_trade(exchange, symbol, signal, amount=1):
    try:
        if signal == 'buy':
            logging.info("Executing Buy Order")
            exchange.create_market_buy_order(symbol, amount)
        elif signal == 'sell':
            logging.info("Executing Sell Order")
            exchange.create_market_sell_order(symbol, amount)
    except ccxt.BaseError as e:
        logging.error(f"Error executing {signal} order: %s", e)
        raise e

# Perform backtesting with historical data
def perform_backtesting(exchange):
    try:
        df = fetch_ohlcv_with_retry(exchange, 'BTCUSDT', timeframe='1h', limit=500)
        df = calculate_indicators(df)
        df = trading_strategy(df)
        logging.info("Backtesting complete")
        print(df.tail())
    except Exception as e:
        logging.error("Error during backtesting: %s", e)

# Detect trading patterns in the data
def detect_patterns(data):
    """
    Detects key trading patterns in historical price data.
    Patterns: Moving Average Crossovers, Double Top/Bottom, Support/Resistance, Breakouts
    """
    required_columns = {'close', 'SMA_50', 'SMA_200', 'high', 'low', 'volume'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(data.columns)}")

    data['Golden_Cross'] = (data['SMA_50'] > data['SMA_200']) & (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))
    data['Death_Cross'] = (data['SMA_50'] < data['SMA_200']) & (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))
    data['Double_Top'] = (data['high'].shift(2) < data['high'].shift(1)) & (data['high'].shift(1) > data['high']) & (data['high'].shift(1) > data['high'].shift(3))
    data['Double_Bottom'] = (data['low'].shift(2) > data['low'].shift(1)) & (data['low'].shift(1) < data['low']) & (data['low'].shift(1) < data['low'].shift(3))
    data['Support_Level'] = data['low'].rolling(window=20).min()
    data['Resistance_Level'] = data['high'].rolling(window=20).max()
    data['Breakout_Up'] = (data['close'] > data['Resistance_Level']) & (data['volume'] > data['volume'].rolling(20).mean() * 1.5)
    data['Breakout_Down'] = (data['close'] < data['Support_Level']) & (data['volume'] > data['volume'].rolling(20).mean() * 1.5)
    pattern_columns = ['Golden_Cross', 'Death_Cross', 'Double_Top', 'Double_Bottom', 'Breakout_Up', 'Breakout_Down']
    data[pattern_columns] = data[pattern_columns].fillna(False)
    return data

# Synchronize time with the exchange (placeholder)
def synchronize_time(exchange):
    """
    Synchronize local time with the exchange server time.
    Placeholder: assumes no offset; implement if needed.
    """
    time_offset = 0  # Replace with actual logic if required
    return time_offset

# Main execution logic
def main():
    # Set up logging
    logging.info("Main function started.")
    setup_logging()
    
    # Load API credentials
    api_key, api_secret = load_api_credentials()
    
    # Initialize exchange
    exchange = initialize_exchange(api_key, api_secret)
    
    # Synchronize time
    time_offset = synchronize_time(exchange)
    print(f"Time offset: {time_offset} seconds")
    
    try:
        # Fetch historical data
        df = fetch_data_from_bybit('BTC/USDT:USDT', timeframe='1h', limit=500)
        if not df.empty:
            print(f"Historical data:\n{df.head()}")
        
        # Calculate indicators
        df_with_indicators = calculate_indicators(df)
        print(f"Data with indicators:\n{df_with_indicators.head()}")

        # Detect patterns
        df_with_patterns = detect_patterns(df_with_indicators)
        print(df_with_patterns[['Golden_Cross', 'Death_Cross', 'Double_Top', 'Double_Bottom', 'Breakout_Up', 'Breakout_Down']].tail())

        # Fetch and analyze historical data
        # Ensure data is available in the database before fetching
        symbol = 'BTCUSDT'
        df = fetch_data_from_bybit(symbol, timeframe='1h', limit=500)

        if df.empty:
            raise ValueError(f"No historical data fetched for {symbol}.")

        # Store the fetched data into the database if the table does not exist
        conn = create_db_connection()
        store_data_to_db(conn, symbol, df)  # Ensure data is stored
        conn.close()

        # Now fetch data from the database after storing it
        data = fetch_historical_data(symbol)
        if data.empty:
            raise ValueError(f"No data found in {symbol} table after storage.")

        #Process the data
        data = calculate_technical_indicators(data)
        data = detect_patterns(data)


        data = calculate_technical_indicators(data)
        data = detect_patterns(data)
        
        # Risk management based on market conditions
        if data.iloc[-1]['SMA_50'] > data.iloc[-1]['SMA_200']:
            df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns
            stop_loss = calculate_stop_loss(data.iloc[-1]['close'], 1.0)
            take_profit = calculate_take_profit(data.iloc[-1]['close'], 1.0, stop_loss)
        elif data.iloc[-1]['SMA_50'] < data.iloc[-1]['SMA_200']:
            df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns
            stop_loss = calculate_stop_loss(data.iloc[-1]['close'], 1.0 )
            take_profit = calculate_take_profit(data.iloc[-1]['close'], 1.0, stop_loss)
        else:
            df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns
            stop_loss = calculate_stop_loss(data.iloc[-1]['close'], 1.0)
            take_profit = calculate_take_profit(data.iloc[-1]['close'], 1.0, stop_loss)

        # Fetch more data if needed
        df = fetch_data_from_bybit('BTC/USDT:USDT', timeframe='1h', limit=500)
        if not df.empty:
            print(f"Historical data:\n{df.head()}")

        # Train predictive model
        model, scaler = build_and_train_model(df)

        # Predict prices
        predicted_prices = predict_prices(model, scaler, df)

        # Sentiment analysis
        api_key = 'xx'  # Replace with your actual key
        api_secret = 'xx'  # Replace with your actual secret
        tweets = get_tweets(api_key, 'BTC')
        sentiment_score = analyze_sentiment(tweets)

        # Train RL model
        rl_model = train_rl_model(df)

        # Portfolio optimization
        returns, cov_matrix = calculate_returns(df)
        optimal_weights = optimize_portfolio(returns, cov_matrix)
        print(f"Optimal asset allocation: {optimal_weights}")

        # RL trading decision
        obs = df.iloc[-1].values
        rl_action = rl_trading_decision(rl_model, obs)

        # Sentiment-based decision
        if sentiment_score > 0.1:
            decision = 'buy'
        elif sentiment_score < -0.1:
            decision = 'sell'
        else:
            decision = 'hold'

        # Execute trading decision
        execute_trade(exchange, symbol, decision)

    except ccxt.AuthenticationError as e:
        logging.error("Authentication error: %s. Check your API key and secret.", e)
    except ccxt.NetworkError as e:
        logging.error("Network error: %s. Check your internet connection.", e)
    except ccxt.ExchangeError as e:
        logging.error("Exchange error: %s. Check exchange status or API docs.", e)
    except ValueError as e:
        logging.error("ValueError: %s", e)
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        logging.error(traceback.format_exc())  # Logs full traceback

# Entry point
if __name__ == "__main__":
    main()

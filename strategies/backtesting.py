import pickle
import time
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import logging
import os
import requests
from textblob import TextBlob
from fetch_data import fetch_ohlcv
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def synchronize_time_with_exchange(exchange):
    """Synchronize local time with the exchange time."""
    try:
        server_time = exchange.milliseconds()
        local_time = pd.Timestamp.now().timestamp() * 1000
        time_offset = server_time - local_time
        logging.info("Time synchronized with exchange. Offset: %d milliseconds", time_offset)
        return time_offset
    except ccxt.BaseError as sync_error:
        logging.error("Failed to synchronize time with exchange: %s", sync_error)
        raise sync_error

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def preprocess_data_for_lstm(df):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close', 'SMA_20', 'SMA_50', 'RSI_14']].values)

    # Prepare the dataset for LSTM
    X = []
    y = []
    look_back = 60
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])  # Predicting 'close' price

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(df):
    X, y, scaler = preprocess_data_for_lstm(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_lstm_model(X_train.shape)
    
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
    
    # Save model and scaler for future use
    model.save("lstm_model.h5")
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    logging.info("LSTM model trained and saved.")
    return model, scaler

def predict_with_lstm(df, model, scaler):
    # Use the model to predict future prices or signals
    X, _, _ = preprocess_data_for_lstm(df)
    predictions = model.predict(X)
    
    # Inverse scaling
    predicted_prices = scaler.inverse_transform(predictions)
    return predicted_prices


def fetch_data(exchange, symbol='BTCUSDT', timeframe='1h', limit=100):
    """Fetch historical OHLCV data from the exchange."""
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        try:
            time_offset = synchronize_time_with_exchange(exchange)
            params = {'recvWindow': 10000, 'timestamp': exchange.milliseconds() + time_offset}
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
        
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logging.info("Fetched OHLCV data for %s", symbol)
            return df
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BaseError) as error:
            attempts += 1
            logging.error(f"Error fetching data (Attempt {attempts}/{max_attempts}): {error}")
            time.sleep(2 ** attempts)
    raise Exception("Failed to fetch data after multiple attempts")


def calculate_indicators(df, sma_short=20, sma_long=50, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9):
    try:
        df.ta.sma(length=sma_short, append=True)
        df.ta.sma(length=sma_long, append=True)
        df.ta.rsi(length=rsi_period, append=True)
        df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
        df.ta.bbands(length=20, std=2, append=True)  # Bollinger Bands
        df.ta.atr(length=14, append=True)  # Average True Range
        df['Volume_MA_20'] = df['volume'].rolling(window=20).mean()
        logging.info("Calculated Volume Moving Average")
        logging.info("Calculated SMA, RSI, MACD, Bollinger Bands, and ATR indicators")
        return df
    except Exception as e:
        logging.error("Error during technical analysis: %s", e)
        raise e

def fetch_multiple_timeframes(exchange, symbol, timeframes):
    df_list = []
    for timeframe in timeframes:
        df = fetch_data(exchange, symbol, timeframe)
        df['timeframe'] = timeframe
        df_list.append(df)
    combined_df = pd.concat(df_list)
    return combined_df

def detect_signals(df, model=None, scaler=None, sma_short=20, sma_long=50, rsi_overbought=70, rsi_oversold=30):
    """Detect trading signals using LSTM predictions, sentiment analysis, technical indicators, and volume."""
    try:
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # Ensure LSTM predictions are used if the model and scaler are provided
        lstm_predictions = predict_with_lstm(df, model, scaler) if model and scaler else None

        # Sentiment Analysis
        sentiment_score = fetch_market_sentiment_data()

        # Check if required columns are available
        if f'SMA_{sma_short}' not in df.columns or f'SMA_{sma_long}' not in df.columns:
            raise ValueError("Required SMA columns are not present in the DataFrame")
        if 'RSI_14' not in df.columns:
            raise ValueError("RSI column is not present in the DataFrame")

        # Fill NaN values with previous values
        df.fillna(method='ffill', inplace=True)

        # Debug prints for indicators
        print(f"Latest SMA_{sma_short}: {latest[f'SMA_{sma_short}']}, SMA_{sma_long}: {latest[f'SMA_{sma_long}']}, RSI_14: {latest['RSI_14']}")
        print(f"Previous SMA_{sma_short}: {previous[f'SMA_{sma_short}']}, SMA_{sma_long}: {previous[f'SMA_{sma_long}']}, RSI_14: {previous['RSI_14']}")

        # Simple Moving Average (SMA) Crossover
        if previous[f'SMA_{sma_short}'] < previous[f'SMA_{sma_long}'] and latest[f'SMA_{sma_short}'] > latest[f'SMA_{sma_long}']:
            print("Buy signal detected based on SMA crossover")
            return 'buy'
        elif previous[f'SMA_{sma_short}'] > previous[f'SMA_{sma_long}'] and latest[f'SMA_{sma_short}'] < latest[f'SMA_{sma_long}']:
            print("Sell signal detected based on SMA crossover")
            return 'sell'

        # Relative Strength Index (RSI)
        if latest['RSI_14'] > rsi_overbought:
            print("Sell signal detected based on RSI overbought")
            return 'sell'
        elif latest['RSI_14'] < rsi_oversold:
            print("Buy signal detected based on RSI oversold")
            return 'buy'

        # Volume-based analysis
        if latest['volume'] > latest['Volume_MA_20']:
            logging.info("High volume detected")
            return 'hold'

        # Combine LSTM prediction, sentiment, and technical analysis
        if lstm_predictions and sentiment_score > 0.5 and lstm_predictions[-1] > latest['close'] and latest[f'SMA_{sma_short}'] > latest[f'SMA_{sma_long}']:
            logging.info("LSTM predicts price increase with positive sentiment, buy signal")
            return 'buy'
        elif lstm_predictions and sentiment_score < -0.5 and lstm_predictions[-1] < latest['close'] and latest[f'SMA_{sma_short}'] < latest[f'SMA_{sma_long}']:
            logging.info("LSTM predicts price decrease with negative sentiment, sell signal")
            return 'sell'

        # Sentiment-only signals
        if sentiment_score > 0.5:
            logging.info("Positive market sentiment detected")
            if latest[f'SMA_{sma_short}'] > latest[f'SMA_{sma_long}'] and latest['RSI_14'] < rsi_overbought:
                return 'buy'
        elif sentiment_score < -0.5:
            logging.info("Negative market sentiment detected")
            if latest[f'SMA_{sma_short}'] < latest[f'SMA_{sma_long}'] and latest['RSI_14'] > rsi_oversold:
                return 'sell'

        return 'hold'
    except Exception as e:
        logging.error("Error detecting signals: %s", e)
        raise e

def adaptive_risk_management(df, risk_tolerance=2):
    # Dynamically adjust stop loss and take profit based on volatility or other factors
    current_atr = df.iloc[-1]['ATR_14']  # Average True Range as a volatility measure
    stop_loss_pct = risk_tolerance * current_atr
    take_profit_pct = risk_tolerance * current_atr * 1.5  # A simple ratio
    
    logging.info(f"Adaptive Stop Loss: {stop_loss_pct:.2f}, Take Profit: {take_profit_pct:.2f}")
    return stop_loss_pct, take_profit_pct



def fetch_real_time_balance(exchange, currency='USDT'):
    """Fetch real-time balance from the exchange."""
    try:
        balance = exchange.fetch_balance()
        real_time_balance = balance['total'][currency]
        logging.info("Fetched real-time balance: %.2f %s", real_time_balance, currency)
        return real_time_balance
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BaseError) as error:
        logging.error("Error fetching real-time balance: %s", error)
        raise error

def analyze_market_sentiment():
    # Placeholder function for market sentiment analysis
    sentiment_score = fetch_market_sentiment_data()  # Fetch sentiment data from an external source
    logging.info(f"Market sentiment score: {sentiment_score}")
    return sentiment_score

def fetch_market_sentiment_data():
    # Example: Fetch news or social media data from an external API
    # Let's assume we fetch some tweets or news headlines about the cryptocurrency market
    try:
        response = requests.get('https://newsapi.org/v2/everything?q=cryptocurrency&apiKey=YOUR_NEWS_API_KEY')
        data = response.json()
        headlines = [article['title'] for article in data['articles']]
        
        # Sentiment analysis
        sentiment_score = sum([TextBlob(headline).sentiment.polarity for headline in headlines]) / len(headlines)
        return sentiment_score
    except Exception as e:
        logging.error("Error fetching market sentiment data: %s", e)
        return 0

def backtest_strategy(df, initial_capital=1000, position_size=1, transaction_cost=0.001, stop_loss_pct=0.02, take_profit_pct=0.05):
    try:
        df['signal'] = df.apply(lambda row: detect_signals(df), axis=1)
        df['position'] = 0  # 1 for long, -1 for short, 0 for no position
        df['capital'] = initial_capital
        df['balance'] = initial_capital
        current_position = 0
        entry_price = 0

        for i in range(1, len(df)):
            if df.at[i, 'signal'] == 'buy' and current_position == 0:
                current_position = position_size
                entry_price = df.at[i, 'close']
                df.at[i, 'position'] = current_position
                df.at[i, 'capital'] -= df.at[i, 'close'] * position_size * (1 + transaction_cost)  # Deduct cost of buying including transaction cost
                df.at[i, 'balance'] = df.at[i - 1, 'balance'] - df.at[i, 'close'] * position_size * (1 + transaction_cost)

            elif df.at[i, 'signal'] == 'sell' and current_position == position_size:
                current_position = 0
                df.at[i, 'position'] = current_position
                df.at[i, 'capital'] += df.at[i, 'close'] * position_size * (1 - transaction_cost)  # Add profit from selling minus transaction cost
                df.at[i, 'balance'] = df.at[i - 1, 'balance'] + df.at[i, 'close'] * position_size * (1 - transaction_cost)

            if current_position == position_size:
                if df.at[i, 'close'] <= entry_price * (1 - stop_loss_pct):
                    current_position = 0
                    df.at[i, 'position'] = current_position
                    df.at[i, 'capital'] += df.at[i, 'close'] * position_size * (1 - transaction_cost)
                    df.at[i, 'balance'] = df.at[i - 1, 'balance'] + df.at[i, 'close'] * position_size * (1 - transaction_cost)
                    logging.info("Stop-loss triggered")

                if df.at[i, 'close'] >= entry_price * (1 + take_profit_pct):
                    current_position = 0
                    df.at[i, 'position'] = current_position
                    df.at[i, 'capital'] += df.at[i, 'close'] * position_size * (1 - transaction_cost)
                    df.at[i, 'balance'] = df.at[i - 1, 'balance'] + df.at[i, 'close'] * position_size * (1 - transaction_cost)
                    logging.info("Take-profit triggered")

        final_balance = df.iloc[-1]['balance']
        logging.info("Backtesting completed. Final balance: %.2f", final_balance)
        return df, final_balance
    except Exception as e:
        logging.error("Error during backtesting: %s", e)
        raise e

def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss_price):
    risk_amount = capital * (risk_per_trade / 100)
    position_size = risk_amount / abs(entry_price - stop_loss_price)
    return position_size

def perform_backtesting(exchange):
    """
    Perform backtesting on BTCUSDT pair using the provided exchange.
    """
    try:
        # Fetch historical data
        df = fetch_ohlcv(exchange, 'BTCUSDT', timeframe='1h', limit=500)

        # Calculate indicators
        df = calculate_indicators(df)

        # Run backtest
        backtest_df, final_capital = backtest_strategy(df)

        # Output results
        print(backtest_df.tail())
        logging.info("Final capital after backtesting: %.2f", final_capital)
    except Exception as e:
        logging.error("Error during backtesting: %s", e)

def main():
    """Main function to run the enhanced backtesting with adaptive risk management."""
    try:
        # Initialize the exchange
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        
        # Fetch real-time balance
        real_time_balance = fetch_real_time_balance(exchange, currency='USDT')
        
        # Fetch historical data
        df = fetch_data(exchange, symbol='BTCUSDT', timeframe='1h', limit=500)
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Train or load LSTM model
        model, scaler = train_lstm_model(df)
        
        # Run backtest with adaptive risk management
        stop_loss_pct, take_profit_pct = adaptive_risk_management(df)
        backtest_df, final_capital = backtest_strategy(df, initial_capital=real_time_balance, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
        
        # Output results
        print(backtest_df.tail())
        logging.info("Final capital after backtesting: %.2f", final_capital)
    except Exception as e:
        logging.error("An error occurred in the main function: %s", e)

if __name__ == "__main__":
    main()


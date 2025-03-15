#python trading\Placing_Orders.py 

import ccxt
import numpy as np
import pandas as pd
import logging
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv(dotenv_path=r'D:\\RAJESH FOLDER\\PROJECTS\\trade-robot\\config\\API.env')

def initialize_exchange(api_key: str, api_secret: str) -> ccxt.Exchange:
    """
    Initialize the exchange with API key and secret.
    """
    try:
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })

        # Test connection
        try:
            balance = exchange.fetch_balance()
            logging.info(f"Connected to Bybit. Balance: {balance}")
        except Exception as e:
            logging.error(f"Error connecting to Bybit API: {e}")
            raise  # Ensure failure stops execution

        return exchange  # ✅ Fix: Ensure exchange is returned!

    except Exception as e:
        logging.error(f"Error initializing exchange: {e}")
        raise  # Stop execution on failure


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
    """
    Fetch OHLCV data.
    """
    try:
        params = {'category': 'linear'}  # Increase recv_window to 30 seconds
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info("Fetched OHLCV data")
        return df
    except ccxt.BaseError as e:
        logging.error("Failed to fetch OHLCV data: %s", e)
        raise

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for trading strategy.
    """
    try:
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()
        logging.info("Calculated SMA_10 and SMA_30")
        return df
    except Exception as e:
        logging.error("Failed to calculate technical indicators: %s", e)
        raise

class LSTMModel(nn.Module):
    """
    PyTorch implementation of an LSTM model for time series prediction.
    """
    def __init__(self, input_size: int, hidden_layer_size: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

def train_lstm(df: pd.DataFrame) -> tuple:
    """
    Train the LSTM model on the historical data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close']].values)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X = torch.tensor(np.array(X)).float().view(-1, 60, 1)
    y = torch.tensor(y).float().view(-1, 1)

    model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        logging.info(f"Epoch {epoch+1}/100, Loss: {loss.item()}")
    
    logging.info("Training completed")
    return model, scaler

def predict_price(df: pd.DataFrame, model, scaler) -> float:
    """
    Predict the price using the trained LSTM model.
    """
    scaled_data = scaler.transform(df[['close']].values)
    input_data = torch.tensor(scaled_data[-60:]).float().view(1, 60, 1)
    
    model.eval()
    with torch.no_grad():
        prediction = model(input_data)
    predicted_price = scaler.inverse_transform(prediction.cpu().numpy())
    
    logging.info(f"Predicted next price: {predicted_price[0][0]}")
    return predicted_price[0][0]

def define_trading_strategy(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    try:
        signals = ['hold']
        predicted_prices = []

        for i in range(1, len(df)):
            if i > 60:
                predicted_price = predict_price(df.iloc[:i], model, scaler)
                predicted_prices.append(predicted_price)

                if df['SMA_10'][i] > df['SMA_30'][i] and predicted_price > df['close'][i]:
                    signals.append('buy')
                elif df['SMA_10'][i] < df['SMA_30'][i] and predicted_price < df['close'][i]:
                    signals.append('sell')
                else:
                    signals.append('hold')
            else:
                signals.append('hold')

        df['signal'] = signals
        df['predicted_price'] = predicted_prices + [None] * (len(df) - len(predicted_prices))

        logging.info("Applied trading strategy with price prediction")
        return df
    except Exception as e:
        logging.error("Failed to define trading strategy: %s", e)
        raise

def manage_leverage(account_balance, risk_level, max_leverage, current_position):
    """
    Manage leverage dynamically based on account balance and risk level.
    """
    try:
        leverage = account_balance * risk_level
        leverage = min(leverage, max_leverage)

        if current_position == 'long':
            leverage *= 1.1
        elif current_position == 'short':
            leverage *= 0.9

        logging.info(f"Calculated leverage: {leverage}")
        return leverage
    except Exception as e:
        logging.error("Failed to manage leverage: %s", e)
        raise

def place_order(exchange: ccxt.Exchange, symbol: str, order_type: str, side: str, amount: float, price=None):
    try:
        try:
            market = exchange.market(symbol)
            min_amount = market['limits']['amount']['min']
            if amount < min_amount:
                raise ValueError(f"Amount {amount} is less than the minimum allowed {min_amount}")
        except Exception as e:
            logging.error("An error occurred while checking the minimum amount: %s", e)
            raise
        if order_type == 'market':
            order = exchange.create_market_order(symbol, side, amount)
        elif order_type == 'limit':
            order = exchange.create_limit_order(symbol, side, amount, price)
        logging.info("Placed %s order for %s %s at %s", side, amount, symbol, price if price else 'market price')
        return order
    except ValueError as ve:
        logging.error(ve)
    except ccxt.InsufficientFunds as insf:
        logging.error("Insufficient funds: %s", insf)
    except ccxt.InvalidOrder as invord:
        logging.error("Invalid order: %s", invord)
    except ccxt.NetworkError as neterr:
        logging.error("Network error: %s", neterr)
    except ccxt.BaseError as e:
        logging.error("An error occurred: %s", e)

def execute_trading_strategy(exchange: ccxt.Exchange, df: pd.DataFrame, symbol: str, amount: float, risk_percent: float):
    try:
        for i in range(len(df)):
            if df['signal'][i] in ['buy', 'sell']:
                account_balance = exchange.fetch_balance()['total']['BTC']
                leverage = manage_leverage(account_balance, risk_percent, max_leverage=10, current_position='long')

                if df['signal'][i] == 'buy':
                    place_order(exchange, symbol, 'market', 'buy', amount)
                elif df['signal'][i] == 'sell':
                    place_order(exchange, symbol, 'market', 'sell', amount)
    except ccxt.BaseError as e:
        logging.error("An error occurred: %s", e)
        raise

def main():
    try:
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')

        if not api_key or not api_secret:
            raise ValueError("BYBIT_API_KEY or BYBIT_API_SECRET environment variables are not set.")

        account_balance = 1000  # Example account balance in USD
        risk_level = 0.02  # Example risk level as a percentage
        max_leverage = 10  # Example maximum leverage

        symbol = 'BTCUSD'
        amount = 1
        risk_percent = 0.02

        exchange = initialize_exchange(api_key, api_secret)

        df = fetch_ohlcv(exchange, symbol)
        df = calculate_technical_indicators(df)

        model, scaler = train_lstm(df)

        df = define_trading_strategy(df, model, scaler)

        execute_trading_strategy(exchange, df, symbol, amount, risk_percent)
    except Exception as e:
        logging.error("An error occurred in the main execution: %s", e)

if __name__ == '__main__':
    main()

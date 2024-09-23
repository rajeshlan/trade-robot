import ccxt
import pandas as pd
import logging
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv(dotenv_path='F:\trading\improvised-code-of-the-pdf-GPT-main/API.env')

def initialize_exchange(api_key: str, api_secret: str) -> ccxt.Exchange:
    """
    Initialize the exchange with API key and secret.
    """
    try:
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        logging.info("Initialized Bybit exchange")
        return exchange
    except Exception as e:
        logging.error("Failed to initialize exchange: %s", e)
        raise

def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
    """
    Fetch OHLCV data.
    """
    try:
        params = {'recvWindow': 30000}  # Increase recv_window to 30 seconds
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
        # Use smaller windows for SMA to ensure crossovers in the limited dataset
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()
        logging.info("Calculated SMA_10 and SMA_30")
        return df
    except Exception as e:
        logging.error("Failed to calculate technical indicators: %s", e)
        raise

def define_trading_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define the trading strategy based on technical indicators.
    """
    try:
        signals = ['hold']
        for i in range(1, len(df)):
            if df['SMA_10'][i] > df['SMA_30'][i] and df['SMA_10'][i-1] <= df['SMA_30'][i-1]:
                signals.append('buy')
            elif df['SMA_10'][i] < df['SMA_30'][i] and df['SMA_10'][i-1] >= df['SMA_30'][i-1]:
                signals.append('sell')
            else:
                signals.append('hold')
        df['signal'] = signals
        logging.info("Applied trading strategy")
        
        # Log the first few signals for debugging
        logging.info("First few signals: %s", df['signal'].head(10).tolist())
        return df
    except Exception as e:
        logging.error("Failed to define trading strategy: %s", e)
        raise

def place_order(exchange: ccxt.Exchange, symbol: str, order_type: str, side: str, amount: float, price=None):
    """
    Place an order on the exchange.
    """
    try:
        if order_type == 'market':
            order = exchange.create_market_order(symbol, side, amount)
        elif order_type == 'limit':
            order = exchange.create_limit_order(symbol, side, amount, price)
        logging.info("Placed %s order for %s %s at %s", side, amount, symbol, price if price else 'market price')
        return order
    except ccxt.InsufficientFunds as insf:
        logging.error("Insufficient funds: %s", insf)
    except ccxt.InvalidOrder as invord:
        logging.error("Invalid order: %s", invord)
    except ccxt.NetworkError as neterr:
        logging.error("Network error: %s", neterr)
    except ccxt.BaseError as e:
        logging.error("An error occurred: %s", e)

def manage_leverage(exchange: ccxt.Exchange, symbol: str, amount: float, risk_percent: float):
    """
    Dynamically manage leverage based on account balance and risk management.
    """
    try:
        balance = exchange.fetch_balance()
        available_margin = balance['total']['USDT']
        logging.info("Available margin: %s USDT", available_margin)
        
        # Calculate maximum allowable loss
        max_loss = available_margin * risk_percent
        
        # Fetch ticker and leverage limit
        ticker = exchange.fetch_ticker(symbol)
        max_leverage = exchange.markets[symbol]['limits']['leverage']['max'] if symbol in exchange.markets else 1
        
        # Calculate maximum leverage
        calculated_leverage = min(max_loss / (amount * ticker['last']), max_leverage)
        
        # Set the leverage on the exchange if within bounds
        if 1 <= calculated_leverage <= max_leverage:
            exchange.set_leverage(calculated_leverage, symbol)
            logging.info("Dynamically set leverage to %.2f for %s based on risk management", calculated_leverage, symbol)
        else:
            logging.warning("Calculated leverage %.2f is out of bounds for %s. Using default leverage of 1.", calculated_leverage, symbol)
            exchange.set_leverage(50, symbol)  # Set to default leverage of 1

        return calculated_leverage
    except KeyError as ke:
        logging.error("Symbol %s not found in markets: %s", symbol, ke)
        raise
    except ccxt.BaseError as e:
        logging.error("Failed to manage leverage: %s", e)
        raise

def place_order(exchange, symbol, order_type, side, amount, price=None):
    try:
        if order_type == 'limit':
            order = exchange.create_order(symbol, order_type, side, amount, price)
        elif order_type == 'market':
            order = exchange.create_order(symbol, order_type, side, amount)
        return order
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def place_order(exchange, side, price, symbol, amount):
    try:
        order = exchange.create_order(symbol, 'market', side, amount)
        logging.info(f"Placed {side} order for {amount} {symbol} at {price}")
        return order
    except ccxt.NetworkError as e:
        logging.error(f"Failed to place order: {e}")
        raise

def execute_trading_decision(signal):
    api_key = 'your_api_key'
    api_secret = 'your_api_secret'
    exchange = ccxt.bybit({'apiKey': api_key, 'secret': api_secret})

    if signal == 'buy':
        place_order(exchange, 'BTCUSDT', 'market', 'buy', 0.01)
    elif signal == 'sell':
        place_order(exchange, 'BTCUSDT', 'market', 'sell', 0.01)

def execute_trading_strategy(exchange: ccxt.Exchange, df: pd.DataFrame, symbol: str, amount: float, risk_percent: float):
    """
    Execute the trading strategy based on signals.
    """
    try:
        markets = exchange.load_markets()
        market = exchange.market(symbol)

        # Log market structure for debugging
        logging.info("Market structure for %s: %s", symbol, market)

        for i in range(len(df)):
            logging.info("Processing signal: %s at index %d", df['signal'][i], i)
            if df['signal'][i] in ['buy', 'sell']:
                # Dynamically manage leverage before placing an order
                manage_leverage(exchange, symbol, amount, risk_percent)
                
                if df['signal'][i] == 'buy':
                    logging.info("Buy Signal - Placing Buy Order")
                    place_order(exchange, symbol, 'market', 'buy', amount)
                elif df['signal'][i] == 'sell':
                    logging.info("Sell Signal - Placing Sell Order")
                    place_order(exchange, symbol, 'market', 'sell', amount)

    except ccxt.BaseError as e:
        logging.error("An error occurred: %s", e)
        raise

def main():
    """
    Main function to execute the trading strategy.
    """
    try:
        # Retrieve API keys and secrets from environment variables
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')

        if not api_key or not api_secret:
            raise ValueError("BYBIT_API_KEY or BYBIT_API_SECRET environment variables are not set.")

        symbol = 'BTCUSDT'  # Example symbol for derivative trading with leverage
        amount = 0.001  # Example amount to trade
        risk_percent = 0.01  # Risk 1% of the available margin per trade

        # Initialize exchange
        exchange = initialize_exchange(api_key, api_secret)
        
        # Fetch historical data
        df = fetch_ohlcv(exchange, symbol)
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Define trading strategy
        df = define_trading_strategy(df)
        
        # Execute trading strategy
        execute_trading_strategy(exchange, df, symbol, amount, risk_percent)
                
    except ccxt.NetworkError as e:
        logging.error("A network error occurred: %s", e)
    except ccxt.BaseError as e:
        logging.error("An error occurred: %s", e)
    except ValueError as e:
        logging.error("Value error occurred: %s", e)

if __name__ == "__main__":
    main()

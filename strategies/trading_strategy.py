from email.policy import HTTP
import subprocess
import sys

from sentiment_analysis import fetch_real_time_sentiment

# Ensure the required libraries are installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "stable-baselines3==1.6.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "gym==0.26.0"])

import ccxt
import pandas as pd
import pandas_ta as ta
import logging
import os
import time
import ntplib
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime
from fetch_data import main

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()])

# Environment variables for API credentials
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, leverage=1):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.done = False
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.position = None  # None, 'long', 'short'
        self.position_size = 0
        self.entry_price = 0
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(df.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.balance = self.initial_balance
        self.position = None
        self.position_size = 0
        self.entry_price = 0
        return self.df.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.done = True

        reward = self._take_action(action)
        obs = self._next_observation()

        return obs, reward, self.done, {}

    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0

        if action == 1:  # Buy/Long
            if self.position is None:
                self.position = 'long'
                self.position_size = (self.balance * self.leverage) / current_price
                self.entry_price = current_price

        elif action == 2:  # Sell/Short
            if self.position is None:
                self.position = 'short'
                self.position_size = (self.balance * self.leverage) / current_price
                self.entry_price = current_price

        if self.position == 'long':
            reward = (current_price - self.entry_price) * self.position_size
        elif self.position == 'short':
            reward = (self.entry_price - current_price) * self.position_size

        self.balance += reward

        if self.position and action == 0:  # Hold position
            reward = 0

        if action == 1 and self.position == 'short':
            self.balance += self.position_size * (self.entry_price - current_price)
            self.position = None
            self.position_size = 0
            self.entry_price = 0

        if action == 2 and self.position == 'long':
            self.balance += self.position_size * (current_price - self.entry_price)
            self.position = None
            self.position_size = 0
            self.entry_price = 0

        return reward

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Position: {self.position}')
        print(f'Position Size: {self.position_size}')
        print(f'Entry Price: {self.entry_price}')

    def _next_observation(self):
        return self.df.iloc[self.current_step].values

# Example usage
if __name__ == "__main__":
    # Sample DataFrame with stock data
    data = {
        'Open': [1.0, 1.2, 1.1, 1.3, 1.4],
        'High': [1.1, 1.3, 1.2, 1.4, 1.5],
        'Low': [0.9, 1.1, 1.0, 1.2, 1.3],
        'Close': [1.0, 1.2, 1.1, 1.3, 1.4],
    }
    df = pd.DataFrame(data)

    env = TradingEnv(df, leverage=10)
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, _ = env.step(action)
        env.render()

# Example usage
if __name__ == "__main__":
    # Sample DataFrame with stock data
    data = {
        'Open': [1.0, 1.2, 1.1, 1.3, 1.4],
        'High': [1.1, 1.3, 1.2, 1.4, 1.5],
        'Low': [0.9, 1.1, 1.0, 1.2, 1.3],
        'Close': [1.0, 1.2, 1.1, 1.3, 1.4],
    }
    df = pd.DataFrame(data)

    env = TradingEnv(df)
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, _ = env.step(action)
        env.render()

# Prepare the data
data = {
    'feature1': np.random.rand(1000),
    'feature2': np.random.rand(1000),
    'feature3': np.random.rand(1000)
}
df = pd.DataFrame(data)

# Create and wrap the environment
def create_env():
    return TradingEnv(df)

vec_env = make_vec_env(create_env, n_envs=4)

# Create the PPO model
model = PPO('MlpPolicy', vec_env, verbose=1)

# Define a callback for saving the model at certain intervals
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_trading_model')

# Train the model
model.learn(total_timesteps=10000, callback=checkpoint_callback)

# Save the trained model
model.save("ppo_trading_model")

# Load the trained model (optional)
model = PPO.load("ppo_trading_model")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

def train_rl_model(df):
    env = TradingEnv(df)
    env = Monitor(env)
    model = PPO(
        'MlpPolicy',
        vec_env,
        n_steps=2048,  # Increased steps for better learning
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=2.5e-4,
        clip_range=0.2,
        verbose=1
    )

    # Set up a checkpoint callback to save the model at intervals
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_trading_model')

    # Train the model
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Save the final model
    model.save("ppo_trading_model_final")

    return model

def rl_trading_decision(model, obs):
    action, _ = model.predict(obs)
    return action

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

def detect_signals(df):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # Example strategy combining SMA crossover and RSI
    if (previous['SMA_20'] < previous['SMA_50'] and latest['SMA_20'] > latest['SMA_50']) and latest['RSI'] < 70:
        return 'buy'
    elif (previous['SMA_20'] > previous['SMA_50'] and latest['SMA_20'] < latest['SMA_50']) and latest['RSI'] > 30:
        return 'sell'
    return 'hold'

def generate_signals(df):
    """
    Generate trading signals based on technical analysis of OHLCV data.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].

    Returns:
    pd.DataFrame: DataFrame with additional columns for trading signals.
    """
    # Calculate technical indicators
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['RSI'] = ta.rsi(df['close'], length=14)

    # Generate signals based on indicators
    signals = []
    for i in range(len(df)):
        if df['SMA_50'][i] > df['SMA_200'][i] and df['RSI'][i] > 50:
            signals.append('buy')
        elif df['SMA_50'][i] < df['SMA_200'][i] and df['RSI'][i] < 50:
            signals.append('sell')
        else:
            signals.append('hold')

    df['signal'] = signals

    logging.info("Generated buy and sell signals")
    return df

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
    Calculate technical indicators using ta library.
    """
    try:
        # Simple Moving Averages
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
        
        # Exponential Moving Averages
        df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        
        logging.info("Calculated technical indicators")
    except Exception as e:
        logging.error("Error calculating indicators: %s", e)
        raise e
    return df

def prepare_data(df, n_features):
    data = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data)-n_features-1):
        X.append(scaled_data[i:(i+n_features), 0])
        y.append(scaled_data[i + n_features, 0])
    return np.array(X), np.array(y), scaler

def build_and_train_model(df):
    n_features = 60
    X, y, scaler = prepare_data(df, n_features)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(n_features, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32)

    return model, scaler

def predict_prices(model, scaler, df):
    n_features = 60
    X, _, _ = prepare_data(df, n_features)
    predicted = model.predict(X)
    return scaler.inverse_transform(predicted)

def trading_strategy(df):
    """
    Define the trading strategy based on SMA crossover.
    """
    signals = ['hold']  # Initialize with 'hold' for the first entry
    for i in range(1, len(df)):
        if df['SMA_50'][i] > df['SMA_200'][i] and df['SMA_50'][i-1] <= df['SMA_200'][i-1]:
            signals.append('buy')
        elif df['SMA_50'][i] < df['SMA_200'][i] and df['SMA_50'][i-1] >= df['SMA_200'][i-1]:
            signals.append('sell')
        else:
            signals.append('hold')
    df['signal'] = signals
    logging.info("Defined trading strategy")
    return df

def execute_trade():
    # Bybit API initialization
    api_key = 'YOUR_BYBIT_API_KEY'
    api_secret = 'YOUR_BYBIT_API_SECRET'
    session = HTTP("https://api.bybit.com", api_key=api_key, api_secret=api_secret)

    # Fetch sentiment data
    sentiment_score = fetch_real_time_sentiment()

    if sentiment_score is not None:
        if sentiment_score > 0.5:
            # Example logic to place a buy order
            order = session.place_active_order(
                symbol="BTCUSD",
                side="Buy",
                order_type="Market",
                qty=0.002,  # Adjust the quantity as needed
                time_in_force="GoodTillCancel"
            )
            logging.info(f"Placed buy order: {order}")
        elif sentiment_score < -0.5:
            # Example logic to place a sell order
            order = session.place_active_order(
                symbol="BTCUSD",
                side="Sell",
                order_type="Market",
                qty=0.002,  # Adjust the quantity as needed
                time_in_force="GoodTillCancel"
            )
            logging.info(f"Placed sell order: {order}")
        else:
            logging.info("Sentiment score neutral, no action taken.")
    else:
        logging.error("Failed to fetch sentiment data.")

def _update_balance_and_position(self, action, current_price):
    if action == 1 and self.position == 'short':
        self.balance += self.position_size * (self.entry_price - current_price)
        self.position = None
    elif action == 2 and self.position == 'long':
        self.balance += self.position_size * (current_price - self.entry_price)
        self.position = None

def step(self, action):
    self.current_step += 1
    if self.current_step >= len(self.df) - 1:
        self.done = True

    current_price = self.df.iloc[self.current_step]['Close']
    reward = self._take_action(action, current_price)
    self._update_balance_and_position(action, current_price)
    
    obs = self._next_observation()
    return obs, reward, self.done, {}


    
if __name__ == "__main__":
    main()

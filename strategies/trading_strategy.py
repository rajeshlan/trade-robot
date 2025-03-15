#python -m strategies.trading_strategy (RUN WITH THIS) needs checking

import subprocess
import sys
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ccxt
import pandas as pd
import pandas_ta as ta
import logging
import os
import time
import ntplib
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym  # Updated to Gymnasium for compatibility
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from data.sentiment_analysis import fetch_real_time_sentiment

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()])

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the LSTM
        return out

# Prepare data for the LSTM model
def prepare_data(df, n_features):
    # Drop non-numeric columns (e.g., 'timestamp') if they exist
    df = df.select_dtypes(include=[np.number])

    # Convert DataFrame to NumPy array
    data = df.values

    # Initialize MinMaxScaler and transform only numeric data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - n_features - 1):
        X.append(scaled_data[i:(i + n_features), 0])
        y.append(scaled_data[i + n_features, 0])

    # Convert to PyTorch tensors
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # Add channel dimension
    y = torch.tensor(y, dtype=torch.float32)

    return X, y, scaler

# Build and train the LSTM model
def build_and_train_model(df):
    n_features = 60
    X, y, scaler = prepare_data(df, n_features)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 20
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model, scaler

# Predict prices using the trained LSTM model
def predict_prices(model, scaler, df):
    n_features = 60
    X, _, _ = prepare_data(df, n_features)

    with torch.no_grad():
        predictions = model(X).squeeze().numpy()  # Convert tensor to NumPy

    # Ensure predictions match the expected shape for inverse_transform
    num_features = scaler.n_features_in_  # Get the number of features used during fit_transform
    dummy_array = np.zeros((predictions.shape[0], num_features))  # Create a dummy array with the right shape
    dummy_array[:, -1] = predictions  # Insert predictions in the last column (assuming 'close' was last)

    # Apply inverse transform and return only the predicted 'close' values
    return scaler.inverse_transform(dummy_array)[:, -1]

# Calculate technical indicators
def calculate_indicators(df):
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df.fillna(0, inplace=True)  # Replace NaN values with 0
    return df

# Define trading environment for reinforcement learning
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

        # Observation space size = number of columns in df + portfolio state (balance, position_size)
        obs_space_size = len(df.columns) + 2
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_space_size,), dtype=np.float32)

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        # Handle seeding for reproducibility
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.balance = self.initial_balance
        self.position = None
        self.position_size = 0
        self.entry_price = 0
        obs = self._next_observation()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # Check if the current step exceeds the data length
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # In this example, there is no explicit truncation condition

        if 'close' not in self.df.columns:
            raise KeyError(f"Expected column 'Close' not found in DataFrame. Available columns: {self.df.columns}")

        current_price = self.df.iloc[self.current_step]['Close']

        reward = self._take_action(action, current_price)
        obs = self._next_observation()

        print("Available columns in df:", self.df.columns)


        # Return observation, reward, terminated flag, truncated flag, and an empty info dictionary
        return obs, reward, terminated, truncated, {}


    def _take_action(self, action, current_price):
        reward = 0
        transaction_cost = 0.001  # Simulate 0.1% trading fee
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

        # Calculate rewards
        if self.position == 'long':
            reward = (current_price - self.entry_price) * self.position_size
        elif self.position == 'short':
            reward = (self.entry_price - current_price) * self.position_size

        # Deduct transaction costs
        reward -= transaction_cost * abs(self.position_size)

        # Update balance
        self.balance += reward

        # Normalize reward
        max_possible_reward = self.initial_balance * self.leverage
        normalized_reward = reward / max_possible_reward

        return normalized_reward

    

    def _next_observation(self):
        df_filtered = self.df.drop(columns=['timestamp'], errors='ignore')
        obs = np.array(df_filtered.iloc[self.current_step].values, dtype=np.float32)
        obs[np.isnan(obs)] = 0  # Replace NaNs with zeros

        # Ensure obs has exactly 6 features before adding portfolio state
        while obs.shape[0] < 6:
            obs = np.append(obs, 0)  # Pad with zeros if needed

        while obs.shape[0] > 6:
            obs = obs[:-1]  # Trim extra features if necessary

        # Portfolio state (balance and position size)
        portfolio_state = np.array([float(self.balance), float(self.position_size)], dtype=np.float32)

        # Concatenate to ensure 8 features in total
        obs = np.concatenate((obs, portfolio_state))

        # Print for debugging
        print(f"Final observation shape: {obs.shape}")

        return obs

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}')

# Fetch real-time sentiment data and make trading decisions
def fetch_and_trade_sentiment(session, symbol):
    try:
        # Fetch account balance
        balance = session.fetch_balance()
        available_balance = balance['total']['USDT']  # Adjust for your account currency
        logging.info(f"Available balance: {available_balance} USDT")

        sentiment_score = fetch_real_time_sentiment()
        logging.info(f"Fetched sentiment score: {sentiment_score}")

        buy_threshold = 0.3  # Adjust threshold
        sell_threshold = -0.3
        trade_amount = 0.002  # Example trade amount

        if sentiment_score is not None:
            if sentiment_score > buy_threshold:
                if available_balance >= trade_amount * 1000:  # Approx price assumption
                    order = session.create_market_buy_order(symbol, trade_amount)
                    logging.info(f"Placed buy order: {order}")
                else:
                    logging.warning("Insufficient balance for buy order.")
            elif sentiment_score < sell_threshold:
                if available_balance >= trade_amount * 1000:  # Approx price assumption
                    order = session.create_market_sell_order(symbol, trade_amount)
                    logging.info(f"Placed sell order: {order}")
                else:
                    logging.warning("Insufficient balance for sell order.")
            else:
                logging.info("Sentiment score neutral, no action taken.")
        else:
            logging.error("Failed to fetch sentiment data.")

    except ccxt.base.errors.InsufficientFunds as e:
        logging.error(f"Trade failed due to insufficient funds: {e}")
    except Exception as e:
        logging.error(f"An error occurred during trading: {e}")

def train_rl_model(df):
    env = TradingEnv(df)
    print("Expected observation space:", env.observation_space.shape)
    
    vec_env = make_vec_env(lambda: env, n_envs=1)
    
    obs, _ = env.reset()
    print("Initial observation shape:", obs.shape)  # Debugging output
    
    ppo_model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01
    )
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_trading_model')
    
    ppo_model.learn(total_timesteps=10000, callback=checkpoint_callback)
    ppo_model.save("ppo_trading_model")
    
    return ppo_model


def rl_trading_decision(rl_model, obs):
    action, _states = rl_model.predict(obs)
    return action

def execute_trade():
    # Placeholder; implement trade execution logic as needed
    logging.info("Executing trade (placeholder)")

# Synchronize system time
def synchronize_system_time():
    try:
        response = ntplib.NTPClient().request('pool.ntp.org')
        current_time = datetime.fromtimestamp(response.tx_time)
        logging.info(f"System time synchronized: {current_time}")
        return int((current_time - datetime.utcnow()).total_seconds() * 1000)  # Return time offset in ms
    except Exception as e:
        logging.error("Time synchronization failed: %s", e)
        return 0

# Main script
if __name__ == "__main__":
    data = {
        'Open': np.random.rand(1000),
        'High': np.random.rand(1000),
        'Low': np.random.rand(1000),
        'Close': np.random.rand(1000)
    }
    df = pd.DataFrame(data)

    # Calculate indicators
    df = calculate_indicators(df)

    # Train LSTM model
    model, scaler = build_and_train_model(df[['Close']])
    predictions = predict_prices(model, scaler, df[['Close']])
    print(predictions)

    # Set up Trading Environment
    env = TradingEnv(df)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Train PPO model
    ppo_model = PPO(
    policy='MlpPolicy',
    env=vec_env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01
)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_trading_model')
    ppo_model.learn(total_timesteps=10000, callback=checkpoint_callback)

    # Save PPO model
    ppo_model.save("ppo_trading_model")

    # Evaluate PPO model
    mean_reward, std_reward = evaluate_policy(ppo_model, vec_env, n_eval_episodes=10)
    logging.info(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Load environment variables
    load_dotenv(dotenv_path='D:\\RAJESH FOLDER\\PROJECTS\\trade-robot\\config\\API.env')

    # Initialize exchange session
    API_KEY = os.getenv('BYBIT_API_KEY')
    API_SECRET = os.getenv('BYBIT_API_SECRET')
    session = ccxt.bybit({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True
    })

    # Fetch real-time sentiment and trade
    fetch_and_trade_sentiment(session, "BTC/USDT")

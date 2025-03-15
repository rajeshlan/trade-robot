#python trading\portfolio_management.py

import logging
import pandas as pd
import ccxt
import numpy as np
import gym
from stable_baselines3 import PPO
from sklearn.neural_network import MLPRegressor
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from transformers import pipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path='D:\\RAJESH FOLDER\\PROJECTS\\trade-robot\\config\\API.env')

# Bybit API credentials
bybit_api_key = os.getenv("BYBIT_API_KEY")
bybit_api_secret = os.getenv("BYBIT_API_SECRET")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom Trading Environment for Reinforcement Learning
class TradingEnv(gym.Env):
    def __init__(self, asset_data, initial_balance=1000):
        super(TradingEnv, self).__init__()
        self.asset_data = asset_data
        self.balance = initial_balance
        self.current_step = 0
        self.done = False

        # Observation space: Portfolio value, step count, and asset price
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.done = False

        obs = self._next_observation()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # Check if the step is out of data range
        if self.current_step >= len(self.asset_data) - 1:
            self.done = True

        reward = self._calculate_reward(action)
        obs = self._next_observation()

        return obs, reward, self.done, False, {}

    def _calculate_reward(self, action):
        """Simple reward function: Increase balance if action was profitable."""
        current_price = self.asset_data[self.current_step]
        reward = 0

        if action == 1:  # Buy
            self.balance -= current_price  # Deduct from balance
        elif action == 2:  # Sell
            self.balance += current_price  # Add to balance

        # Reward = current balance as a measure of performance
        return self.balance / self.initial_balance

    def _next_observation(self):
        """Returns a normalized observation"""
        current_price = self.asset_data[self.current_step]
        obs = np.array([self.balance, current_price], dtype=np.float32)

        return obs

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Price: {self.asset_data[self.current_step]}')


# Training the RL agent
def train_rl_agent(env):
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

def optimize_portfolio_with_rl(env, model):
    obs = env.reset()
    done = False
    actions = []
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        actions.append(action)
        logging.info(f"RL optimized allocation: {action}")
    return actions[-1]  # Return the last action taken

# Sentiment Analysis Functions
def fetch_news_headlines(asset):
    # Mock function for fetching news headlines
    return [f"{asset} is performing well", f"Concerns over {asset}", f"{asset} market update"]

def fetch_market_sentiment(asset):
    sentiment_analysis = pipeline("sentiment-analysis")
    news_headlines = fetch_news_headlines(asset)
    sentiments = sentiment_analysis(news_headlines)
    positive_score = sum([1 for s in sentiments if s['label'] == 'POSITIVE'])
    negative_score = sum([1 for s in sentiments if s['label'] == 'NEGATIVE'])
    return positive_score - negative_score

def adjust_portfolio_based_on_sentiment(portfolio):
    for index, row in portfolio.iterrows():
        sentiment = fetch_market_sentiment(row['asset'])
        if sentiment > 0:
            row['weight'] *= 1.1
        else:
            row['weight'] *= 0.9
        portfolio.at[index, 'weight'] = row['weight']
    logging.info("Adjusted portfolio weights based on sentiment.")

# Portfolio Optimization Functions
def calculate_returns(df):
    returns = df.pct_change().mean() * 252
    cov_matrix = df.pct_change().cov() * 252
    return returns, cov_matrix

def portfolio_performance(weights, returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def optimize_portfolio(returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(returns)
    args = (returns, cov_matrix, risk_free_rate)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(portfolio_performance, num_assets * [1. / num_assets, ], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def build_portfolio_model(input_dim):
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500)
    return model

def train_price_prediction_model(asset_prices):
    X = asset_prices.drop('future_price', axis=1)
    y = asset_prices['future_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = np.mean(np.abs(predictions - y_test) / y_test)
    logging.info(f"Prediction accuracy: {1 - accuracy}")
    return model

# Bybit Exchange Functions
exchange = ccxt.bybit({
    'apiKey': bybit_api_key,
    'secret': bybit_api_secret,
    'enableRateLimit': True,
})

def fetch_derivative_positions():
    try:
        positions = exchange.fetch_positions()
        derivative_positions = []
        for position in positions:
            if position['info']['side'] != 'None':
                derivative_positions.append({
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'quantity': position['contracts'],
                    'entry_price': position['entryPrice']
                })
        logging.info(f"Fetched derivative positions: {derivative_positions}")
        return derivative_positions
    except Exception as e:
        logging.error(f"Error fetching derivative positions: {e}")
        return []

def fetch_current_prices(assets):
    prices = {}
    for asset in assets:
        try:
            ticker = exchange.fetch_ticker(asset)
            prices[asset] = ticker['last']
            logging.info(f"Fetched price for {asset}: {ticker['last']}")
        except Exception as e:
            logging.error(f"Error fetching price for {asset}: {e}")
    return prices

# Portfolio Management
def track_portfolio_performance(portfolio):
    if portfolio.empty:
        logging.warning("Portfolio is empty. No performance to track.")
        return
    total_value = portfolio['value'].sum()
    portfolio['weighted_performance'] = portfolio['value'] * portfolio['weight']
    total_weighted_performance = portfolio['weighted_performance'].sum()
    logging.info(f"Total Portfolio Value: {total_value:.2f}")
    logging.info(f"Total Weighted Performance: {total_weighted_performance:.2f}")
    for index, row in portfolio.iterrows():
        logging.info(f"Asset: {row['asset']}, Quantity: {row['quantity']:.6f}, Value: {row['value']:.2f}, Weight: {row['weight']:.2f}, Weighted Performance: {row['weighted_performance']:.2f}")

def rebalance_portfolio(portfolio, target_weights):
    if portfolio.empty:
        logging.warning("Portfolio is empty. Cannot rebalance.")
        return
    assets = portfolio['asset'].tolist()
    current_prices = fetch_current_prices(assets)
    total_value = portfolio['value'].sum()
    for index, row in portfolio.iterrows():
        asset = row['asset']
        if asset in target_weights:
            target_weight = target_weights[asset]
            target_value = total_value * target_weight
            current_price = current_prices.get(asset, 0)
            target_quantity = target_value / current_price if current_price else 0
            logging.info(f"Rebalancing {asset}: current value = {row['value']:.2f}, target value = {target_value:.2f}, current quantity = {row['quantity']:.6f}, target quantity = {target_quantity:.6f}")
            portfolio.at[index, 'weight'] = target_weight
            portfolio.at[index, 'value'] = target_value
            portfolio.at[index, 'quantity'] = target_quantity
        else:
            logging.warning(f"Target weight for {asset} not found. Skipping.")
    logging.info("Portfolio rebalanced.")

# Main Execution Flow
def main():
    # Initialize trading environment and RL agent
    derivative_positions = fetch_derivative_positions()
    if not derivative_positions:
        logging.error("No derivative positions found. Ensure your account has open positions and API credentials are correct.")
        return
    portfolio_data = {
        'asset': [pos['symbol'] for pos in derivative_positions],
        'quantity': [pos['quantity'] for pos in derivative_positions],
        'entry_price': [pos['entry_price'] for pos in derivative_positions],
        'side': [pos['side'] for pos in derivative_positions]
    }
    portfolio = pd.DataFrame(portfolio_data)
    current_prices = fetch_current_prices(portfolio['asset'].tolist())
    portfolio['value'] = portfolio.apply(lambda row: row['quantity'] * (current_prices.get(row['asset'], 0)), axis=1)

    env = TradingEnv(portfolio['value'].values)
    model = train_rl_agent(env)

    # Optimize portfolio based on RL
    optimized_weights = optimize_portfolio_with_rl(env, model)
    portfolio['weight'] = optimized_weights
    adjust_portfolio_based_on_sentiment(portfolio)

    # Fetch price data for price prediction model training
    price_data = pd.DataFrame()  # Assume this is filled with historical prices
    price_model = train_price_prediction_model(price_data)

    # Track portfolio performance
    track_portfolio_performance(portfolio)

    # Rebalance portfolio
    target_weights = dict(zip(portfolio['asset'], optimized_weights))
    rebalance_portfolio(portfolio, target_weights)

if __name__ == "__main__":
    main()

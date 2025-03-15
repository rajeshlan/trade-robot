#python -m trading.price_prediction

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import inspect
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dotenv import load_dotenv
from pybit.unified_trading import HTTP, WebSocket
import joblib
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator
from collections import deque


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
WINDOW_SIZE = 60
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
DROPOUT = 0.3
HIDDEN_SIZE = 128
NUM_LAYERS = 3
PATIENCE = 10
SYMBOL = "BTCUSDT"
INTERVAL = "1"
MODEL_PATH = "best_model.pth"
BUFFER_SIZE = 200

# Load environment variables
load_dotenv()

class SecureBybitClient:
    def __init__(self):
        # Ensure mainnet usage by setting testnet=False unless explicitly enabled
        self.client = HTTP(
            api_key=os.getenv("BYBIT_API_KEY"),
            api_secret=os.getenv("BYBIT_API_SECRET"),
            testnet=os.getenv("TESTNET", "False") == "True"
        )
        
    def fetch_historical_data(self, limit=1000):
        retries = 3
        for attempt in range(retries):
            try:
                end_time = int(time.time() * 1000)
                response = self.client.get_kline(
                    category="linear",
                    symbol=SYMBOL,
                    interval=INTERVAL,
                    end=end_time,
                    limit=limit
                )
                if 'result' not in response or not response['result']['list']:
                    raise ValueError("Empty response from API")
            
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                data = [[float(candle[i]) for i in range(6)] for candle in response['result']['list']]
                df = pd.DataFrame(data, columns=columns)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                logging.info(f"Fetched {len(df)} data points successfully")
                return df
            except Exception as e:
                logging.error(f"Data fetch error: {e}")
                if attempt < retries - 1:
                    delay = 5 * (2 ** attempt)  # Exponential backoff: 5s, 10s, 20s
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    return None

class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def prepare_data(df):
    try:
        if df is None or len(df) < WINDOW_SIZE + 1:
            raise ValueError(f"Need at least {WINDOW_SIZE+1} data points, got {len(df)}")

        df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")
        df["SMA_14"] = SMAIndicator(df["close"], window=14).sma_indicator()
        df["EMA_14"] = EMAIndicator(df["close"], window=14).ema_indicator()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.interpolate(method='linear', inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        if df.isna().sum().sum() > 0:
            logging.warning(f"Dropping rows with NaN. Rows before: {len(df)}")
            df.dropna(inplace=True)
            logging.info(f"Rows after dropna: {len(df)}")

        if len(df) < WINDOW_SIZE:
            logging.error(f"Not enough data: {len(df)} required {WINDOW_SIZE}")
            return None

        exclude = ['timestamp', 'close']
        features = [col for col in df.columns if col not in exclude]
        if len(features) != 92:
            logging.error(f"Feature mismatch! Expected 92, got {len(features)}")
            return None

        logging.info(f"Using {len(features)} features for training.")
        logging.info(f"Training close price range: Min {df['close'].min()}, Max {df['close'].max()}")

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])

        # Use StandardScaler for target
        target_scaler = StandardScaler()
        target_scaler.fit(df[['close']])
        scaled_target = target_scaler.transform(df[['close']])
        logging.info(f"Target scaler mean: {target_scaler.mean_[0]}, std: {np.sqrt(target_scaler.var_[0])}")

        X, y = [], []
        for i in range(WINDOW_SIZE, len(scaled_features)):
            X.append(scaled_features[i - WINDOW_SIZE:i])
            y.append(scaled_target[i])

        X, y = np.array(X), np.array(y)
        split_idx = int(len(X) * 0.8)
        return (torch.tensor(X[:split_idx], dtype=torch.float32),
                torch.tensor(y[:split_idx], dtype=torch.float32),
                torch.tensor(X[split_idx:], dtype=torch.float32),
                torch.tensor(y[split_idx:], dtype=torch.float32),
                scaler, target_scaler)
    except Exception as e:
        logging.error(f"Data preparation error: {e}")
        return None

def train_model(model, X_train, y_train, X_val, y_val, target_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logging.info("Early stopping triggered")
                break

        scheduler.step(avg_val_loss)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_val.to(device)).cpu().numpy()
        y_true = y_val.numpy()
        y_pred_price = target_scaler.inverse_transform(y_pred)
        y_true_price = target_scaler.inverse_transform(y_true)
        mae_price = mean_absolute_error(y_true_price, y_pred_price)
        rmse_price = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
        logging.info(f"Validation MAE in price units: {mae_price:.2f}, RMSE in price units: {rmse_price:.2f}")

class TradingBot:
    def __init__(self, model, scalers, interval="1", symbol="BTCUSDT", testnet_mode=True):
        self.model = model
        self.scalers = scalers
        self.interval = interval
        self.symbol = symbol
        self.testnet_mode = testnet_mode
        self.data_buffer = []
        self.last_processed_timestamp = 0
        self.current_kline = None  # Track the current open kline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.scaler, self.target_scaler = scalers
        bybit = SecureBybitClient()
        self.preload_historical_data(bybit, limit=200)
        self.connect_websocket()

    def connect_websocket(self, max_retries=5):
        def run_websocket():
            retries = 0
            while retries < max_retries:
                try:
                    logging.info("Connecting to WebSocket...")
                    self.ws = WebSocket(testnet=self.testnet_mode, channel_type="linear")
                    self.ws.kline_stream(
                        interval=self.interval,
                        symbol=self.symbol,
                        callback=self.handle_ws_message
                    )
                    logging.info(f"WebSocket connected and subscribed to kline.{self.interval}.{self.symbol}")
                    return
                except Exception as e:
                    retries += 1
                    logging.error(f"WebSocket connection error (attempt {retries}/{max_retries}): {e}")
                    if retries >= max_retries:
                        logging.error("Max retries reached. Exiting WebSocket connection attempts.")
                        raise
                    time.sleep(5)

    # Start WebSocket in a background thread
        threading.Thread(target=self.connect_websocket, daemon=True).start()
        logging.info("WebSocket connection initiated in background thread.")

    def handle_ws_message(self, message):
        logging.info(f"Received WebSocket message: {message}")
        try:
            if 'topic' not in message or 'data' not in message:
                logging.warning("Message lacks 'topic' or 'data'. Skipping.")
                return
            kline_data = message['data']
            if isinstance(kline_data, list):
                for kline in kline_data:
                    self.process_kline(kline)
            else:
                self.process_kline(kline)
        except Exception as e:
            logging.error(f"Error processing WebSocket message: {e}")

    def process_kline(self, kline_data):
        timestamp = int(kline_data['start'])
        is_closed = kline_data.get('confirm', False)  # True if kline is closed

        if is_closed:
            if timestamp <= self.last_processed_timestamp:
                logging.info(f"Duplicate closed kline for timestamp {timestamp}. Skipping.")
                return
            # Append the previous open kline (now closed) to buffer
            if self.current_kline:
                self.data_buffer.append(self.current_kline)
                self.last_processed_timestamp = timestamp
                if len(self.data_buffer) > BUFFER_SIZE:
                    self.data_buffer.pop(0)
                # Process prediction with closed kline
                buffer_df = pd.DataFrame(self.data_buffer)
                if len(buffer_df) >= WINDOW_SIZE:
                    prediction = self.process_data(buffer_df)
                    if prediction is not None:
                        logging.info(f"Predicted price: {prediction:.2f} | Actual: {buffer_df['close'].iloc[-1]:.2f}")
            self.current_kline = None  # Reset for next open kline
        else:
            # Update the current open kline
            self.current_kline = {
                'timestamp': pd.to_datetime(timestamp, unit='ms'),
                'open': float(kline_data['open']),
                'high': float(kline_data['high']),
                'low': float(kline_data['low']),
                'close': float(kline_data['close']),
                'volume': float(kline_data['volume'])
            }

    def preload_historical_data(self, bybit_client, limit=200):
        df = bybit_client.fetch_historical_data(limit=limit)
        if df is not None:
            self.data_buffer.extend(df.to_dict('records'))
            timestamps_ms = (df['timestamp'].astype('int64') // 10**6).astype('int64')
            self.last_processed_timestamp = timestamps_ms.max()
            logging.info(f"Preloaded {len(df)} historical data points")
            logging.info(f"Historical timestamp range: {timestamps_ms.min()} to {self.last_processed_timestamp} (ms)")
        else:
            logging.error("Failed to preload historical data.")
            self.last_processed_timestamp = 0

    def process_data(self, buffer_data):
        try:
            if len(buffer_data) < WINDOW_SIZE:
                logging.error(f"Not enough data: {len(buffer_data)} rows")
                return None

            new_data = add_all_ta_features(buffer_data, open="open", high="high",
                                          low="low", close="close", volume="volume")
            new_data["SMA_14"] = SMAIndicator(new_data["close"], window=14).sma_indicator()
            new_data["EMA_14"] = EMAIndicator(new_data["close"], window=14).ema_indicator()

            new_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            new_data.interpolate(method='linear', inplace=True)
            new_data.ffill(inplace=True)
            new_data.bfill(inplace=True)

            last_60 = new_data.iloc[-WINDOW_SIZE:]
            if last_60.isna().sum().sum() > 0:
                logging.warning(f"NaN values in last {WINDOW_SIZE} rows")
                return None

            features = [col for col in last_60.columns if col not in ['timestamp', 'close']]
            if len(features) != 92:
                logging.error(f"Feature count mismatch: Expected 92, got {len(features)}")
                return None

            logging.info(f"Real-time close price range: Min {buffer_data['close'].min()}, Max {buffer_data['close'].max()}")
            scaled_features = self.scaler.transform(last_60[features])
            logging.info(f"Scaled features min: {scaled_features.min()}, max: {scaled_features.max()}")
            sequence = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                prediction = self.model(sequence.to(self.device))
                scaled_pred = prediction.cpu().numpy()
                predicted_price = self.target_scaler.inverse_transform(scaled_pred.reshape(1, -1))[0][0]
                return predicted_price
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None

if __name__ == "__main__":
    bybit = SecureBybitClient()
    df = bybit.fetch_historical_data(limit=1000)
    if df is None:
        logging.error("Failed to fetch historical data. Exiting.")
        exit(1)
    processed = prepare_data(df)
    if processed is None:
        logging.error("Data preparation failed. Exiting.")
        exit(1)
    
    X_train, y_train, X_val, y_val, scaler, target_scaler = processed
    joblib.dump(scaler, "feature_scaler.save")
    joblib.dump(target_scaler, "target_scaler.save")
    
    model = AdvancedLSTMModel(input_size=X_train.shape[2],
                             hidden_size=HIDDEN_SIZE,
                             num_layers=NUM_LAYERS,
                             dropout=DROPOUT)
    train_model(model, X_train, y_train, X_val, y_val, target_scaler)
    
    bot = TradingBot(model, (scaler, target_scaler))
    
    # Keep the main thread alive
    logging.info("Trading bot started. Listening for WebSocket messages...")
    while True:
        time.sleep(1)  # Prevent tight CPU loop
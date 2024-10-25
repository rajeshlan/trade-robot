import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import os

def prepare_data(df, window_size=60):
    """
    Prepare data for LSTM model by creating sequences of historical prices.
    
    Args:
    - df (pd.DataFrame): DataFrame containing historical OHLCV data.
    - window_size (int): The number of previous time steps to use for predicting the next price.
    
    Returns:
    - X_train (np.array): Feature set for training.
    - y_train (np.array): Target set for training.
    - scaler (MinMaxScaler): Fitted scaler object for data normalization.
    """
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

        X_train, y_train = [], []
        for i in range(window_size, len(scaled_data)):
            X_train.append(scaled_data[i - window_size:i, 0])
            y_train.append(scaled_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        return X_train, y_train, scaler
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None, None

def create_lstm_model(input_shape):
    """
    Build and compile the LSTM model.
    
    Args:
    - input_shape (tuple): Shape of the input data (time steps, features).
    
    Returns:
    - model (Sequential): Compiled LSTM model.
    """
    try:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Predicting the next closing price

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def train_lstm(df, epochs=5, batch_size=32, window_size=60, model_save_path=None):
    """
    Train the LSTM model using historical OHLCV data.
    
    Args:
    - df (pd.DataFrame): DataFrame containing historical OHLCV data.
    - epochs (int): Number of epochs for training.
    - batch_size (int): Batch size for training.
    - window_size (int): The number of previous time steps to use for predicting the next price.
    - model_save_path (str): Path to save the trained model.
    
    Returns:
    - model (Sequential): Trained LSTM model.
    - scaler (MinMaxScaler): Fitted scaler object.
    """
    X_train, y_train, scaler = prepare_data(df, window_size)

    if X_train is None or y_train is None:
        print("Failed to prepare data for training.")
        return None, None

    model = create_lstm_model((X_train.shape[1], 1))
    if model is None:
        print("Failed to create LSTM model.")
        return None, None

    try:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        if model_save_path:
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
        
        return model, scaler
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None

def predict_price(df, model, scaler, window_size=60):
    """
    Predict future price using the trained LSTM model.
    
    Args:
    - df (pd.DataFrame): DataFrame containing historical OHLCV data.
    - model (Sequential): Trained LSTM model.
    - scaler (MinMaxScaler): Fitted scaler object.
    - window_size (int): The number of previous time steps to use for predicting the next price.
    
    Returns:
    - float: Predicted closing price.
    """
    try:
        scaled_data = scaler.transform(df['close'].values.reshape(-1, 1))
        last_sequence = scaled_data[-window_size:]
        last_sequence = np.reshape(last_sequence, (1, window_size, 1))

        predicted_price = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_price)

        return predicted_price[0][0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def load_trained_model(model_path):
    """
    Load a trained LSTM model from file.
    
    Args:
    - model_path (str): Path to the saved model file.
    
    Returns:
    - model (Sequential): Loaded LSTM model.
    """
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return None
    
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

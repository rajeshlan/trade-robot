import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def prepare_data(df, window_size=60):
    """
    Prepare data for LSTM model by creating sequences of historical prices.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(window_size, len(scaled_data)):
        X_train.append(scaled_data[i - window_size:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler

def create_lstm_model(input_shape):
    """
    Build and compile the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predicting the next closing price

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm(df):
    """
    Train the LSTM model using historical OHLCV data.
    """
    X_train, y_train, scaler = prepare_data(df)

    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    return model, scaler

def predict_price(df, model, scaler, window_size=60):
    """
    Predict future price using the trained LSTM model.
    """
    scaled_data = scaler.transform(df['close'].values.reshape(-1, 1))
    last_sequence = scaled_data[-window_size:]
    last_sequence = np.reshape(last_sequence, (1, window_size, 1))

    predicted_price = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price[0][0]

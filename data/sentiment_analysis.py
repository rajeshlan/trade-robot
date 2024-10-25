#HOW TO RUN IS BELOW 

# python data\sentiment_analysis.py --train en

#python data\sentiment_analysis.py --evaluate en

#python data\sentiment_analysis.py --analyze "I love this product!" --real-time-score 

import pandas as pd
import numpy as np
import json
import logging
import os
import argparse
import requests
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from langdetect import detect
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Supported languages for sentiment analysis
SUPPORTED_LANGUAGES = ['en', 'es']  # Add other languages as needed

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data['text'].values, data['label'].values

# Function to preprocess text for sentiment analysis
def preprocess_text(tweets, tokenizer=None):
    logging.info("Preprocessing tweets...")
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    data = pad_sequences(sequences, maxlen=100)
    logging.info("Text preprocessing complete.")
    return data, tokenizer

# Save Tokenizer to File
def save_tokenizer(tokenizer, path='tokenizer.json'):
    logging.info(f"Saving tokenizer to {path}...")
    with open(path, 'w') as f:
        json.dump(tokenizer.to_json(), f)
    logging.info("Tokenizer saved.")

# Load Tokenizer from File
def load_tokenizer(path='tokenizer.json'):
    logging.info(f"Loading tokenizer from {path}...")
    with open(path) as f:
        tokenizer_json = json.load(f)
    tokenizer = Tokenizer()
    tokenizer_config = json.loads(tokenizer_json)

    if 'word_index' in tokenizer_config:
        tokenizer.word_index = tokenizer_config['word_index']
        logging.info("Tokenizer loaded successfully.")
    else:
        logging.error("Key 'word_index' not found in the tokenizer JSON.")
        raise KeyError("Key 'word_index' not found in the tokenizer JSON.")

# Build the Sentiment Analysis Model
def build_sentiment_model(input_length, embedding_dim=128, num_filters=128, kernel_size=5):
    logging.info("Building sentiment analysis model...")
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=embedding_dim, input_length=input_length))
    model.add(Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logging.info("Model built successfully.")
    return model

# Function to create and train the model
def train_model(X_train, y_train, X_val, y_val, vocab_size, embedding_dim, max_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(LSTM(64))
    model.add(Dropout(0.5))  # Added dropout layer for regularization
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'sentiment_model.h5', save_best_only=True, monitor='val_loss', verbose=1)

    logging.info("Training model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, checkpoint])
    return model

def generate_sentiment_heatmap(sentiment_data):
    """
    Generate a heatmap to visualize sentiment scores against price changes.

    :param sentiment_data: DataFrame containing sentiment scores and price data.
    """
    # Assuming the data has columns like 'Sentiment_Score' and 'Price_Change'
    sentiment_correlation = sentiment_data.corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sentiment_correlation, annot=True, cmap="Blues", linewidths=0.5)
    plt.title("Sentiment vs. Price Movement Heatmap")
    plt.show()


# Function to evaluate the model
def evaluate_model(model, X_val, y_val):
    logging.info("Evaluating model...")
    predictions = (model.predict(X_val) > 0.5).astype("int32")
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    logging.info(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Fetch Real-Time Sentiment Score from API
def fetch_real_time_sentiment():
    url = 'https://api.bybit.com'  # Ensure this is the correct endpoint
    headers = {'Authorization': 'u3Bm7IRMcAAc7RKCpn'}  # Replace with your actual API key
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses
            sentiment_data = response.json()
            logging.info(f"API response: {sentiment_data}")

            if 'score' in sentiment_data:
                sentiment_score = sentiment_data['score']
                logging.info(f"Real-time sentiment score: {sentiment_score}")
                return sentiment_score
            else:
                logging.error(f"Unexpected response format: {sentiment_data}")

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logging.error(f"Other error occurred: {err}")

    return None  # Return None after retries

# Detect Language of Text
def detect_language(text):
    logging.info("Detecting language...")
    return detect(text)

# Main Function for CLI Argument Parsing and Handling Different Functionalities
def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Bot')
    parser.add_argument('--train', type=str, help='Train the sentiment analysis model for a specified language (e.g., en, es)')
    parser.add_argument('--evaluate', type=str, help='Evaluate the sentiment analysis model for a specified language (e.g., en, es)')
    parser.add_argument('--analyze', type=str, help='Analyze sentiment of new tweets (comma-separated)')
    parser.add_argument('--real-time-score', action='store_true', help='Fetch real-time sentiment score')
    
    args = parser.parse_args()

    if args.train:
        lang = args.train
        if lang not in SUPPORTED_LANGUAGES:
            logging.error(f"Unsupported language: {lang}")
            return

        logging.info(f"Training model for language: {lang}...")
        # Load and preprocess data
        X, y = load_data('data/tweets.csv')  # Update with your data file path
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Tokenization and padding
        X_train_pad, tokenizer = preprocess_text(X_train)
        X_val_pad, _ = preprocess_text(X_val, tokenizer)

        # Save tokenizer
        save_tokenizer(tokenizer, f'tokenizer_{lang}.json')
        
        # Train the model
        model = train_model(X_train_pad, y_train, X_val_pad, y_val, vocab_size=len(tokenizer.word_index) + 1, embedding_dim=100, max_length=X_train_pad.shape[1])

        # Evaluate the model on validation set
        evaluate_model(model, X_val_pad, y_val)

    if args.evaluate:
        lang = args.evaluate
        if lang not in SUPPORTED_LANGUAGES:
            logging.error(f"Unsupported language: {lang}")
            return

        logging.info(f"Evaluating model for language: {lang}...")
        model = load_sentiment_model(f'sentiment_model_{lang}.h5')
        tokenizer = load_tokenizer(f'tokenizer_{lang}.json')

        # Load and preprocess validation data
        X, y = load_data('data/tweets.csv')  # Update with your data file path
        X_val, y_val = X, y  # Here we assume we're evaluating on the same dataset; adjust as necessary
        X_val_pad, _ = preprocess_text(X_val, tokenizer)

        evaluate_model(model, X_val_pad, y_val)

    if args.analyze:
        logging.info("Analyzing new tweets...")
        new_tweets = args.analyze.split(',')
        real_time_score = None

        if args.real_time_score:
            real_time_score = fetch_real_time_sentiment()

        for tweet in new_tweets:
            language = detect_language(tweet)
            logging.info(f"Detected language: {language}")

            if language in SUPPORTED_LANGUAGES:
                model = load_sentiment_model(f'sentiment_model_{language}.h5')
                tokenizer = load_tokenizer(f'tokenizer_{language}.json')
                data, _ = preprocess_text([tweet], tokenizer)
                prediction = model.predict(data)
                sentiment = 'positive' if prediction[0] > 0.5 else 'negative'
                logging.info(f"Sentiment for '{tweet}': {sentiment}")
            else:
                logging.error(f"Unsupported language detected: {language}")

if __name__ == "__main__":
    main()


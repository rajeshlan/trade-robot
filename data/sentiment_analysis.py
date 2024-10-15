# Required Imports
import logging
import os
import json
import requests
import numpy as np
import argparse
import unittest
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from langdetect import detect
from unittest.mock import patch

# Configure logging
logging.basicConfig(level=logging.INFO)

# Supported languages for sentiment analysis
SUPPORTED_LANGUAGES = ['en', 'es']  # Add other languages as needed

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

# Preprocess Text for Sentiment Analysis
def preprocess_text(tweets, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    data = pad_sequences(sequences, maxlen=100)
    return data, tokenizer

# Save Tokenizer to File
def save_tokenizer(tokenizer, path='tokenizer.json'):
    with open(path, 'w') as f:
        json.dump(tokenizer.to_json(), f)

# Load Tokenizer from File
def load_tokenizer(path='tokenizer.json'):
    with open(path) as f:
        tokenizer_json = json.load(f)
    tokenizer = Tokenizer()
    tokenizer_config = json.loads(tokenizer_json)

    if 'word_index' in tokenizer_config:
        tokenizer.word_index = tokenizer_config['word_index']
    else:
        logging.error("Key 'word_index' not found in the tokenizer JSON.")
        raise KeyError("Key 'word_index' not found in the tokenizer JSON.")

    return tokenizer

# Build the Sentiment Analysis Model
def build_sentiment_model(input_length, embedding_dim=128, num_filters=128, kernel_size=5):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=embedding_dim))
    model.add(Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Save the Sentiment Analysis Model to File
def save_model(model, path='sentiment_model.h5'):
    model.save(path)

# Load the Sentiment Analysis Model from File
def load_sentiment_model(path='sentiment_model.h5'):
    return load_model(path)

# Evaluate the Model using Precision, Recall, and F1 Score
def evaluate_model(model, X_val, y_val):
    predictions = (model.predict(X_val) > 0.5).astype("int32")
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    logging.info(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Analyze Sentiment of Tweets
def analyze_sentiment(tweets, model, tokenizer, real_time_score=None):
    data, _ = preprocess_text(tweets, tokenizer)
    predictions = model.predict(data)
    average_sentiment = np.mean(predictions)

    if real_time_score is not None:
        average_sentiment = (average_sentiment + real_time_score) / 2

    return average_sentiment

# Detect Language of Text
def detect_language(text):
    return detect(text)

# Main Function for CLI Argument Parsing and Handling Different Functionalities
def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Bot')
    parser.add_argument('--train', type=str, help='Train the sentiment analysis model for a specified language (e.g., en, es)')
    parser.add_argument('--evaluate', type=str, help='Evaluate the sentiment analysis model for a specified language (e.g., en, es)')
    parser.add_argument('--analyze', type=str, help='Analyze sentiment of new tweets (comma-separated)')
    parser.add_argument('--real-time-score', action='store_true', help='Fetch real-time sentiment score')
    parser.add_argument('--update-model', type=str, help='Update the sentiment analysis model with new data (comma-separated) for a specified language')

    args = parser.parse_args()

    if args.train:
        lang = args.train
        if lang not in SUPPORTED_LANGUAGES:
            logging.error(f"Unsupported language: {lang}")
            return

        # Sample data (replace with actual data)
        tweets = ["I love this!", "I hate this!", "This is the best!", "This is the worst!"]
        labels = [1, 0, 1, 0]  # Replace with actual labels

        data, tokenizer = preprocess_text(tweets)
        save_tokenizer(tokenizer, f'tokenizer_{lang}.json')
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

        model = build_sentiment_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_val, y_val))

        save_model(model, f'sentiment_model_{lang}.h5')
        evaluate_model(model, X_val, y_val)

    if args.evaluate:
        lang = args.evaluate
        if lang not in SUPPORTED_LANGUAGES:
            logging.error(f"Unsupported language: {lang}")
            return

        model = load_sentiment_model(f'sentiment_model_{lang}.h5')
        tokenizer = load_tokenizer(f'tokenizer_{lang}.json')

        # Sample data (replace with actual data)
        tweets = ["I love this!", "I hate this!", "This is the best!", "This is the worst!"]
        labels = [1, 0, 1, 0]  # Replace with actual labels
        data, _ = preprocess_text(tweets, tokenizer)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

        evaluate_model(model, X_val, y_val)

    if args.analyze:
        model = None
        tokenizer = None
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
            else:
                logging.error(f"Unsupported language: {language}")
                continue

            average_sentiment = analyze_sentiment([tweet], model, tokenizer, real_time_score)
            logging.info(f"Tweet: {tweet}, Sentiment score: {average_sentiment}")

    if args.update_model:
        lang = args.update_model
        if lang not in SUPPORTED_LANGUAGES:
            logging.error(f"Unsupported language: {lang}")
            return

        model = load_sentiment_model(f'sentiment_model_{lang}.h5')
        tokenizer = load_tokenizer(f'tokenizer_{lang}.json')

        # Sample new data (replace with actual new data)
        new_tweets = args.update_model.split(',')
        new_labels = [1, 0]  # Replace with actual new labels
        data, _ = preprocess_text(new_tweets, tokenizer)
        label_encoder = LabelEncoder()
        new_labels = label_encoder.fit_transform(new_labels)

        X_train, X_val, y_train, y_val = train_test_split(data, new_labels, test_size=0.2, random_state=42)

        model.fit(X_train, y_train, epochs=5, batch_size=2, validation_data=(X_val, y_val))

        save_model(model, f'sentiment_model_{lang}.h5')
        logging.info("Model updated successfully.")

# Unit Tests
class TestSentimentAnalysis(unittest.TestCase):
    def test_language_detection(self):
        text = "This is a test."
        language = detect_language(text)
        self.assertEqual(language, 'en')

    @patch('requests.get')
    def test_fetch_real_time_sentiment(self, mock_get):
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {'score': 0.75}
        score = fetch_real_time_sentiment()
        self.assertEqual(score, 0.75)

    def test_preprocess_text(self):
        tweets = ["I love this!", "I hate this!"]
        data, tokenizer = preprocess_text(tweets)
        self.assertEqual(len(data), len(tweets))
        self.assertIsNotNone(tokenizer)

# Run tests if this file is executed directly
if __name__ == "__main__":
    main()
    unittest.main()

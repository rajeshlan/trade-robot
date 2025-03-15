#HOW TO RUN IS BELOW 

# python data\sentiment_analysis.py --train en

# python data\sentiment_analysis.py --evaluate en

# python data\sentiment_analysis.py --analyze "I love this product!" --real-time-score 

import pandas as pd
import numpy as np
import json
import logging
import os
import argparse
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from langdetect import detect
from textblob import TextBlob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Supported languages for sentiment analysis
SUPPORTED_LANGUAGES = ['en', 'es']  # Add other languages as needed

def fetch_real_time_sentiment():
    text = "The market looks strong and bullish."  # Example input
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return polarity  # Returns a score between -1 (negative) and 1 (positive)

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    logging.info(f"Class distribution: {data['label'].value_counts()}")
    return data['text'].values, data['label'].values

# Preprocess text using TF-IDF Vectorizer
def preprocess_text_tfidf(tweets, vectorizer=None):
    logging.info("Preprocessing tweets using TF-IDF...")
    
    if vectorizer is None:
        # If no vectorizer is provided, fit a new one
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(tweets)
    else:
        # Check if the vectorizer is fitted
        if not hasattr(vectorizer, 'vocabulary_'):
            # If not fitted, fit the vectorizer first
            logging.info("Vectorizer not fitted. Fitting vectorizer...")
            X = vectorizer.fit_transform(tweets)
        else:
            # If fitted, just transform the tweets
            X = vectorizer.transform(tweets)
    
    logging.info("TF-IDF preprocessing complete.")
    return X, vectorizer

# Save Vectorizer to File
def save_vectorizer(vectorizer, path='vectorizer.json'):
    logging.info(f"Saving vectorizer to {path}...")
    # Save the fitted vectorizer with joblib
    import joblib
    joblib.dump(vectorizer, path)  # Save the entire fitted vectorizer
    logging.info("Vectorizer saved.")

# Load Vectorizer from File
def load_vectorizer(path='vectorizer.json'):
    logging.info(f"Loading vectorizer from {path}...")
    import joblib
    return joblib.load(path)  # Load the entire fitted vectorizer

# Build and Train the Sentiment Analysis Model
def train_model(X_train, y_train):
    logging.info("Training Logistic Regression model for sentiment analysis...")
    model = LogisticRegression(max_iter=500, class_weight='balanced')  # Add class_weight='balanced' here
    model.fit(X_train, y_train)
    logging.info("Model trained successfully.")
    return model

# Save the Model
def save_model(model, path='sentiment_model.pkl'):
    import joblib
    logging.info(f"Saving model to {path}...")
    joblib.dump(model, path)
    logging.info("Model saved.")

# Load the Model
def load_model(path='sentiment_model.pkl'):
    import joblib
    logging.info(f"Loading model from {path}...")
    return joblib.load(path)

# Evaluate the Model
def evaluate_model(model, X_val, y_val):
    logging.info("Evaluating model...")
    predictions = model.predict(X_val)
    
    precision = precision_score(y_val, predictions, average='binary', zero_division=1)
    recall = recall_score(y_val, predictions, average='binary', zero_division=1)
    f1 = f1_score(y_val, predictions, average='binary', zero_division=1)
    
    logging.info(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    logging.info("Classification Report:\n" + classification_report(y_val, predictions, zero_division=1))
    
    # Return the metrics
    return precision, recall, f1

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

        # TF-IDF Vectorization
        X_train_tfidf, vectorizer = preprocess_text_tfidf(X_train)
        X_val_tfidf, _ = preprocess_text_tfidf(X_val, vectorizer)

        # Save vectorizer
        save_vectorizer(vectorizer, f'vectorizer_{lang}.json')

        # Train the model
        model = train_model(X_train_tfidf, y_train)

        # Save the model
        save_model(model, f'sentiment_model_{lang}.pkl')

        # Evaluate the model on validation set
        evaluate_model(model, X_val_tfidf, y_val)

    if args.evaluate:
        lang = args.evaluate
        if lang not in SUPPORTED_LANGUAGES:
            logging.error(f"Unsupported language: {lang}")
            return

        logging.info(f"Evaluating model for language: {lang}...")
        model = load_model(f'sentiment_model_{lang}.pkl')
        vectorizer = load_vectorizer(f'vectorizer_{lang}.json')

        # Load and preprocess validation data
        X, y = load_data('data/tweets.csv')  # Update with your data file path
        X_val_tfidf, _ = preprocess_text_tfidf(X, vectorizer)

        evaluate_model(model, X_val_tfidf, y)

    if args.analyze:
        logging.info("Analyzing new tweets...")
        new_tweets = args.analyze.split(',')
        for tweet in new_tweets:
            language = detect_language(tweet)
            logging.info(f"Detected language: {language}")

            if language in SUPPORTED_LANGUAGES:
                model = load_model(f'sentiment_model_{language}.pkl')
                vectorizer = load_vectorizer(f'vectorizer_{language}.json')
                data, _ = preprocess_text_tfidf([tweet], vectorizer)
                prediction = model.predict(data)
                sentiment = 'positive' if prediction[0] > 0.5 else 'negative'
                logging.info(f"Sentiment for '{tweet}': {sentiment}")
            else:
                logging.error(f"Unsupported language detected: {language}")

if __name__ == "__main__":
    main()

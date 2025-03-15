#python -m unittest tests.test_sentiment_analysis (runs on this)


import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from data.sentiment_analysis import (
    preprocess_text_tfidf,
    train_model,
    evaluate_model,
)



class TestSentimentAnalysis(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.tweets = [
            "I love this product!",
            "This is the worst experience.",
            "Had a fantastic day!",
            "Absolutely terrible service."
        ]
        self.labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

    def test_preprocess_text_tfidf(self):
        # Test the TF-IDF preprocessing
        vectorizer = TfidfVectorizer(max_features=10)
        X, vec = preprocess_text_tfidf(self.tweets, vectorizer)
        self.assertEqual(X.shape[1], 10)  # Ensure max_features is respected
        self.assertEqual(len(vec.get_feature_names_out()), 10)

    def test_train_model(self):
        # Test model training
        vectorizer = TfidfVectorizer(max_features=10)
        X, _ = preprocess_text_tfidf(self.tweets, vectorizer)
        model = train_model(X, self.labels)
        self.assertIsInstance(model, LogisticRegression)

    def test_evaluate_model(self):
        # Test model evaluation
        vectorizer = TfidfVectorizer(max_features=10)
        X, _ = preprocess_text_tfidf(self.tweets, vectorizer)
        X_train, X_test = X[:3], X[3:]
        y_train, y_test = self.labels[:3], self.labels[3:]
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        precision, recall, f1 = evaluate_model(model, X_test, y_test)
        self.assertGreaterEqual(precision, 0)  # Precision should be valid
        self.assertGreaterEqual(recall, 0)  # Recall should be valid
        self.assertGreaterEqual(f1, 0)  # F1 score should be valid

if __name__ == "__main__":
    unittest.main()

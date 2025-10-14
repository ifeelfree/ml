import numpy as np
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class NaiveBayesSentiment:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Smoothing parameter
        self.class_priors = {}
        self.word_probs = {}
        self.vocab = set()
        self.class_word_counts = {}
        self.class_total_words = {}

    def preprocess(self, text):
        """Simple text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

    def fit(self, X, y):
        """Train the Naive Bayes classifier"""
        # Count documents per class
        class_counts = defaultdict(int)
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_total_words = defaultdict(int)

        # Process each document
        for doc, label in zip(X, y):
            class_counts[label] += 1
            words = self.preprocess(doc)

            for word in words:
                self.vocab.add(word)
                self.class_word_counts[label][word] += 1
                self.class_total_words[label] += 1

        # Calculate prior probabilities
        total_docs = len(X)
        for label in class_counts:
            self.class_priors[label] = class_counts[label] / total_docs

        # Calculate word probabilities with smoothing
        vocab_size = len(self.vocab)
        self.word_probs = {}

        for label in class_counts:
            self.word_probs[label] = {}
            for word in self.vocab:
                count = self.class_word_counts[label][word]
                self.word_probs[label][word] = (
                        (count + self.alpha) /
                        (self.class_total_words[label] + self.alpha * vocab_size)
                )

    def predict_proba(self, X):
        """Predict probabilities for each class"""
        predictions = []

        for doc in X:
            words = self.preprocess(doc)
            class_scores = {}

            for label in self.class_priors:
                # Start with log prior
                log_prob = np.log(self.class_priors[label])

                # Add log probabilities of words
                for word in words:
                    if word in self.vocab:
                        log_prob += np.log(self.word_probs[label][word])
                    else:
                        # Handle unknown words with smoothing
                        log_prob += np.log(
                            self.alpha /
                            (self.class_total_words[label] + self.alpha * len(self.vocab))
                        )

                class_scores[label] = log_prob

            # Convert log probabilities back to probabilities
            max_log_prob = max(class_scores.values())
            exp_scores = {label: np.exp(score - max_log_prob)
                          for label, score in class_scores.items()}
            total = sum(exp_scores.values())
            probabilities = {label: score / total
                             for label, score in exp_scores.items()}
            predictions.append(probabilities)

        return predictions

    def predict(self, X):
        """Predict the class with highest probability"""
        probas = self.predict_proba(X)
        return [max(proba, key=proba.get) for proba in probas]


# Example usage
if __name__ == "__main__":
    # Training data
    X_train = [
        "This movie is absolutely fantastic and amazing",
        "I love this film, it's wonderful",
        "Best movie I've ever seen",
        "Terrible movie, waste of time",
        "I hate this film, it's awful",
        "Worst movie ever, completely boring"
    ]
    y_train = ["positive", "positive", "positive",
               "negative", "negative", "negative"]

    # Create and train classifier
    nb = NaiveBayesSentiment(alpha=1.0)
    nb.fit(X_train, y_train)

    # Test data
    X_test = [
        "This is a fantastic film",
        "Terrible and boring movie",
        "I think this movie is okay"
    ]

    # Predictions
    predictions = nb.predict(X_test)
    probabilities = nb.predict_proba(X_test)

    print("Predictions:")
    for text, pred, proba in zip(X_test, predictions, probabilities):
        print(f"\nText: '{text}'")
        print(f"Prediction: {pred}")
        print(f"Probabilities: {proba}")

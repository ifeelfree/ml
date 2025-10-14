from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np


class ProfessionalSentimentAnalyzer:
    def __init__(self):
        """
        Initialize with TF-IDF vectorizer and Naive Bayes classifier

        TF-IDF: A technique that converts text to numbers while considering
                word importance (frequent words in a document but rare overall
                are considered more important)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Use only top 1000 most frequent words
            stop_words='english',  # Remove common words like 'the', 'is', 'at'
            ngram_range=(1, 2)  # Consider single words and word pairs
        )
        self.classifier = MultinomialNB()

    def prepare_data(self, texts, labels):
        """Convert text reviews into numerical features"""
        # Transform text into numerical features
        features = self.vectorizer.fit_transform(texts)
        return features

    def train(self, texts, labels):
        """Train the sentiment analyzer"""
        print("Preparing training data...")
        features = self.prepare_data(texts, labels)

        print("Training the model...")
        self.classifier.fit(features, labels)

        # Calculate training accuracy
        predictions = self.classifier.predict(features)
        accuracy = accuracy_score(labels, predictions)
        print(f"Training complete! Accuracy on training data: {accuracy:.2%}")

    def predict(self, texts):
        """Predict sentiment for new reviews"""
        # Transform text using the same vectorizer
        features = self.vectorizer.transform(texts)

        # Get predictions and probabilities
        predictions = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)

        return predictions, probabilities

    def analyze_reviews(self, reviews):
        """Analyze reviews with detailed output"""
        predictions, probabilities = self.predict(reviews)

        print("\n" + "=" * 60)
        print("SENTIMENT ANALYSIS RESULTS")
        print("=" * 60)

        for review, pred, prob in zip(reviews, predictions, probabilities):
            sentiment = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
            confidence = max(prob) * 100

            print(f"\nReview: '{review}'")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.1f}%")
            print(f"Probability breakdown: Negative={prob[0]:.2%}, Positive={prob[1]:.2%}")

    def show_important_words(self):
        """Display which words are most indicative of each sentiment"""
        # Get feature names (words)
        feature_names = self.vectorizer.get_feature_names_out()

        # Get the log probabilities for each word in each class
        neg_log_probs = self.classifier.feature_log_prob_[0]
        pos_log_probs = self.classifier.feature_log_prob_[1]

        # Find most positive and negative words
        word_scores = []
        for i, word in enumerate(feature_names):
            score = pos_log_probs[i] - neg_log_probs[i]
            word_scores.append((word, score))

        word_scores.sort(key=lambda x: x[1])

        print("\n" + "=" * 60)
        print("MOST INDICATIVE WORDS")
        print("=" * 60)

        print("\n🔴 Top words indicating NEGATIVE sentiment:")
        for word, score in word_scores[:10]:
            print(f"   • {word}")

        print("\n🟢 Top words indicating POSITIVE sentiment:")
        for word, score in word_scores[-10:]:
            print(f"   • {word}")

    def evaluate(self, test_texts, test_labels):
        """Evaluate model performance on test data"""
        predictions, probabilities = self.predict(test_texts)

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        print(f"\nAccuracy: {accuracy:.2%}")

        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(test_labels, predictions,
                                    target_names=['Negative', 'Positive']))

        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        print("Confusion Matrix:")
        print("                 Predicted")
        print("                 Neg   Pos")
        print(f"Actual Negative: {cm[0, 0]:3d}   {cm[0, 1]:3d}")
        print(f"       Positive: {cm[1, 0]:3d}   {cm[1, 1]:3d}")

        return accuracy


# Demonstration with a larger dataset
if __name__ == "__main__":
    # Create a more substantial training dataset
    training_data = {
        'review': [
            # Positive reviews
            "This movie exceeded all my expectations absolutely brilliant",
            "Fantastic storyline with amazing special effects loved it",
            "Best film of the year outstanding performances",
            "Highly recommend this masterpiece to everyone",
            "Wonderful experience great acting and direction",
            "Absolutely loved this movie from start to finish",
            "Brilliant cinematography and excellent soundtrack",
            "Amazing film that touches your heart deeply",
            "Perfect movie for the whole family to enjoy",
            "Exceptional storytelling with memorable characters",
            "Outstanding performances by all actors truly remarkable",
            "Beautifully crafted film with stunning visuals",
            "Incredible movie that keeps you engaged throughout",
            "Superb direction and flawless execution loved every minute",
            "Must watch movie of the year absolutely phenomenal",

            # Negative reviews
            "Terrible movie waste of time and money",
            "Boring plot with awful acting throughout",
            "Worst film I have ever seen completely disappointing",
            "Poorly written script and terrible direction",
            "Complete disaster avoid at all costs",
            "Awful movie with no redeeming qualities whatsoever",
            "Disappointing film that fails on every level",
            "Terrible acting and boring storyline throughout",
            "Waste of talent and resources utterly bad",
            "Horrible movie experience want my money back",
            "Predictable plot and poor character development",
            "Extremely boring and poorly executed film",
            "Terrible screenplay with awful dialogue",
            "Complete waste of time terrible in every way",
            "Disappointing and boring from start to finish"
        ],
        'sentiment': [1] * 15 + [0] * 15  # 1=positive, 0=negative
    }

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(training_data)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'],
        df['sentiment'],
        test_size=0.3,
        random_state=42,
        stratify=df['sentiment']  # Ensure balanced split
    )

    print("=" * 60)
    print("PROFESSIONAL SENTIMENT ANALYZER")
    print("=" * 60)
    print(f"\nDataset size: {len(df)} reviews")
    print(f"Training set: {len(X_train)} reviews")
    print(f"Test set: {len(X_test)} reviews")
    print(f"Class distribution: {sum(df['sentiment'])} positive, {len(df) - sum(df['sentiment'])} negative")

    # Initialize and train the analyzer
    analyzer = ProfessionalSentimentAnalyzer()

    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    analyzer.train(X_train.tolist(), y_train.tolist())

    # Evaluate on test set
    analyzer.evaluate(X_test.tolist(), y_test.tolist())

    # Show important words
    analyzer.show_important_words()

    # Test on new, unseen reviews
    print("\n" + "=" * 60)
    print("TESTING ON NEW REVIEWS")
    print("=" * 60)

    new_reviews = [
        "This movie is absolutely fantastic and amazing",
        "Terrible film, complete waste of time",
        "Not bad but not great either, just okay",
        "Brilliant performances and outstanding cinematography",
        "Boring and disappointing, fell asleep halfway",
        "One of the best movies I've seen this year",
        "Awful acting and terrible script",
        "Surprisingly good, exceeded my expectations"
    ]

    analyzer.analyze_reviews(new_reviews)

    # Additional analysis: Feature importance visualization
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    # Get vocabulary statistics
    vocab = analyzer.vectorizer.vocabulary_
    print(f"\nVocabulary size: {len(vocab)} unique features")
    print(f"(includes both single words and word pairs)")

    # Show some example features
    sample_features = list(vocab.keys())[:20]
    print(f"\nSample features: {sample_features}")

    # Analyze prediction confidence distribution
    _, all_probs = analyzer.predict(new_reviews)
    confidences = [max(prob) * 100 for prob in all_probs]

    print(f"\nPrediction Confidence Statistics:")
    print(f"  Average confidence: {np.mean(confidences):.1f}%")
    print(f"  Min confidence: {np.min(confidences):.1f}%")
    print(f"  Max confidence: {np.max(confidences):.1f}%")

    # Performance summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ Model trained successfully")
    print("✓ TF-IDF features extracted")
    print("✓ Naive Bayes classifier fitted")
    print("✓ Ready for sentiment prediction")

    # Interactive prediction (optional)
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("You can now test the analyzer with your own reviews!")
    print("Type 'quit' to exit\n")

    while True:
        user_review = input("Enter a movie review: ")
        if user_review.lower() == 'quit':
            print("Thank you for using the Sentiment Analyzer!")
            break

        if user_review.strip():
            analyzer.analyze_reviews([user_review])
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # VADER's lexicon can be accessed via analyzer.lexicon
        self.positive_words = set(word for word, score in self.analyzer.lexicon.items() if score > 0)
        self.negative_words = set(word for word, score in self.analyzer.lexicon.items() if score < 0)

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of a given text using VADER.
        Returns a dictionary with 'neg', 'neu', 'pos', 'compound' scores,
        and 'positive_word_count', 'negative_word_count'.
        """
        sentiment_scores = self.analyzer.polarity_scores(text)

        # Calculate emotional word density
        words = re.findall(r'\b\w+\b', text.lower()) # Simple word tokenization
        positive_word_count = sum(1 for word in words if word in self.positive_words)
        negative_word_count = sum(1 for word in words if word in self.negative_words)

        sentiment_scores['positive_word_count'] = positive_word_count
        sentiment_scores['negative_word_count'] = negative_word_count

        return sentiment_scores

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of a given text using VADER.
        Returns a dictionary with 'neg', 'neu', 'pos', and 'compound' scores.
        """
        return self.analyzer.polarity_scores(text)

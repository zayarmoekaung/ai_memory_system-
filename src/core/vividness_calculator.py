from typing import Dict, Any
import datetime
import time
import math
from datetime import datetime

from config.settings import settings
from src.core.sentiment_analyzer import SentimentAnalyzer
from src.core.entity_extractor import EntityExtractor
from src.core.chunk_optimizer import ChunkOptimizer

class VividnessCalculator:
    def __init__(self, sentiment_analyzer: SentimentAnalyzer, entity_extractor: EntityExtractor, chunk_optimizer: ChunkOptimizer):
        self.sentiment_analyzer = sentiment_analyzer
        self.entity_extractor = entity_extractor
        self.chunk_optimizer = chunk_optimizer

    def calculate_initial_vividness(self, chunk_content: str) -> float:
        """
        Calculates an initial vividness score based on content richness, emotional density, and specificity.
        """
        # 1. Text Length and Complexity (using token count as a proxy)
        token_count = self.chunk_optimizer.count_tokens(chunk_content)
        # Normalize token count to a score (e.g., max 500 tokens = 1.0, adjust as needed)
        length_score = min(1.0, token_count / 50.0) # Adjusted divisor for better range

        # 2. Emotional Word Density
        sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(chunk_content)
        positive_word_count = sentiment_analysis.get('positive_word_count', 0)
        negative_word_count = sentiment_analysis.get('negative_word_count', 0)
        total_emotional_words = positive_word_count + negative_word_count
        
        # Avoid division by zero if no words are found
        words_in_chunk = len(chunk_content.split())
        emotional_density_score = min(1.0, total_emotional_words / words_in_chunk) if words_in_chunk > 0 else 0.0

        # 3. Specificity/Detail
        specificity_metrics = self.entity_extractor.quantify_specificity(chunk_content, settings.RELEVANT_ENTITY_TYPES)
        total_entities = specificity_metrics.get('total_entities_count', 0)
        unique_entity_types = specificity_metrics.get('unique_entity_types_count', 0)
        
        # Normalize specificity scores (adjust divisors as needed based on expected max values)
        # Example weighting: more weight on unique types for diversity of detail
        specificity_score = min(1.0, (total_entities / 5.0) + (unique_entity_types / 3.0)) # Adjusted divisors

        # Combine factors (these weights can be tuned in settings if desired)
        # For now, a simple average or weighted sum
        combined_score = (
            0.35 * length_score +
            0.35 * emotional_density_score +
            0.30 * specificity_score
        )
        
        # Clamp between 0 and 1
        return max(0.0, min(1.0, combined_score))

    def apply_decay(self, initial_vividness: float, timestamp: float) -> float:
        """
        Applies temporal (exponential) decay to the vividness score.
        """
        now = time.time()
        age_seconds = now - timestamp
        
        # Ensure age_seconds is non-negative
        if age_seconds < 0: 
            print(f"  [DEBUG] Decay - Warning: Negative age_seconds detected ({age_seconds:.2f}). Setting to 0.")
            age_seconds = 0 # Should not happen if timestamps are correct

        # Use an exponential decay model: initial_vividness * e^(-decay_rate * age)
        # settings.VIVIDNESS_DECAY_RATE is expected to be a small positive number
        exp_term = math.exp(-settings.VIVIDNESS_DECAY_RATE * age_seconds)
        decayed_vividness = initial_vividness * exp_term
        
        print(f"  [DEBUG] Decay - Initial vividness: {initial_vividness:.4f}, Timestamp used: {datetime.fromtimestamp(timestamp)}, Current time: {datetime.fromtimestamp(now)}, Age seconds: {age_seconds:.2f}, Decay factor: {settings.VIVIDNESS_DECAY_RATE}, Exp term: {exp_term:.4f}, Calculated decayed: {decayed_vividness:.4f}")

        # Ensure vividness doesn't go below zero (though exp will keep it positive)
        return max(0.0, decayed_vividness)

from transformers import pipeline

class ContextTagger:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        # Define default candidate labels for context tagging
        self.default_candidate_labels = [
            "personal reflection", "technical detail", "task management",
            "emotional state", "learning experience", "social interaction",
            "planning", "decision making", "environmental observation"
        ]

    def tag_context(self, text: str, candidate_labels: list[str] = None) -> list[str]:
        """
        Tags the context of a given text using zero-shot classification.
        Returns a list of the top predicted tags.
        """
        if candidate_labels is None:
            candidate_labels = self.default_candidate_labels

        # Perform zero-shot classification
        result = self.classifier(text, candidate_labels, multi_label=True)

        # Sort by score and return the labels above a certain threshold, or top N
        # For simplicity, let's take the top N highest-scoring labels
        # You can adjust this threshold or N based on desired granularity
        threshold = 0.6 # Example threshold
        context_tags = [
            label for label, score in zip(result['labels'], result['scores'])
            if score > threshold
        ]
        return context_tags if context_tags else [result['labels'][0]] # Fallback to highest if none above threshold

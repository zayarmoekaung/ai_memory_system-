# Phase 1: Detailed Implementation Plan for NLP Processing Functions

**Objective:** Replace current basic heuristic placeholder functions in `RetrievalManager` with actual production-ready NLP logic for Sentiment Analysis, Named Entity Recognition, and Context Tag Extraction.

---

## 1. Sentiment Analysis (`_simple_sentiment_analysis`)

*   **Current:** Basic keyword counting heuristic.
*   **Target:** Use `VADER` for sentiment analysis.

**Detailed Steps:**

1.  **Add Dependency:**
    *   Open `ai_memory_system/requirements.txt`.
    *   Add `vaderSentiment` to a new line in the file.
    *   Run `pip install -r requirements.txt` in the `ai_memory_system` directory (after activating the virtual environment, if applicable).

2.  **Create Sentiment Analyzer Class:**
    *   Create a new file: `ai_memory_system/src/core/sentiment_analyzer.py`.
    *   **Content for `sentiment_analyzer.py`:**
        ```python
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
        ```

3.  **Integrate into `RetrievalManager`:**
    *   Open `ai_memory_system/src/core/retrieval_manager.py`.
    *   **Import:** Add `from src.core.sentiment_analyzer import SentimentAnalyzer` at the top.
    *   **Instantiation:** In the `RetrievalManager.__init__` method, add `self.sentiment_analyzer = SentimentAnalyzer()`.
    *   **Replace Placeholder:** Locate the `_extract_enhanced_metadata` method (or wherever sentiment is currently being extracted) and modify it to call the new `SentimentAnalyzer` and assign `emotional_valence`.
        ```python
        # Example modification within RetrievalManager._extract_enhanced_metadata
        def _extract_enhanced_metadata(self, text: str) -> dict:
            # ... existing metadata extraction ...

            # Sentiment Analysis
            sentiment_scores = self.sentiment_analyzer.analyze_sentiment(text)
            emotional_valence = sentiment_scores['compound'] # Using compound score for overall valence

            return {
                # ... other metadata ...
                "emotional_valence": emotional_valence,
                "sentiment_scores": sentiment_scores, # Optionally store all scores
            }
        ```
    *   Ensure the `ingest_memory` method calls this updated metadata extraction logic.

4.  **Configuration (Optional but Recommended):**
    *   Open `ai_memory_system/config/settings.py`.
    *   Add a setting like `SENTIMENT_ANALYZER_MODEL: str = "vader"` if there's a future need to switch between different sentiment models.

---

## 2. Named Entity Recognition (`_simple_entity_extraction`)

*   **Current:** Capitalized word detection heuristic.
*   **Target:** Use `spaCy` with `en_core_web_sm` model.

**Detailed Steps:**

1.  **Add Dependencies and Download Model:**
    *   Open `ai_memory_system/requirements.txt`.
    *   Add `spacy` to a new line.
    *   Run `pip install -r requirements.txt` in the `ai_memory_system` directory.
    *   **After installing spaCy, download the English model:**
        ```bash
        python -m spacy download en_core_web_sm
        ```

2.  **Create Entity Extractor Class:**
    *   Create a new file: `ai_memory_system/src/core/entity_extractor.py`.
    *   **Content for `entity_extractor.py`:**
        ```python
        import spacy

        class EntityExtractor:
            def __init__(self, model_name: str = "en_core_web_sm"):
                try:
                    self.nlp = spacy.load(model_name)
                except OSError:
                    print(f"SpaCy model '{model_name}' not found. Please run 'python -m spacy download {model_name}'")
                    raise

            def extract_entities(self, text: str) -> list[dict]:
                """
                Extracts named entities from a given text using spaCy.
                Returns a list of dictionaries, each with 'text' and 'label' (entity type).
                """
                doc = self.nlp(text)
                entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
                return entities

            def filter_entities_by_type(self, entities: list[dict], entity_types: list[str]) -> list[str]:
                """
                Filters extracted entities by a list of desired entity types (e.g., ["PERSON", "ORG", "GPE"]).
                Returns a list of unique entity texts.
                """
                filtered = []
                for ent in entities:
                    if ent['label'] in entity_types:
                        filtered.append(ent['text'])
                return list(set(filtered))
        ```

3.  **Integrate into `RetrievalManager`:**
    *   Open `ai_memory_system/src/core/retrieval_manager.py`.
    *   **Import:** Add `from src.core.entity_extractor import EntityExtractor` at the top.
    *   **Instantiation:** In the `RetrievalManager.__init__` method, add `self.entity_extractor = EntityExtractor()`.
    *   **Replace Placeholder:** Locate the `_extract_enhanced_metadata` method and modify it:
        ```python
        # Example modification within RetrievalManager._extract_enhanced_metadata
        def _extract_enhanced_metadata(self, text: str) -> dict:
            # ... existing metadata extraction ...

            # Named Entity Recognition
            all_entities = self.entity_extractor.extract_entities(text)
            # Filter for specific entity types that are useful for association
            relevant_entity_types = ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"] # Define based on need
            associated_entities = self.entity_extractor.filter_entities_by_type(all_entities, relevant_entity_types)

            return {
                # ... other metadata ...
                "associated_entities": associated_entities,
                "all_extracted_entities": all_entities, # Optionally store all extracted entities
            }
        ```
    *   Ensure `ingest_memory` uses this updated logic and that `AssociativeNetwork` correctly uses `associated_entities`.

4.  **Configuration:**
    *   Open `ai_memory_system/config/settings.py`.
    *   Add `NER_MODEL_NAME: str = "en_core_web_sm"` and `RELEVANT_ENTITY_TYPES: list[str] = ["PERSON", "ORG", "GPE", "LOC"]` to allow easy modification of the spaCy model and desired entity types.

---

## 3. Context Tag Extraction (`_simple_context_tag_extraction`)

*   **Current:** Hardcoded keyword matching.
*   **Target:** Use zero-shot classification with a HuggingFace transformer model.

**Detailed Steps:**

1.  **Add Dependency:**
    *   Open `ai_memory_system/requirements.txt`.
    *   Add `transformers` to a new line.
    *   Run `pip install -r requirements.txt` in the `ai_memory_system` directory.

2.  **Create Context Tagger Class:**
    *   Create a new file: `ai_memory_system/src/core/context_tagger.py`.
    *   **Content for `context_tagger.py`:**
        ```python
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
        ```

3.  **Integrate into `RetrievalManager`:**
    *   Open `ai_memory_system/src/core/retrieval_manager.py`.
    *   **Import:** Add `from src.core.context_tagger import ContextTagger` at the top.
    *   **Instantiation:** In the `RetrievalManager.__init__` method, add `self.context_tagger = ContextTagger()`.
    *   **Replace Placeholder:** Locate the `_extract_enhanced_metadata` method and modify it:
        ```python
        # Example modification within RetrievalManager._extract_enhanced_metadata
        def _extract_enhanced_metadata(self, text: str) -> dict:
            # ... existing metadata extraction ...

            # Context Tag Extraction
            context_tags = self.context_tagger.tag_context(text)

            return {
                # ... other metadata ...
                "context_tags": context_tags,
            }
        ```

4.  **Configuration:**
    *   Open `ai_memory_system/config/settings.py`.
    *   Add `CONTEXT_TAGGER_MODEL: str = "facebook/bart-large-mnli"` and `DEFAULT_CONTEXT_LABELS: list[str] = [...]` (matching the `default_candidate_labels` in `ContextTagger`) to enable easy updates to the model and labels.

---

### **General Considerations for All Implementations:**

*   **Error Handling:** Add robust `try-except` blocks around NLP model loading and inference calls to handle potential errors (e.g., model not found, inference failure).
*   **Logging:** Incorporate logging to track the NLP processing, especially for debugging purposes.
*   **Performance:** Be mindful of the performance impact. Initializing `pipeline` for transformers can be slow, so ensure it's done once in the constructor. If performance becomes an issue, consider a smaller model or batch processing.
*   **Virtual Environment:** Always perform `pip install` commands and run scripts within the activated virtual environment to avoid conflicts.
*   **Testing:** After implementing each component, write unit tests to ensure it functions as expected before integrating it fully. The `examples/` directory is a good place to add simple tests for these new features.

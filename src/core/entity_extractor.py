import spacy
from collections import Counter

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

    def quantify_specificity(self, text: str, relevant_entity_types: list[str]) -> dict:
        """
        Quantifies the specificity of the text by counting unique occurrences of relevant entity types.
        Returns a dictionary with 'total_entities_count' and 'unique_entity_types_count'.
        """
        doc = self.nlp(text)
        entities = [ent.label_ for ent in doc.ents if ent.label_ in relevant_entity_types]
        
        total_entities_count = len(entities)
        unique_entity_types_count = len(set(entities))

        return {
            "total_entities_count": total_entities_count,
            "unique_entity_types_count": unique_entity_types_count
        }

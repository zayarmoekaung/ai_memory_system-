from sentence_transformers import SentenceTransformer
from typing import List
import os
from ..config.settings import settings

class EmbeddingManager:
    def __init__(self):
        """
        Initializes the EmbeddingManager by loading the pre-trained SentenceTransformer model.
        The model name is fetched from the application settings.
        """
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates a vector embedding for the given text.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        embeddings = self.model.encode(text)
        return embeddings.tolist()

    def get_tokenizer(self):
        """
        Returns the tokenizer associated with the loaded SentenceTransformer model.
        This tokenizer can be used for consistent token counting.

        Returns:
            PreTrainedTokenizer: The tokenizer instance.
        """
        return self.model.tokenizer

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Temporarily set up environment for testing if .env is not loaded by default
    # In a real application, settings would be loaded once at startup
    os.environ["EMBEDDING_MODEL_NAME"] = "all-MiniLM-L6-v2"

    embedding_manager = EmbeddingManager()
    text_to_embed = "This is a test sentence for embedding."
    embedding = embedding_manager.get_embedding(text_to_embed)
    print(f"Text: {text_to_embed}")
    print(f"Embedding shape: {len(embedding)}")
    print(f"First 5 embedding dimensions: {embedding[:5]}")

    tokenizer = embedding_manager.get_tokenizer()
    tokens = tokenizer.tokenize(text_to_embed)
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # Core Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHROMA_DB_PATH: Path = Path("./data/chroma_db")
    MAX_CONTEXT_TOKENS: int = 4000  # Example token limit for a typical LLM context window

    # Retrieval Weighting Coefficients (tuned for human-like recall)
    RECENCY_WEIGHT: float = 0.5
    IMPORTANCE_WEIGHT: float = 0.2
    TASK_RELATEDNESS_WEIGHT: float = 0.2 # Increased significance
    ASSOCIATIVE_STRENGTH_WEIGHT: float = 0.1 # New weight for explicit links
    EMOTIONAL_SALIENCE_WEIGHT: float = 0.1 # New weight for emotional valence
    VIVIDNESS_WEIGHT: float = 0.05 # New weight for vividness, can be adjusted

    # Chunking Parameters
    CHUNK_SIZE_SENTENCES: int = 3  # Target number of sentences per chunk
    CHUNK_OVERLAP_SENTENCES: int = 1 # Overlap between chunks to maintain context

    # Human-like Memory Specific Settings
    WORKING_MEMORY_CAPACITY: int = 10 # Number of recent items to keep in working memory
    MEMORY_CONSOLIDATION_INTERVAL_SECONDS: int = 3600 # How often background consolidation runs (e.g., 1 hour)

    # NLP Models for Metadata Extraction (placeholders for now)
    SENTIMENT_MODEL_NAME: str = "distilbert-base-uncased-sentiment" # Example sentiment model
    ENTITY_EXTRACTION_MODEL_NAME: str = "dslim/bert-base-NER" # Example NER model
    
    # Decay rates for dynamic scores (conceptual, for future tuning)
    VIVIDNESS_DECAY_RATE: float = 0.01 # How quickly vividness fades over time
    ASSOCIATIVE_STRENGTH_DECAY_RATE: float = 0.05 # How quickly associative links weaken if not reinforced

    # Optional: API Settings (if FastAPI is used)
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    @property
    def CHROMA_COLLECTION_NAME(self) -> str:
        return "ai_memory_collection"

    @property
    def CHROMA_ASSOCIATIVE_COLLECTION_NAME(self) -> str:
        return "ai_associative_links"

# Instantiate settings
settings = Settings()

# Ensure the ChromaDB path exists
settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

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

    # Retrieval Weighting Coefficients
    RECENCY_WEIGHT: float = 0.6
    IMPORTANCE_WEIGHT: float = 0.3
    TASK_RELATEDNESS_WEIGHT: float = 0.1 # Placeholder, will be implemented later

    # Chunking Parameters
    CHUNK_SIZE_SENTENCES: int = 3  # Target number of sentences per chunk
    CHUNK_OVERLAP_SENTENCES: int = 1 # Overlap between chunks to maintain context

    # Optional: API Settings (if FastAPI is used)
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    @property
    def CHROMA_COLLECTION_NAME(self) -> str:
        return "ai_memory_collection"

# Instantiate settings
settings = Settings()

# Ensure the ChromaDB path exists
settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# Example of how to use environment variables for sensitive data
# API_KEY: str = os.getenv("OPENAI_API_KEY", "your_default_api_key")

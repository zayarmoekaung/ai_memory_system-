# AI Agent Memory System - Implementation Notes

This document details the step-by-step implementation plan for the AI Agent Memory System, outlining the purpose of each feature and the methodology for its implementation.

## 1. Core Language and Environment Setup

*   **Purpose:** Establish the foundational Python environment and install necessary dependencies.
*   **Implementation Steps:**
    1.  [x] Ensure Python 3.9+ is installed.
    2.  [x] Navigate to the `ai_memory_system/` directory.
    3.  [ ] Create and activate a virtual environment (`python -m venv venv && source venv/bin/activate`).
    4.  [ ] Install dependencies from `requirements.txt` (`pip install -r requirements.txt`).

## 2. Configuration Management (`config/settings.py`)

*   **Purpose:** Centralize all configurable parameters, such as embedding model names, token limits, weighting coefficients for retrieval, and ChromaDB settings, for easy management and access.
*   **Implementation Steps:**
    1.  [x] Create `config/settings.py` file.
    2.  [x] Define a `Settings` class using Pydantic for robust validation of configuration parameters.
    3.  [x] Load environment variables using `python-dotenv` for sensitive or environment-specific settings.
    4.  [x] Include parameters for:
        *   `EMBEDDING_MODEL_NAME`: (e.g., `all-MiniLM-L6-v2`)
        *   `CHROMA_DB_PATH`: Path for ChromaDB persistence (e.g., `./data/chroma_db`)
        *   `MAX_CONTEXT_TOKENS`: Maximum tokens for the main LLM context window.
        *   `RECENCY_WEIGHT`, `IMPORTANCE_WEIGHT`, `TASK_RELATEDNESS_WEIGHT`: Coefficients for weighted retrieval.
        *   `CHUNK_SIZE_SENTENCES`, `CHUNK_OVERLAP_SENTENCES`: For memory chunking.
    5.  [x] Provide methods to easily access these settings.

## 3. Embedding Manager (`src/core/embedding_manager.py`)

*   **Purpose:** Handle the loading of the chosen embedding model and the conversion of text into high-dimensional vector representations.
*   **Implementation Steps:**
    1.  [x] Create `src/core/embedding_manager.py` file.
    2.  [x] Import `SentenceTransformer` from `sentence_transformers`.
    3.  [x] Define an `EmbeddingManager` class.
    4.  [x] In the constructor, load the embedding model using `Settings.EMBEDDING_MODEL_NAME`.
    5.  [x] Implement a `get_embedding(text: str) -> list[float]` method that takes text and returns its vector embedding.
    6.  [x] Implement a `get_tokenizer()` method to return the model's tokenizer for token counting.

## 4. Memory Store (`src/core/memory_store.py`)

*   **Purpose:** Manage the storage and retrieval of memory chunks and their associated metadata using ChromaDB.
*   **Implementation Steps:**
    1.  [x] Create `src/core/memory_store.py` file.
    2.  [x] Import `chromadb` and `tiktoken`.
    3.  [x] Define a `MemoryStore` class.
    4.  [x] In the constructor:
        *   Initialize ChromaDB client, specifying persistence path (`Settings.CHROMA_DB_PATH`).
        *   Get or create a ChromaDB collection for memories.
    5.  [x] Implement `add_memory_chunk(chunk_id: str, content: str, embedding: list[float], metadata: dict)`:
        *   Adds a single memory chunk to the ChromaDB collection.
        *   Metadata should include `timestamp`, `importance_score`, `source_id`, `original_text_start_index`, etc.
    6.  [x] Implement `search_memories(query_embedding: list[float], n_results: int, min_distance: float = 0.5) -> list[dict]`:
        *   Performs a similarity search in ChromaDB.
        *   Returns relevant memory chunks and their metadata.
    7.  [x] Implement `get_memory_by_id(chunk_id: str) -> dict`:
        *   Retrieves a specific memory chunk by its ID.

## 5. Chunk Optimizer (`src/core/chunk_optimizer.py`)

*   **Purpose:** Segment raw memories into meaningful chunks, accurately count tokens, and dynamically select/truncate chunks to fit within context window limits.
*   **Implementation Steps:**
    1.  [x] Create `src/core/chunk_optimizer.py` file.
    2.  [x] Import `tiktoken` and `Settings`.
    3.  [x] Define a `ChunkOptimizer` class.
    4.  [x] In the constructor, initialize the tokenizer using `tiktoken.encoding_for_model("gpt-4")` or the `EmbeddingManager`'s tokenizer.
    5.  [x] Implement `chunk_text(text: str) -> list[str]`:
        *   Segments raw text into smaller, meaningful chunks (e.g., by sentences or short paragraphs), using parameters from `Settings`.
    6.  [x] Implement `count_tokens(text: str) -> int`:
        *   Uses the tokenizer to accurately count tokens for a given text.
    7.  [x] Implement `optimize_chunks_for_context(chunks: list[dict], query_tokens: int) -> list[dict]`:
        *   Takes a list of retrieved memory chunks (with their content and scores) and the token count of the query.
        *   Iteratively selects the highest-scoring chunks until `Settings.MAX_CONTEXT_TOKENS` is approached.
        *   If necessary, implements truncation strategies for individual chunks (e.g., keeping only the most relevant sentences) while respecting linguistic boundaries.
        *   Returns the optimized list of chunks that fit within the token limit.

## 6. Retrieval Manager (`src/core/retrieval_manager.py`)

*   **Purpose:** Orchestrate the retrieval process, combining vector similarity with weighted scoring based on recency, importance, and task-relatedness.
*   **Implementation Steps:**
    1.  [x] Create `src/core/retrieval_manager.py` file.
    2.  [x] Import `MemoryStore`, `EmbeddingManager`, `ChunkOptimizer`, and `Settings`.
    3.  [x] Define a `RetrievalManager` class.
    4.  [x] In the constructor, initialize instances of `MemoryStore`, `EmbeddingManager`, and `ChunkOptimizer`.
    5.  [x] Implement `ingest_memory(raw_text: str, importance_score: float = 0.5, source_id: str = "agent_observation")`:
        *   Takes raw text, chunks it using `ChunkOptimizer`.
        *   Generates embeddings for each chunk using `EmbeddingManager`.
        *   Adds each chunk to `MemoryStore` with metadata (timestamp, importance, etc.).
    6.  [x] Implement `retrieve_relevant_memories(query: str, n_results: int = 10) -> list[dict]`:
        *   Generates embedding for the `query` using `EmbeddingManager`.
        *   Performs initial similarity search using `MemoryStore`.
        *   Applies weighted scoring:
            *   Calculate recency score (decay function based on `timestamp`).
            *   Incorporate `importance_score` from metadata.
            *   (Future) Implement logic for task-relatedness weighting.
            *   Combine scores using `Settings.RECENCY_WEIGHT`, `IMPORTANCE_WEIGHT`, etc.
        *   Sorts chunks by combined score.
        *   Passes top `n_results` chunks to `ChunkOptimizer.optimize_chunks_for_context()` to fit the token window.
        *   Returns the final list of optimized, highly relevant memory chunks.

## 7. API Integration (`src/api/main.py` - Optional)

*   **Purpose:** Provide a RESTful API for external services or agents to interact with the memory system.
*   **Implementation Steps:**
    1.  [ ] Create `src/api/main.py` file.
    2.  [ ] Import `FastAPI` and `RetrievalManager`.
    3.  [ ] Define API endpoints:
        *   `/ingest_memory` (POST): To add new memories.
        *   `/retrieve_memories` (POST): To query and retrieve relevant memories.
    4.  [ ] Use Pydantic models for request and response body validation.

## 8. Examples (`examples/`)

*   **Purpose:** Demonstrate the usage of the memory system and provide interactive testing.
*   **Implementation Steps:**
    1.  [x] Create `examples/basic_usage.py` file.
    2.  [x] Create `examples/interactive_test.py` file.

## 9. README.md

*   **Purpose:** Provide a clear overview of the project, setup instructions, and usage examples.
*   **Implementation Steps:**
    1.  [x] Create `README.md` file.
    2.  [ ] Describe the project's goal.
    3.  [ ] Include setup instructions (virtual environment, `pip install`).
    4.  [ ] Explain how to run examples.
    5.  [ ] Detail the project structure and key modules.
    6.  [ ] Mention chosen technologies and their purpose.
    7.  [ ] Provide a basic usage example.

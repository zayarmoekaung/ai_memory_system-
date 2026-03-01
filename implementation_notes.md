# AI Agent Memory System - Implementation Notes

This document details the step-by-step implementation plan for the AI Agent Memory System, outlining the purpose of each feature and the methodology for its implementation.

## 1. Core Language and Environment Setup

- **Purpose:** Establish the foundational Python environment and install necessary dependencies.
- **Implementation Steps:**
    1.  [x] Ensure Python 3.9+ is installed.
    2.  [x] Navigate to the `ai_memory_system/` directory.
    3.  [ ] Create and activate a virtual environment (`python -m venv venv && source venv/bin/activate`).
    4.  [ ] Install dependencies from `requirements.txt` (`pip install -r requirements.txt`).

## 2. Configuration Management (`config/settings.py`)

- **Purpose:** Centralize all configurable parameters, such as embedding model names, token limits, weighting coefficients for retrieval, and ChromaDB settings, for easy management and access.
- **Implementation Steps:**
    1.  [x] Create `config/settings.py` file.
    2.  [x] Define a `Settings` class using Pydantic for robust validation of configuration parameters.
    3.  [x] Load environment variables using `python-dotenv` for sensitive or environment-specific settings.
    4.  [x] Include parameters for:
        - `EMBEDDING_MODEL_NAME`: (e.g., `all-MiniLM-L6-v2`)
        - `CHROMA_DB_PATH`: Path for ChromaDB persistence (e.g., `./data/chroma_db`)
        - `MAX_CONTEXT_TOKENS`: Maximum tokens for the main LLM context window.
        - `RECENCY_WEIGHT`, `IMPORTANCE_WEIGHT`, `TASK_RELATEDNESS_WEIGHT`: Coefficients for weighted retrieval.
        - `CHUNK_SIZE_SENTENCES`, `CHUNK_OVERLAP_SENTENCES`: For memory chunking.
        - `WORKING_MEMORY_CAPACITY`: Capacity for the Sensory Input Buffer.
        - `MEMORY_CONSOLIDATION_INTERVAL_SECONDS`: Interval for background consolidation.
        - `SENTIMENT_MODEL_NAME`, `ENTITY_EXTRACTION_MODEL_NAME`: NLP models for metadata extraction.
        - `VIVIDNESS_DECAY_RATE`, `ASSOCIATIVE_STRENGTH_DECAY_RATE`: Decay rates for dynamic scores.
    5.  [x] Provide methods to easily access these settings.

## 3. Embedding Manager (`src/core/embedding_manager.py`)

- **Purpose:** Handle the loading of the chosen embedding model and the conversion of text into high-dimensional vector representations.
- **Implementation Steps:**
    1.  [x] Create `src/core/embedding_manager.py` file.
    2.  [x] Import `SentenceTransformer` from `sentence_transformers`.
    3.  [x] Define an `EmbeddingManager` class.
    4.  [x] In the constructor, load the embedding model using `Settings.EMBEDDING_MODEL_NAME`.
    5.  [x] Implement a `get_embedding(text: str) -> list[float]` method that takes text and returns its vector embedding.
    6.  [x] Implement a `get_tokenizer()` method to return the model's tokenizer for token counting.

## 4. Memory Store (`src/core/memory_store.py`)

- **Purpose:** Manage the storage and retrieval of memory chunks and their associated metadata using ChromaDB.
- **Implementation Steps:**
    1.  [x] Create `src/core/memory_store.py` file.
    2.  [x] Import `chromadb` and `tiktoken`.
    3.  [x] Define a `MemoryStore` class.
    4.  [x] In the constructor:
        - Initialize ChromaDB client, specifying persistence path (`Settings.CHROMA_DB_PATH`).
        - Get or create a ChromaDB collection for memories.
        - Get or create a separate ChromaDB collection for associative links (`Settings.CHROMA_ASSOCIATIVE_COLLECTION_NAME`).
    5.  [x] Implement `add_memory_chunk(chunk_id: str, content: str, embedding: list[float], metadata: dict)`:
        - Adds a single memory chunk to the ChromaDB collection.
        - Metadata should include `timestamp`, `importance_score`, `source_id`, `original_text_start_index`, `associated_entities`, `emotional_valence`, `vividness_score`, `context_tags`, `event_sequence_id`.
    6.  [x] Implement `search_memories(query_embedding: list[float], n_results: int, min_distance: float = 0.5) -> list[dict]`:
        - Performs a similarity search in ChromaDB.
        - Returns relevant memory chunks and their metadata.
    7.  [x] Implement `get_memory_by_id(chunk_id: str) -> dict`:
        - Retrieves a specific memory chunk by its ID.

## 5. Chunk Optimizer (`src/core/chunk_optimizer.py`)

- **Purpose:** Segment raw memories into meaningful chunks, accurately count tokens, and dynamically select/truncate chunks to fit within context window limits.
- **Implementation Steps:**
    1.  [x] Create `src/core/chunk_optimizer.py` file.
    2.  [x] Import `tiktoken` and `Settings`.
    3.  [x] Define a `ChunkOptimizer` class.
    4.  [x] In the constructor, initialize the tokenizer using `tiktoken.encoding_for_model("gpt-4")` or the `EmbeddingManager`'s tokenizer.
    5.  [x] Implement `chunk_text(text: str) -> list[str]`:
        - Segments raw text into smaller, meaningful chunks (e.g., by sentences or short paragraphs), using parameters from `Settings`.
    6.  [x] Implement `count_tokens(text: str) -> int`:
        - Uses the tokenizer to accurately count tokens for a given text.
    7.  [x] Implement `optimize_chunks_for_context(chunks: list[dict], query_tokens: int) -> list[dict]`:
        - Takes a list of retrieved memory chunks (with their content and scores) and the token count of the query.
        - Iteratively selects the highest-scoring chunks until `Settings.MAX_CONTEXT_TOKENS` is approached.
        - If necessary, implements truncation strategies for individual chunks (e.g., keeping only the most relevant sentences) while respecting linguistic boundaries.
        - Includes conceptual placeholder for dynamic synthesis/summarization.
        - Returns the optimized list of chunks that fit within the token limit.

## 6. Retrieval Manager (`src/core/retrieval_manager.py`)

- **Purpose:** Orchestrate the retrieval process, combining vector similarity with weighted scoring based on recency, importance, and task-relatedness.
- **Implementation Steps:**
    1.  [x] Create `src/core/retrieval_manager.py` file.
    2.  [x] Import `MemoryStore`, `EmbeddingManager`, `ChunkOptimizer`, `WorkingMemory`, `MemoryConsolidation`, `AssociativeNetwork`, and `Settings`.
    3.  [x] Define a `RetrievalManager` class.
    4.  [x] In the constructor, initialize instances of `MemoryStore`, `EmbeddingManager`, `ChunkOptimizer`, `WorkingMemory`, `MemoryConsolidation`, and `AssociativeNetwork`. Set the tokenizer for `ChunkOptimizer`.
    5.  [x] Implement `ingest_memory(raw_text: str, importance_score: float = 0.5, source_id: str = "agent_observation")`:
        - Adds raw text to `WorkingMemory`.
        - Chunks it using `ChunkOptimizer`.
        - Generates embeddings for each chunk using `EmbeddingManager`.
        - Extracts enhanced metadata (`emotional_valence`, `associated_entities`, `context_tags`) using simple heuristic placeholder functions.
        - Adds each chunk to `MemoryStore` with metadata.
        - Links `chunk_id` to extracted `associated_entities` in the `AssociativeNetwork`.
    6.  [x] Implement `_calculate_recency_score(timestamp: float) -> float` for recency scoring.
    7.  [x] Implement `_calculate_vividness_score(metadata: Dict[str, Any]) -> float` for vividness scoring (placeholder).
    8.  [x] Implement `_calculate_emotional_saliency_score(metadata: Dict[str, Any]) -> float` for emotional saliency scoring.
    9.  [x] Implement `_calculate_associative_strength_score(chunk_id: str, query_entities: List[str]) -> float` for associative strength scoring (placeholder).
    10. [x] Implement `retrieve_relevant_memories(query: str, n_results: int = 10) -> list[dict]`:
        - Prioritizes initial recall from `WorkingMemory`.
        - Performs initial similarity search using `MemoryStore`.
        - Incorporates `AssociativeNetwork` for spreading activation using query entities.
        - Applies a refined weighted scoring combining similarity, recency, importance, emotional salience, vividness, and associative strength.
        - Optimizes top chunks using `ChunkOptimizer` to fit the token window.
        - Returns the final list of optimized, highly relevant memory chunks.

## 7. API Integration (`src/api/main.py` - Optional)

- **Purpose:** Provide a RESTful API for external services or agents to interact with the memory system.
- **Implementation Steps:**
    1.  [x] Create `src/api/main.py` file.
    2.  [x] Import `FastAPI` and `RetrievalManager`.
    3.  [x] Define API endpoints:
        - `/ingest_memory` (POST): To add new memories.
        - `/retrieve_memories` (POST): To query and retrieve relevant memories.
    4.  [x] Use Pydantic models for request and response body validation.

## 8. Examples (`examples/`)

- **Purpose:** Demonstrate the usage of the memory system and provide interactive testing.
- **Implementation Steps:**
    1.  [x] Create `examples/basic_usage.py` file.
    2.  [x] Create `examples/interactive_test.py` file.

## 9. README.md

- **Purpose:** Provide a clear overview of the project, setup instructions, and usage examples.
- **Implementation Steps:**
    1.  [x] Create `README.md` file.
    2.  [x] Describe the project's goal.
    3.  [x] Include setup instructions (virtual environment, `pip install`).
    4.  [x] Explain how to run examples.
    5.  [x] Detail the project structure and key modules.
    6.  [x] Mention chosen technologies and their purpose.
    7.  [x] Provide a basic usage example.

## 10. New Human-like Memory Prototype Components

- **Purpose:** Introduce and integrate new modules for the human-like memory system design.
- **Implementation Steps:**
    1.  [x] Create `src/core/working_memory.py` for the Sensory Input Buffer.
    2.  [x] Create `src/core/memory_consolidation.py` for the Memory Consolidation background process (conceptual placeholder).
    3.  [x] Create `src/core/associative_network.py` for the Associative Network/Knowledge Graph.
    4.  [x] Update `src/core/__init__.py` to include the new modules.
    5.  [x] Update `config/settings.py` with new settings (`WORKING_MEMORY_CAPACITY`, NLP models, decay rates, associative collection name).
    6.  [x] Update `src/core/memory_store.py` to prepare for richer metadata and add associative collection.
    7.  [x] Update `src/core/retrieval_manager.py` to integrate new components and flows for ingestion and retrieval.
    8.  [x] Update `src/core/chunk_optimizer.py` to reflect enhancements for dynamic synthesis (conceptual placeholder).
    9.  [ ] Update `README.md` with information about the human-like memory prototype.
    10. [ ] Update `Shion.md` with details of this prototype implementation.

# AI Agent Memory System

This project implements a sophisticated memory system for AI agents, designed to efficiently store, retrieve, and optimize contextual information. It leverages vector embeddings for semantic search, weighted retrieval based on factors like recency and importance, and dynamic chunk selection to manage token usage within large language model (LLM) context windows.

## Features

- **Vectorizing Memories:** Converts raw text memories into high-dimensional vector embeddings using pre-trained `sentence-transformers` models.
- **Weighted Retrieval:** Retrieves relevant memory chunks based on a combination of vector similarity, recency, and importance scores.
- **Dynamic Chunk Selection:** Segments raw text into manageable chunks, accurately counts tokens, and intelligently selects/truncates chunks to fit within specified token limits.
- **Persistent Storage:** Utilizes ChromaDB for efficient and persistent storage of memory chunks and their associated metadata.
- **Configurable:** All key parameters (embedding model, token limits, weighting coefficients) are centralized in a `settings.py` file.
- **Optional API:** Provides a FastAPI interface for external services or agents to interact with the memory system programmatically.

## Human-like Memory Prototype (Work in Progress)

This section outlines the conceptual enhancements and initial structural changes aimed at making the AI Memory System emulate human memory more closely. This involves a richer understanding of memory elements and their interconnections.

### Key Prototype Components & Concepts

- **Sensory Input Buffer (Working Memory):** A new, temporary, limited-capacity in-memory store for immediate context and ongoing thoughts, acting as the fastest recall layer.
- **Memory Consolidation:** A conceptual background process for reviewing working memories, summarizing them, assigning enhanced metadata, and integrating them into long-term storage.
- **Associative Network:** A lightweight graph structure for managing explicit links and relationships between memory chunks and extracted entities, enabling "spreading activation" during retrieval.
- **Enhanced Long-Term Memory Metadata:** ChromaDB is extended to store richer metadata for each chunk, including `associated_entities`, `emotional_valence`, `vividness_score`, `context_tags`, and `event_sequence_id`.
- **Adaptive Ingestion Flow:** Introduces intelligent selection of memory candidates from working memory based on information density, topic shifts, and actionable items, along with pre-processing for basic NLP extraction of metadata.
- **Dynamic & Associative Retrieval Flow:** Prioritizes working memory, incorporates associative spreading, and uses a refined weighted scoring that includes emotional salience, associative strength, and vividness, alongside similarity, recency, and importance.
- **Dynamic Synthesis & Focal Attention:** The `ChunkOptimizer` is conceptually enhanced to not just truncate but also to potentially synthesize or summarize relevant chunks for more coherent recollections.

## Project Structure

```
ai_memory_system/
├── src/
│   ├── core/
│   │   ├── memory_store.py         # Manages memory chunks, metadata, and interaction with ChromaDB
│   │   ├── embedding_manager.py    # Handles embedding model loading and vectorization
│   │   ├── retrieval_manager.py    # Implements weighted retrieval logic
│   │   ├── chunk_optimizer.py      # Handles dynamic chunk selection, token counting, and truncation
│   │   ├── working_memory.py       # Sensory Input Buffer (new)
│   │   ├── memory_consolidation.py # Background process for memory consolidation (new)
│   │   ├── associative_network.py  # Manages explicit links between memories/entities (new)
│   │   └── __init__.py
│   ├── api/                      # REST API interface using FastAPI
│   │   ├── main.py
│   │   └── __init__.py
│   └── __init__.py
├── tests/                        # Unit and integration tests
├── config/
│   ├── settings.py               # Centralized configuration parameters
├── data/                         # Directory for ChromaDB persistence
├── examples/                     # Demonstration and interactive usage scripts
│   ├── basic_usage.py
│   ├── interactive_test.py
├── requirements.txt              # Project dependencies
├── implementation_notes.md       # Detailed implementation plan and progress
├── Shion.md                      # ShionAide's contributions and agent instructions
└── README.md                     # Project overview and instructions
```

## Technologies Used

- **Python 3.9+:** Core programming language.
- **ChromaDB:** Vector database for memory storage and similarity search (now with two collections: main memory and associative links).
- **`sentence-transformers`:** For generating robust text embeddings.
- **`tiktoken`:** For accurate token counting, consistent with OpenAI models.
- **`pydantic` / `pydantic-settings`:** For robust data validation and configuration management.
- **`numpy`:** For efficient numerical operations.
- **`FastAPI` & `Uvicorn`:** (Optional) For building a high-performance web API.
- **`python-dotenv`:** For loading environment variables.
- **`nltk`:** For sentence tokenization (requires `punkt` data).
- **`networkx`:** (New, for prototype) For the conceptual `AssociativeNetwork` graph structure.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone git@github.com:zayarmoekaung/ai_memory_system-.git
    cd ai_memory_system
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data (required for ChunkOptimizer):**
    The `ChunkOptimizer` uses `nltk` for sentence tokenization. You will need to download the `punkt` tokenizer data.
    You can do this by running Python:
    ```python
    import nltk
    nltk.download(\'punkt\')
    ```

5.  **Create an `.env` file (optional but recommended):**
    While default values are provided in `config/settings.py`, you can override them by creating a `.env` file in the `ai_memory_system/` directory. For example:
    ```env
    EMBEDDING_MODEL_NAME=\"all-MiniLM-L6-v2\"
    MAX_CONTEXT_TOKENS=8000
    WORKING_MEMORY_CAPACITY=15
    # ... other settings
    ```

## Usage Examples

### Basic Usage

Run the `basic_usage.py` script to see a simple demonstration of memory ingestion and retrieval:

```bash
python examples/basic_usage.py
```

### Interactive Test

Run the `interactive_test.py` script to interact with the memory system via a command-line interface:

```bash
python examples/interactive_test.py
```

### Running the API (Optional)

If you want to expose the memory system as a REST API, you can run the FastAPI application:

```bash
# Ensure you are in the ai_memory_system/ directory
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

This will start the API server, typically accessible at `http://localhost:8000/docs` for interactive documentation.

## Testing

(Details on how to run tests will be added here as tests are implemented.)

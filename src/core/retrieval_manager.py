import time
import uuid
from typing import List, Dict, Any
from datetime import datetime, timedelta

from .memory_store import MemoryStore
from .embedding_manager import EmbeddingManager
from .chunk_optimizer import ChunkOptimizer
from  config.settings import settings

class RetrievalManager:
    def __init__(self):
        """
        Initializes the RetrievalManager with instances of MemoryStore, EmbeddingManager, and ChunkOptimizer.
        Also sets the tokenizer for ChunkOptimizer from EmbeddingManager.
        """
        self.memory_store = MemoryStore()
        self.embedding_manager = EmbeddingManager()
        self.chunk_optimizer = ChunkOptimizer()
        self.chunk_optimizer.set_tokenizer(self.embedding_manager.get_tokenizer())

    def ingest_memory(self, raw_text: str, importance_score: float = 0.5, source_id: str = "agent_observation"):
        """
        Ingests raw text by chunking it, generating embeddings, and storing each chunk with metadata.

        Args:
            raw_text (str): The raw text of the memory.
            importance_score (float): A score indicating the importance of this memory (0.0 to 1.0).
            source_id (str): An identifier for the source of this memory.
        """
        chunks = self.chunk_optimizer.chunk_text(raw_text)
        current_timestamp = time.time()

        for i, chunk_content in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            embedding = self.embedding_manager.get_embedding(chunk_content)
            metadata = {
                "timestamp": current_timestamp,
                "importance_score": importance_score,
                "source_id": source_id,
                "original_text_start_index": raw_text.find(chunk_content) # Simple approach, can be refined
            }
            self.memory_store.add_memory_chunk(chunk_id, chunk_content, embedding, metadata)
        print(f"Ingested {len(chunks)} memory chunks from source: {source_id}")

    def _calculate_recency_score(self, timestamp: float) -> float:
        """
        Calculates a recency score for a memory chunk based on its timestamp.
        Newer memories get higher scores. The decay function can be adjusted.
        """
        now = time.time()
        age_seconds = now - timestamp

        # Example decay: memories from last 24 hours get high score, then linearly decay over 7 days
        # This is a placeholder and can be made more sophisticated.
        one_day_seconds = 24 * 3600
        seven_days_seconds = 7 * one_day_seconds

        if age_seconds <= one_day_seconds:
            return 1.0 # Very recent
        elif age_seconds < seven_days_seconds:
            # Linearly decay from 1.0 to 0.0 over the remaining 6 days
            decay_factor = (seven_days_seconds - age_seconds) / (seven_days_seconds - one_day_seconds)
            return max(0.0, decay_factor) # Ensure score doesn't go below 0
        else:
            return 0.0 # Older than 7 days, gets 0 score

    def retrieve_relevant_memories(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves relevant memories based on a query, combining vector similarity with weighted scoring.

        Args:
            query (str): The query text.
            n_results (int): The number of top relevant chunks to retrieve before optimization.

        Returns:
            List[Dict[str, Any]]: A list of optimized, highly relevant memory chunks, each with 'content', 'id', etc.
        """
        query_embedding = self.embedding_manager.get_embedding(query)
        query_tokens = self.chunk_optimizer.count_tokens(query)

        # 1. Initial similarity search
        raw_retrieved_chunks = self.memory_store.search_memories(query_embedding, n_results=n_results)

        scored_chunks = []
        for chunk_data in raw_retrieved_chunks:
            metadata = chunk_data.get('metadata', {})
            distance = chunk_data.get('distance', 1.0) # Assume 1.0 if not present, will be inverse for similarity
            # Convert distance to similarity (lower distance -> higher similarity). Max distance can be ~2.0 for L2
            # A simple inverse for L2: similarity = 1 - (distance / max_possible_distance)
            # For now, let's normalize distance to a 0-1 range if it's L2, where 0 is identical, 2 is max diff
            # More robust similarity conversion might be needed based on ChromaDB's exact distance metric.
            similarity_score = 1.0 - (distance / 2.0) if distance <= 2.0 else 0.0 # Assuming max L2 is 2.0
            similarity_score = max(0.0, min(1.0, similarity_score)) # Clamp between 0 and 1

            recency_score = self._calculate_recency_score(metadata.get('timestamp', 0))
            importance_score = metadata.get('importance_score', 0.5) # Default to 0.5 if not set
            # task_relatedness_score = ... # Placeholder for future implementation
            task_relatedness_score = 0.5 # Default placeholder

            # Combine scores using weighted sum from settings
            combined_score = (
                settings.RECENCY_WEIGHT * recency_score +
                settings.IMPORTANCE_WEIGHT * importance_score +
                settings.TASK_RELATEDNESS_WEIGHT * task_relatedness_score +
                (1.0 - settings.RECENCY_WEIGHT - settings.IMPORTANCE_WEIGHT - settings.TASK_RELATEDNESS_WEIGHT) * similarity_score
            )
            # Ensure combined score is between 0 and 1
            combined_score = max(0.0, min(1.0, combined_score))

            scored_chunks.append({
                'id': chunk_data['id'],
                'content': chunk_data['content'],
                'metadata': chunk_data['metadata'],
                'embedding': chunk_data['embedding'],
                'score': combined_score,
                'similarity': similarity_score, # For debugging/analysis
                'recency': recency_score,
                'importance': importance_score,
                'task_relatedness': task_relatedness_score
            })
        
        # 2. Optimize chunks for context window
        # Pass all scored chunks and let optimizer select and truncate based on token limits
        optimized_chunks = self.chunk_optimizer.optimize_chunks_for_context(scored_chunks, query_tokens)

        return optimized_chunks

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Make sure NLTK punkt tokenizer data is downloaded for sent_tokenize
    # import nltk
    # try:
    #     sent_tokenize("test")
    # except LookupError:
    #     nltk.download('punkt')
    # Ensure the data directory exists for ChromaDB persistence
    settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

    retrieval_manager = RetrievalManager()

    # Clear existing memories for a clean test run
    # retrieval_manager.memory_store.delete_collection()

    # Ingest some test memories
    print("\nIngesting memories...")
    retrieval_manager.ingest_memory(
        "The capital of France is Paris. Paris is known for its Eiffel Tower and Louvre Museum. I visited Paris last year and it was beautiful.",
        importance_score=0.9,
        source_id="travel_log"
    )
    time.sleep(1) # Simulate time passing
    retrieval_manager.ingest_memory(
        "Today's weather forecast predicts sunny skies with a high of 25 degrees Celsius. Perfect for a walk in the park.",
        importance_score=0.7,
        source_id="daily_news"
    )
    time.sleep(2) # Simulate more time passing
    retrieval_manager.ingest_memory(
        "The project meeting is scheduled for tomorrow at 10 AM. We need to discuss the AI memory system implementation details.",
        importance_score=1.0,
        source_id="work_calendar"
    )
    time.sleep(0.5) # Simulate time passing
    retrieval_manager.ingest_memory(
        "I remember reading about vector databases like ChromaDB and Pinecone for efficient similarity search. They are crucial for AI memory systems.",
        importance_score=0.85,
        source_id="research_notes"
    )

    # Retrieve memories based on a query
    print("\nRetrieving memories for query: 'What is the weather like and what about the AI project?'")
    relevant_memories = retrieval_manager.retrieve_relevant_memories(
        "What is the weather like and what about the AI project?", n_results=5
    )
    print("\n--- Retrieved and Optimized Memories ---")
    total_tokens_in_retrieved = 0
    for i, mem in enumerate(relevant_memories):
        content_tokens = retrieval_manager.chunk_optimizer.count_tokens(mem['content'])
        total_tokens_in_retrieved += content_tokens
        print(f"Memory {i+1} (Score: {mem['score']:.4f}, Tokens: {content_tokens}): {mem['content']}")
        print(f"  Metadata: Recency={mem['recency']:.2f}, Importance={mem['importance']:.2f}, Similarity={mem['similarity']:.2f}")

    query_tokens_test = retrieval_manager.chunk_optimizer.count_tokens("What is the weather like and what about the AI project?")
    print(f"\nQuery tokens: {query_tokens_test}")
    print(f"Total memory tokens in retrieved: {total_tokens_in_retrieved}")
    print(f"Total tokens (query + memories): {query_tokens_test + total_tokens_in_retrieved}")
    print(f"Max context tokens (from settings): {settings.MAX_CONTEXT_TOKENS}")

    # Test with a very specific query that might only hit one chunk
    print("\nRetrieving memories for query: 'Eiffel Tower'")
    eiffel_memories = retrieval_manager.retrieve_relevant_memories("Eiffel Tower", n_results=1)
    for i, mem in enumerate(eiffel_memories):
        print(f"Memory {i+1} (Score: {mem['score']:.4f}): {mem['content']}")

    print("\nRetrieving memories for query: 'vector databases'")
    vector_db_memories = retrieval_manager.retrieve_relevant_memories("vector databases", n_results=1)
    for i, mem in enumerate(vector_db_memories):
        print(f"Memory {i+1} (Score: {mem['score']:.4f}): {mem['content']}")

import project_path
from src.core.retrieval_manager import RetrievalManager
from src.core.chunk_optimizer import ChunkOptimizer
from src.core.embedding_manager import EmbeddingManager
import time
from config.settings import settings

def run_chunk_optimization_demo():
    """
    Demonstrates the chunk optimization functionality for context window management.
    Shows how chunks are selected and potentially truncated to fit within token limits.
    """
    print("=== Chunk Optimization Demo ===")
    print("This demo shows how the system optimizes memory chunks for context windows.\n")

    # Initialize components
    embedding_manager = EmbeddingManager()
    chunk_optimizer = ChunkOptimizer()
    chunk_optimizer.set_tokenizer(embedding_manager.get_tokenizer())

    # Ensure data directory exists
    settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    retrieval_manager = RetrievalManager()

    # Clear existing data for clean demo
    try:
        retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_ASSOCIATIVE_COLLECTION_NAME)
    except:
        pass

    print("1. Ingesting memories of varying lengths...")

    # Create memories with different lengths and characteristics
    test_memories = [
        # Short memory
        ("Paris is beautiful.", 0.8, "short"),

        # Medium memory
        ("The Eiffel Tower is an iron lattice tower located in Paris, France. It was built in 1889 and stands 324 meters tall.", 0.9, "medium"),

        # Long memory
        ("Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. The process involves training algorithms on large datasets to recognize patterns and make predictions or decisions without being explicitly programmed for each specific task. Common applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles.", 0.95, "long"),

        # Very long memory
        ("The history of artificial intelligence began in ancient times with philosophers contemplating the nature of thought and reasoning. Modern AI research started in the 1950s with the development of the first neural networks and expert systems. Key milestones include the creation of the perceptron in 1957, the development of expert systems in the 1970s, the AI winter periods of reduced funding and interest, the resurgence with machine learning in the 1990s and 2000s, and the current deep learning revolution starting around 2012 with convolutional neural networks and large language models. Today, AI systems power everything from search engines and social media algorithms to medical diagnosis tools and autonomous driving systems.", 0.9, "very_long"),
    ]

    for text, importance, length_category in test_memories:
        retrieval_manager.ingest_memory(text, importance_score=importance, source_id=f"demo_{length_category}")
        time.sleep(0.2)

    print("\n2. Testing chunk optimization with different query complexities...")

    test_queries = [
        ("What is Paris?", 5),  # Simple query, should fit all
        ("Tell me about machine learning and AI history", 50),  # Complex query, may need optimization
        ("Explain artificial intelligence comprehensively", 30),  # Should trigger truncation
    ]

    for query, max_results in test_queries:
        print(f"\nQuery: '{query}' (max results: {max_results})")

        # Get raw retrieved memories before optimization
        query_embedding = embedding_manager.get_embedding(query)
        raw_memories = retrieval_manager.memory_store.search_memories(query_embedding, n_results=max_results)

        # Add scoring information
        scored_memories = []
        for mem in raw_memories:
            metadata = mem.get('metadata', {})
            distance = mem.get('distance', 1.0)
            similarity_score = 1.0 - (distance / 2.0) if distance <= 2.0 else 0.0

            scored_memories.append({
                'id': mem['id'],
                'content': mem['content'],
                'metadata': metadata,
                'embedding': mem.get('embedding'),
                'score': similarity_score,
                'similarity': similarity_score,
                'recency': 1.0,  # Simplified for demo
                'importance': metadata.get('importance_score', 0.5),
                'task_relatedness': 0.5,
                'emotional_saliency': 0.5,
                'vividness': 0.5,
                'associative_strength': 0.0
            })

        # Apply optimization
        query_tokens = chunk_optimizer.count_tokens(query)
        optimized_memories = chunk_optimizer.optimize_chunks_for_context(scored_memories, query_tokens)

        print(f"Query tokens: {query_tokens}")
        print(f"Raw memories retrieved: {len(scored_memories)}")
        print(f"Optimized memories: {len(optimized_memories)}")

        total_raw_tokens = sum(chunk_optimizer.count_tokens(mem['content']) for mem in scored_memories)
        total_optimized_tokens = sum(chunk_optimizer.count_tokens(mem['content']) for mem in optimized_memories)

        print(f"Total raw memory tokens: {total_raw_tokens}")
        print(f"Total optimized memory tokens: {total_optimized_tokens}")
        print(f"Max context tokens: {settings.MAX_CONTEXT_TOKENS}")
        print(f"Combined tokens (query + optimized): {query_tokens + total_optimized_tokens}")

        print("\nOptimized memories:")
        for i, mem in enumerate(optimized_memories):
            content_tokens = chunk_optimizer.count_tokens(mem['content'])
            content_preview = mem['content'][:80] + "..." if len(mem['content']) > 80 else mem['content']
            print(f"  {i+1}. Tokens: {content_tokens}, Score: {mem['score']:.4f}")
            print(f"      {content_preview}")

    print("\n3. Demonstrating token counting and chunking...")

    sample_text = "This is a sample text that will be chunked into smaller pieces for processing. Each chunk should be meaningful and contain complete thoughts when possible. The chunking algorithm considers sentence boundaries and paragraph breaks."

    print(f"Original text: {sample_text}")
    print(f"Original token count: {chunk_optimizer.count_tokens(sample_text)}")

    chunks = chunk_optimizer.chunk_text(sample_text)
    print(f"\nChunked into {len(chunks)} chunks:")

    for i, chunk in enumerate(chunks):
        chunk_tokens = chunk_optimizer.count_tokens(chunk)
        print(f"  Chunk {i+1} ({chunk_tokens} tokens): '{chunk}'")

    print("\nChunk optimization demo complete!")

if __name__ == "__main__":
    run_chunk_optimization_demo()
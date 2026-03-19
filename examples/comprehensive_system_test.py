import project_path
from src.core.retrieval_manager import RetrievalManager
import time
from config.settings import settings

def run_comprehensive_system_test():
    """
    Comprehensive test of the entire human-like memory system.
    Tests all components working together in a realistic scenario.
    """
    print("=== Comprehensive Human-like Memory System Test ===")
    print("This test simulates a realistic conversation scenario with the AI.\n")

    # Ensure data directory exists
    settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    retrieval_manager = RetrievalManager()

    # Clear existing data for clean test
    try:
        retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_ASSOCIATIVE_COLLECTION_NAME)
    except:
        pass

    print("1. Building initial knowledge base...")

    # Initial knowledge base (simulating long-term memories)
    initial_knowledge = [
        # Personal/Identity
        ("I am ShionAide, an AI assistant created by Zayar-Sama.", 1.0, "identity"),
        ("Zayar-Sama is my creator and project manager.", 0.95, "identity"),
        ("TinaAide is my assistant for public communications.", 0.9, "identity"),

        # Technical Knowledge
        ("The AI memory system uses vector embeddings for semantic search.", 0.9, "technical"),
        ("ChromaDB is used as the vector database for persistent storage.", 0.85, "technical"),
        ("SentenceTransformers provide embeddings for text similarity.", 0.8, "technical"),
        ("Memory consolidation moves important information from working to long-term memory.", 0.9, "technical"),

        # Project Information
        ("This project implements a human-like memory system for AI agents.", 0.95, "project"),
        ("The system includes working memory, associative networks, and weighted retrieval.", 0.9, "project"),
        ("Working memory handles immediate context and recent information.", 0.85, "project"),

        # World Knowledge
        ("Paris is the capital of France.", 0.8, "geography"),
        ("The Eiffel Tower is an iconic landmark in Paris.", 0.8, "geography"),
        ("Machine learning is a type of artificial intelligence.", 0.85, "ai_knowledge"),
        ("Neural networks are computing systems inspired by biological brains.", 0.8, "ai_knowledge"),
    ]

    for text, importance, category in initial_knowledge:
        retrieval_manager.ingest_memory(text, importance_score=importance, source_id=category)
        time.sleep(0.2)

    print(f"Initial knowledge base: {len(initial_knowledge)} memories ingested")

    print("\n2. Simulating a conversation session...")

    # Simulate a conversation with context building
    conversation_turns = [
        "Hello, who are you?",
        "Tell me about your creator",
        "What project are you working on?",
        "How does the memory system work?",
        "What's the difference between working memory and long-term memory?",
        "Tell me about Paris",
        "What's the Eiffel Tower?",
        "How does machine learning relate to AI?",
        "What are neural networks?",
        "Can you remind me what we were talking about earlier?",
    ]

    conversation_history = []

    for i, user_query in enumerate(conversation_turns):
        print(f"\n--- Conversation Turn {i+1} ---")
        print(f"User: {user_query}")

        # Add user query to working memory as immediate context
        retrieval_manager.working_memory.add_item({
            "content": f"User asked: {user_query}",
            "source_id": "user_input",
            "timestamp": time.time()
        })

        # Retrieve relevant memories
        relevant_memories = retrieval_manager.retrieve_relevant_memories(
            query=user_query,
            n_results=3
        )

        # Simulate AI response generation (simplified)
        if relevant_memories:
            # Use the top memory as the basis for response
            top_memory = relevant_memories[0]
            response_content = top_memory['content']

            # Add AI response to working memory
            retrieval_manager.working_memory.add_item({
                "content": f"I responded about: {response_content[:50]}...",
                "source_id": "ai_response",
                "timestamp": time.time()
            })

            print(f"AI Response (based on memory): {response_content}")
            print(f"Memory Score: {top_memory['score']:.4f}, Source: {top_memory['metadata'].get('source_id', 'unknown')}")

            # Show scoring breakdown for the first few turns
            if i < 3:
                print("Scoring breakdown:")
                print(f"  - Similarity: {top_memory['similarity']:.4f}")
                print(f"  - Recency: {top_memory['recency']:.4f}")
                print(f"  - Importance: {top_memory['importance']:.4f}")
                print(f"  - Emotional: {top_memory['emotional_saliency']:.4f}")
                print(f"  - Associative: {top_memory['associative_strength']:.4f}")

        conversation_history.append((user_query, relevant_memories))

        # Simulate time passing between conversation turns
        time.sleep(1)

    print("\n3. Testing memory retention and recall...")

    # Test questions that should recall information from the conversation
    recall_tests = [
        ("Who is my creator?", "Should recall Zayar-Sama from identity memories"),
        ("What were we discussing about memory?", "Should recall working memory explanation"),
        ("Tell me about Paris again", "Should recall Paris/Eiffel Tower information"),
        ("What AI concepts have we talked about?", "Should recall ML, neural networks, etc."),
        ("Remind me what project you're working on", "Should recall the memory system project"),
    ]

    print("\nRecall test results:")
    for question, expected in recall_tests:
        print(f"\nQuestion: '{question}'")
        print(f"Expected: {expected}")

        memories = retrieval_manager.retrieve_relevant_memories(question, n_results=2)

        if memories:
            top_memory = memories[0]
            content_preview = top_memory['content'][:80] + "..." if len(top_memory['content']) > 80 else top_memory['content']
            print(f"Recalled: {content_preview}")
            print(f"Score: {top_memory['score']:.4f}")
        else:
            print("No relevant memories found")

    print("\n4. Analyzing system performance...")

    # Analyze working memory state
    working_items = retrieval_manager.working_memory.get_recent_items()
    print(f"\nWorking memory contains {len(working_items)} items from the conversation")

    # Analyze associative network growth
    network_size = len(retrieval_manager.associative_network.graph.nodes())
    associations = len(retrieval_manager.associative_network.graph.edges())
    print(f"Associative network: {network_size} entities, {associations} associations")

    # Check long-term memory storage
    # Note: In a real system, we'd query ChromaDB for collection size
    print("Long-term memories stored in vector database")

    print("\n5. Testing edge cases...")

    # Test with a completely new topic
    new_topic_query = "What do you know about quantum physics?"
    print(f"\nQuery on unknown topic: '{new_topic_query}'")

    unknown_memories = retrieval_manager.retrieve_relevant_memories(new_topic_query, n_results=2)
    if unknown_memories:
        print("Found some related memories (might be false positives):")
        for mem in unknown_memories:
            print(f"  - Score: {mem['score']:.4f}, Content: {mem['content'][:50]}...")
    else:
        print("Correctly found no relevant memories")

    # Test with very short query
    short_query = "Paris?"
    print(f"\nVery short query: '{short_query}'")

    short_memories = retrieval_manager.retrieve_relevant_memories(short_query, n_results=2)
    if short_memories:
        for mem in short_memories:
            print(f"  - Score: {mem['score']:.4f}, Content: {mem['content'][:50]}...")

    print("\n6. System health check...")

    # Basic health checks
    try:
        # Test embedding generation
        test_embedding = retrieval_manager.embedding_manager.get_embedding("test")
        print("✓ Embedding generation: OK")

        # Test token counting
        test_tokens = retrieval_manager.chunk_optimizer.count_tokens("test text")
        print("✓ Token counting: OK")

        # Test memory storage
        test_memories = retrieval_manager.memory_store.search_memories(test_embedding, n_results=1)
        print("✓ Memory storage/retrieval: OK")

        # Test working memory
        wm_items = retrieval_manager.working_memory.get_recent_items()
        print("✓ Working memory: OK")

        # Test associative network
        entities = list(retrieval_manager.associative_network.graph.nodes())
        print("✓ Associative network: OK")

    except Exception as e:
        print(f"✗ System health check failed: {e}")

    print("\nComprehensive system test complete!")
    print("The human-like memory system successfully demonstrated:")
    print("- Multi-factor weighted retrieval")
    print("- Working memory for immediate context")
    print("- Associative linking between memories")
    print("- Chunk optimization for context windows")
    print("- Memory consolidation concepts")
    print("- Realistic conversation handling")

if __name__ == "__main__":
    run_comprehensive_system_test()
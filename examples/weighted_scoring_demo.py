import project_path
from src.core.retrieval_manager import RetrievalManager
import time
from config.settings import settings

def run_weighted_scoring_demo():
    """
    Demonstrates the multi-factor weighted scoring system in the human-like memory system.
    Shows how different factors (recency, importance, emotional salience, etc.) contribute to final scores.
    """
    print("=== Weighted Scoring System Demo ===")
    print("This demo shows how multiple factors contribute to memory relevance scoring.\n")

    # Ensure data directory exists
    settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    retrieval_manager = RetrievalManager()

    # Clear existing data for clean demo
    try:
        retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_ASSOCIATIVE_COLLECTION_NAME)
    except:
        pass

    print("1. Ingesting memories with varying characteristics...")

    # Create memories with different timestamps, importance levels, and emotional content
    memories_data = [
        # Recent, high importance, positive emotion
        ("I just finished an excellent presentation on AI memory systems. Everyone was impressed!", 1.0, "presentation", "positive"),
        # Recent, medium importance, neutral
        ("The weather today is partly cloudy with mild temperatures.", 0.5, "weather", "neutral"),
        # Older, high importance, negative emotion
        ("The server crashed during the critical deployment last week. It was very stressful.", 0.9, "incident", "negative"),
        # Medium age, low importance, positive
        ("I had a nice coffee at the new cafe downtown yesterday.", 0.3, "personal", "positive"),
        # Old, medium importance, neutral
        ("Vector databases store embeddings for similarity search.", 0.7, "technical", "neutral"),
        # Very old, high importance, positive
        ("Zayar-Sama provided outstanding guidance on the project architecture.", 1.0, "feedback", "positive"),
    ]

    # Simulate different timestamps by sleeping between ingestions
    for i, (text, importance, source, emotion_type) in enumerate(memories_data):
        retrieval_manager.ingest_memory(text, importance_score=importance, source_id=source)

        # Simulate time passing (except for the last few which should be recent)
        if i < len(memories_data) - 2:
            time.sleep(3)  # 3 seconds to create clear time differences

    print("\n2. Testing queries with different scoring priorities...")

    test_queries = [
        ("How was the presentation?", "Should prioritize recent, high-importance, positive memory"),
        ("What happened with the server?", "Should find the older but high-importance negative memory"),
        ("Tell me about vector databases", "Should find technical memory despite age"),
        ("What positive feedback have you received?", "Should prioritize positive emotional valence"),
        ("What's the weather like?", "Should find recent weather memory"),
    ]

    for query, explanation in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: {explanation}")

        relevant_memories = retrieval_manager.retrieve_relevant_memories(query=query, n_results=3)

        print("Top memories with scoring breakdown:")
        for i, mem in enumerate(relevant_memories):
            metadata = mem['metadata']
            content_preview = mem['content'][:60] + "..." if len(mem['content']) > 60 else mem['content']

            print(f"  {i+1}. Score: {mem['score']:.4f}")
            print(f"      Content: {content_preview}")
            print(f"      Breakdown: Recency={mem['recency']:.2f}, Importance={mem['importance']:.2f}, "
                  f"Similarity={mem['similarity']:.2f}, Emotional={mem['emotional_saliency']:.2f}, "
                  f"Vividness={mem['vividness']:.2f}, Assoc={mem['associative_strength']:.2f}")
            print(f"      Source: {metadata.get('source_id', 'unknown')}, "
                  f"Age: {time.time() - metadata.get('timestamp', 0):.1f}s ago")

    print("\n3. Demonstrating scoring weight adjustments...")

    # Show how changing weights affects results
    original_weights = {
        'RECENCY_WEIGHT': settings.RECENCY_WEIGHT,
        'IMPORTANCE_WEIGHT': settings.IMPORTANCE_WEIGHT,
        'EMOTIONAL_SALIENCE_WEIGHT': settings.EMOTIONAL_SALIENCE_WEIGHT,
    }

    print("\nOriginal weights:", original_weights)

    # Temporarily adjust weights to emphasize recency
    settings.RECENCY_WEIGHT = 0.8
    settings.IMPORTANCE_WEIGHT = 0.1
    settings.EMOTIONAL_SALIENCE_WEIGHT = 0.05

    print("Adjusted weights (emphasizing recency):", {
        'RECENCY_WEIGHT': settings.RECENCY_WEIGHT,
        'IMPORTANCE_WEIGHT': settings.IMPORTANCE_WEIGHT,
        'EMOTIONAL_SALIENCE_WEIGHT': settings.EMOTIONAL_SALIENCE_WEIGHT,
    })

    query_recent = "Tell me something recent"
    recent_memories = retrieval_manager.retrieve_relevant_memories(query=query_recent, n_results=2)

    print(f"\nQuery with recency emphasis: '{query_recent}'")
    for i, mem in enumerate(recent_memories):
        age = time.time() - mem['metadata'].get('timestamp', 0)
        content_preview = mem['content'][:50] + "..." if len(mem['content']) > 50 else mem['content']
        print(f"  {i+1}. Age: {age:.1f}s, Score: {mem['score']:.4f} - {content_preview}")

    # Restore original weights
    settings.RECENCY_WEIGHT = original_weights['RECENCY_WEIGHT']
    settings.IMPORTANCE_WEIGHT = original_weights['IMPORTANCE_WEIGHT']
    settings.EMOTIONAL_SALIENCE_WEIGHT = original_weights['EMOTIONAL_SALIENCE_WEIGHT']

    print("\nWeighted scoring demo complete!")

if __name__ == "__main__":
    run_weighted_scoring_demo()
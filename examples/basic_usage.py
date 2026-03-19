import project_path
from src.core.retrieval_manager import RetrievalManager
import time
from config.settings import settings

def run_basic_usage():
    print("Initializing RetrievalManager...")
    # Ensure data directory exists
    settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    retrieval_manager = RetrievalManager()

    # Optional: Clear existing memories for a clean test run
    # print("Deleting existing memory collections...")
    # retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
    # retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_ASSOCIATIVE_COLLECTION_NAME)

    print("\n--- Ingesting Memories ---")
    memories_to_ingest = [
        ("The capital of France is Paris. Paris is known for its Eiffel Tower and Louvre Museum.", 0.9, "travel_log"),
        ("Today's weather forecast predicts sunny skies with a high of 25 degrees Celsius. Perfect for a walk in the park.", 0.7, "daily_news"),
        ("The project meeting is scheduled for tomorrow at 10 AM. We need to discuss the AI memory system implementation details.", 1.0, "work_calendar"),
        ("I remember reading about vector databases like ChromaDB and Pinecone for efficient similarity search. They are crucial for AI memory systems.", 0.85, "research_notes"),
        ("ShionAide is an AI assistant, built to help Zayar-Sama with tasks and project management.", 0.95, "self_description"),
        ("TinaAide is ShionAide's assistant, focusing on public-facing communications.", 0.8, "self_description"),
    ]

    for text, importance, source in memories_to_ingest:
        print(f"Ingesting from {source}: '{text[:50]}...'")
        retrieval_manager.ingest_memory(raw_text=text, importance_score=importance, source_id=source)
        time.sleep(0.1) # Small delay for different timestamps

    print("\n--- Retrieving Memories ---")

    queries = [
        "What is the weather like?",
        "Tell me about the AI project meeting.",
        "What are vector databases?",
        "Who is ShionAide?",
        "Where is the Eiffel Tower?"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        relevant_memories = retrieval_manager.retrieve_relevant_memories(query=query, n_results=3)
        if relevant_memories:
            print("  Relevant memories found:")
            for i, mem in enumerate(relevant_memories):
                content_preview = mem['content'][:70] + "..." if len(mem['content']) > 70 else mem['content']
                print(f"    {i+1}. (Score: {mem['score']:.4f}, Source: {mem['metadata'].get('source_id', 'N/A')}, Recency: {mem.get('recency', 0):.2f}, Importance: {mem.get('importance', 0):.2f}, Emotional: {mem.get('emotional_saliency', 0):.2f}, Vividness: {mem.get('vividness', 0):.2f}, Assoc: {mem.get('associative_strength', 0):.2f}) {content_preview}")
        else:
            print("  No relevant memories found.")

    print("\nBasic usage demonstration complete.")

if __name__ == "__main__":
    run_basic_usage()

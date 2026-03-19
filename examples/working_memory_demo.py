import project_path
from src.core.retrieval_manager import RetrievalManager
import time
from config.settings import settings

def run_working_memory_demo():
    """
    Demonstrates working memory prioritization in the human-like memory system.
    Working memory should prioritize very recent items and immediate context.
    """
    print("=== Working Memory Prioritization Demo ===")
    print("This demo shows how the system prioritizes working memory for immediate context.\n")

    # Ensure data directory exists
    settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    retrieval_manager = RetrievalManager()

    # Clear existing data for clean demo
    try:
        retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_ASSOCIATIVE_COLLECTION_NAME)
    except:
        pass

    print("1. Ingesting some older memories (simulating long-term memory)...")
    old_memories = [
        "The capital of France is Paris. Paris is known for its Eiffel Tower.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
    ]

    for memory in old_memories:
        retrieval_manager.ingest_memory(memory, importance_score=0.7, source_id="long_term")
        time.sleep(2)  # Simulate time passing

    print("\n2. Adding immediate context to working memory...")
    # Simulate a conversation where working memory should be prioritized
    conversation_context = [
        "We're discussing the AI memory system implementation.",
        "The user just asked about vector databases.",
        "I need to recall information about ChromaDB and Pinecone.",
    ]

    for context in conversation_context:
        retrieval_manager.working_memory.add_item({"content": context, "source_id": "conversation"})
        time.sleep(0.1)

    print("\n3. Querying with a question that should prioritize working memory...")
    query = "What are we discussing about vector databases?"

    relevant_memories = retrieval_manager.retrieve_relevant_memories(query=query, n_results=5)

    print(f"\nQuery: '{query}'")
    print("Retrieved memories (should prioritize working memory items):")
    for i, mem in enumerate(relevant_memories):
        source = mem['metadata'].get('source_id', 'unknown')
        is_working_memory = "WORKING MEMORY" if source == "conversation" else "LONG-TERM"
        content_preview = mem['content'][:80] + "..." if len(mem['content']) > 80 else mem['content']
        print(f"  {i+1}. [{is_working_memory}] Score: {mem['score']:.4f} - {content_preview}")

    print("\n4. Demonstrating working memory decay...")
    print("Waiting 10 seconds to simulate time passing...")
    time.sleep(10)

    # Add a new immediate item
    retrieval_manager.working_memory.add_item({"content": "The user wants to know about embedding models.", "source_id": "conversation"})

    query2 = "What about embedding models?"
    relevant_memories2 = retrieval_manager.retrieve_relevant_memories(query=query2, n_results=3)

    print(f"\nQuery after delay: '{query2}'")
    print("Working memory should still prioritize the most recent item:")
    for i, mem in enumerate(relevant_memories2):
        source = mem['metadata'].get('source_id', 'unknown')
        is_working_memory = "WORKING MEMORY" if source == "conversation" else "LONG-TERM"
        content_preview = mem['content'][:80] + "..." if len(mem['content']) > 80 else mem['content']
        print(f"  {i+1}. [{is_working_memory}] Score: {mem['score']:.4f} - {content_preview}")

    print("\nWorking memory demo complete!")

if __name__ == "__main__":
    run_working_memory_demo()
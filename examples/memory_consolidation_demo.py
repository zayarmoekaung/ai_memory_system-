import project_path
from src.core.retrieval_manager import RetrievalManager
from src.core.memory_consolidation import MemoryConsolidation
from src.core.embedding_manager import EmbeddingManager
import time
from config.settings import settings

def run_memory_consolidation_demo():
    """
    Demonstrates the memory consolidation process.
    Shows how working memory items can be processed for long-term storage.
    """
    print("=== Memory Consolidation Demo ===")
    print("This demo shows how working memory items are processed for consolidation.\n")

    # Ensure data directory exists
    settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    retrieval_manager = RetrievalManager()

    # Clear existing data for clean demo
    try:
        retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        retrieval_manager.memory_store.client.delete_collection(settings.CHROMA_ASSOCIATIVE_COLLECTION_NAME)
    except:
        pass

    print("1. Adding items to working memory...")

    # Simulate a conversation or task session
    conversation_items = [
        "User asked about the weather forecast",
        "I checked the weather API and got today's conditions",
        "The forecast shows sunny skies with 25°C high",
        "User then asked about Paris travel plans",
        "I recalled that Paris is the capital of France",
        "User mentioned the Eiffel Tower specifically",
        "I provided information about the Eiffel Tower's history",
        "Conversation shifted to AI memory systems",
        "User asked how the system handles context windows",
        "I explained chunk optimization and token limits",
    ]

    for item_text in conversation_items:
        retrieval_manager.working_memory.add_item({
            "content": item_text,
            "source_id": "conversation",
            "timestamp": time.time()
        })
        time.sleep(0.2)  # Simulate conversation pacing

    print(f"Added {len(conversation_items)} items to working memory")

    print("\n2. Examining working memory contents...")
    working_items = retrieval_manager.working_memory.get_recent_items()
    print(f"Current working memory contains {len(working_items)} items:")

    for i, item in enumerate(working_items):
        content_preview = item['content'][:60] + "..." if len(item['content']) > 60 else item['content']
        age = time.time() - item.get('timestamp', time.time())
        print(f"  {i+1}. ({age:.1f}s ago) {content_preview}")

    print("\n3. Simulating memory consolidation process...")

    # Access the memory consolidation component
    consolidator = retrieval_manager.memory_consolidation

    print("Consolidation would typically happen in the background, but for demo purposes:")
    print("- Working memory items would be analyzed for importance")
    print("- Related items would be grouped and summarized")
    print("- High-value items would be transferred to long-term storage")
    print("- Less important items would decay or be forgotten")

    # Demonstrate the consolidation logic conceptually
    print("\nAnalyzing working memory for consolidation candidates...")

    # Group related items (conceptual demonstration)
    weather_items = [item for item in working_items if 'weather' in item['content'].lower()]
    paris_items = [item for item in working_items if 'paris' in item['content'].lower() or 'eiffel' in item['content'].lower()]
    ai_items = [item for item in working_items if 'ai' in item['content'].lower() or 'memory' in item['content'].lower()]

    print(f"Weather-related items: {len(weather_items)}")
    print(f"Paris-related items: {len(paris_items)}")
    print(f"AI memory-related items: {len(ai_items)}")

    print("\n4. Demonstrating selective consolidation...")

    # Simulate consolidating high-value items
    high_value_items = []

    # Weather info might be consolidated if it's recent and relevant
    if weather_items:
        consolidated_weather = "Weather information: " + weather_items[0]['content']
        high_value_items.append({
            "content": consolidated_weather,
            "importance_score": 0.7,
            "source_id": "consolidated_conversation"
        })

    # Paris/Eiffel Tower info is educational and might be worth keeping
    if paris_items:
        consolidated_paris = "Paris information: " + " ".join([item['content'] for item in paris_items[:2]])
        high_value_items.append({
            "content": consolidated_paris,
            "importance_score": 0.8,
            "source_id": "consolidated_conversation"
        })

    # AI memory discussion is highly relevant to the system itself
    if ai_items:
        consolidated_ai = "AI memory system discussion: " + " ".join([item['content'] for item in ai_items[:2]])
        high_value_items.append({
            "content": consolidated_ai,
            "importance_score": 0.9,
            "source_id": "consolidated_conversation"
        })

    print("Items selected for consolidation:")
    for i, item in enumerate(high_value_items):
        print(f"  {i+1}. Importance: {item['importance_score']}, Content: {item['content'][:80]}...")

    print("\n5. Adding consolidated memories to long-term storage...")

    for item in high_value_items:
        retrieval_manager.ingest_memory(
            raw_text=item['content'],
            importance_score=item['importance_score'],
            source_id=item['source_id']
        )
        print(f"Consolidated: {item['content'][:50]}...")

    print("\n6. Verifying consolidation by querying...")

    test_queries = [
        "What did we discuss about weather?",
        "Tell me about Paris and the Eiffel Tower",
        "What about AI memory systems?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        relevant_memories = retrieval_manager.retrieve_relevant_memories(query=query, n_results=2)

        for i, mem in enumerate(relevant_memories):
            source = mem['metadata'].get('source_id', 'unknown')
            content_preview = mem['content'][:70] + "..." if len(mem['content']) > 70 else mem['content']
            print(f"  {i+1}. [{source}] Score: {mem['score']:.4f} - {content_preview}")

    print("\n7. Working memory cleanup simulation...")

    # Simulate working memory decay/cleanup
    print("In a real system, working memory would gradually decay:")
    print("- Less important items would be forgotten")
    print("- Only highly relevant recent items would remain")
    print("- Consolidation would happen periodically in the background")

    # Show current working memory state
    current_wm = retrieval_manager.working_memory.get_recent_items()
    print(f"\nCurrent working memory still contains {len(current_wm)} items")
    print("(In practice, these would decay over time or be explicitly cleared)")

    print("\nMemory consolidation demo complete!")

if __name__ == "__main__":
    run_memory_consolidation_demo()
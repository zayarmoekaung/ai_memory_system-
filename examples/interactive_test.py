from ai_memory_system.src.core.retrieval_manager import RetrievalManager
import os

def run_interactive_test():
    print("Initializing RetrievalManager for interactive testing...")
    retrieval_manager = RetrievalManager()

    # Optional: Clear existing memories for a clean test run
    # if input("Clear all existing memories? (y/N): ").lower() == 'y':
    #     print("Deleting existing memory collection...")
    #     retrieval_manager.memory_store.delete_collection()
    #     print("Collection deleted. Please restart the script to re-ingest if needed.")
    #     return

    # Basic ingestion to have some data to query
    print("\n--- Initial Memory Ingestion (for testing purposes) ---")
    initial_memories = [
        "My name is ShionAide, and I am an AI assistant for Zayar-Sama.",
        "Zayar-Sama is my owner and I help him manage projects and tasks.",
        "TinaAide is my assistant, handling public communications and some logic tasks.",
        "We are currently developing a sophisticated AI agent memory system.",
        "This memory system involves vectorizing memories, weighted retrieval, and dynamic chunk selection.",
        "ChromaDB is used as the vector database for persistent memory storage.",
        "The meeting about the AI memory system is scheduled for tomorrow at 10 AM.",
        "I need to prepare a summary of the project's current status for the meeting.",
        "The weather today is clear and sunny, perfect for outdoor activities.",
        "I should remember to check Zayar-Sama's calendar for any upcoming important events."
    ]

    for i, mem_text in enumerate(initial_memories):
        importance = 0.5 + (i / len(initial_memories)) * 0.5 # Gradually increasing importance
        source = "interactive_ingest"
        retrieval_manager.ingest_memory(raw_text=mem_text, importance_score=importance, source_id=source)
    print(f"Ingested {len(initial_memories)} initial memories.\n")

    print("--- Interactive Memory Retrieval ---")
    print("Type your query to retrieve relevant memories. Type 'exit' to quit.")

    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break

        n_results_str = input("Number of results to retrieve (default 3): ")
        n_results = int(n_results_str) if n_results_str.isdigit() else 3

        print(f"Searching for up to {n_results} relevant memories for: '{query}'")
        relevant_memories = retrieval_manager.retrieve_relevant_memories(query=query, n_results=n_results)

        if relevant_memories:
            print("  Relevant memories found:")
            for i, mem in enumerate(relevant_memories):
                content_preview = mem['content'][:100] + "..." if len(mem['content']) > 100 else mem['content']
                print(f"    {i+1}. (Score: {mem['score']:.4f}, Source: {mem['metadata'].get('source_id', 'N/A')}, Recency: {mem['recency']:.2f}, Importance: {mem['importance']:.2f})\n       -> {content_preview}")
        else:
            print("  No relevant memories found.")

    print("\nInteractive test complete. Goodbye!")

if __name__ == "__main__":
    # Ensure the data directory exists for ChromaDB persistence
    from ai_memory_system.config.settings import settings
    settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    
    # NLTK punkt tokenizer data might need to be downloaded for sent_tokenize in ChunkOptimizer
    # import nltk
    # try:
    #     from nltk.tokenize import sent_tokenize
    #     sent_tokenize("test sentence")
    # except LookupError:
    #     nltk.download('punkt')

    run_interactive_test()

import project_path
from src.core.retrieval_manager import RetrievalManager
import os
import time
from config.settings import settings

def run_interactive_test():
    print("Initializing RetrievalManager for interactive testing...")
    retrieval_manager = RetrievalManager()

    # Force clear existing memories for a clean test run
    print("Deleting existing memory collections for a clean test run...")
    retrieval_manager.memory_store.delete_collection(settings.CHROMA_COLLECTION_NAME)
    retrieval_manager.memory_store.delete_collection(settings.CHROMA_ASSOCIATIVE_COLLECTION_NAME)
    print("Collections deleted.")

    # --- Vividness Score Verification ---
    print("\n--- Vividness Score Verification ---")
    vivid_memories = [
        "The vivid red sunset painted the sky, casting long, eerie shadows across the ancient ruins. A chill wind whispered secrets through the crumbling stones.",
        "Meeting Zayar-Sama for the first time was an important and memorable event. His guidance is always insightful.",
        "I calculated the total server uptime, which was 99.9% last month. The system performed stably.",
        "A simple note: remember to buy milk tomorrow."
    ]

    ingested_vivid_memory_ids = []

    for i, mem_text in enumerate(vivid_memories):
        importance = 0.6 + (i / len(vivid_memories)) * 0.4 # Vary importance
        source = "vividness_test"
        print(f"\nIngesting memory: '{mem_text}' (Importance: {importance:.2f})")
        retrieval_manager.ingest_memory(raw_text=mem_text, importance_score=importance, source_id=source)
        # Get the ID of the last ingested chunk to retrieve its vividness later
        # This assumes ingest_memory processes chunks and the last one is representative
        # A more robust approach would be to capture all chunk IDs returned by ingest_memory
        # For simplicity, we'll try to retrieve the first chunk associated with this ingestion
        retrieved_chunks = retrieval_manager.memory_store.search_memories(
            retrieval_manager.embedding_manager.get_embedding(mem_text), n_results=1)
        if retrieved_chunks:
            ingested_vivid_memory_ids.append(retrieved_chunks[0]['id'])
            print(f"  Initial Vividness: {retrieved_chunks[0]['metadata'].get('vividness_score', 0.0):.4f}")

    print("\n--- Simulating Time Passage for Vividness Decay ---")
    time_to_pass = 5 * 24 * 3600 # 5 days in seconds
    print(f"  Pausing for {time_to_pass} seconds (simulated) to observe decay...")
    # In a real scenario, you'd actually wait or modify timestamps directly
    # For this test, we'll assume time has passed and just re-calculate.
    # In a real application, the timestamp would be read from the DB and decay applied
    # For the purpose of this interactive test, we will re-query and the decay will be applied then.
    # This part of the test is more conceptual for now until we can directly manipulate timestamps easily.

    print("\n--- Re-evaluating Vividness After Simulated Time Passage (Directly retrieving by ID) ---")
    for i, chunk_id in enumerate(ingested_vivid_memory_ids):
        mem_data = retrieval_manager.memory_store.get_memory_by_id(chunk_id)
        if mem_data:
            # Manually calculate vividness score (this path will call _calculate_vividness_score and apply decay)
            # We need to construct a "fake" chunk_data dict for scoring, as _calculate_vividness_score expects metadata
            # from a scored chunk, which is what retrieve_relevant_memories usually generates.
            # For direct testing, we'll mimic that structure.
            # First, extract relevant info from mem_data:
            metadata_from_store = mem_data.get('metadata', {})
            content_from_store = mem_data.get('content', '')

            # Now, call the internal decay calculation. This will ensure _calculate_vividness_score gets the right metadata.
            decayed_vividness_value = retrieval_manager._calculate_vividness_score(metadata_from_store)
            print(f"  [DEBUG][Test Script] Calculated decayed_vividness_value: {decayed_vividness_value:.4f}")

            print(f"  Memory ID: {chunk_id[:8]}... (Content: '{content_from_store[:50]}...')")
            print(f"  Initial Vividness (from store): {metadata_from_store.get('vividness_score', 0.0):.4f}")
            print(f"  Decayed Vividness (after simulated time): {decayed_vividness_value:.4f}")
        else:
            print(f"  Memory data for ID {chunk_id} not found.")

    # --- Associative Strength Score Verification ---
    print("\n--- Associative Strength Score Verification ---")

    # Ingest memories with clear entities and relationships
    associative_memories = [
        "Zayar-Sama is working on the AI Memory System project with TinaAide.",
        "The AI Memory System uses ChromaDB for vector storage and is a key project.",
        "TinaAide is responsible for public-facing communications and some logic tasks.",
        "ChromaDB is a powerful vector database used in many AI applications.",
        "The project meeting for the AI Memory System is tomorrow. Zayar-Sama will lead it."
    ]

    for i, mem_text in enumerate(associative_memories):
        importance = 0.7 + (i / len(associative_memories)) * 0.3
        source = "associative_test"
        print(f"\nIngesting associative memory: '{mem_text}' (Importance: {importance:.2f})")
        retrieval_manager.ingest_memory(raw_text=mem_text, importance_score=importance, source_id=source)

    print("\n--- Querying to observe Associative Strength ---")

    queries_for_associative_strength = [
        "What is TinaAide working on with Zayar-Sama?", # Direct link
        "Tell me about the key project using vector databases.", # Indirect link via entities
        "Who is leading tomorrow's meeting?" # Direct entity query
    ]

    for query in queries_for_associative_strength:
        print(f"\nQuery: '{query}'")
        relevant_memories = retrieval_manager.retrieve_relevant_memories(query, n_results=5)
        if relevant_memories:
            for i, mem in enumerate(relevant_memories):
                content_preview = mem['content'][:100] + "..." if len(mem['content']) > 100 else mem['content']
                print(f"    {i+1}. (Score: {mem['score']:.4f}, Source: {mem['metadata'].get('source_id', 'N/A')})\n       -> Assoc. Strength: {mem['associative_strength']:.4f}, Content: {content_preview}")
        else:
            print("    No relevant memories found.")

    # --- Interactive Memory Retrieval (Existing functionality) ---
    print("\n--- Interactive Memory Retrieval ---")
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
                print(f"    {i+1}. (Score: {mem['score']:.4f}, Source: {mem['metadata'].get('source_id', 'N/A')}, Recency: {mem['recency']:.2f}, Importance: {mem['importance']:.2f}, Vividness: {mem['vividness']:.2f}, Emotional: {mem['emotional_saliency']:.2f}, AssocStrength: {mem['associative_strength']:.2f})\n       -> {content_preview}")
        else:
            print("  No relevant memories found.")

    print("\nInteractive test complete. Goodbye!")

if __name__ == "__main__":
    # Ensure the data directory exists for ChromaDB persistence
    settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    
    # NLTK punkt tokenizer data might need to be downloaded for sent_tokenize in ChunkOptimizer
    # import nltk
    # try:
    #     from nltk.tokenize import sent_tokenize
    #     sent_tokenize("test sentence")
    # except LookupError:
    #     nltk.download('punkt')

    run_interactive_test()

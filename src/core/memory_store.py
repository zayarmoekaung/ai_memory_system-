import chromadb
from typing import List, Dict, Any
from config.settings import settings

class MemoryStore:
    def __init__(self):
        """
        Initializes the MemoryStore by setting up the ChromaDB client and collection.
        The database path and collection name are fetched from the application settings.
        """
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
        # Use the collection name from settings
        self.collection = self.client.get_or_create_collection(name=settings.CHROMA_COLLECTION_NAME)

    def add_memory_chunk(self, chunk_id: str, content: str, embedding: List[float], metadata: Dict[str, Any]):
        """
        Adds a single memory chunk to the ChromaDB collection.

        Args:
            chunk_id (str): A unique identifier for the memory chunk.
            content (str): The textual content of the memory chunk.
            embedding (List[float]): The vector embedding of the chunk's content.
            metadata (Dict[str, Any]): A dictionary of associated metadata (e.g., timestamp, importance_score, source_id).
        """
        try:
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[chunk_id]
            )
        except Exception as e:
            print(f"Error adding memory chunk {chunk_id}: {e}")
            raise

    def search_memories(self, query_embedding: List[float], n_results: int = 5, min_distance: float = 0.5) -> List[Dict[str, Any]]:
        """
        Performs a similarity search in ChromaDB using a query embedding.

        Args:
            query_embedding (List[float]): The embedding of the query.
            n_results (int): The maximum number of results to return.
            min_distance (float): The minimum distance (e.g., 0.0 to 2.0 for cosine distance) for filtering results.
                                  Lower distance means higher similarity. Currently, ChromaDB uses L2 distance by default.
                                  This parameter would typically be used to filter out less relevant results.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains
                                  'id', 'content', 'embedding', 'metadata', and 'distance' for a retrieved chunk.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                # where={'$and': [{'distance': {'$lt': min_distance}}]},
                # ChromaDB does not directly support distance filtering in query() with L2 distance as readily as cosine.
                # We will filter post-query if needed or adjust based on distance metric.
                # For now, simply return results and let higher-level logic handle relevance filtering.
                include=['documents', 'embeddings', 'metadatas', 'distances']
            )

            formatted_results = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'embedding': results['embeddings'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            return formatted_results
        except Exception as e:
            print(f"Error searching memories: {e}")
            return []

    def get_memory_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """
        Retrieves a specific memory chunk by its unique identifier.

        Args:
            chunk_id (str): The unique identifier of the memory chunk.

        Returns:
            Dict[str, Any]: A dictionary containing the chunk's 'id', 'content', 'embedding', and 'metadata',
                            or an empty dictionary if not found.
        """
        try:
            result = self.collection.get(ids=[chunk_id], include=['documents', 'embeddings', 'metadatas'])
            if result and result['ids'] and result['ids'][0]:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'embedding': result['embeddings'][0],
                    'metadata': result['metadatas'][0]
                }
            return {}
        except Exception as e:
            print(f"Error retrieving memory {chunk_id}: {e}")
            return {}

    def delete_collection(self):
        """
        Deletes the entire ChromaDB collection.
        Use with caution, primarily for testing or resetting the memory store.
        """
        try:
            self.client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
            print(f"Collection '{settings.CHROMA_COLLECTION_NAME}' deleted.")
            # Re-initialize collection after deletion to ensure it's ready for new data
            self.collection = self.client.get_or_create_collection(name=settings.CHROMA_COLLECTION_NAME)
        except Exception as e:
            print(f"Error deleting collection {settings.CHROMA_COLLECTION_NAME}: {e}")
            raise


# Example usage (for testing purposes)
if __name__ == "__main__":
    # This part assumes settings are correctly loaded (e.g., via .env or default values)
    print(f"ChromaDB will persist to: {settings.CHROMA_DB_PATH}")
    memory_store = MemoryStore()

    # Ensure collection is empty for a clean test run
    # Uncomment the following line to clear the database for each run if needed
    # memory_store.delete_collection()

    from uuid import uuid4
    import time

    # Add a memory chunk
    chunk_id_1 = str(uuid4())
    content_1 = "The quick brown fox jumps over the lazy dog."
    # For testing, we can use a dummy embedding if EmbeddingManager is not yet integrated
    dummy_embedding_1 = [0.1] * 384  # Assuming 384 dimensions for all-MiniLM-L6-v2
    metadata_1 = {"timestamp": time.time(), "importance_score": 0.8, "source_id": "test_source_1", "original_text_start_index": 0}
    memory_store.add_memory_chunk(chunk_id_1, content_1, dummy_embedding_1, metadata_1)
    print(f"Added memory chunk 1: {chunk_id_1}")

    chunk_id_2 = str(uuid4())
    content_2 = "The dog is very lazy and sleeps all day."
    dummy_embedding_2 = [0.2] * 384
    metadata_2 = {"timestamp": time.time() - 3600, "importance_score": 0.6, "source_id": "test_source_1", "original_text_start_index": len(content_1) + 1}
    memory_store.add_memory_chunk(chunk_id_2, content_2, dummy_embedding_2, metadata_2)
    print(f"Added memory chunk 2: {chunk_id_2}")

    # Search for memories (using a dummy query embedding for now)
    query_embedding = [0.15] * 384
    print("\nSearching for memories...")
    found_memories = memory_store.search_memories(query_embedding, n_results=2)
    for mem in found_memories:
        print(f"  ID: {mem['id']}, Content: {mem['content']}, Distance: {mem['distance']:.4f}")

    # Retrieve a specific memory
    print(f"\nRetrieving memory by ID: {chunk_id_1}")
    retrieved_memory = memory_store.get_memory_by_id(chunk_id_1)
    if retrieved_memory:
        print(f"  Found: {retrieved_memory['content']}")
    else:
        print("  Memory not found.")

    # Test deleting the collection (use with caution)
    # print("\nDeleting collection...")
    # memory_store.delete_collection()
    # print("Searching after deletion...")
    # found_memories_after_delete = memory_store.search_memories(query_embedding, n_results=2)
    # print(f"Found memories: {len(found_memories_after_delete)}")

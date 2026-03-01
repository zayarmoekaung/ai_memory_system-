import time
import uuid
from typing import List, Dict, Any
from datetime import datetime, timedelta

from .memory_store import MemoryStore
from .embedding_manager import EmbeddingManager
from .chunk_optimizer import ChunkOptimizer
from .working_memory import WorkingMemory
from .memory_consolidation import MemoryConsolidation
from .associative_network import AssociativeNetwork
from config.settings import settings # Updated import path

# Placeholder for NLTK if not globally downloaded
# import nltk
# try:
#     from nltk.tokenize import sent_tokenize
#     sent_tokenize("test")
# except LookupError:
#     nltk.download('punkt')

# Placeholder for basic sentiment and NER if full NLP models are not used directly in prototype
def _simple_sentiment_analysis(text: str) -> float:
    # A very basic heuristic: count positive/negative words
    positive_keywords = ["good", "great", "excellent", "happy", "love", "nice", "perfect"]
    negative_keywords = ["bad", "terrible", "poor", "sad", "hate", "error", "issue"]
    text_lower = text.lower()
    positive_count = sum(text_lower.count(k) for k in positive_keywords)
    negative_count = sum(text_lower.count(k) for k in negative_keywords)
    if positive_count > negative_count: return 0.8
    if negative_count > positive_count: return -0.8
    return 0.0 # Neutral

def _simple_entity_extraction(text: str) -> List[str]:
    # A very basic heuristic: capitalized words not at start of sentence, or known entities
    # In a real system, use spaCy or a dedicated NER model
    words = text.split()
    entities = set()
    for word in words:
        cleaned_word = word.strip("',.!?;:""''")
        if cleaned_word and cleaned_word[0].isupper() and cleaned_word.lower() not in ["the", "a", "an", "and", "is", "are", "was", "were", "i", "you", "he", "she", "it", "we", "they", "this", "that", "for", "on", "in", "at", "with", "from", "by"]:
            entities.add(cleaned_word)
    return list(entities)

def _simple_context_tag_extraction(text: str) -> List[str]:
    # Very basic: check for predefined keywords
    text_lower = text.lower()
    tags = []
    if "memory system" in text_lower or "ai agent" in text_lower: tags.append("ai_project")
    if "weather" in text_lower or "forecast" in text_lower: tags.append("weather")
    if "meeting" in text_lower or "schedule" in text_lower: tags.append("calendar")
    return tags

class RetrievalManager:
    def __init__(self):
        """
        Initializes the RetrievalManager with instances of all core memory components.
        Sets the tokenizer for ChunkOptimizer from EmbeddingManager.
        """
        self.memory_store = MemoryStore()
        self.embedding_manager = EmbeddingManager()
        self.chunk_optimizer = ChunkOptimizer()
        self.chunk_optimizer.set_tokenizer(self.embedding_manager.get_tokenizer())
        
        self.working_memory = WorkingMemory(capacity=settings.WORKING_MEMORY_CAPACITY)
        self.associative_network = AssociativeNetwork()
        self.memory_consolidation = MemoryConsolidation(
            memory_store=self.memory_store,
            embedding_manager=self.embedding_manager,
            chunk_optimizer=self.chunk_optimizer
        )

    def ingest_memory(self, raw_text: str, importance_score: float = 0.5, source_id: str = "agent_observation"):
        """
        Ingests raw text by chunking it, generating embeddings, and storing each chunk with enhanced metadata.
        Also updates working memory and associative network.

        Args:
            raw_text (str): The raw text of the memory.
            importance_score (float): A score indicating the importance of this memory (0.0 to 1.0).
            source_id (str): An identifier for the source of this memory.
        """
        # 1. Add raw text to working memory (for immediate context/future consolidation)
        self.working_memory.add_item({"content": raw_text, "source_id": source_id})

        # 2. Chunk and Pre-process for long-term storage
        chunks = self.chunk_optimizer.chunk_text(raw_text)
        current_timestamp = time.time()
        event_sequence_id = str(uuid.uuid4()) # Generate a new event ID for this ingestion sequence

        for i, chunk_content in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            embedding = self.embedding_manager.get_embedding(chunk_content)

            # Placeholder for NLP extraction (can be replaced with actual model calls)
            emotional_valence = _simple_sentiment_analysis(chunk_content)
            associated_entities = _simple_entity_extraction(chunk_content)
            context_tags = _simple_context_tag_extraction(chunk_content)

            metadata = {
                "timestamp": current_timestamp,
                "importance_score": importance_score,
                "source_id": source_id,
                "original_text_start_index": raw_text.find(chunk_content), # Simple approach, can be refined
                "associated_entities": associated_entities, # Enhanced metadata
                "emotional_valence": emotional_valence, # Enhanced metadata
                "vividness_score": 0.5, # Default, will be dynamically learned/adjusted (Enhanced metadata)
                "context_tags": context_tags, # Enhanced metadata
                "event_sequence_id": event_sequence_id # Enhanced metadata
            }
            self.memory_store.add_memory_chunk(chunk_id, chunk_content, embedding, metadata)
            
            # 3. Update Associative Network
            if associated_entities:
                self.associative_network.link_chunk_to_entities(chunk_id, associated_entities)

        print(f"Ingested {len(chunks)} memory chunks from source: {source_id}")

        # Optional: Trigger consolidation from working memory (e.g., after a few turns or on a timer)
        # For prototype, this might be called explicitly or through a simple loop
        # self.memory_consolidation.process_working_memory_for_consolidation(self.working_memory.get_recent_items())

    def _calculate_recency_score(self, timestamp: float) -> float:
        now = time.time()
        age_seconds = now - timestamp
        # Decay function can be made more sophisticated, potentially non-linear
        one_day_seconds = 24 * 3600
        seven_days_seconds = 7 * one_day_seconds

        if age_seconds <= one_day_seconds:
            return 1.0 
        elif age_seconds < seven_days_seconds:
            decay_factor = (seven_days_seconds - age_seconds) / (seven_days_seconds - one_day_seconds)
            return max(0.0, decay_factor)
        else:
            return 0.0

    def _calculate_vividness_score(self, metadata: Dict[str, Any]) -> float:
        # Placeholder: could be based on length, specific keywords, or learned over time
        return metadata.get('vividness_score', 0.5)

    def _calculate_emotional_saliency_score(self, metadata: Dict[str, Any]) -> float:
        # Directly use emotional_valence from metadata. Normalize -1 to 1 to 0 to 1 if needed
        valence = metadata.get('emotional_valence', 0.0)
        return (valence + 1.0) / 2.0 # Normalize -1 to 1 to 0 to 1 range

    def _calculate_associative_strength_score(self, chunk_id: str, query_entities: List[str]) -> float:
        # Conceptual: how strongly is this chunk connected to entities in the query via associative network
        # For prototype, a simple count of shared entities or direct links
        score = 0.0
        chunk_entities = self.memory_store.get_memory_by_id(chunk_id).get('metadata', {}).get('associated_entities', [])
        shared_entities = set(chunk_entities).intersection(set(query_entities))
        score += len(shared_entities) * 0.2 # Simple heuristic for shared entities
        
        # Further, check direct links in associative network if entities from query are linked to chunk_id
        for entity in query_entities:
            if self.associative_network.graph.has_edge(chunk_id, entity):
                score += self.associative_network.graph[chunk_id][entity].get('weight', 0.5)
        return min(1.0, score) # Clamp score

    def retrieve_relevant_memories(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves relevant memories based on a query, combining vector similarity with enhanced weighted scoring.

        Args:
            query (str): The query text.
            n_results (int): The number of top relevant chunks to retrieve before optimization.

        Returns:
            List[Dict[str, Any]]: A list of optimized, highly relevant memory chunks, each with 'content', 'id', etc.
        """
        query_embedding = self.embedding_manager.get_embedding(query)
        query_tokens = self.chunk_optimizer.count_tokens(query)

        # 1. Initial Activation (Working Memory First)
        # Prioritize working memory for very recent and active items
        working_memory_results = []
        for item in self.working_memory.get_recent_items():
            # For prototype, a simple content match or high recency for WM items
            if query.lower() in item.get('content', '').lower():
                working_memory_results.append({
                    'id': f"wm_{str(uuid.uuid4())}", # Temporary ID for WM items
                    'content': item['content'],
                    'metadata': {"source_id": "working_memory", "timestamp": item['timestamp'], "importance_score": 1.0}, # High importance
                    'embedding': self.embedding_manager.get_embedding(item['content']), # Embed WM item for scoring
                    'score': 1.0 # High score for working memory match
                })
        # For now, append working memory results. Later, we'll integrate scoring better.
        all_retrieved_chunks_pre_scoring = working_memory_results # Start with WM, then add long-term

        # 2. Associative Spreading & Triggering (Long-Term Memory Search - Broad)
        # Perform initial semantic search on long-term store
        raw_retrieved_chunks = self.memory_store.search_memories(query_embedding, n_results=n_results)
        all_retrieved_chunks_pre_scoring.extend(raw_retrieved_chunks)

        # Further activate memories via AssociativeNetwork using query entities
        query_entities = _simple_entity_extraction(query) # Extract entities from the query
        activated_chunk_ids_from_associative_net = set()
        for entity in query_entities:
            activated_chunk_ids_from_associative_net.update(self.associative_network.get_chunks_by_entity(entity))
            # Also consider broader related nodes if needed (depth > 1)
            # activated_chunk_ids_from_associative_net.update(self.associative_network.get_related_nodes(entity, depth=1))

        # Fetch content for activated chunks not already in raw_retrieved_chunks
        for chunk_id in activated_chunk_ids_from_associative_net:
            # Check if this chunk is already in our list (by ID or some other unique identifier)
            if not any(c.get('id') == chunk_id for c in all_retrieved_chunks_pre_scoring):
                # Retrieve the full chunk data from memory store
                full_chunk_data = self.memory_store.get_memory_by_id(chunk_id)
                if full_chunk_data:
                    # For now, just add it with a default score, will be properly scored next step
                    full_chunk_data['distance'] = 1.0 # Default distance for associatively retrieved
                    all_retrieved_chunks_pre_scoring.append(full_chunk_data)

        # 3. Refined Weighted Scoring for ALL collected chunks
        scored_chunks = []
        for chunk_data in all_retrieved_chunks_pre_scoring:
            metadata = chunk_data.get('metadata', {})
            distance = chunk_data.get('distance', 1.0) 
            similarity_score = 1.0 - (distance / 2.0) if distance <= 2.0 else 0.0
            similarity_score = max(0.0, min(1.0, similarity_score))

            recency_score = self._calculate_recency_score(metadata.get('timestamp', 0))
            importance_score = metadata.get('importance_score', 0.5)
            task_relatedness_score = metadata.get('task_relatedness_score', 0.5) # Assuming it's in metadata now
            emotional_saliency_score = self._calculate_emotional_saliency_score(metadata)
            vividness_score = self._calculate_vividness_score(metadata)
            associative_strength_score = self._calculate_associative_strength_score(chunk_data['id'], query_entities)

            # Combine scores using weighted sum from settings
            combined_score = (
                settings.RECENCY_WEIGHT * recency_score +
                settings.IMPORTANCE_WEIGHT * importance_score +
                settings.TASK_RELATEDNESS_WEIGHT * task_relatedness_score +
                settings.ASSOCIATIVE_STRENGTH_WEIGHT * associative_strength_score +
                settings.EMOTIONAL_SALIENCE_WEIGHT * emotional_saliency_score +
                settings.VIVIDNESS_WEIGHT * vividness_score +
                (1.0 - (settings.RECENCY_WEIGHT + settings.IMPORTANCE_WEIGHT + settings.TASK_RELATEDNESS_WEIGHT + settings.ASSOCIATIVE_STRENGTH_WEIGHT + settings.EMOTIONAL_SALIENCE_WEIGHT + settings.VIVIDNESS_WEIGHT)) * similarity_score
            )
            # Ensure combined score is between 0 and 1
            combined_score = max(0.0, min(1.0, combined_score))

            scored_chunks.append({
                'id': chunk_data['id'],
                'content': chunk_data['content'],
                'metadata': metadata,
                'embedding': chunk_data.get('embedding'),
                'score': combined_score,
                'similarity': similarity_score, 
                'recency': recency_score,
                'importance': importance_score,
                'task_relatedness': task_relatedness_score,
                'emotional_saliency': emotional_saliency_score,
                'vividness': vividness_score,
                'associative_strength': associative_strength_score
            })
        
        # 4. Optimize chunks for context window
        # Pass all scored chunks and let optimizer select and truncate based on token limits
        optimized_chunks = self.chunk_optimizer.optimize_chunks_for_context(scored_chunks, query_tokens)

        return optimized_chunks

# Example usage (for testing purposes)
if __name__ == "__main__":
    print("\n--- Initializing RetrievalManager for human-like memory prototype ---")
    # Ensure data directory exists
    settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

    retrieval_manager = RetrievalManager()

    # Optional: Clear existing memories for a clean test run
    # retrieval_manager.memory_store.delete_collection()
    # retrieval_manager.memory_store.delete_collection(settings.CHROMA_ASSOCIATIVE_COLLECTION_NAME)

    # Ingest some test memories with enhanced metadata
    print("\n--- Ingesting Memories with Enhanced Metadata ---")
    retrieval_manager.ingest_memory(
        "The capital of France is Paris. Paris is known for its Eiffel Tower and Louvre Museum. I visited Paris last year and it was beautiful.",
        importance_score=0.9,
        source_id="travel_log"
    )
    time.sleep(0.5) # Simulate time passing
    retrieval_manager.ingest_memory(
        "Today's weather forecast predicts sunny skies with a high of 25 degrees Celsius. Perfect for a walk in the park.",
        importance_score=0.7,
        source_id="daily_news"
    )
    time.sleep(0.5) # Simulate more time passing
    retrieval_manager.ingest_memory(
        "The project meeting is scheduled for tomorrow at 10 AM. We need to discuss the AI memory system implementation details. TinaAide and ShionAide are key contributors.",
        importance_score=1.0,
        source_id="work_calendar"
    )
    time.sleep(0.5) # Simulate time passing
    retrieval_manager.ingest_memory(
        "I remember reading about vector databases like ChromaDB and Pinecone for efficient similarity search. They are crucial for AI memory systems.",
        importance_score=0.85,
        source_id="research_notes"
    )
    time.sleep(0.5) # Simulate time passing
    retrieval_manager.ingest_memory(
        "Zayar-Sama provided excellent guidance on the human-like memory design concept.",
        importance_score=0.98,
        source_id="conversation_summary"
    )

    print("\n--- Retrieving Memories for specific queries ---")

    # Simulate a thought breakdown and remembrance flow
    conversation_turn = "If the weather is nice let's take my car and go to Paris"
    print(f"\nSimulating response for: \"{conversation_turn}\"")

    # Querying based on the current conversation turn
    relevant_memories = retrieval_manager.retrieve_relevant_memories(
        conversation_turn, n_results=5
    )
    print("\n--- Retrieved and Optimized Memories (Human-like Flow) ---")
    total_tokens_in_retrieved = 0
    if relevant_memories:
        for i, mem in enumerate(relevant_memories):
            content_tokens = retrieval_manager.chunk_optimizer.count_tokens(mem['content'])
            total_tokens_in_retrieved += content_tokens
            print(f"Memory {i+1} (Score: {mem['score']:.4f}, Tokens: {content_tokens}): {mem['content']}")
            print(f"  Metadata: Recency={mem['recency']:.2f}, Importance={mem['importance']:.2f}, Similarity={mem['similarity']:.2f}, Emotional={mem['emotional_saliency']:.2f}, Vividness={mem['vividness']:.2f}, AssocStrength={mem['associative_strength']:.2f}")
    else:
        print("No relevant memories found.")

    query_tokens_test = retrieval_manager.chunk_optimizer.count_tokens(conversation_turn)
    print(f"\nQuery tokens: {query_tokens_test}")
    print(f"Total memory tokens in retrieved: {total_tokens_in_retrieved}")
    print(f"Total tokens (query + memories): {query_tokens_test + total_tokens_in_retrieved}")
    print(f"Max context tokens (from settings): {settings.MAX_CONTEXT_TOKENS}")

    # You can further process 'relevant_memories' to formulate a response to Zayar-Sama
    # based on the retrieved information.

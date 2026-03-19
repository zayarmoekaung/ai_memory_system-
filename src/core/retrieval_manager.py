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
import networkx as nx # Added for associative strength calculations
from src.core.sentiment_analyzer import SentimentAnalyzer
from src.core.entity_extractor import EntityExtractor
from src.core.context_tagger import ContextTagger
from src.core.vividness_calculator import VividnessCalculator

# Placeholder for NLTK if not globally downloaded
# import nltk
# try:
#     from nltk.tokenize import sent_tokenize
#     sent_tokenize("test")
# except LookupError:
#     nltk.download('punkt')

# NOTE: _simple_sentiment_analysis has been replaced by SentimentAnalyzer class.
# These simple placeholder functions will be replaced by dedicated NLP classes in Phase 1.

# DEPRECATED: _simple_context_tag_extraction has been replaced by ContextTagger class.
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
        
        self.sentiment_analyzer = SentimentAnalyzer() # Initialize SentimentAnalyzer
        self.entity_extractor = EntityExtractor() # Initialize EntityExtractor
        self.context_tagger = ContextTagger() # Initialize ContextTagger
        self.vividness_calculator = VividnessCalculator(self.sentiment_analyzer, self.entity_extractor, self.chunk_optimizer) # Initialize VividnessCalculator
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

            # Sentiment Analysis
            sentiment_scores = self.sentiment_analyzer.analyze_sentiment(chunk_content)
            emotional_valence = sentiment_scores['compound']

            # Named Entity Recognition
            all_entities = self.entity_extractor.extract_entities(chunk_content)
            # Manually add known agent names if present in the chunk_content
            if "Zayar-Sama" in chunk_content:
                all_entities.append({'text': "Zayar-Sama", 'label': "PERSON"})
            if "TinaAide" in chunk_content:
                all_entities.append({'text': "TinaAide", 'label': "PERSON"})
            if "ShionAide" in chunk_content:
                all_entities.append({'text': "ShionAide", 'label': "PERSON"})

            relevant_entity_types = ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"] # Define based on need
            associated_entities = self.entity_extractor.filter_entities_by_type(all_entities, relevant_entity_types)

            # Context Tag Extraction
            context_tags = self.context_tagger.tag_context(chunk_content)

            metadata = {
                "timestamp": current_timestamp,
                "importance_score": importance_score,
                "source_id": source_id,
                "original_text_start_index": raw_text.find(chunk_content), # Simple approach, can be refined
                "emotional_valence": emotional_valence, # Enhanced metadata
                "vividness_score": self.vividness_calculator.calculate_initial_vividness(chunk_content), # Dynamically calculated (Enhanced metadata)
                "event_sequence_id": event_sequence_id # Enhanced metadata
            }
            if associated_entities:
                metadata["associated_entities"] = associated_entities
                print(f"  [DEBUG][Ingest] Storing associated entities: {associated_entities}")
            if context_tags:
                metadata["context_tags"] = context_tags
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
        initial_vividness = metadata.get('vividness_score', 0.5)
        timestamp = metadata.get('timestamp', time.time()) # Use current time if timestamp is missing
        print(f"  [DEBUG] Vividness - Initial from metadata: {initial_vividness:.4f}, Timestamp from metadata: {datetime.fromtimestamp(timestamp)}")
        return self.vividness_calculator.apply_decay(initial_vividness, timestamp)

    def _calculate_emotional_saliency_score(self, metadata: Dict[str, Any]) -> float:
        # Directly use emotional_valence from metadata. Normalize -1 to 1 to 0 to 1 if needed
        valence = metadata.get('emotional_valence', 0.0)
        return (valence + 1.0) / 2.0 # Normalize -1 to 1 to 0 to 1 range

    def _calculate_associative_strength_score(self, chunk_id: str, query_entities: List[str]) -> float:
        score = 0.0
        
        # Ensure the chunk_id exists in the graph for centrality and path calculations
        if not self.associative_network.graph.has_node(chunk_id):
            print(f"  [DEBUG][AssocStrength] Chunk ID {chunk_id[:8]}... not in associative network. Returning 0.0.")
            return 0.0 # No associative strength if chunk not in network

        print(f"  [DEBUG][AssocStrength] Calculating for chunk {chunk_id[:8]}..., Query entities: {query_entities}")

        # 1. Direct Links / Shared Entities (Existing Logic, refined)
        chunk_metadata = self.memory_store.get_memory_by_id(chunk_id).get('metadata', {})
        chunk_entities = chunk_metadata.get('associated_entities', [])
        shared_entities = set(chunk_entities).intersection(set(query_entities))
        shared_entity_contribution = len(shared_entities) * settings.ASSOCIATIVE_SHARED_ENTITY_WEIGHT
        score += shared_entity_contribution
        print(f"  [DEBUG][AssocStrength]   Shared entities: {shared_entities}, Contribution: {shared_entity_contribution:.4f}")

        # 2. Path-finding Contribution
        path_score = 0.0
        for q_entity in query_entities:
            if self.associative_network.graph.has_node(q_entity):
                try:
                    # Shortest path: shorter paths mean stronger association
                    path_length = nx.shortest_path_length(self.associative_network.graph, source=chunk_id, target=q_entity)
                    # Normalize path length to a score (e.g., inverse of length, clamped)
                    # Example: path_length 1 -> 1.0; 2 -> 0.5; 3 -> 0.33; 4 -> 0.25 (for max_path_length_considered=4)
                    entity_path_contribution = max(0.0, 1.0 - ((path_length - 1) / (settings.ASSOCIATIVE_MAX_PATH_LENGTH_CONSIDERED -1 ))) if settings.ASSOCIATIVE_MAX_PATH_LENGTH_CONSIDERED > 1 else 1.0
                    path_score += entity_path_contribution
                    print(f"  [DEBUG][AssocStrength]     Path from {chunk_id[:8]}... to {q_entity}: Length={path_length}, Individual Path Contribution: {entity_path_contribution:.4f}")
                except nx.NetworkXNoPath:
                    print(f"  [DEBUG][AssocStrength]     No path from {chunk_id[:8]}... to {q_entity}")
                    pass # No path, no score for this entity
        
        path_contribution = path_score * settings.ASSOCIATIVE_PATH_WEIGHT
        score += path_contribution
        print(f"  [DEBUG][AssocStrength]   Total Path Score: {path_score:.4f}, Total Path Contribution: {path_contribution:.4f}")


        # 3. Centrality Bonus (PageRank for global importance)
        centrality_scores = self.associative_network.calculate_node_centrality(centrality_type="pagerank")
        chunk_pagerank = centrality_scores.get(chunk_id, 0.0)
        pagerank_contribution = chunk_pagerank * settings.ASSOCIATIVE_PAGERANK_WEIGHT
        score += pagerank_contribution
        print(f"  [DEBUG][AssocStrength]   Chunk PageRank: {chunk_pagerank:.4f}, PageRank Contribution: {pagerank_contribution:.4f}")

        # Combine direct edge weights as before, but with updated logic if dynamic weights were added
        direct_link_weight_contribution = 0.0
        for entity in query_entities:
            if self.associative_network.graph.has_edge(chunk_id, entity):
                link_weight = self.associative_network.graph[chunk_id][entity].get('weight', 0.0)
                direct_link_weight_contribution += link_weight * settings.ASSOCIATIVE_DIRECT_LINK_WEIGHT
                print(f"  [DEBUG][AssocStrength]   Direct link {chunk_id[:8]}...-{entity} with weight {link_weight:.2f}, Contribution: {link_weight * settings.ASSOCIATIVE_DIRECT_LINK_WEIGHT:.4f}")
        score += direct_link_weight_contribution
        print(f"  [DEBUG][AssocStrength]   Total Direct Link Weight Contribution: {direct_link_weight_contribution:.4f}")

        final_score_unclamped = score
        # Ensure score is clamped between 0 and 1
        score = max(0.0, min(1.0, score))
        print(f"  [DEBUG][AssocStrength]   Final Associative Score (Unclamped): {final_score_unclamped:.4f}, Clamped: {score:.4f}")
        return score

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
                # For working memory items, we simulate some metadata for consistent scoring
                # A proper implementation might store more detailed metadata in WorkingMemory itself
                wm_metadata = {
                    "source_id": "working_memory",
                    "timestamp": item['timestamp'], # Use the actual timestamp from WorkingMemory
                    "importance_score": 1.0, # High importance for active working memory
                    "emotional_valence": self.sentiment_analyzer.analyze_sentiment(item['content']).get('compound', 0.0),
                    "vividness_score": self.vividness_calculator.calculate_initial_vividness(item['content']), # Calculate initial vividness for WM
                    "event_sequence_id": str(uuid.uuid4()) # Assign a unique event ID
                }
                working_memory_results.append({
                    'id': f"wm_{str(uuid.uuid4())}", # Temporary ID for WM items
                    'content': item['content'],
                    'metadata': wm_metadata,
                    'embedding': self.embedding_manager.get_embedding(item['content']), # Embed WM item for scoring
                    'score': 1.0 # High score for working memory match (will be re-scored)
                })
        # For now, append working memory results. Later, we'll integrate scoring better.
        all_retrieved_chunks_pre_scoring = working_memory_results # Start with WM, then add long-term

        # 2. Associative Spreading & Triggering (Long-Term Memory Search - Broad)
        # Perform initial semantic search on long-term store
        raw_retrieved_chunks = self.memory_store.search_memories(query_embedding, n_results=n_results)
        all_retrieved_chunks_pre_scoring.extend(raw_retrieved_chunks)

        # Further activate memories via AssociativeNetwork using query entities
        # Extract entities from the query using the new EntityExtractor
        query_all_entities = self.entity_extractor.extract_entities(query)
        print(f"  [DEBUG][RetrievalManager] Raw query entities from EntityExtractor: {query_all_entities}")
        # Use relevant entity types from settings for filtering query entities
        query_entities = self.entity_extractor.filter_entities_by_type(query_all_entities, settings.RELEVANT_ENTITY_TYPES)
        print(f"  [DEBUG][RetrievalManager] Query entities for associative score: {query_entities}")
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

from typing import List, Dict, Any
import networkx as nx

class AssociativeNetwork:
    def __init__(self):
        """
        Initializes the AssociativeNetwork. This is a conceptual placeholder
        for managing explicit links and relationships between memory chunks.
        Uses NetworkX as a potential underlying graph structure for the prototype.
        """
        self.graph = nx.Graph() # Nodes will be chunk_ids or entities/concepts
        self.entity_to_chunks = {} # Map entities/concepts to relevant chunk_ids
        print("AssociativeNetwork initialized (placeholder). Ready for linking memories.")

    def add_node(self, node_id: str, node_type: str, attributes: Dict[str, Any] = None):
        """
        Adds a node to the associative network. A node can be a memory chunk or an entity.
        """
        if attributes is None:
            attributes = {}
        attributes['node_type'] = node_type
        self.graph.add_node(node_id, **attributes)

    def add_edge(self, node1_id: str, node2_id: str, relation_type: str = "related_to", weight: float = 1.0):
        """
        Adds a weighted edge (relationship) between two nodes.
        """
        self.graph.add_edge(node1_id, node2_id, type=relation_type, weight=weight)

    def link_chunk_to_entities(self, chunk_id: str, entities: List[str], relation_type: str = "mentions"):
        """
        Links a memory chunk to a list of extracted entities.
        """
        if chunk_id not in self.graph:
            self.add_node(chunk_id, "memory_chunk")

        for entity in entities:
            if entity not in self.graph:
                self.add_node(entity, "entity", {'name': entity})
            self.add_edge(chunk_id, entity, relation_type=relation_type, weight=1.0)
            
            if entity not in self.entity_to_chunks:
                self.entity_to_chunks[entity] = set()
            self.entity_to_chunks[entity].add(chunk_id)

    def get_related_nodes(self, node_id: str, depth: int = 1) -> List[str]:
        """
        Finds nodes related to a given node within a specified depth.
        """
        if node_id not in self.graph:
            return []

        related = set()
        # Breadth-first search for neighbors up to the specified depth
        for _ in range(depth):
            current_layer_neighbors = set()
            for n in ([node_id] if _ == 0 else related):
                current_layer_neighbors.update(self.graph.neighbors(n))
            related.update(current_layer_neighbors)
        
        # Remove the original node itself if it's in the related set
        if node_id in related:
            related.remove(node_id)
            
        return list(related)

    def get_chunks_by_entity(self, entity: str) -> List[str]:
        """
        Retrieves a list of chunk IDs associated with a specific entity.
        """
        return list(self.entity_to_chunks.get(entity, set()))

# Example usage (for testing purposes)
if __name__ == "__main__":
    associative_network = AssociativeNetwork()

    # Add a memory chunk and link it to entities
    chunk_id_1 = "chunk_1"
    entities_1 = ["AI Memory System", "ChromaDB", "Zayar-Sama"]
    associative_network.link_chunk_to_entities(chunk_id_1, entities_1)
    print(f"Linked {chunk_id_1} to {entities_1}")

    chunk_id_2 = "chunk_2"
    entities_2 = ["AI Memory System", "TinaAide", "Memory Consolidation"]
    associative_network.link_chunk_to_entities(chunk_id_2, entities_2)
    print(f"Linked {chunk_id_2} to {entities_2}")

    # Test getting related nodes
    print(f"Nodes related to 'AI Memory System' (depth 1): {associative_network.get_related_nodes('AI Memory System', depth=1)}")
    print(f"Nodes related to 'chunk_1' (depth 1): {associative_network.get_related_nodes(chunk_id_1, depth=1)}")

    # Test getting chunks by entity
    print(f"Chunks mentioning 'TinaAide': {associative_network.get_chunks_by_entity('TinaAide')}")

    # Add a direct link between two entities
    associative_network.add_edge("Zayar-Sama", "TinaAide", "collaborates_with", weight=0.8)
    print(f"Added direct link between Zayar-Sama and TinaAide.")
    print(f"Nodes related to 'Zayar-Sama' (depth 1): {associative_network.get_related_nodes('Zayar-Sama', depth=1)}")

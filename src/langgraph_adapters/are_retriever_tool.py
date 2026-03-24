from langchain_core.tools import tool
from src.core.retrieval_manager import RetrievalManager

@tool
def recall_from_associative_memory(query: str) -> str:
    """Use the Associative Recall Engine to fetch emotionally salient, associatively linked memories."""
    rm = RetrievalManager()
    memories = rm.retrieve_relevant_memories(query=query, n_results=10)
    return "\n\n".join([f"- {m['content']}" for m in memories])
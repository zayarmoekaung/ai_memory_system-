from langgraph.graph import StateGraph
from src.core.retrieval_manager import RetrievalManager
from .are_state import AREAgentState

def are_memory_node(state: AREAgentState, retrieval_manager: RetrievalManager) -> AREAgentState:
    """Node that runs full ARE retrieval before the LLM sees the prompt."""
    last_user_message = state["user_input"] if state["user_input"] else ""

    retrieved = retrieval_manager.retrieve_relevant_memories(
        query=last_user_message,
        n_results=10,
        include_working_memory=True
    )

    context = "\n\n".join([m["content"] for m in retrieved])

    return {
        "are_context": f"""--- Relevant memories from Associative Recall Engine ---\n{context if context else "No relevant memories yet."}\n--- End of memories ---"""
    }
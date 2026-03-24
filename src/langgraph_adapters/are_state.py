from typing import Annotated, TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from src.core.retrieval_manager import RetrievalManager

class AREAgentState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    are_context: Annotated[str, lambda x,y: y or x]
    thread_id: str
    
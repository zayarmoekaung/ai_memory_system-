from typing import Annotated, TypedDict, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class AREAgentState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    user_input: str
    are_context: Annotated[str, lambda x,y: y or x]
    thread_id: str
    conversation_topic: Optional[str]
    conversation_heading: Optional[str]
    
import os
import project_path
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode

from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.retrieval_manager import RetrievalManager
from src.langgraph_adapters.are_state import AREAgentState
from src.langgraph_adapters.are_memory_node import are_memory_node
from src.langgraph_adapters.are_retriever_tool import recall_from_associative_memory
from src.langgraph_adapters.post_process_node import post_process_node

load_dotenv()
USE_MESSAGE_HISTORY = True

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=2024,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

rm = RetrievalManager()

def estimate_tokens(text: str) -> int:
    return len(text) // 4


def agent_node(state: AREAgentState):
    messages = state["messages"][-4:] if USE_MESSAGE_HISTORY else []
    user_input = state.get("user_input", "")
    are_context = state.get("are_context", "")
    topic = state.get("conversation_topic","")
    heading = state.get("conversation_heading")
    
    full_messages = [
        ("system", f"""You are a helpful assistant with human-like long-term memory.
        Use the provided memories naturally when relevant.
        Memories are already weighted by recency, emotion, vividness and associations.
        Topic of conversation: {topic}\n
        Conversation Heading: {heading}\n
        {are_context}"""),
        ("human", user_input)
        ]
    
    context_chars = len(are_context)
    messages_chars = sum(len(m.content) for m in messages)
    estimated_tokens = estimate_tokens(are_context + " ".join(m.content for m in messages))

    print("\n Context Stats:")
    print(f"  ARE context chars: {context_chars}")
    print(f"  ARE context : {are_context}")
    print(f"  Messages chars: {messages_chars}")
    print(f"  Estimated input tokens: {estimated_tokens}")

    response = llm.invoke(full_messages)

    #Extract token usage (if available)
    usage = getattr(response, "usage_metadata", None)
    print(response.response_metadata)
    if usage:
        print("\n Token Usage:")
        print(f"  Prompt tokens: {usage.get('input_tokens')}")
        print(f"  Completion tokens: {usage.get('output_tokens')}")
        print(f"  Total tokens: {usage.get('total_tokens')}")
    else:
        print("\nNo token usage metadata returned")

    return {"messages": [AIMessage(content=response.content)]}



# ====================== Build the Graph ======================
workflow = StateGraph(AREAgentState)
workflow.add_node(
    "are_memory",
    lambda state: are_memory_node(state, retrieval_manager=rm)
)
workflow.add_node("agent", agent_node)
workflow.add_node(
    "post_process",
    lambda state: post_process_node(state, llm=llm, retrieval_manager=rm)
)
workflow.add_edge(START, "are_memory")
workflow.add_edge("are_memory", "agent")
workflow.add_edge("agent", "post_process")
workflow.add_edge("post_process", END)

# Persistent checkpointer (short-term conversation memory)
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ====================== Run Interactive Test ======================
print("Associative Recall Engine + Gemini Agent Ready!")
print("Type 'exit' or 'quit' to stop.\n")

thread_config = {"configurable": {"thread_id": "test-shion-001"}}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    # Run the agent
    input_state = {"messages": [HumanMessage(content=user_input)],
                    "user_input": user_input}
    
    response = app.invoke(input_state, config=thread_config)
    
    print(f"Agent: {response['messages'][-1].content}\n")
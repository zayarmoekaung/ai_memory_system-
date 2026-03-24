import json
import re
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage

from src.langgraph_adapters.are_state import AREAgentState
from src.core.retrieval_manager import RetrievalManager


def safe_json_parse(text: str):
    if not text:
        return None

    # Remove markdown code fences
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)  # remove opening ```json
        text = re.sub(r"\n?```$", "", text)          # remove closing ```

    try:
        return json.loads(text)
    except Exception as e:
        print("[DEBUG] JSON parse failed:", e)
        print("[DEBUG] Cleaned text:", text)
        return None


def build_conversation_text(messages: List[BaseMessage]) -> str:
    lines = []
    for m in messages:
        role = "User" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def post_process_node(state: AREAgentState, llm, retrieval_manager: RetrievalManager) -> Dict[str, Any]:
    """
    Post-response cognitive processing:
    - Extract topic + heading
    - Extract memory candidates
    - Ingest into ARE
    """

    messages = state.get("messages", [])[-4:]
    print(f"  [DEBUG][PostProcess] Starting post_process_node with {len(messages)} messages")

    if not messages:
        print("  [DEBUG][PostProcess] No messages found; returning empty state")
        return {}

    convo_text = build_conversation_text(messages)
    print(f"  [DEBUG][PostProcess] Conversation text:\n{convo_text}")

    # =========================
    # 1. Extract topic + heading
    # =========================
    topic_prompt = [
        ("system", """Extract structured info from conversation.

            Return STRICT JSON:
            {
            "topic": "main topic",
            "heading": "short title (max 8 words)"
            }
            """),
                    ("human", convo_text)
    ]

    topic_res = llm.invoke(topic_prompt)
    print("[DEBUG] topic_res:", topic_res)
    print("[DEBUG] topic_res.content:", getattr(topic_res, "content", None))
    topic_data = safe_json_parse(topic_res.content) or {}

    topic = topic_data.get("topic", "")
    heading = topic_data.get("heading", "")

    # =========================
    # 2. Extract memory candidates
    # =========================
    memory_prompt = [
        ("system", """Extract important long-term memory candidates.

        Return STRICT JSON array with MAX 3 items.
        Each "content" MUST be under 200 characters.:
        [
        {
            "content": "memory text",
            "importance": 0.0-1.0
        }
        ]

        Rules:
        - Only meaningful long-term info
        - No duplicates
        - No trivial chat
        """),
        ("human", convo_text)
    ]

    mem_res = llm.invoke(memory_prompt)
    print("[DEBUG] mem_res:", mem_res)
    print("[DEBUG] mem_res.content:", getattr(mem_res, "content", None))
    memories = safe_json_parse(mem_res.content) or []

    # =========================
    # 3. Dedup + ingest
    # =========================
    seen = set()

    for m in memories:
        content = m.get("content", "").strip()

        if not content or len(content) < 20:
            continue

        if content in seen:
            continue

        seen.add(content)

        retrieval_manager.ingest_memory(
            raw_text=content,
            importance_score=m.get("importance", 0.5),
            source_id="conversation_digest"
        )

    # =========================
    # 4. Return updated state
    # =========================
    print(f"  [DEBUG][PostProcess] Return Values:\nconversation_topic:{topic}\nconversation_heading: {heading}")

    return {
        "conversation_topic": topic,
        "conversation_heading": heading
    }
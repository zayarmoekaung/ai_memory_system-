from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ..core.retrieval_manager import RetrievalManager

# Initialize FastAPI app
app = FastAPI(
    title="AI Memory System API",
    description="API for ingesting and retrieving memories for AI agents.",
    version="0.1.0",
)

# Initialize RetrievalManager
retrieval_manager = RetrievalManager()

# Pydantic models for request and response validation
class IngestMemoryRequest(BaseModel):
    raw_text: str
    importance_score: float = 0.5
    source_id: str = "api_ingestion"

class RetrieveMemoriesRequest(BaseModel):
    query: str
    n_results: int = 10

class MemoryChunkResponse(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    similarity: float
    recency: float
    importance: float
    task_relatedness: float

@app.post("/ingest_memory", summary="Ingest a new memory into the system")
async def ingest_memory_endpoint(request: IngestMemoryRequest):
    """
    Ingest a new piece of raw text memory into the AI Memory System.
    The text will be chunked, embedded, and stored with provided metadata.
    """
    try:
        retrieval_manager.ingest_memory(
            raw_text=request.raw_text,
            importance_score=request.importance_score,
            source_id=request.source_id
        )
        return {"message": "Memory ingested successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest memory: {str(e)}")

@app.post("/retrieve_memories", response_model=List[MemoryChunkResponse], summary="Retrieve relevant memories based on a query")
async def retrieve_memories_endpoint(request: RetrieveMemoriesRequest):
    """
    Retrieve a list of memory chunks most relevant to the given query.
    Memories are scored based on similarity, recency, and importance, then optimized for token usage.
    """
    try:
        relevant_memories = retrieval_manager.retrieve_relevant_memories(
            query=request.query,
            n_results=request.n_results
        )
        return relevant_memories
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {str(e)}")

# To run this API, you would typically use:
# uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

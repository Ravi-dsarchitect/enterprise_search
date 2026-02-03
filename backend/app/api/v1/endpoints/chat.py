from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
from app.services.rag.service import RAGService

router = APIRouter()

class Message(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Message]] = Field(default=None, description="Previous conversation messages for context")
    use_hyde: bool = Field(default=False, description="Enable HyDE (hypothetical document embeddings)")
    use_decomposition: bool = Field(default=False, description="Enable query decomposition")
    use_hybrid_search: bool = Field(default=False, description="Enable hybrid search (BM25 + Vector)")
    use_auto_filters: bool = Field(default=True, description="Auto-extract metadata filters from query")
    metadata_filters: Optional[Dict[str, Any]] = Field(default=None, description="Manual filter override (e.g., {'source': 'doc.pdf'})")
    limit: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    
class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]

@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """Non-streaming query endpoint (returns complete response)."""
    service = RAGService()
    try:
        # Convert Pydantic models to dicts for the service layer
        history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history] if request.conversation_history else None
        
        result = await service.answer_query(
            query=request.query,
            conversation_history=history,
            use_hyde=request.use_hyde,
            use_decomposition=request.use_decomposition,
            use_hybrid_search=request.use_hybrid_search,
            use_auto_filters=request.use_auto_filters,
            metadata_filters=request.metadata_filters,
            limit=request.limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/stream")
async def chat_query_stream(request: ChatRequest):
    """
    Streaming query endpoint using Server-Sent Events (SSE).
    
    Event types:
    - metadata: Query info and retrieval timing
    - citations: Retrieved document citations
    - token: Individual LLM response tokens
    - done: Completion signal with full answer
    """
    service = RAGService()
    
    async def event_generator():
        try:
            # Convert Pydantic models to dicts for the service layer
            history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history] if request.conversation_history else None
            
            async for event in service.answer_query_stream(
                query=request.query,
                conversation_history=history,
                use_hyde=request.use_hyde,
                use_decomposition=request.use_decomposition,
                use_hybrid_search=request.use_hybrid_search,
                use_auto_filters=request.use_auto_filters,
                metadata_filters=request.metadata_filters,
                limit=request.limit
            ):
                # Format as Server-Sent Event
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_event = {
                "type": "error",
                "data": {"message": str(e)}
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


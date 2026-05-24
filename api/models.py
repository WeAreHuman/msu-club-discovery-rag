"""
Pydantic request/response models for the MSU Club Discovery API.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ConversationMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=500)
    conversation_history: List[ConversationMessage] = []
    top_k: Optional[int] = None
    vibe: str = "scholar"
    org_name: Optional[str] = None
    chunk_type: Optional[str] = None
    apply_filters: bool = True
    return_citations: bool = True


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=500)
    top_k: Optional[int] = None
    vibe: str = "scholar"
    apply_filters: bool = True
    return_citations: bool = True


class Citation(BaseModel):
    source_number: int
    org_name: str
    chunk_type: str
    relevance_score: float
    text_snippet: str
    metadata: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    filters_applied: Dict[str, Any] = {}
    num_chunks: int = 0


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    index_name: str

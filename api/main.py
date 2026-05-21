"""
MSU Club Discovery — FastAPI headless backend.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).parent.parent))

import config
from api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    QueryRequest,
)
from src.rag_engine import RAGEngine

_rag_engine: RAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_engine
    config.validate_config()
    _rag_engine = RAGEngine()
    yield
    _rag_engine = None


app = FastAPI(
    title="MSU Club Discovery API",
    description="Headless RAG API for MSU student club discovery",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_engine() -> RAGEngine:
    if _rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return _rag_engine


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        llm_provider=config.LLM_PROVIDER,
        llm_model=config.LLM_MODEL,
        index_name=config.PINECONE_INDEX_NAME,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    engine = _get_engine()
    history = [{"role": m.role, "content": m.content} for m in req.conversation_history]

    result = engine.chat(
        question=req.question,
        conversation_history=history,
        top_k=req.top_k,
        apply_filters=req.apply_filters,
        return_citations=req.return_citations,
        vibe=req.vibe,
        org_name=req.org_name,
        chunk_type=req.chunk_type,
    )

    return ChatResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        filters_applied=result.get("filters_applied", {}),
        num_chunks=len(result.get("retrieved_chunks", [])),
    )


@app.post("/query", response_model=ChatResponse)
def query(req: QueryRequest):
    engine = _get_engine()

    result = engine.query(
        question=req.question,
        top_k=req.top_k,
        apply_filters=req.apply_filters,
        return_citations=req.return_citations,
        vibe=req.vibe,
    )

    return ChatResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        filters_applied=result.get("filters_applied", {}),
        num_chunks=len(result.get("retrieved_chunks", [])),
    )

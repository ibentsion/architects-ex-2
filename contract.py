"""
Exercise 2 API contract -- your system MUST expose exactly this interface.

The blind evaluation calls POST /ask on your endpoint with an AskRequest and
expects an AskResponse. Fields you don't fill (e.g. cost_usd) simply score
worse on the efficiency component; fields with wrong types fail validation.

Run this stub as-is to see the contract in action:

    uvicorn contract:app --port 8000
    curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
         -d '{"question": "האם הביטוח מכסה נזק מפגיעת ברק?"}'

Replace `answer_question` with your actual system. Do not change the models.
"""
import os
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.cli.query import build_answer_engine
from rag.config import ConfigError, load_config
from rag.index.manifest import ManifestError, ManifestMismatchError
from rag.types import Answer


class AskRequest(BaseModel):
    question: str = Field(..., description="Customer question, usually Hebrew")
    session_id: Optional[str] = Field(None, description="For multi-turn context (optional)")


class Citation(BaseModel):
    file: str = Field(..., description="Source document path or URL")
    page: Optional[int] = Field(None, description="1-based page number for PDFs")
    quote: Optional[str] = Field(None, description="The supporting passage (optional but persuasive)")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The answer, in the language of the question")
    citations: List[Citation] = Field(default_factory=list)
    domain: Optional[str] = Field(None, description="Routed insurance domain, e.g. 'travel'")
    confidence: Optional[float] = Field(None, ge=0, le=1)
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = Field(None, description="Estimated $ cost of answering this question")


DEFAULT_RAG_CONFIG = "configs/ship.yaml"

_engine_lock = threading.Lock()


def _rag_config_path() -> str:
    return os.environ.get("RAG_CONFIG", DEFAULT_RAG_CONFIG)


@lru_cache(maxsize=1)
def _get_engine():
    config_path = _rag_config_path()
    config = load_config(config_path)
    return build_answer_engine(config, "agent")


def answer_question(question: str) -> AskResponse:
    """Run the production agentic RAG pipeline behind the FastAPI contract."""
    t0 = time.monotonic()
    try:
        # The submit runner is serial, but a shared retriever/generator instance
        # has mutable diagnostics state. Serialize endpoint calls for correctness.
        with _engine_lock:
            answer = _get_engine().answer(question)
    except (ConfigError, ManifestError, ManifestMismatchError) as exc:
        raise HTTPException(status_code=500, detail=f"RAG configuration error: {exc}") from exc

    latency_ms = answer.latency_ms
    if latency_ms is None:
        latency_ms = (time.monotonic() - t0) * 1000
    return _to_ask_response(answer, latency_ms=latency_ms)


def _to_ask_response(answer: Answer, *, latency_ms: float | None = None) -> AskResponse:
    return AskResponse(
        answer=answer.text,
        citations=[
            Citation(file=c.file, page=c.page, quote=c.quote)
            for c in answer.citations
        ],
        domain=answer.category,
        confidence=answer.confidence,
        latency_ms=latency_ms,
        cost_usd=answer.cost_estimate,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        if _get_engine.cache_info().currsize:
            _get_engine().close()
            _get_engine.cache_clear()


app = FastAPI(title="APEX Exercise 2 -- Harel Support Agent", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return answer_question(req.question)


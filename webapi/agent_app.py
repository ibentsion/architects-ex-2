"""The app that runs ON the GPU node for the web-UI demo, instead of
``contract:app``.

Same engine, same config — the only difference is that this one streams the
pipeline's own trace records as they happen (Server-Sent Events) so the
browser can show classification tags, sub-questions and per-sub-question
retrieval counts while retrieval and synthesis are still running.

One streaming contract. There is deliberately no non-streaming ``/ask`` here:
``contract.py`` is that endpoint, and it is the graded one.

    RAG_CONFIG=configs/ship.yaml python -m uvicorn webapi.agent_app:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncIterator

import anyio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.cli.query import build_answer_engine
from rag.config import load_config
from webapi.schema import answer_to_pair

logger = logging.getLogger(__name__)

DEFAULT_RAG_CONFIG = "configs/ship.yaml"

#: Comment frames keep an idle SSH tunnel from dropping a silent stream —
#: retrieval plus DeepSeek synthesis regularly runs past 60 s with nothing to
#: report in between.
HEARTBEAT_SECONDS = 15.0

# The retriever and generator carry mutable diagnostics state, so two
# concurrent queries would corrupt each other's stats. One query at a time —
# the same reason contract.py serializes.
_engine_lock = threading.Lock()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Customer question, usually Hebrew")


def _rag_config_path() -> str:
    return os.environ.get("RAG_CONFIG", DEFAULT_RAG_CONFIG)


@lru_cache(maxsize=1)
def _get_engine():
    return build_answer_engine(load_config(_rag_config_path()), "agent")


def _frame(event: str, payload: Any) -> str:
    """One SSE frame. ``ensure_ascii=False`` because everything here is
    Hebrew and escaping it triples the byte count for no benefit."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream(question: str) -> AsyncIterator[str]:
    """Drain the engine's trace records to the client as they are produced,
    then the finished answer.

    The engine call is blocking, so it runs in a worker thread and pushes each
    record onto this loop's queue; the generator below is what actually writes
    to the socket.
    """
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def sink(record: dict[str, Any]) -> None:
        # Called from the worker thread. Raising here is harmless — the
        # engine's _emit swallows sink failures (e.g. after a disconnect).
        loop.call_soon_threadsafe(events.put_nowait, record)

    def run_engine():
        with _engine_lock:
            return _get_engine().answer(question, event_sink=sink)

    task = asyncio.ensure_future(anyio.to_thread.run_sync(run_engine))

    # Race the next record against the engine finishing, so the stream is never
    # left waiting out a heartbeat after the last step.
    pending: asyncio.Future | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(events.get())
            done, _ = await asyncio.wait(
                {pending, task},
                timeout=HEARTBEAT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pending in done:  # records first, so none is dropped on the way out
                yield _frame("step", pending.result())
                pending = None
            elif task in done:
                break
            else:
                yield ": ping\n\n"
    finally:
        if pending is not None:
            pending.cancel()

    while not events.empty():  # queued in the instant before the thread returned
        yield _frame("step", events.get_nowait())

    try:
        answer = task.result()
    except Exception as exc:  # config error, index mismatch, LLM outage...
        logger.exception("query failed")
        yield _frame("error", {"message": f"{type(exc).__name__}: {exc}"})
        return

    pair = answer_to_pair(answer, pair_id=f"live-{uuid.uuid4().hex[:12]}", question=question)
    yield _frame("answer", pair.model_dump(mode="json"))
    yield _frame("done", {})


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        if _get_engine.cache_info().currsize:
            _get_engine().close()
            _get_engine.cache_clear()


app = FastAPI(title="APEX Exercise 2 -- agent stream (web UI bonus)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

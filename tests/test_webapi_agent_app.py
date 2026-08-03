"""The GPU-node SSE app, exercised against a fake engine.

No GPU and no Token Factory here: the app's job is framing and ordering, and
that is what these assert.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag.types import Answer, Citation
from webapi import agent_app


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """(event, data) per frame; ``:`` comment frames are heartbeats, not events."""
    frames = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        frames.append((event, data))
    return frames


class FakeEngine:
    """Emits trace records exactly the way AgentEngine does, then answers."""

    def __init__(self, delay: float = 0.0, fail: Exception | None = None):
        self.delay = delay
        self.fail = fail
        self.questions: list[str] = []
        self.closed = False

    def close(self):
        self.closed = True

    def answer(self, question, category=None, *, event_sink=None):
        self.questions.append(question)
        if self.fail is not None:
            raise self.fail
        for record in ({"step": "hint", "top_category": "apartment"},
                       {"step": "classify", "categories": ["apartment"]},
                       {"step": "retrieve", "n_gated": 4},
                       {"step": "synthesize", "model": "fake"}):
            event_sink(record)
        time.sleep(self.delay)
        return Answer(
            text="תשובה סופית",
            citations=[Citation(file="apartment/files/a.pdf", page=2, quote="ציטוט")],
            category="apartment",
            confidence=0.8,
            latency_ms=120.0,
            cost_estimate=0.003,
            trace=[{"step": "hint"}],
        )


def client_with(engine, monkeypatch) -> TestClient:
    """Patch what the engine cache BUILDS, not the cache itself — the lifespan
    handler needs the real lru_cache to release the engine on shutdown."""
    monkeypatch.setattr(agent_app, "load_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(agent_app, "build_answer_engine", lambda config, kind: engine)
    agent_app._get_engine.cache_clear()
    return TestClient(agent_app.app)


def test_health_is_what_the_node_health_poll_reads(monkeypatch):
    with client_with(FakeEngine(), monkeypatch) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_the_engine_is_built_once_and_released_on_shutdown(monkeypatch):
    engine = FakeEngine()
    with client_with(engine, monkeypatch) as client:
        client.post("/query", json={"question": "אחת"})
        client.post("/query", json={"question": "שתיים"})
        assert agent_app._get_engine.cache_info().currsize == 1
        assert engine.closed is False
    assert engine.closed is True
    assert agent_app._get_engine.cache_info().currsize == 0


def test_query_streams_steps_then_the_answer(monkeypatch):
    engine = FakeEngine()
    with client_with(engine, monkeypatch) as client:
        response = client.post("/query", json={"question": "שאלה"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert engine.questions == ["שאלה"]

    frames = parse_sse(response.text)
    assert [event for event, _ in frames] == [
        "step", "step", "step", "step", "answer", "done"
    ]
    assert [data["step"] for _e, data in frames[:4]] == [
        "hint", "classify", "retrieve", "synthesize"
    ]
    pair = frames[4][1]
    assert pair["answer"] == "תשובה סופית"
    assert pair["domain"] == "apartment"
    assert pair["citations"][0]["page_number"] == 2
    assert pair["citations"][0]["thumbnail_url"].startswith("/api/citation/thumbnail?")


def test_hebrew_is_streamed_unescaped(monkeypatch):
    with client_with(FakeEngine(), monkeypatch) as client:
        response = client.post("/query", json={"question": "שאלה"})
    assert "תשובה סופית" in response.text  # not תש...


def test_engine_failure_becomes_a_terminal_error_frame(monkeypatch):
    engine = FakeEngine(fail=RuntimeError("index missing"))
    with client_with(engine, monkeypatch) as client:
        response = client.post("/query", json={"question": "שאלה"})

    # The response is already committed to 200 by the time the engine runs, so
    # the browser learns why from the stream, not from a status code.
    assert response.status_code == 200
    frames = parse_sse(response.text)
    assert frames == [("error", {"message": "RuntimeError: index missing"})]


def test_a_slow_query_gets_heartbeats_instead_of_silence(monkeypatch):
    monkeypatch.setattr(agent_app, "HEARTBEAT_SECONDS", 0.02)
    with client_with(FakeEngine(delay=0.3), monkeypatch) as client:
        response = client.post("/query", json={"question": "שאלה"})

    assert ": ping" in response.text
    assert [event for event, _ in parse_sse(response.text)][-1] == "done"


def test_an_empty_question_is_rejected_before_the_engine(monkeypatch):
    engine = FakeEngine()
    with client_with(engine, monkeypatch) as client:
        assert client.post("/query", json={"question": ""}).status_code == 422
    assert engine.questions == []


def test_there_is_no_non_streaming_fallback(monkeypatch):
    with client_with(FakeEngine(), monkeypatch) as client:
        assert client.post("/ask", json={"question": "שאלה"}).status_code == 404

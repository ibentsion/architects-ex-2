"""The browser-facing bridge: SSE relay, dataset endpoints, pluggable STT.

The agent app is never actually contacted here — an httpx.MockTransport stands
in for the GPU node, which is what makes the relay's framing and its two
failure modes (dead before the stream, dying mid-stream) testable at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from webapi import bridge_app, paths, stt

AGENT_SSE = (
    b'event: step\ndata: {"step": "classify"}\n\n'
    b'event: step\ndata: {"step": "retrieve"}\n\n'
    b'event: answer\ndata: {"id": "live-1", "answer": "\xd7\xaa\xd7\xa9\xd7\x95\xd7\x91\xd7\x94"}\n\n'
    b"event: done\ndata: {}\n\n"
)


def mock_agent(handler) -> None:
    """Point the bridge's httpx client at a fake node."""
    def factory():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=httpx.Timeout(None, connect=10.0),
        )
    return factory


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AGENT_BASE_URL", "http://node:8000")
    return TestClient(bridge_app.app)


# --------------------------------------------------------------------------- #
# SSE relay
# --------------------------------------------------------------------------- #


def test_relays_the_agent_sse_bytes_through_unchanged(client, monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=AGENT_SSE,
                              headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(bridge_app, "_client", mock_agent(handler))
    response = client.post("/api/query", json={"question": "שאלה"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert seen["url"] == "http://node:8000/query"
    assert seen["body"] == {"question": "שאלה"}
    # Byte-identical: the bridge does not parse or re-frame SSE.
    assert response.content == AGENT_SSE


def test_agent_base_url_is_read_per_request_not_at_import(client, monkeypatch):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, content=b"event: done\ndata: {}\n\n")

    monkeypatch.setattr(bridge_app, "_client", mock_agent(handler))
    client.post("/api/query", json={"question": "א"})
    monkeypatch.setenv("AGENT_BASE_URL", "https://public.example/agent")
    client.post("/api/query", json={"question": "ב"})

    assert urls == ["http://node:8000/query", "https://public.example/agent/query"]


def test_an_unreachable_agent_is_a_502_before_the_stream_starts(client, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(bridge_app, "_client", mock_agent(handler))
    response = client.post("/api/query", json={"question": "שאלה"})

    assert response.status_code == 502
    assert "http://node:8000" in response.json()["detail"]


def test_an_agent_error_response_is_a_502_with_its_detail(client, monkeypatch):
    def handler(request):
        return httpx.Response(500, json={"detail": "RAG configuration error"})

    monkeypatch.setattr(bridge_app, "_client", mock_agent(handler))
    response = client.post("/api/query", json={"question": "שאלה"})

    assert response.status_code == 502
    assert "RAG configuration error" in response.json()["detail"]


def test_a_mid_stream_failure_becomes_a_terminal_error_frame(client, monkeypatch):
    async def dying_stream():
        yield b'event: step\ndata: {"step": "classify"}\n\n'
        raise httpx.ReadError("tunnel dropped")

    def handler(request):
        return httpx.Response(200, content=dying_stream())

    monkeypatch.setattr(bridge_app, "_client", mock_agent(handler))
    response = client.post("/api/query", json={"question": "שאלה"})

    # Already committed to 200 — the browser learns about it from the stream.
    assert response.status_code == 200
    assert response.text.startswith('event: step\ndata: {"step": "classify"}')
    assert "event: error" in response.text
    tail = response.text.split("event: error\ndata: ")[1].strip()
    assert "ReadError" in json.loads(tail)["message"]


def test_an_empty_question_never_reaches_the_agent(client, monkeypatch):
    def handler(request):
        raise AssertionError("agent must not be called")

    monkeypatch.setattr(bridge_app, "_client", mock_agent(handler))
    assert client.post("/api/query", json={"question": "   "}).status_code == 400


# --------------------------------------------------------------------------- #
# Voice: pluggable, off by default, never a fake transcript
# --------------------------------------------------------------------------- #


def test_audio_with_no_stt_backend_is_a_501_with_the_remediation_text(client, monkeypatch):
    monkeypatch.delenv("STT_MODEL", raising=False)

    def handler(request):
        raise AssertionError("agent must not be called")

    monkeypatch.setattr(bridge_app, "_client", mock_agent(handler))
    response = client.post("/api/query", files={"audio": ("clip.webm", b"\x00\x01", "audio/webm")})

    assert response.status_code == 501
    detail = response.json()["detail"]
    assert "STT backend not configured" in detail
    assert "faster-whisper" in detail and "STT_MODEL" in detail


def test_stt_is_off_when_the_package_is_missing_even_with_a_model_set(monkeypatch):
    monkeypatch.setenv("STT_MODEL", "base")
    monkeypatch.setattr(stt, "_backend_installed", lambda: False)
    with pytest.raises(stt.SttNotConfigured):
        stt.transcribe(b"\x00", "audio/webm")


def test_stt_is_off_when_no_model_is_configured(monkeypatch):
    monkeypatch.delenv("STT_MODEL", raising=False)
    monkeypatch.setattr(stt, "_backend_installed", lambda: True)
    with pytest.raises(stt.SttNotConfigured):
        stt.transcribe(b"\x00", "audio/webm")


def test_a_configured_backend_transcribes_hebrew_and_the_bridge_relays_it(
    client, monkeypatch
):
    monkeypatch.setenv("STT_MODEL", "base")
    monkeypatch.setattr(stt, "_backend_installed", lambda: True)
    seen = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            seen["path"] = path
            seen["kwargs"] = kwargs
            seen["bytes"] = Path(path).read_bytes()
            segments = [type("S", (), {"text": " מה תקופת"})(), type("S", (), {"text": " ההתיישנות?"})()]
            return segments, {"language": "he"}

    monkeypatch.setattr(stt, "_load_model", lambda: FakeModel())

    def handler(request):
        seen["question"] = json.loads(request.content)["question"]
        return httpx.Response(200, content=b"event: done\ndata: {}\n\n")

    monkeypatch.setattr(bridge_app, "_client", mock_agent(handler))
    response = client.post(
        "/api/query", files={"audio": ("clip.webm", b"AUDIOBYTES", "audio/webm")}
    )

    assert response.status_code == 200
    assert seen["bytes"] == b"AUDIOBYTES"
    assert seen["kwargs"]["language"] == "he"
    assert seen["question"] == "מה תקופת ההתיישנות?"
    # The UI shows what was heard before the answer starts arriving.
    assert response.text.startswith('event: transcript\ndata: ')
    first = json.loads(response.text.split("\n")[1][len("data: "):])
    assert first == {"text": "מה תקופת ההתיישנות?"}
    assert not Path(seen["path"]).exists()  # temp file cleaned up


def test_multipart_without_an_audio_part_is_a_400(client):
    assert client.post("/api/query", data={"note": "no audio"}).status_code == 400


# --------------------------------------------------------------------------- #
# Offline datasets
# --------------------------------------------------------------------------- #


@pytest.fixture
def offline_repo(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(paths, "REPO_ROOT", root)
    (root / ".env").write_text("NEBIUS_API_KEY=secret", encoding="utf-8")
    (root / "reference_questions.json").write_text(
        json.dumps([{"id": "q1", "question": "שאלה", "ground_truth_answer": "אמת"}]),
        encoding="utf-8",
    )
    (root / "rag_answers_full.jsonl").write_text(
        json.dumps({"id": "q1", "answer": "תשובה",
                    "citations": [{"file": "apartment/files/a.pdf", "page": 2}]},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return root


def test_datasets_lists_what_discovery_found(client, offline_repo):
    body = client.get("/api/datasets").json()
    assert [d["id"] for d in body] == ["rag_answers_full.jsonl"]
    assert body[0]["kind"] == "answers" and body[0]["n_pairs"] == 1


def test_offline_pairs_returns_the_joined_pair(client, offline_repo):
    body = client.get("/api/offline-pairs", params={"dataset": "rag_answers_full.jsonl"}).json()
    assert body["dataset"] == "rag_answers_full.jsonl"
    assert body["total"] == 1
    pair = body["pairs"][0]
    assert (pair["question"], pair["answer"], pair["reference_answer"]) == (
        "שאלה", "תשובה", "אמת")
    assert pair["citations"][0]["thumbnail_url"].endswith("&page=2")


def test_offline_pairs_refuses_an_unknown_dataset(client, offline_repo):
    response = client.get("/api/offline-pairs", params={"dataset": ".env"})
    assert response.status_code == 400
    assert "secret" not in response.text


def test_offline_pairs_refuses_traversal(client, offline_repo):
    for attack in ("../../etc/passwd", "/etc/passwd"):
        response = client.get("/api/offline-pairs", params={"dataset": attack})
        assert response.status_code == 400, attack


def test_offline_pairs_paginates(client, offline_repo):
    (offline_repo / "rag_answers_full.jsonl").write_text(
        "".join(json.dumps({"id": f"q{i}", "answer": "a"}) + "\n" for i in range(10)),
        encoding="utf-8",
    )
    body = client.get("/api/offline-pairs", params={
        "dataset": "rag_answers_full.jsonl", "limit": 3, "offset": 5}).json()
    assert body["total"] == 10
    assert [p["id"] for p in body["pairs"]] == ["q5", "q6", "q7"]


def test_the_bridge_adds_no_permissive_cors(client, offline_repo):
    """The Vite dev server proxies /api, so the browser is always same-origin;
    an Access-Control-Allow-Origin: * here would be a hole with no purpose."""
    response = client.get("/api/datasets", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}

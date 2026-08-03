"""The browser-facing bridge — the only origin the web UI ever talks to.

It runs on the laptop and does three things the GPU node cannot:
  * relays ``/api/query`` to the agent app at ``AGENT_BASE_URL`` (an SSH tunnel
    by default), transcribing an audio upload first when voice input falls back
    to the backend;
  * serves the QA-History view over the repo's own answer/judgment files;
  * serves citation previews and page thumbnails from the local ``corpus/``.

No CORS middleware: the Vite dev server proxies ``/api``, so the browser is
always same-origin and a permissive header here would be a hole with nothing
to open.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from webapi import corpus_view, stt
from webapi.corpus_view import CorpusUnavailable
from webapi.datasets import DatasetInfo, UnknownDataset, discover_datasets, load_pairs
from webapi.paths import PathEscape
from webapi.stt import SttNotConfigured

logger = logging.getLogger(__name__)

DEFAULT_AGENT_BASE_URL = "http://localhost:8000"

app = FastAPI(title="APEX Exercise 2 -- support UI bridge")


def _agent_base_url() -> str:
    """Read per request, not at import: pointing the UI at a public node URL
    is an env change, not a code change."""
    return os.environ.get("AGENT_BASE_URL", DEFAULT_AGENT_BASE_URL).rstrip("/")


def _agent_headers() -> dict[str, str]:
    """Bearer token for the agent, when it is behind one.

    A Nebius `ai endpoint` created with `--auth token` is reachable at a managed
    public https:// URL, so it must not be left open — the agent spends the
    shared Token Factory key on every question. Unset for a localhost agent or
    an SSH tunnel, where there is nothing to authenticate against.
    """
    token = os.environ.get("AGENT_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _client() -> httpx.AsyncClient:
    # No read timeout: a streamed answer can be silent for a minute between
    # frames. The connect timeout is what catches a node that isn't there.
    return httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))


def _frame(event: str, payload: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


# --------------------------------------------------------------------------- #
# Live query relay
# --------------------------------------------------------------------------- #


async def _question_from(request: Request) -> tuple[str, str | None]:
    """(question, transcript). ``transcript`` is set only when the question
    came from audio, so the UI can show what was heard."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("audio")
        if upload is None or isinstance(upload, str):
            raise HTTPException(400, detail="multipart request without an `audio` file part")
        try:
            transcript = stt.transcribe(await upload.read(), upload.content_type or "")
        except SttNotConfigured as exc:
            # 501, not 500: the server understood, it just has no such feature.
            raise HTTPException(501, detail=str(exc)) from exc
        return transcript.strip(), transcript.strip()

    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(400, detail=f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, detail="body must be a JSON object")
    return str(body.get("question") or "").strip(), None


@app.post("/api/query")
async def query(request: Request) -> StreamingResponse:
    question, transcript = await _question_from(request)
    if not question:
        raise HTTPException(400, detail="empty question")

    base = _agent_base_url()
    client = _client()
    stream = client.stream(
        "POST", f"{base}/query", json={"question": question}, headers=_agent_headers()
    )

    # Enter the upstream stream BEFORE returning a response, so an agent that
    # is not there is an honest 502 rather than a 200 that immediately errors.
    try:
        upstream = await stream.__aenter__()
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(502, detail=f"agent unreachable at {base}: {exc}") from exc

    if upstream.status_code != 200:
        detail = (await upstream.aread()).decode("utf-8", "replace")[:500]
        await stream.__aexit__(None, None, None)
        await client.aclose()
        raise HTTPException(502, detail=f"agent at {base} returned {upstream.status_code}: {detail}")

    async def body() -> AsyncIterator[bytes]:
        try:
            if transcript is not None:
                yield _frame("transcript", {"text": transcript})
            # Straight through: SSE is framed once, in the agent app.
            async for chunk in upstream.aiter_bytes():
                yield chunk
        except Exception as exc:
            logger.warning("relay failed mid-stream (%s: %s)", type(exc).__name__, exc)
            yield _frame("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            await stream.__aexit__(None, None, None)
            await client.aclose()

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# QA history (read-only)
# --------------------------------------------------------------------------- #


@app.get("/api/datasets")
def datasets() -> list[DatasetInfo]:
    return discover_datasets()


@app.get("/api/offline-pairs")
def offline_pairs(
    dataset: str = Query(..., description="Dataset id from /api/datasets"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    try:
        total, pairs = load_pairs(dataset, limit=limit, offset=offset)
    except (PathEscape, UnknownDataset) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    return {"dataset": dataset, "total": total, "pairs": pairs}


# --------------------------------------------------------------------------- #
# Citations (local corpus/, which is gitignored and often absent)
# --------------------------------------------------------------------------- #


@app.get("/api/citation/thumbnail")
def citation_thumbnail(
    file: str = Query(..., description="Category-relative corpus path"),
    page: int = Query(..., description="1-based page number"),
) -> FileResponse:
    try:
        rendered = corpus_view.page_thumbnail(file, page)
    except PathEscape as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except CorpusUnavailable as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    return FileResponse(
        rendered,
        media_type="image/jpeg",
        # Immutable once rendered: the corpus is a fixed snapshot.
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/citation/content")
def citation_content(
    file: str = Query(..., description="Category-relative corpus path"),
    page: int | None = Query(None, description="1-based page number, PDFs only"),
) -> dict[str, Any]:
    is_pdf = file.lower().endswith(".pdf")
    try:
        # Nothing is re-parsed on a UI request: a PDF page's text comes from
        # the Docling parse cache the index was built from (and is None when
        # that cache has no entry for the document).
        if is_pdf:
            corpus_view.page_thumbnail(file, page or 1)  # 404s if it isn't there
            text, url = corpus_view.page_preview(file, page or 1), corpus_view.source_url(file)
        else:
            text, url = corpus_view.page_text(file)
    except PathEscape as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except CorpusUnavailable as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    return {
        "kind": "pdf" if is_pdf else "text",
        "text": text,
        "source_url": url,
        "file_name": file,
        "page_number": page,
    }

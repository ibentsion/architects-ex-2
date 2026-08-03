"""Citation previews and PDF page thumbnails.

``corpus/`` is gitignored and is often simply not on the machine running the
UI, so "no preview available" is a first-class outcome here, not an error path
that happens to work.
"""
from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from webapi import bridge_app, corpus_view, paths, schema
from webapi.corpus_view import CorpusUnavailable

ANCHOR_PDF = "apartment/files/הודעה-על-תקופת-התיישנות.pdf"
ANCHOR_TXT = "apartment/pages/cancellation.txt"


@pytest.fixture
def corpus_repo(tmp_path, monkeypatch, mini_corpus_dir) -> Path:
    """A repo root whose corpus/ is the real mini-corpus fixture (a real PDF
    and real Hebrew TXTs). Copied, not symlinked — a symlink would resolve out
    of the fake root and be rejected, correctly."""
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copytree(mini_corpus_dir, root / "corpus")
    monkeypatch.setattr(paths, "REPO_ROOT", root)
    corpus_view._manifest.cache_clear()
    return root


@pytest.fixture
def empty_repo(tmp_path, monkeypatch) -> Path:
    """A checkout with no corpus/ at all — the normal laptop case."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(paths, "REPO_ROOT", root)
    corpus_view._manifest.cache_clear()
    return root


@pytest.fixture
def client() -> TestClient:
    return TestClient(bridge_app.app)


# --------------------------------------------------------------------------- #
# Thumbnails
# --------------------------------------------------------------------------- #


def test_pdf_page_renders_once_and_is_served_from_cache(corpus_repo, client):
    params = {"file": ANCHOR_PDF, "page": 1}

    first = client.get("/api/citation/thumbnail", params=params)
    assert first.status_code == 200
    assert first.headers["content-type"] == "image/jpeg"
    assert "max-age=86400" in first.headers["cache-control"]

    image = Image.open(io.BytesIO(first.content))
    assert image.format == "JPEG"
    assert 0 < image.width <= corpus_view.MAX_THUMBNAIL_WIDTH

    cached = list((corpus_repo / "cache" / "webui_thumbs").glob("*.jpg"))
    assert len(cached) == 1
    stamp = cached[0].stat().st_mtime_ns

    second = client.get("/api/citation/thumbnail", params=params)
    assert second.status_code == 200
    assert second.content == first.content
    assert cached[0].stat().st_mtime_ns == stamp  # not re-rendered


@pytest.mark.parametrize("page", [0, -1, 9999])
def test_a_page_outside_the_document_is_404_not_a_traceback(corpus_repo, client, page):
    response = client.get("/api/citation/thumbnail", params={"file": ANCHOR_PDF, "page": page})
    assert response.status_code == 404


def test_thumbnail_traversal_is_400(corpus_repo, client):
    response = client.get("/api/citation/thumbnail",
                          params={"file": "../../etc/passwd", "page": 1})
    assert response.status_code == 400
    assert not (corpus_repo / "cache" / "webui_thumbs").exists()


def test_a_txt_source_has_no_page_image(corpus_repo, client):
    response = client.get("/api/citation/thumbnail", params={"file": ANCHOR_TXT, "page": 1})
    assert response.status_code == 404


def test_thumbnails_degrade_when_the_corpus_is_absent(empty_repo, client):
    response = client.get("/api/citation/thumbnail", params={"file": ANCHOR_PDF, "page": 1})
    assert response.status_code == 404
    assert "corpus" in response.json()["detail"]
    # ...and the app is still serving.
    assert client.get("/api/datasets").status_code == 200


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #


def test_txt_citation_content_carries_the_page_text_and_its_source_url(corpus_repo, client):
    body = client.get("/api/citation/content", params={"file": ANCHOR_TXT}).json()

    assert body["kind"] == "text"
    assert body["file_name"] == ANCHOR_TXT
    assert body["page_number"] is None
    assert body["text"].strip()
    assert body["source_url"] == (
        "https://www.harel-group.co.il/insurance/apartment/requests/cancellation"
    )


def test_pdf_citation_content_points_at_the_thumbnail_not_a_re_parse(corpus_repo, client):
    body = client.get("/api/citation/content",
                      params={"file": ANCHOR_PDF, "page": 1}).json()

    assert body["kind"] == "pdf"
    assert body["page_number"] == 1
    # Re-parsing a PDF page on a UI request would take seconds and duplicate
    # the ingest pipeline; the page image is the preview.
    assert body["text"] is None
    assert body["source_url"].endswith(".pdf")


def test_content_traversal_is_400(corpus_repo, client):
    assert client.get("/api/citation/content",
                      params={"file": "../../.env"}).status_code == 400


def test_content_without_a_corpus_is_404(empty_repo, client):
    response = client.get("/api/citation/content", params={"file": ANCHOR_TXT})
    assert response.status_code == 404
    assert client.get("/api/datasets").status_code == 200  # app still serving


# --------------------------------------------------------------------------- #
# The Task-2 loop: a quote-less TXT citation still gets a preview
# --------------------------------------------------------------------------- #


def test_a_quoteless_txt_citation_previews_from_the_corpus(corpus_repo):
    pair = schema.record_to_pair(
        {"id": "q1", "answer": "a", "citations": [{"file": ANCHOR_TXT, "page": None}]},
        pair_id="q1",
    )
    preview = pair.citations[0].content_preview
    assert preview and len(preview) <= corpus_view.PREVIEW_CHARS
    assert preview in (corpus_repo / "corpus" / ANCHOR_TXT).read_text(encoding="utf-8")


def test_a_quote_always_wins_over_the_corpus_text(corpus_repo):
    pair = schema.record_to_pair(
        {"id": "q1", "citations": [{"file": ANCHOR_TXT, "quote": "הציטוט מהתשובה"}]},
        pair_id="q1",
    )
    assert pair.citations[0].content_preview == "הציטוט מהתשובה"


def test_previews_are_none_rather_than_an_error_without_a_corpus(empty_repo):
    pair = schema.record_to_pair(
        {"id": "q1", "citations": [{"file": ANCHOR_TXT}, {"file": ANCHOR_PDF, "page": 2}]},
        pair_id="q1",
    )
    assert [c.content_preview for c in pair.citations] == [None, None]
    assert pair.citations[1].thumbnail_url is not None  # the card still offers the image


def test_preview_never_leaves_the_repo(corpus_repo):
    assert corpus_view.preview_text("../../etc/passwd") is None


# --------------------------------------------------------------------------- #
# Module-level contract
# --------------------------------------------------------------------------- #


def test_page_thumbnail_and_page_text_raise_corpus_unavailable(empty_repo):
    with pytest.raises(CorpusUnavailable):
        corpus_view.page_thumbnail(ANCHOR_PDF, 1)
    with pytest.raises(CorpusUnavailable):
        corpus_view.page_text(ANCHOR_TXT)


def test_manifest_is_optional(corpus_repo):
    (corpus_repo / "corpus" / "manifest.json").unlink()
    corpus_view._manifest.cache_clear()
    text, url = corpus_view.page_text(ANCHOR_TXT)
    assert text.strip() and url is None


def test_a_corrupt_manifest_does_not_take_the_endpoint_down(corpus_repo):
    (corpus_repo / "corpus" / "manifest.json").write_text("{not json", encoding="utf-8")
    corpus_view._manifest.cache_clear()
    text, url = corpus_view.page_text(ANCHOR_TXT)
    assert text.strip() and url is None


def test_manifest_keys_are_the_citation_paths(corpus_repo):
    raw = json.loads((corpus_repo / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    assert ANCHOR_TXT in raw and ANCHOR_PDF in raw

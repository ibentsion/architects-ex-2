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
def corpus_repo(tmp_path, monkeypatch, mini_corpus_dir, repo_root) -> Path:
    """A repo root whose corpus/ is the real mini-corpus fixture (a real PDF
    and real Hebrew TXTs), plus the Docling parse-cache entries for its PDFs.

    Copied, not symlinked — a symlink would resolve out of the fake root and be
    rejected, correctly. The parse entries come from the shared repo cache, so
    this triggers no Docling run (both fixture PDFs are already parsed there).
    """
    from rag.parsing import discover

    root = tmp_path / "repo"
    root.mkdir()
    shutil.copytree(mini_corpus_dir, root / "corpus")

    parsed = root / "cache" / "parsed"
    parsed.mkdir(parents=True)
    for source in discover(mini_corpus_dir):
        entry = repo_root / "cache" / "parsed" / f"{source.sha256}.json"
        if entry.is_file():
            shutil.copyfile(entry, parsed / entry.name)

    monkeypatch.setattr(paths, "REPO_ROOT", root)
    corpus_view._manifest.cache_clear()
    corpus_view.reset_page_store()
    return root


@pytest.fixture
def empty_repo(tmp_path, monkeypatch) -> Path:
    """A checkout with no corpus/ at all — the normal laptop case."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(paths, "REPO_ROOT", root)
    corpus_view._manifest.cache_clear()
    corpus_view.reset_page_store()
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


def test_pdf_citation_content_serves_the_cached_page_text(corpus_repo, client):
    body = client.get("/api/citation/content",
                      params={"file": ANCHOR_PDF, "page": 1}).json()

    assert body["kind"] == "pdf"
    assert body["page_number"] == 1
    # From the parse cache, not a fresh parse — the same text the index holds.
    assert body["text"] and "התיישנות" in body["text"]
    assert body["source_url"].endswith(".pdf")


def test_pdf_citation_content_without_a_parse_cache_still_serves_the_image(
    corpus_repo, client
):
    for entry in (corpus_repo / "cache" / "parsed").glob("*.json"):
        entry.unlink()
    corpus_view.reset_page_store()

    body = client.get("/api/citation/content",
                      params={"file": ANCHOR_PDF, "page": 1}).json()
    assert body["kind"] == "pdf" and body["text"] is None
    assert client.get("/api/citation/thumbnail",
                      params={"file": ANCHOR_PDF, "page": 1}).status_code == 200


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
    assert corpus_view.preview_text("../../etc/passwd", 1) is None


# --------------------------------------------------------------------------- #
# PDF page previews — the citation accordion's text, from the parse cache
# --------------------------------------------------------------------------- #


def test_a_pdf_citation_previews_the_cited_page(corpus_repo):
    """The page text comes from the same corpus-walk + Docling parse cache the
    citation judge uses, so the card shows the text the retriever indexed."""
    pair = schema.record_to_pair(
        {"id": "q1", "citations": [{"file": ANCHOR_PDF, "page": 1}]}, pair_id="q1"
    )
    preview = pair.citations[0].content_preview
    assert preview, "the anchor PDF's page 1 must resolve"
    assert len(preview) <= corpus_view.PREVIEW_CHARS
    assert "התיישנות" in preview  # the ground-truth anchor's page-1 content


def test_the_preview_is_the_cited_page_not_just_the_document(corpus_repo):
    """A preview of the wrong page is worse than none, which is why the page
    number is threaded all the way through the adapter."""
    page_one = corpus_view.page_preview(ANCHOR_PDF, 1)
    assert page_one and "התיישנות" in page_one
    assert corpus_view.page_preview(ANCHOR_PDF, 2) is None  # 1-page document


@pytest.mark.parametrize(
    ("file", "page"),
    [
        (ANCHOR_PDF, 9999),  # page outside the document
        (ANCHOR_PDF, 0),  # not a real page number
        ("apartment/files/no-such-document.pdf", 1),  # unknown file
        (ANCHOR_PDF, None),  # a paged document cited without a page
    ],
)
def test_unresolvable_page_previews_are_none_not_exceptions(corpus_repo, file, page):
    assert corpus_view.preview_text(file, page) is None


def test_a_pdf_with_no_parse_cache_entry_previews_as_none(corpus_repo):
    """cache/parsed/ is gitignored and may be empty — a preview must degrade,
    and must not try to parse a PDF on a UI request."""
    for entry in (corpus_repo / "cache" / "parsed").glob("*.json"):
        entry.unlink()
    corpus_view.reset_page_store()

    assert corpus_view.preview_text(ANCHOR_PDF, 1) is None
    # ...and the citation list still renders.
    pair = schema.record_to_pair(
        {"id": "q1", "citations": [{"file": ANCHOR_PDF, "page": 1}]}, pair_id="q1"
    )
    assert pair.citations[0].content_preview is None
    assert pair.citations[0].thumbnail_url is not None


def test_page_previews_never_raise_out_of_the_adapter(corpus_repo, monkeypatch):
    """Whatever the store does — and it raises FileNotFoundError for a missing
    parse — a citation list must render."""
    class Exploding:
        def resolve(self, file, page):
            raise RuntimeError("parse cache is on fire")

    monkeypatch.setattr(corpus_view, "_page_store", lambda: Exploding())
    assert corpus_view.page_preview(ANCHOR_PDF, 1) is None

    pair = schema.record_to_pair(
        {"id": "q1", "citations": [{"file": ANCHOR_PDF, "page": 1}]}, pair_id="q1"
    )
    assert pair.citations[0].content_preview is None


def test_the_page_store_is_built_once_per_repo_root(corpus_repo):
    corpus_view.preview_text(ANCHOR_PDF, 1)
    first = corpus_view._page_store()
    corpus_view.preview_text(ANCHOR_PDF, 1)
    assert corpus_view._page_store() is first  # no re-walk of the corpus


def test_a_quote_still_wins_over_the_resolved_page(corpus_repo):
    pair = schema.record_to_pair(
        {"id": "q1", "citations": [{"file": ANCHOR_PDF, "page": 1, "quote": "הציטוט"}]},
        pair_id="q1",
    )
    assert pair.citations[0].content_preview == "הציטוט"


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

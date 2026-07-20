"""Parsing-phase tests (rag_plan.md §9 test_parsing.py list).

Fast tests mock the converter / use synthetic docling dicts; tests that run a
real Docling parse are marked ``slow`` (first run downloads ~500 MB of layout
models; afterwards the shared parse cache makes them cheap).
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from rag.parsing import ParsedDoc, SourceFile, discover, sha256_file
from rag.parsing.cache import CachedParser, ParseCache
from rag.parsing.canary import (
    CanaryError,
    check_anchor,
    check_file,
    doc_text,
    iter_docling_texts,
    run_canary,
)
from rag.parsing.docling_parser import DoclingParser
from rag.parsing.txt_parser import TxtParser
from tests.conftest import ANCHOR_PDF_REL

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

FAKE_DOCLING = {
    "texts": [
        {"text": "הודעה על ביטוח דירה של הראל", "prov": [{"page_no": 1}]},
        {"text": "תנאי הפוליסה המלאים", "prov": [{"page_no": 2}]},
    ],
    "tables": [],
}


def make_source(rel_path: str = "apartment/files/doc.pdf", kind: str = "pdf", sha: str = "a" * 64) -> SourceFile:
    category = rel_path.split("/", 1)[0]
    return SourceFile(
        abs_path=Path("/nonexistent") / rel_path,
        rel_path=rel_path,
        category=category,
        kind=kind,  # type: ignore[arg-type]
        sha256=sha,
        source_url=None,
    )


def make_pdf_doc(texts_with_pages: list[tuple[str, int]], rel_path: str = "apartment/files/doc.pdf") -> ParsedDoc:
    return ParsedDoc(
        source=make_source(rel_path),
        docling={
            "texts": [{"text": t, "prov": [{"page_no": p}]} for t, p in texts_with_pages],
            "tables": [],
        },
    )


class CountingParser:
    """Stub PDF parser that counts calls (cache-behavior tests)."""

    def __init__(self) -> None:
        self.calls = 0

    def parse(self, source: SourceFile) -> ParsedDoc:
        self.calls += 1
        return ParsedDoc(source=source, docling=dict(FAKE_DOCLING))


# --------------------------------------------------------------------------- #
# Stage 1 — discovery
# --------------------------------------------------------------------------- #


def test_discover_finds_all_fixture_files(mini_corpus_dir, mini_sources):
    rel_paths = {s.rel_path for s in mini_sources}
    assert rel_paths == {
        "apartment/files/הודעה-על-תקופת-התיישנות.pdf",
        "apartment/pages/cancellation.txt",
        "travel/files/הודעה-על-הגדרת-ספורט-אתגרי.pdf",
        "travel/pages/benefits.txt",
    }
    by_rel = {s.rel_path: s for s in mini_sources}
    assert by_rel[ANCHOR_PDF_REL].kind == "pdf"
    assert by_rel[ANCHOR_PDF_REL].category == "apartment"
    assert by_rel["travel/pages/benefits.txt"].kind == "txt"


def test_discover_rel_path_is_nfc_and_matches_ground_truth(mini_sources):
    anchor = [s for s in mini_sources if s.kind == "pdf" and s.category == "apartment"][0]
    # Byte-identical to the reference_questions.json convention (NFC).
    assert anchor.rel_path == ANCHOR_PDF_REL
    assert anchor.rel_path == unicodedata.normalize("NFC", anchor.rel_path)
    assert "\\" not in anchor.rel_path and not anchor.rel_path.startswith("./")


def test_discover_sha256_matches_file_content(mini_sources):
    for source in mini_sources:
        assert source.sha256 == sha256_file(source.abs_path)
        assert len(source.sha256) == 64


def test_discover_source_url_from_manifest(mini_sources):
    by_rel = {s.rel_path: s for s in mini_sources}
    assert (
        by_rel["travel/pages/benefits.txt"].source_url
        == "https://www.harel-group.co.il/insurance/travel/benefits"
    )
    assert by_rel[ANCHOR_PDF_REL].source_url is not None
    assert by_rel[ANCHOR_PDF_REL].source_url.startswith("https://")


def test_discover_missing_corpus_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="get_corpus.py"):
        discover(tmp_path / "no-such-corpus")


def test_discover_skips_non_pdf_txt_assets(tmp_path):
    files_dir = tmp_path / "apartment" / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "policy.pdf").write_bytes(b"%PDF-fake")
    (files_dir / "image.png").write_bytes(b"\x89PNG")
    sources = discover(tmp_path)
    assert [s.rel_path for s in sources] == ["apartment/files/policy.pdf"]


# --------------------------------------------------------------------------- #
# Stage 2 — TXT parser
# --------------------------------------------------------------------------- #


def test_txt_parser_reads_utf8_and_has_no_pages(mini_sources):
    txt_source = [s for s in mini_sources if s.rel_path == "apartment/pages/cancellation.txt"][0]
    parsed = TxtParser().parse(txt_source)
    assert parsed.kind == "txt"
    assert parsed.docling is None
    assert parsed.text
    assert parsed.text == txt_source.abs_path.read_text(encoding="utf-8")


def test_txt_parser_rejects_pdf_source():
    with pytest.raises(ValueError, match="only parses TXT"):
        TxtParser().parse(make_source(kind="pdf"))


def test_docling_parser_rejects_txt_source():
    with pytest.raises(ValueError, match="only parses PDF"):
        DoclingParser().parse(make_source("apartment/pages/x.txt", kind="txt"))


# --------------------------------------------------------------------------- #
# Stage 2 — parse cache
# --------------------------------------------------------------------------- #


def test_cache_hit_skips_converter(tmp_cache_dir, monkeypatch):
    source = make_source()
    ParseCache(tmp_cache_dir).store(source.sha256, FAKE_DOCLING)

    real_parser = DoclingParser()
    monkeypatch.setattr(
        real_parser,
        "_get_converter",
        lambda: pytest.fail("DocumentConverter must not be constructed on a cache hit"),
    )
    parsed = CachedParser(real_parser, tmp_cache_dir).parse(source)
    assert parsed.docling == FAKE_DOCLING
    assert parsed.source is source


def test_cache_miss_parses_then_second_call_hits(tmp_cache_dir):
    counting = CountingParser()
    cached = CachedParser(counting, tmp_cache_dir)
    source = make_source()
    cached.parse(source)
    cached.parse(source)
    assert counting.calls == 1
    assert ParseCache(tmp_cache_dir).load(source.sha256) == FAKE_DOCLING


def test_sha256_change_triggers_reparse(tmp_cache_dir):
    counting = CountingParser()
    cached = CachedParser(counting, tmp_cache_dir)
    cached.parse(make_source(sha="a" * 64))
    cached.parse(make_source(sha="b" * 64))  # same path, new content hash
    assert counting.calls == 2


def test_corrupt_cache_entry_reparses(tmp_cache_dir):
    source = make_source()
    cache = ParseCache(tmp_cache_dir)
    cache.parsed_dir.mkdir(parents=True)
    cache.path_for(source.sha256).write_text("{not json", encoding="utf-8")
    counting = CountingParser()
    CachedParser(counting, tmp_cache_dir).parse(source)
    assert counting.calls == 1


def test_force_reparse_bypasses_cache(tmp_cache_dir):
    source = make_source()
    ParseCache(tmp_cache_dir).store(source.sha256, {"texts": [], "tables": []})
    counting = CountingParser()
    CachedParser(counting, tmp_cache_dir, force_reparse=True).parse(source)
    assert counting.calls == 1


def test_txt_bypasses_cache(tmp_cache_dir, mini_sources):
    txt_source = [s for s in mini_sources if s.kind == "txt"][0]
    parsed = CachedParser(TxtParser(), tmp_cache_dir).parse(txt_source)
    assert parsed.text
    assert not (tmp_cache_dir / "parsed").exists()


# --------------------------------------------------------------------------- #
# Stage 2 — RTL canary (synthetic, fast)
# --------------------------------------------------------------------------- #


def test_canary_passes_on_normal_hebrew():
    docs = [
        make_pdf_doc([("ביטוח דירה מקיף", 1), ("הראל חברה לביטוח", 1)]),
        make_pdf_doc([("תנאי הפוליסה", 1)], rel_path="travel/files/other.pdf"),
    ]
    result = run_canary(docs)
    assert result.ok
    assert all(r.ok for r in result.files)
    assert result.missing_tokens == []


def test_canary_fails_on_reversed_hebrew():
    good = make_pdf_doc([("ביטוח דירה של הראל לפי הפוליסה", 1)])
    reversed_doc = make_pdf_doc(
        [("הריד חוטיב לע עדימ", 1)], rel_path="travel/files/reversed.pdf"
    )
    with pytest.raises(CanaryError) as excinfo:
        run_canary([good, reversed_doc])
    result = excinfo.value.result
    assert not result.ok
    bad = [r for r in result.files if not r.ok]
    assert [r.rel_path for r in bad] == ["travel/files/reversed.pdf"]
    assert "חוטיב" in bad[0].reversals_found
    assert "travel/files/reversed.pdf" in str(excinfo.value)  # per-file diagnosis


def test_canary_fails_when_tokens_missing_from_sample():
    docs = [make_pdf_doc([("טקסט כללי ללא מילות עוגן", 1)])]
    with pytest.raises(CanaryError) as excinfo:
        run_canary(docs)
    assert set(excinfo.value.result.missing_tokens) == {"ביטוח", "הראל", "פוליסה"}


def test_canary_reversal_needs_word_boundary():
    # "לארה״ב" (to the USA) contains the reversal of הראל as a substring —
    # must NOT trip the canary (legitimate travel-insurance text).
    doc = make_pdf_doc([("ביטוח נסיעות לארה״ב של הראל לפי הפוליסה", 1)])
    assert run_canary([doc]).ok


def test_check_anchor_on_synthetic_page():
    doc = make_pdf_doc(
        [("תקופת התיישנות של שלוש   שנים", 1), ("עמוד שני", 2)],
        rel_path=ANCHOR_PDF_REL,
    )
    report = check_anchor(doc)
    assert report.ok  # whitespace-normalized phrase match
    assert report.page == 1


def test_check_anchor_fails_when_phrase_on_wrong_page():
    doc = make_pdf_doc([("כלום", 1), ("התיישנות שלוש שנים", 2)], rel_path=ANCHOR_PDF_REL)
    report = check_anchor(doc)
    assert not report.ok
    assert set(report.phrases_missing) == {"התיישנות", "שלוש שנים"}


def test_doc_text_covers_tables():
    doc = ParsedDoc(
        source=make_source(),
        docling={
            "texts": [{"text": "כותרת", "prov": [{"page_no": 1}]}],
            "tables": [
                {
                    "prov": [{"page_no": 1}],
                    "data": {"table_cells": [{"text": "פוליסה"}, {"text": "5,000 ₪"}]},
                }
            ],
        },
    )
    text = doc_text(doc)
    assert "פוליסה" in text and "5,000 ₪" in text


# --------------------------------------------------------------------------- #
# Real Docling parses (slow; served from the shared parse cache after run 1)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_real_pdf_every_text_item_has_valid_page(parsed_anchor):
    assert parsed_anchor.docling is not None
    items = list(iter_docling_texts(parsed_anchor.docling))
    assert items, "anchor PDF parsed to zero text items"
    for text, page_no in items:
        assert page_no is not None and page_no >= 1, f"bad page_no for item: {text[:40]!r}"


@pytest.mark.slow
def test_real_canary_passes_on_fixture_pdfs(parsed_anchor, parsed_travel_pdf):
    result = run_canary([parsed_anchor, parsed_travel_pdf], anchor_doc=parsed_anchor)
    assert result.ok
    assert result.missing_tokens == []
    assert result.anchor is not None and result.anchor.ok


@pytest.mark.slow
def test_real_anchor_page1_contains_ground_truth(parsed_anchor):
    """Validates plan assumption A1: Docling page_no is 1-based and the
    ground-truth citation {file, page: 1} lands on the right text."""
    report = check_anchor(parsed_anchor)
    assert report.ok, f"anchor check failed: missing {report.phrases_missing}"
    per_file = check_file(parsed_anchor)
    assert per_file.ok, f"reversed Hebrew in anchor PDF: {per_file.reversals_found}"

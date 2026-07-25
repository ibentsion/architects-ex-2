"""Ingestion report tests (rag/report.py).

Covers the three signals the report exists for: structured warning
attribution (file/category/page reach the collector), context-attributed
library warnings (Docling/Stanza-style calls with no ``extra=`` support), and
the content-loss reconciliation check (the one thing that must never be
silently wrong: did chunking actually drop document text).
"""
from __future__ import annotations

import json
import logging

from rag.chunking.common import TokenCounter, pack_paragraphs
from rag.report import (
    EVENT_LONG_PARAGRAPH,
    ReportBuilder,
    collecting,
    current_source,
    load_previous,
    render_markdown,
    render_summary_line,
    save_report_history,
    write_report_files,
)

logger = logging.getLogger("rag.report.test_helper")


def test_pack_paragraphs_long_paragraph_carries_file_context() -> None:
    """The 'kept whole' warning must include enough context to find the file
    without re-reading raw logs — that was the whole point of the report."""
    counter = TokenCounter()
    long_para = "מילה " * 2000  # far exceeds any max_tokens
    warnings: list[dict] = []
    with collecting(warnings):
        packs = pack_paragraphs(
            [long_para],
            max_tokens=10,
            counter=counter,
            context={"file": "apartment/files/doc.pdf", "category": "apartment", "page": 3, "chunker": "per_page"},
        )
    assert packs == [long_para]  # kept whole, never split or dropped
    events = [w for w in warnings if w["rag_event"] == EVENT_LONG_PARAGRAPH]
    assert len(events) == 1
    assert events[0]["file"] == "apartment/files/doc.pdf"
    assert events[0]["category"] == "apartment"
    assert events[0]["page"] == 3
    assert events[0]["chunker"] == "per_page"
    assert events[0]["detail"]["max_tokens"] == 10


def test_pack_paragraphs_without_context_still_warns() -> None:
    """context= is optional — callers that don't pass it still get a usable
    (if file-less) warning rather than a crash."""
    counter = TokenCounter()
    warnings: list[dict] = []
    with collecting(warnings):
        packs = pack_paragraphs(["מילה " * 2000], max_tokens=10, counter=counter)
    assert len(packs) == 1
    assert any(w["rag_event"] == EVENT_LONG_PARAGRAPH for w in warnings)


def test_current_source_attributes_library_warnings_without_extra() -> None:
    """Simulates a Docling/Stanza-style warning that can't pass extra= —
    the collector must still recover file/category from the contextvar."""
    warnings: list[dict] = []
    with collecting(warnings):
        with current_source("car/files/policy.pdf", "car"):
            logger.warning("RapidOCR returned empty result!")
    assert len(warnings) == 1
    assert warnings[0]["file"] == "car/files/policy.pdf"
    assert warnings[0]["category"] == "car"
    assert warnings[0]["rag_event"] is None  # no extra= passed — attributed via context only


def test_current_source_does_not_leak_after_exit() -> None:
    warnings: list[dict] = []
    with collecting(warnings):
        with current_source("car/files/policy.pdf", "car"):
            pass
        logger.warning("unrelated warning outside the context")
    assert warnings[-1]["file"] is None


def test_record_doc_flags_content_loss_on_shrinkage() -> None:
    builder = ReportBuilder()
    builder.record_doc(
        file="apartment/files/doc.pdf",
        category="apartment",
        chunker="per_page",
        raw_chars=1000,
        chunks=["x" * 500],  # 50% kept -> well below the 0.98 floor
    )
    assert len(builder.content_loss) == 1
    assert builder.content_loss[0]["ratio"] == 0.5


def test_record_doc_does_not_flag_kept_whole_or_growth() -> None:
    builder = ReportBuilder()
    # per_table prepends a heading -> chunked chars can exceed raw chars.
    builder.record_doc(
        file="apartment/files/doc.pdf",
        category="apartment",
        chunker="per_table",
        raw_chars=1000,
        chunks=["x" * 1200],
    )
    # per_page keeping an oversized paragraph whole -> ratio == 1.0
    builder.record_doc(
        file="travel/files/doc.pdf",
        category="travel",
        chunker="per_page",
        raw_chars=1000,
        chunks=["x" * 1000],
    )
    assert builder.content_loss == []


def test_finalize_and_render_markdown_smoke() -> None:
    builder = ReportBuilder()
    builder.record_doc(
        file="apartment/files/doc.pdf", category="apartment", chunker="per_page",
        raw_chars=100, chunks=["x" * 100],
    )
    report = builder.finalize(
        raw_warnings=[],
        config_identity="deadbeef",
        impls={"chunker": "per_page"},
        categories_filter=["apartment"],
        files_meta=[{"file": "apartment/files/doc.pdf", "sha256": "a" * 64, "status": "ok"}],
        chunk_counts={"apartment": 1},
        canary={"passed": True},
        embedding_tokens=42,
        wall_seconds=12.3,
        stage_seconds={},
    )
    assert report["chunks"]["total"] == 1
    assert report["content_loss_suspected"] == []
    md = render_markdown(report, previous=None)
    assert "Content-loss check" in md
    assert "0 files flagged" in md
    assert render_summary_line(report)  # doesn't raise, returns something


def test_report_history_round_trip(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    index_dir = tmp_path / "rag_index" / "default"
    assert load_previous(cache_dir, index_dir) is None

    builder = ReportBuilder()
    report_v1 = builder.finalize(
        raw_warnings=[], config_identity="id1", impls={}, categories_filter=None,
        files_meta=[], chunk_counts={"apartment": 5}, canary=None,
        embedding_tokens=0, wall_seconds=1.0, stage_seconds={},
    )
    save_report_history(cache_dir, index_dir, report_v1)

    previous = load_previous(cache_dir, index_dir)
    assert previous is not None
    assert previous["chunks"]["by_category"] == {"apartment": 5}

    history_files = list((cache_dir / "reports").rglob("*.json"))
    assert any(f.name == "latest.json" for f in history_files)
    assert any(f.name != "latest.json" for f in history_files)  # timestamped copy kept too

    # A second run's delta section should be able to see the first run.
    report_v2 = builder.finalize(
        raw_warnings=[], config_identity="id1", impls={}, categories_filter=None,
        files_meta=[], chunk_counts={"apartment": 5, "travel": 3}, canary=None,
        embedding_tokens=0, wall_seconds=1.0, stage_seconds={},
    )
    md = render_markdown(report_v2, previous)
    assert "Comparison to previous run" in md
    assert "New categories since previous run: travel" in md


def test_write_report_files_keeps_timestamped_copy_alongside_latest(tmp_path) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    report = ReportBuilder().finalize(
        raw_warnings=[], config_identity="id1", impls={}, categories_filter=None,
        files_meta=[], chunk_counts={"apartment": 5}, canary=None,
        embedding_tokens=0, wall_seconds=1.0, stage_seconds={},
    )
    write_report_files(build_dir, report, render_markdown(report, None))

    names = {f.name for f in build_dir.iterdir()}
    assert {"ingest_report.md", "ingest_report.json"} <= names
    timestamped = [n for n in names if n not in ("ingest_report.md", "ingest_report.json")]
    assert len(timestamped) == 2  # one .md + one .json, both stamped with the same run timestamp
    assert all(n.startswith("ingest_report_") for n in timestamped)
    # timestamped and "latest" content are byte-identical for this run
    ts_md = next(n for n in timestamped if n.endswith(".md"))
    assert (build_dir / ts_md).read_text(encoding="utf-8") == (build_dir / "ingest_report.md").read_text(encoding="utf-8")

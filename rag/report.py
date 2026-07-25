"""Ingestion report — inline warning collection + inspection/debugging output.

Answers three questions every ingest run must be able to answer without
re-reading raw logs: (1) did any file lose text during chunking, (2) which
files/categories triggered which warnings, and (3) how does this run compare
to the previous one for the same ``index_dir``.

Two collection mechanisms, both attached only for the duration of
:func:`rag.cli.ingest.run_ingestion`:

* **Structured events** — call sites that already know full context (file,
  category, page, chunker) pass it via ``logger.warning(..., extra={...})``
  with keys prefixed ``rag_`` (avoids colliding with reserved
  :class:`logging.LogRecord` attribute names). Collected verbatim.
* **Context-attributed library warnings** — Docling/Stanza emit warnings
  (OCR empty result, Stanza fallback) from inside library calls that don't
  accept ``extra=``. :func:`current_source` stamps the file/category being
  processed into a :class:`contextvars.ContextVar`; the handler reads it at
  emit time. Ingestion is strictly sequential, so this is unambiguous.

A third signal has no logging involved at all: :meth:`ReportBuilder.record_doc`
compares each document's raw extracted character count against its chunked
character count. "Kept whole" never loses text (verified by the invariant
below); only a dropped chunk/item can shrink the ratio, so this is an
independent, log-warning-agnostic check for the one failure mode that
actually matters — silently missing document text.
"""
from __future__ import annotations

import json
import logging
import os
import re
import statistics
from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# --------------------------------------------------------------------------- #
# Structured event names (stamped via extra={"rag_event": ...})
# --------------------------------------------------------------------------- #

EVENT_LONG_PARAGRAPH = "long_paragraph_kept_whole"
EVENT_EMPTY_CHUNK = "empty_chunk_dropped"
EVENT_NO_PAGE = "item_without_page_dropped"
EVENT_STANZA_FALLBACK = "stanza_fallback"
EVENT_CACHE_CORRUPT = "cache_corrupt"

#: Events that indicate chunk text was PRESERVED (just oversized) — never a
#: content-loss signal on their own.
LOSSLESS_EVENTS = {EVENT_LONG_PARAGRAPH}

#: Events that mean a piece of document text was dropped and never indexed.
LOSSY_EVENTS = {EVENT_EMPTY_CHUNK, EVENT_NO_PAGE}

#: Content-loss reconciliation ratio floor (chunked_chars / raw_chars) below
#: which a doc is flagged for inspection — "kept whole" and heading-prepend
#: (per_table) only grow the ratio, so a drop below this is a real signal.
CONTENT_LOSS_RATIO = 0.98

_OCR_EMPTY_RE = re.compile(r"RapidOCR returned empty result")


# --------------------------------------------------------------------------- #
# Context attribution for library warnings without extra= support
# --------------------------------------------------------------------------- #

_CURRENT_SOURCE: ContextVar[dict[str, str] | None] = ContextVar(
    "rag_report_current_source", default=None
)


@contextmanager
def current_source(file: str, category: str) -> Iterator[None]:
    """Stamp the file/category being processed for the duration of a library
    call so the collecting handler can attribute warnings it can't tag
    itself (Docling OCR, Stanza fallback). Ingestion is single-threaded and
    sequential, so this is unambiguous."""
    token = _CURRENT_SOURCE.set({"file": file, "category": category})
    try:
        yield
    finally:
        _CURRENT_SOURCE.reset(token)


# --------------------------------------------------------------------------- #
# Collecting handler
# --------------------------------------------------------------------------- #


class _CollectingHandler(logging.Handler):
    def __init__(self, sink: list[dict[str, Any]]) -> None:
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        source = _CURRENT_SOURCE.get()
        entry: dict[str, Any] = {
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "rag_event": getattr(record, "rag_event", None),
            "file": getattr(record, "rag_file", None) or (source or {}).get("file"),
            "category": getattr(record, "rag_category", None) or (source or {}).get("category"),
            "page": getattr(record, "rag_page", None),
            "chunker": getattr(record, "rag_chunker", None),
            "detail": getattr(record, "rag_detail", None),
        }
        self.sink.append(entry)


@contextmanager
def collecting(sink: list[dict[str, Any]]) -> Iterator[None]:
    """Attach a WARNING+ collecting handler to the root logger for the scope
    of the ``with`` block (used around the whole ``run_ingestion`` call)."""
    handler = attach_collector(sink)
    try:
        yield
    finally:
        detach_collector(handler)


def attach_collector(sink: list[dict[str, Any]]) -> logging.Handler:
    """Attach and return a WARNING+ collecting handler (try/finally style,
    for call sites that can't cleanly wrap their whole body in a ``with``)."""
    handler = _CollectingHandler(sink)
    logging.getLogger().addHandler(handler)
    return handler


def detach_collector(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)


# --------------------------------------------------------------------------- #
# Report builder
# --------------------------------------------------------------------------- #


class ReportBuilder:
    """Accumulates per-run stats as the ingestion CLI progresses through its
    stages; :meth:`finalize` turns it into the report dict."""

    def __init__(self) -> None:
        self.chunk_char_lengths: dict[str, list[int]] = defaultdict(list)  # by chunker
        self.content_loss: list[dict[str, Any]] = []

    def record_doc(
        self, *, file: str, category: str, chunker: str, raw_chars: int, chunks: list[str]
    ) -> None:
        chunk_chars = sum(len(c) for c in chunks)
        self.chunk_char_lengths[chunker].extend(len(c) for c in chunks)
        if raw_chars <= 0:
            return
        ratio = chunk_chars / raw_chars
        if ratio < CONTENT_LOSS_RATIO:
            self.content_loss.append(
                {
                    "file": file,
                    "category": category,
                    "chunker": chunker,
                    "raw_chars": raw_chars,
                    "chunk_chars": chunk_chars,
                    "ratio": round(ratio, 4),
                }
            )

    def finalize(
        self,
        *,
        raw_warnings: list[dict[str, Any]],
        config_identity: str,
        impls: dict[str, str],
        categories_filter: list[str] | None,
        files_meta: list[dict[str, Any]],
        chunk_counts: dict[str, int],
        canary: dict[str, Any] | None,
        embedding_tokens: int,
        wall_seconds: float,
        stage_seconds: dict[str, float],
    ) -> dict[str, Any]:
        events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        misc: Counter[tuple[str, str]] = Counter()
        misc_examples: dict[tuple[str, str], str] = {}
        ocr_empty: list[dict[str, str]] = []

        for w in raw_warnings:
            event = w["rag_event"]
            if event is not None:
                events[event].append(
                    {
                        "file": w["file"],
                        "category": w["category"],
                        "page": w["page"],
                        "chunker": w["chunker"],
                        "detail": w["detail"],
                        "message": w["message"],
                    }
                )
                continue
            if w["logger"].startswith("docling") and _OCR_EMPTY_RE.search(w["message"]):
                ocr_empty.append({"file": w["file"], "category": w["category"]})
                continue
            key = (w["logger"], w["message"][:100])
            misc[key] += 1
            misc_examples.setdefault(key, w["message"])

        chunk_size_stats: dict[str, dict[str, float]] = {}
        for chunker, lengths in self.chunk_char_lengths.items():
            if not lengths:
                continue
            sorted_lens = sorted(lengths)
            chunk_size_stats[chunker] = {
                "n": len(lengths),
                "min": sorted_lens[0],
                "max": sorted_lens[-1],
                "mean": round(statistics.mean(lengths), 1),
                "p50": sorted_lens[len(sorted_lens) // 2],
                "p95": sorted_lens[int(len(sorted_lens) * 0.95)],
            }

        n_failed = sum(1 for f in files_meta if f["status"] == "failed")
        n_cached = sum(1 for f in files_meta if f["status"] == "cached")
        failed_files = [f for f in files_meta if f["status"] == "failed"]

        return {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_identity": config_identity,
            "impls": impls,
            "categories_filter": categories_filter,
            "wall_seconds": round(wall_seconds, 1),
            "stage_seconds": {k: round(v, 1) for k, v in stage_seconds.items()},
            "files": {
                "total": len(files_meta),
                "cached": n_cached,
                "parsed_fresh": len(files_meta) - n_cached - n_failed,
                "failed": n_failed,
                "failed_files": failed_files,
            },
            "chunks": {
                "total": sum(chunk_counts.values()),
                "by_category": chunk_counts,
                "size_chars": chunk_size_stats,
            },
            "warnings": {
                "long_paragraph_kept_whole": {
                    "count": len(events[EVENT_LONG_PARAGRAPH]),
                    "note": "text preserved as one oversized chunk — not a loss",
                    "examples": events[EVENT_LONG_PARAGRAPH][:50],
                },
                "empty_chunk_dropped": {
                    "count": len(events[EVENT_EMPTY_CHUNK]),
                    "note": "whitespace-only piece dropped — not a loss",
                    "examples": events[EVENT_EMPTY_CHUNK][:50],
                },
                "item_without_page_dropped": {
                    "count": len(events[EVENT_NO_PAGE]),
                    "note": "REAL TEXT LOSS: a parsed item/chunk had no page provenance and was dropped",
                    "examples": events[EVENT_NO_PAGE][:50],
                },
                "stanza_fallback": {
                    "count": len(events[EVENT_STANZA_FALLBACK]),
                    "note": "whitespace tokenization used instead of Stanza lemmas for this chunk (BM25 recall only, not a text loss)",
                    "examples": events[EVENT_STANZA_FALLBACK][:20],
                },
                "cache_corrupt": {
                    "count": len(events[EVENT_CACHE_CORRUPT]),
                    "examples": events[EVENT_CACHE_CORRUPT][:20],
                },
                "ocr_empty_result": {
                    "count": len(ocr_empty),
                    "note": "docling's OCR pass returned nothing for some region — check these files for missing scanned/image text",
                    "files": ocr_empty,
                },
                "misc": [
                    {"logger": k[0], "message": k[1], "count": n, "example": misc_examples[k]}
                    for k, n in misc.most_common(20)
                ],
            },
            "content_loss_suspected": self.content_loss,
            "canary": canary,
            "embedding_input_tokens": embedding_tokens,
        }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _fmt_examples(examples: list[dict[str, Any]], cap: int = 10) -> str:
    if not examples:
        return "  (none)\n"
    lines = []
    for e in examples[:cap]:
        loc = f"{e.get('file', '?')}"
        if e.get("page") is not None:
            loc += f" p.{e['page']}"
        if e.get("chunker"):
            loc += f" [{e['chunker']}]"
        detail = f" — {e['detail']}" if e.get("detail") else ""
        lines.append(f"  - {loc}{detail}")
    if len(examples) > cap:
        lines.append(f"  - … {len(examples) - cap} more (see ingest_report.json)")
    return "\n".join(lines) + "\n"


def render_markdown(report: dict[str, Any], previous: dict[str, Any] | None) -> str:
    lines: list[str] = []
    lines.append("# Ingestion Report")
    lines.append("")
    lines.append(f"- **Run:** {report['created_at']}")
    lines.append(f"- **Config identity:** `{report['config_identity']}`")
    lines.append(
        f"- **Impls:** "
        + ", ".join(f"{k}={v}" for k, v in report["impls"].items())
    )
    cf = report["categories_filter"]
    lines.append(f"- **Categories filter:** {', '.join(cf) if cf else 'all (full corpus)'}")
    lines.append(f"- **Wall time:** {report['wall_seconds']:.1f}s")
    lines.append("")

    lines.append("## Files")
    f = report["files"]
    lines.append(
        f"- Total: {f['total']} · cached: {f['cached']} · parsed fresh: {f['parsed_fresh']} "
        f"· **failed: {f['failed']}**"
    )
    if f["failed_files"]:
        for ff in f["failed_files"]:
            lines.append(f"  - FAILED: {ff['file']}")
    lines.append("")

    lines.append("## Chunks")
    c = report["chunks"]
    lines.append(f"- Total: {c['total']}")
    lines.append(
        "- By category: " + ", ".join(f"{k}={v}" for k, v in sorted(c["by_category"].items()))
    )
    for chunker, stats in c["size_chars"].items():
        lines.append(
            f"- `{chunker}` chunk size (chars): n={stats['n']} min={stats['min']} "
            f"mean={stats['mean']} p50={stats['p50']} p95={stats['p95']} max={stats['max']}"
        )
    lines.append("")

    lines.append("## Content-loss check (chunked vs. raw extracted text)")
    loss = report["content_loss_suspected"]
    if not loss:
        lines.append(f"**0 files flagged** — every parsed document's chunked text is within "
                      f"{int((1 - CONTENT_LOSS_RATIO) * 100)}% of its raw extracted length.")
    else:
        lines.append(f"**{len(loss)} file(s) flagged — chunked text is noticeably shorter than raw extraction:**")
        for entry in loss:
            lines.append(
                f"  - {entry['file']} [{entry['chunker']}]: {entry['chunk_chars']}/{entry['raw_chars']} "
                f"chars kept ({entry['ratio']:.0%})"
            )
    lines.append("")

    lines.append("## Warnings")
    w = report["warnings"]
    lines.append(
        f"- **Long paragraph kept whole** (no text lost, just an oversized chunk): "
        f"{w['long_paragraph_kept_whole']['count']}"
    )
    lines.append(_fmt_examples(w["long_paragraph_kept_whole"]["examples"]))
    lines.append(f"- **Empty chunk dropped** (whitespace-only, no loss): {w['empty_chunk_dropped']['count']}")
    lines.append(_fmt_examples(w["empty_chunk_dropped"]["examples"]))
    lines.append(
        f"- **⚠ Item dropped for missing page provenance (REAL TEXT LOSS)**: "
        f"{w['item_without_page_dropped']['count']}"
    )
    lines.append(_fmt_examples(w["item_without_page_dropped"]["examples"]))
    lines.append(f"- **Stanza fallback (BM25 recall only)**: {w['stanza_fallback']['count']}")
    lines.append(_fmt_examples(w["stanza_fallback"]["examples"]))
    lines.append(f"- **OCR empty result** (check for missing scanned/image text): {w['ocr_empty_result']['count']}")
    if w["ocr_empty_result"]["files"]:
        for e in w["ocr_empty_result"]["files"][:10]:
            lines.append(f"  - {e['file']}")
    else:
        lines.append("  (none)")
    lines.append(f"- **Cache-corrupt entries** (auto-recovered by re-processing): {w['cache_corrupt']['count']}")
    if w["misc"]:
        lines.append("- **Other warnings (misc, deduped)**:")
        for m in w["misc"]:
            lines.append(f"  - [{m['logger']}] {m['example']} (×{m['count']})")
    lines.append("")

    if report["canary"] is not None:
        lines.append("## RTL canary")
        lines.append(f"```\n{json.dumps(report['canary'], ensure_ascii=False, indent=2)}\n```")
        lines.append("")

    lines.append("## Embedding usage")
    lines.append(f"- Input tokens this run: {report['embedding_input_tokens']} (cached docs cost 0)")
    lines.append("")

    if previous is not None:
        lines.append("## Comparison to previous run")
        lines.append(f"- Previous run: {previous.get('created_at', '?')}")
        pf, cfl = previous.get("files", {}), report["files"]
        lines.append(f"- Files: {pf.get('total', '?')} → {cfl['total']} (failed {pf.get('failed', '?')} → {cfl['failed']})")
        pc, cc = previous.get("chunks", {}), report["chunks"]
        lines.append(f"- Chunks: {pc.get('total', '?')} → {cc['total']}")
        prev_cats = pc.get("by_category", {})
        cur_cats = cc["by_category"]
        new_cats = sorted(set(cur_cats) - set(prev_cats))
        if new_cats:
            lines.append(f"- New categories since previous run: {', '.join(new_cats)}")
        pw, cw = previous.get("warnings", {}), report["warnings"]
        for key in ("long_paragraph_kept_whole", "empty_chunk_dropped", "item_without_page_dropped", "ocr_empty_result"):
            pv = pw.get(key, {}).get("count", "?")
            cvv = cw[key]["count"]
            lines.append(f"- {key}: {pv} → {cvv}")
        pl, cl = len(previous.get("content_loss_suspected", [])), len(loss)
        lines.append(f"- Content-loss-flagged files: {pl} → {cl}")
        lines.append("")

    return "\n".join(lines)


def render_summary_line(report: dict[str, Any]) -> str:
    """One-line stderr summary for the final ingest stage log."""
    w = report["warnings"]
    loss = len(report["content_loss_suspected"])
    return (
        f"warnings: {w['long_paragraph_kept_whole']['count']} long-paragraph (safe), "
        f"{w['item_without_page_dropped']['count']} page-dropped (REAL LOSS), "
        f"{w['ocr_empty_result']['count']} OCR-empty, {loss} content-loss-flagged files "
        f"— see ingest_report.md"
    )


# --------------------------------------------------------------------------- #
# History persistence (survives the index_dir atomic swap — lives in
# cache_dir, keyed by a sanitized index_dir name)
# --------------------------------------------------------------------------- #


def _history_dir(cache_dir: Path, index_dir: Path) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(index_dir).strip("/"))
    return Path(cache_dir) / "reports" / safe_name


def load_previous(cache_dir: Path, index_dir: Path) -> dict[str, Any] | None:
    latest = _history_dir(cache_dir, index_dir) / "latest.json"
    if not latest.is_file():
        return None
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _ts_slug(report: dict[str, Any]) -> str:
    """Filesystem-safe timestamp from ``report["created_at"]`` (UTC ISO8601),
    shared by the cache-dir history snapshots and the index_dir report files
    so both name the same run identically."""
    return report["created_at"].replace(":", "").replace("+00:00", "Z")


def save_report_history(cache_dir: Path, index_dir: Path, report: dict[str, Any]) -> None:
    hist_dir = _history_dir(cache_dir, index_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts_slug(report)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    (hist_dir / f"{ts}.json").write_text(payload, encoding="utf-8")
    tmp = hist_dir / "latest.json.tmp"
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, hist_dir / "latest.json")


def write_report_files(build_dir: Path, report: dict[str, Any], markdown: str) -> None:
    """Writes both a timestamped, permanent pair (``ingest_report_<ts>.{md,json}``
    -- never overwritten by a later run) and the stable ``ingest_report.{md,json}``
    "latest" pointer that existing log messages/docs reference by name."""
    ts = _ts_slug(report)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    (build_dir / f"ingest_report_{ts}.md").write_text(markdown, encoding="utf-8")
    (build_dir / f"ingest_report_{ts}.json").write_text(payload, encoding="utf-8")
    (build_dir / "ingest_report.md").write_text(markdown, encoding="utf-8")
    (build_dir / "ingest_report.json").write_text(payload, encoding="utf-8")

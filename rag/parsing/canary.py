"""RTL sanity canary (rag_plan.md §5 stage 2) — blocking, first-run.

The project's #1 risk is reversed/jumbled Hebrew from PDF extraction (docling
#1938; AI21's benchmark shows all major parsers degrade on Hebrew). Before
anything is built downstream, a sample of parsed PDFs must show the canary
tokens (ביטוח, הראל, פוליסה) in correct order and their reversals (חוטיב,
לארה, הסילופ) in NO file. A ground-truth anchor check additionally asserts
that apartment/files/הודעה-על-תקופת-התיישנות.pdf page 1 contains התיישנות and
שלוש שנים — which also empirically validates that Docling ``prov.page_no`` is
1-based (plan assumption A1).

Failure semantics: HARD STOP with per-file diagnosis (remediation ladder:
marker-pdf per-file fallback -> BiDi line repair -> human escalation).
"""
from __future__ import annotations

import re
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

from rag.parsing import ParsedDoc

#: Tokens that must appear (in correct letter order) somewhere in the sample.
CANARY_TOKENS: tuple[str, ...] = ("ביטוח", "הראל", "פוליסה")

#: Ground-truth anchor phrases (rel path is tests/conftest ANCHOR_PDF_REL).
ANCHOR_PAGE = 1
ANCHOR_PHRASES: tuple[str, ...] = ("התיישנות", "שלוש שנים")

#: Word-internal characters for boundary purposes: Hebrew letters plus the
#: geresh/gershayim family (״ ׳ and their ASCII stand-ins " ') — acronyms like
#: לארה״ב are single words.
_WORD_CHARS = "א-ת׳״\"'"


def _reversal_pattern(token: str) -> re.Pattern[str]:
    """Reversed token as a standalone Hebrew word.

    Word boundaries matter: reversed ``הראל`` is ``לארה`` which is a substring
    of the legitimate ``לארה״ב`` ("to the USA") — bare substring matching
    would false-positive on travel documents. In genuinely reversed text the
    reversal appears whitespace/punctuation-delimited, so boundary matching
    loses no recall.
    """
    return re.compile(
        f"(?<![{_WORD_CHARS}]){re.escape(token[::-1])}(?![{_WORD_CHARS}])"
    )


_REVERSALS: dict[str, re.Pattern[str]] = {t: _reversal_pattern(t) for t in CANARY_TOKENS}


def iter_docling_texts(doc_dict: dict[str, Any]) -> Iterator[tuple[str, int | None]]:
    """Yield ``(text, page_no)`` for every text item in a DoclingDocument dict.

    Reads the ``export_to_dict()`` JSON directly (no model reconstruction —
    the canary must stay cheap enough to run on every ingest).
    """
    for item in doc_dict.get("texts", []):
        text = item.get("text") or ""
        if not text.strip():
            continue
        prov = item.get("prov") or []
        page_no = prov[0].get("page_no") if prov else None
        yield text, page_no
    # Table cells: canary tokens frequently live in coverage tables.
    for table in doc_dict.get("tables", []):
        prov = table.get("prov") or []
        page_no = prov[0].get("page_no") if prov else None
        cells = (table.get("data") or {}).get("table_cells") or []
        for cell in cells:
            text = cell.get("text") or ""
            if text.strip():
                yield text, page_no


def doc_text(parsed: ParsedDoc) -> str:
    """All extracted text of a ParsedDoc, joined with newlines."""
    if parsed.text is not None:
        return parsed.text
    assert parsed.docling is not None
    return "\n".join(text for text, _ in iter_docling_texts(parsed.docling))


def page_text(parsed: ParsedDoc, page_no: int) -> str:
    """Text of one 1-based page of a parsed PDF."""
    assert parsed.docling is not None, "page_text needs a parsed PDF"
    return "\n".join(
        text for text, page in iter_docling_texts(parsed.docling) if page == page_no
    )


class FileReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rel_path: str
    tokens_found: list[str] = Field(default_factory=list)
    reversals_found: list[str] = Field(default_factory=list, description="Reversed forms detected — RTL corruption")
    ok: bool


class AnchorReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rel_path: str
    page: int
    phrases_found: list[str] = Field(default_factory=list)
    phrases_missing: list[str] = Field(default_factory=list)
    ok: bool


class CanaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    files: list[FileReport]
    missing_tokens: list[str] = Field(
        default_factory=list, description="Canary tokens absent from the ENTIRE sample"
    )
    anchor: AnchorReport | None = None

    def diagnosis(self) -> str:
        """Human-readable per-file diagnosis for the hard-stop message."""
        lines: list[str] = []
        for report in self.files:
            status = "OK" if report.ok else "REVERSED-HEBREW"
            lines.append(
                f"  [{status}] {report.rel_path}: tokens={report.tokens_found or '-'}"
                + (f" reversals={report.reversals_found}" if report.reversals_found else "")
            )
        if self.missing_tokens:
            lines.append(f"  MISSING from entire sample: {self.missing_tokens}")
        if self.anchor is not None:
            status = "OK" if self.anchor.ok else "FAILED"
            lines.append(
                f"  [ANCHOR {status}] {self.anchor.rel_path} p.{self.anchor.page}: "
                f"found={self.anchor.phrases_found or '-'} missing={self.anchor.phrases_missing or '-'}"
            )
        return "\n".join(lines)


class CanaryError(Exception):
    """Raised on canary failure — callers must HARD STOP ingestion."""

    def __init__(self, result: CanaryResult) -> None:
        self.result = result
        super().__init__(
            "RTL canary FAILED — reversed/garbled Hebrew detected; do NOT build "
            "downstream on this text. Remediation ladder: marker-pdf fallback for "
            "failing files -> BiDi line repair -> human escalation.\n"
            + result.diagnosis()
        )


def check_file(parsed: ParsedDoc) -> FileReport:
    text = doc_text(parsed)
    tokens_found = [t for t in CANARY_TOKENS if t in text]
    reversals_found = [t[::-1] for t, pat in _REVERSALS.items() if pat.search(text)]
    return FileReport(
        rel_path=parsed.source.rel_path,
        tokens_found=tokens_found,
        reversals_found=reversals_found,
        ok=not reversals_found,
    )


def check_anchor(parsed: ParsedDoc) -> AnchorReport:
    """Ground-truth anchor: p.1 of the anchor PDF contains the known phrases."""
    text = re.sub(r"\s+", " ", page_text(parsed, ANCHOR_PAGE))
    found = [p for p in ANCHOR_PHRASES if p in text]
    missing = [p for p in ANCHOR_PHRASES if p not in text]
    return AnchorReport(
        rel_path=parsed.source.rel_path,
        page=ANCHOR_PAGE,
        phrases_found=found,
        phrases_missing=missing,
        ok=not missing,
    )


def run_canary(
    parsed_docs: list[ParsedDoc], anchor_doc: ParsedDoc | None = None
) -> CanaryResult:
    """Run the gate over a sample of parsed PDFs (+ optional anchor doc).

    Per-file: reversals must be ABSENT. Aggregate: every canary token must
    appear somewhere in the sample (a single form-like PDF legitimately may
    not contain all three, but a 10-file Harel-insurance sample must).
    Raises :class:`CanaryError` on failure — callers hard-stop.
    """
    files = [check_file(doc) for doc in parsed_docs]
    all_text_tokens = {t for report in files for t in report.tokens_found}
    missing = [t for t in CANARY_TOKENS if t not in all_text_tokens]
    anchor = check_anchor(anchor_doc) if anchor_doc is not None else None
    ok = all(r.ok for r in files) and not missing and (anchor is None or anchor.ok)
    result = CanaryResult(ok=ok, files=files, missing_tokens=missing, anchor=anchor)
    if not ok:
        raise CanaryError(result)
    return result

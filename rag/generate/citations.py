"""Sources-block parsing + citation validation (rag_plan.md §6 stage 7).

Every cited {file, page} MUST exist in the retrieved chunk set — anything else
is a fabricated citation and is dropped.
"""
from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from rag.generate.prompts import SOURCES_HEADER
from rag.types import Citation

if TYPE_CHECKING:
    from rag.types import RetrievedChunk

#: look-alike punctuation an LLM may emit instead of the plain ASCII the
#: prompt asks for (maqaf/en-dash/em-dash for "-", fullwidth pipe/colon) —
#: normalized to the ASCII form before the line regex runs.
_PUNCT_LOOKALIKES = {
    "־": "-",  # maqaf
    "–": "-",  # en dash
    "—": "-",  # em dash
    "｜": "|",  # fullwidth vertical bar
    "：": ":",  # fullwidth colon
}

_LINE_RE = re.compile(
    r'^[-•]\s*file:\s*(?P<file>[^|]+?)\s*\|\s*page:\s*(?P<page>-|null|\d+)\s*'
    r'(?:\|\s*quote:\s*"(?P<quote>[^"]*)")?\s*$',
    re.IGNORECASE | re.MULTILINE,
)


def _strip_bidi_controls(text: str) -> str:
    """Drop invisible bidi/format control chars (LRM, RLM, ALM, embeddings,
    overrides, isolates — Unicode category "Cf") an LLM or terminal may
    sprinkle around RTL punctuation. Purely cosmetic chars — safe to strip."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _normalize_punct(text: str) -> str:
    for lookalike, ascii_form in _PUNCT_LOOKALIKES.items():
        text = text.replace(lookalike, ascii_form)
    return text


def parse_sources_block(text: str) -> list[Citation]:
    """Parse the trailing sources block (rag_plan.md §6 stage 6)::

        מקורות:
        - file: apartment/files/....pdf | page: 1

    Tolerant of bidi control characters and RTL-adjacent punctuation
    look-alikes. NFC-normalizes the parsed ``file`` path. Returns ``[]``
    when the ``מקורות:`` header is absent (missing block)."""
    cleaned = _normalize_punct(_strip_bidi_controls(text))
    header_idx = cleaned.rfind(SOURCES_HEADER)
    if header_idx == -1:
        return []
    block = cleaned[header_idx + len(SOURCES_HEADER):]
    citations: list[Citation] = []
    for match in _LINE_RE.finditer(block):
        raw_file = match.group("file").strip()
        if not raw_file:
            continue
        file = unicodedata.normalize("NFC", raw_file)
        raw_page = match.group("page").strip().lower()
        page = None if raw_page in ("-", "null") else int(raw_page)
        quote = match.group("quote")
        citations.append(Citation(file=file, page=page, quote=quote or None))
    return citations


def validate_citations(
    citations: list[Citation], retrieved: "list[RetrievedChunk]"
) -> list[Citation]:
    """Drop any citation whose ``{file, page}`` is not in the retrieved chunk
    set — the fabrication guard (rag_plan.md §6 stage 7). Invalid citations
    are silently dropped; the caller decides what to do if ALL are dropped
    (retry / fallback)."""
    allowed = {
        (unicodedata.normalize("NFC", r.chunk.file), r.chunk.page) for r in retrieved
    }
    return [
        c
        for c in citations
        if (unicodedata.normalize("NFC", c.file), c.page) in allowed
    ]

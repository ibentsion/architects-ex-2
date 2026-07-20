"""Parsing phase: corpus file -> ParsedDoc (rag_plan.md §5 stages 1-2).

Load-bearing rule (§1.1): downstream chunkers consume the DoclingDocument
dict — NEVER a Markdown export (Markdown destroys ``prov.page_no`` and with
it citation pages).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


class ParsedDoc:
    """Placeholder — full shape ``ParsedDoc{source, pages: …}`` is defined by
    wave E2 (rag_plan.md §5 stage 2). PDFs carry the DoclingDocument dict;
    TXT docs carry the whole text with ``page=None``."""


@runtime_checkable
class Parser(Protocol):
    """Parse one corpus file into a ParsedDoc."""

    def parse(self, path: Path) -> ParsedDoc: ...


def _docling_factory(**params: Any) -> Any:
    from rag.parsing.docling_parser import DoclingParser

    return DoclingParser(**params)


#: marker-pdf is a per-file fallback inside the docling parser's remediation
#: ladder (§5 stage 2), not a global registry alternative.
REGISTRY: dict[str, Callable[..., Any]] = {
    "docling": _docling_factory,
}

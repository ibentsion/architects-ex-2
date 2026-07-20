"""Chunking phase: ParsedDoc -> list[Chunk] (rag_plan.md §5 stage 3).

All strategies consume the DoclingDocument dict — never Markdown (§1.1).
Chunk invariants are documented on ``rag.types.Chunk`` (§8).
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from rag.parsing import ParsedDoc
from rag.types import Chunk


@runtime_checkable
class Chunker(Protocol):
    """Split one ParsedDoc into chunks."""

    def chunk(self, doc: ParsedDoc) -> list[Chunk]: ...


def _per_page_factory(**params: Any) -> Any:
    from rag.chunking.per_page import PerPageChunker

    return PerPageChunker(**params)


def _per_paragraph_factory(**params: Any) -> Any:
    from rag.chunking.per_paragraph import PerParagraphChunker

    return PerParagraphChunker(**params)


def _per_table_factory(**params: Any) -> Any:
    from rag.chunking.per_table import PerTableChunker

    return PerTableChunker(**params)


REGISTRY: dict[str, Callable[..., Any]] = {
    "per_page": _per_page_factory,
    "per_paragraph": _per_paragraph_factory,
    "per_table": _per_table_factory,
}

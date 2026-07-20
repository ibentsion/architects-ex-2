"""per_table chunker: every TableItem -> one ATOMIC Markdown chunk with its
section-heading context prepended; non-table prose falls back to
per_paragraph. Implemented in wave E2 (T3)."""
from __future__ import annotations

from typing import Any


class PerTableChunker:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def chunk(self, doc: Any) -> list[Any]:
        raise NotImplementedError("PerTableChunker is implemented in wave E2 (rag_plan.md T3)")

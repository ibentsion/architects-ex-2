"""per_paragraph chunker: Docling ``HybridChunker(tokenizer=<embedder model>,
merge_peers=True)``; page from the chunk's first ``doc_items[].prov[].page_no``.
Implemented in wave E2 (T3)."""
from __future__ import annotations

from typing import Any


class PerParagraphChunker:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def chunk(self, doc: Any) -> list[Any]:
        raise NotImplementedError("PerParagraphChunker is implemented in wave E2 (rag_plan.md T3)")

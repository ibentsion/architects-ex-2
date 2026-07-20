"""Local sentence-transformers embedder (offline fallback). The e5
``query:``/``passage:`` prefix handling lives HERE, keyed on model name —
call sites never know (rag_plan.md §5 stage 5). Implemented in wave E3 (T5)."""
from __future__ import annotations

from typing import Any


class SentenceTransformersEmbedder:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("SentenceTransformersEmbedder is implemented in wave E3 (rag_plan.md T5)")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("SentenceTransformersEmbedder is implemented in wave E3 (rag_plan.md T5)")

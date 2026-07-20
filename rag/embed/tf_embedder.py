"""Token Factory embedder (DEFAULT): ``litellm.embedding`` against the Nebius
/v1/embeddings endpoint; batching, 429/5xx backoff-retry, query-side
``Instruct:``/``Query:`` framing, ``.env`` key loading (rag_plan.md §5 stage 5).
Implemented in wave E3 (T5)."""
from __future__ import annotations

from typing import Any


class TokenFactoryEmbedder:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("TokenFactoryEmbedder is implemented in wave E3 (rag_plan.md T5)")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("TokenFactoryEmbedder is implemented in wave E3 (rag_plan.md T5)")

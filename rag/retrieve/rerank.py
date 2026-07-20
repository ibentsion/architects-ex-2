"""CrossEncoder reranking + relevance gate (rag_plan.md §6 stage 4).

Runs locally on CPU — Token Factory serves no reranker model (verified
2026-07-20). Implemented in wave E4 (T7)."""
from __future__ import annotations

from typing import Any


class BgeReranker:
    """BAAI/bge-reranker-v2-m3 CrossEncoder (Apache-2.0, local CPU)."""

    def __init__(self, **params: Any) -> None:
        self.params = params

    def score(self, question: str, candidates: list[Any]) -> list[Any]:
        raise NotImplementedError("BgeReranker is implemented in wave E4 (rag_plan.md T7)")


class JinaReranker:
    """jina-reranker-v2 CrossEncoder — CC-BY-NC license, non-commercial only."""

    def __init__(self, **params: Any) -> None:
        self.params = params

    def score(self, question: str, candidates: list[Any]) -> list[Any]:
        raise NotImplementedError("JinaReranker is implemented in wave E4 (rag_plan.md T7)")

"""Retrieval orchestration: dense + sparse search -> RRF fuse -> rerank ->
relevance gate (rag_plan.md §6 stages 2-4). Implemented in wave E4 (T7)."""
from __future__ import annotations

from typing import Any


class Retriever:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def retrieve(self, question: str, category: str | None = None) -> list[Any]:
        raise NotImplementedError("Retriever is implemented in wave E4 (rag_plan.md T7)")

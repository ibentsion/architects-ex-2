"""Retrieval phase: dense+sparse -> RRF fusion -> CrossEncoder rerank ->
relevance gate (rag_plan.md §6 stages 2-4).

Gate fail (zero survivors above ``gate_threshold``) skips generation entirely
and returns the Hebrew "not enough information" fallback — zero LLM cost.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from rag.types import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Score (question, chunk_text) pairs; sigmoid score doubles as the
    relevance-gate signal. (Method set is provisional until wave E4/T7.)"""

    def score(self, question: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


def _bge_factory(**params: Any) -> Any:
    from rag.retrieve.rerank import BgeReranker

    return BgeReranker(**params)


def _jina_factory(**params: Any) -> Any:
    from rag.retrieve.rerank import JinaReranker

    return JinaReranker(**params)


RERANK_REGISTRY: dict[str, Callable[..., Any]] = {
    "bge": _bge_factory,
    # WARNING: jina-reranker-v2 is CC-BY-NC — non-commercial use only.
    "jina": _jina_factory,
}

"""Embedding phase (rag_plan.md §5 stage 5).

Default impl is 'tokenfactory' (Nebius Token Factory /v1/embeddings via
litellm — no local embedding compute). 'sentence_transformers' is the fully
local fallback; model-specific prefix logic (e5 query:/passage:) lives inside
that adapter — call sites never know.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Embed documents (plain) and queries (instruct-framed where the model
    supports it). Vector space identity (model + dimensions) is stamped into
    the index manifest — query must match ingest."""

    def embed_docs(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _tokenfactory_factory(**params: Any) -> Any:
    from rag.embed.tf_embedder import TokenFactoryEmbedder

    return TokenFactoryEmbedder(**params)


def _sentence_transformers_factory(**params: Any) -> Any:
    from rag.embed.st_embedder import SentenceTransformersEmbedder

    return SentenceTransformersEmbedder(**params)


REGISTRY: dict[str, Callable[..., Any]] = {
    "tokenfactory": _tokenfactory_factory,
    "sentence_transformers": _sentence_transformers_factory,
}

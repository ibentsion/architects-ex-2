"""Normalization phase: text -> token list, for SPARSE indexing only
(rag_plan.md §5 stage 4).

Normalized tokens exist only inside the sparse index — displayed/stored chunk
text is always the original. The same Normalizer instance/recipe MUST be used
at ingest and query time (asymmetry silently zeroes BM25 recall, §6 stage 1).
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Normalizer(Protocol):
    """Tokenize + normalize text for BM25 (lemma+surface union recipe)."""

    def tokens(self, text: str) -> list[str]: ...


def _stanza_factory(**params: Any) -> Any:
    from rag.normalize.stanza_norm import StanzaNormalizer

    return StanzaNormalizer(**params)


def _trankit_factory(**params: Any) -> Any:
    # Import raises ImportError with a pip hint when trankit is not installed.
    from rag.normalize.trankit_norm import TrankitNormalizer

    return TrankitNormalizer(**params)


def _yap_factory(**params: Any) -> Any:
    # Import raises ImportError with setup instructions (YAP is not a pip package).
    from rag.normalize.yap_norm import YapNormalizer

    return YapNormalizer(**params)


REGISTRY: dict[str, Callable[..., Any]] = {
    "stanza": _stanza_factory,
    "trankit": _trankit_factory,
    "yap": _yap_factory,
}

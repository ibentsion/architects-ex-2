"""Sparse (keyword/BM25) index backends (rag_plan.md §5 stage 6).

Default: in-process ``bm25s`` over Stanza lemma token lists, persisted mmap at
``<index_dir>/bm25``. Elasticsearch/OpenSearch (Docker + Hebrew analyzer) are
optional-dependency alternatives.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class KeywordIndex(Protocol):
    """Sparse retrieval backend over pre-normalized token lists, joined to the
    dense side on ``chunk_id``. (Method set is provisional until wave E3/T5.)"""

    def add(self, chunk_ids: list[str], token_lists: list[list[str]]) -> None: ...

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[str, float]]: ...


class Bm25sIndex:
    """In-process ``bm25s.BM25()`` over pre-tokenized input; ``save()`` mmap
    persistence at ``<index_dir>/bm25``. Implemented in wave E3 (T5)."""

    def __init__(self, **params: Any) -> None:
        self.params = params

    def add(self, chunk_ids: list[str], token_lists: list[list[str]]) -> None:
        raise NotImplementedError("Bm25sIndex is implemented in wave E3 (rag_plan.md T5)")

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[str, float]]:
        raise NotImplementedError("Bm25sIndex is implemented in wave E3 (rag_plan.md T5)")


def _bm25s_factory(**params: Any) -> Any:
    return Bm25sIndex(**params)


def _elasticsearch_factory(**params: Any) -> Any:
    try:
        import elasticsearch  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "sparse_index impl 'elasticsearch' requires the optional dependency "
            "'elasticsearch', which is not installed.\n"
            "Install it with: pip install elasticsearch\n"
            "(and run an Elasticsearch server with a Hebrew analyzer, e.g. via Docker; "
            "or keep the default: sparse_index: {impl: bm25s})"
        ) from err
    raise NotImplementedError("Elasticsearch KeywordIndex adapter is planned (rag_plan.md §2) but not implemented yet")


def _opensearch_factory(**params: Any) -> Any:
    try:
        import opensearchpy  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "sparse_index impl 'opensearch' requires the optional dependency "
            "'opensearch-py', which is not installed.\n"
            "Install it with: pip install opensearch-py\n"
            "(and run an OpenSearch server with a Hebrew analyzer, e.g. via Docker; "
            "or keep the default: sparse_index: {impl: bm25s})"
        ) from err
    raise NotImplementedError("OpenSearch KeywordIndex adapter is planned (rag_plan.md §2) but not implemented yet")


SPARSE_REGISTRY: dict[str, Callable[..., Any]] = {
    "bm25s": _bm25s_factory,
    "elasticsearch": _elasticsearch_factory,
    "opensearch": _opensearch_factory,
}

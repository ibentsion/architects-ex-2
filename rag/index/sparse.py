"""Sparse (keyword/BM25) index backends (rag_plan.md §5 stage 6).

Default: in-process ``bm25s`` over pre-normalized (Stanza lemma+surface)
token lists, persisted at ``<index_dir>/bm25`` with mmap reload at query
time. Chunk ids are stored alongside so fusion joins dense and sparse
results on a single ``chunk_id`` namespace. Elasticsearch/OpenSearch
(Docker + Hebrew analyzer) are optional-dependency alternatives.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

CHUNK_IDS_FILENAME = "chunk_ids.json"


@runtime_checkable
class KeywordIndex(Protocol):
    """Sparse retrieval backend over pre-normalized token lists, joined to
    the dense side on ``chunk_id``."""

    def add(self, chunk_ids: list[str], token_lists: list[list[str]]) -> None: ...

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[str, float]]: ...


class Bm25sIndex:
    """In-process ``bm25s.BM25()`` over pre-tokenized input. ``save``/``load``
    persist to a directory (``<index_dir>/bm25``) with mmap reload."""

    def __init__(self, **params: Any) -> None:
        if params:
            raise TypeError(f"Unknown bm25s params: {sorted(params)}")
        self._retriever: Any = None
        self._chunk_ids: list[str] = []

    def add(self, chunk_ids: list[str], token_lists: list[list[str]]) -> None:
        """Build the index over the full corpus token lists. bm25s builds its
        scoring matrix in one pass — call once with everything (a second call
        replaces the index)."""
        import bm25s

        if len(chunk_ids) != len(token_lists):
            raise ValueError(f"{len(chunk_ids)} chunk_ids but {len(token_lists)} token lists")
        retriever = bm25s.BM25()
        retriever.index(token_lists, show_progress=False)
        self._retriever = retriever
        self._chunk_ids = list(chunk_ids)

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[str, float]]:
        """Top-k ``(chunk_id, score)`` pairs. Query tokens unknown to the
        index vocabulary are dropped (bm25s KeyErrors on them); zero-score
        results are filtered out."""
        if self._retriever is None:
            raise RuntimeError("Bm25sIndex is empty — call add() or load() first")
        vocab = self._retriever.vocab_dict or {}
        known = [token for token in query_tokens if token in vocab]
        if not known:
            return []
        k = min(top_k, len(self._chunk_ids))
        indices, scores = self._retriever.retrieve([known], k=k, show_progress=False)
        return [
            (self._chunk_ids[int(idx)], float(score))
            for idx, score in zip(indices[0], scores[0])
            if score > 0.0
        ]

    # ------------------------------------------------------------------ #
    # Persistence (mmap reload at query time)
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        if self._retriever is None:
            raise RuntimeError("Bm25sIndex is empty — nothing to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(path), corpus=None)
        (path / CHUNK_IDS_FILENAME).write_text(
            json.dumps(self._chunk_ids, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, mmap: bool = True) -> "Bm25sIndex":
        import bm25s

        path = Path(path)
        ids_path = path / CHUNK_IDS_FILENAME
        if not ids_path.is_file():
            raise FileNotFoundError(
                f"BM25 index at {path} is missing {CHUNK_IDS_FILENAME} — re-ingest."
            )
        index = cls()
        index._retriever = bm25s.BM25.load(str(path), mmap=mmap, load_vocab=True)
        index._chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        return index


def _bm25s_factory(**params: Any) -> Bm25sIndex:
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

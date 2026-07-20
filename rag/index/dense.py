"""Dense (vector) index backends (rag_plan.md §5 stage 6).

Default: Qdrant local mode (``QdrantClient(path=…)`` — single-process lock,
ingest and query cannot open it concurrently). Server mode is the same code
path with ``url=``. Chroma/Milvus are optional-dependency alternatives.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from rag.types import Chunk


@runtime_checkable
class VectorIndex(Protocol):
    """Dense retrieval backend. Payload carries full chunk metadata; category
    is a filterable field. (Method set is provisional until wave E3/T5.)"""

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self, vector: list[float], top_k: int, category: str | None = None
    ) -> list[tuple[Chunk, float]]: ...


class QdrantLocalIndex:
    """Qdrant local mode: ``QdrantClient(path=<index_dir>/qdrant)``.
    Implemented in wave E3 (T5)."""

    def __init__(self, **params: Any) -> None:
        self.params = params

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError("QdrantLocalIndex is implemented in wave E3 (rag_plan.md T5)")

    def search(
        self, vector: list[float], top_k: int, category: str | None = None
    ) -> list[tuple[Chunk, float]]:
        raise NotImplementedError("QdrantLocalIndex is implemented in wave E3 (rag_plan.md T5)")


class QdrantServerIndex(QdrantLocalIndex):
    """Qdrant server mode: same code path with ``url=`` instead of ``path=``
    (lifts the local single-process lock). Implemented in wave E3 (T5)."""


def _qdrant_local_factory(**params: Any) -> Any:
    return QdrantLocalIndex(**params)


def _qdrant_server_factory(**params: Any) -> Any:
    return QdrantServerIndex(**params)


def _chroma_factory(**params: Any) -> Any:
    try:
        import chromadb  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "dense_index impl 'chroma' requires the optional dependency "
            "'chromadb', which is not installed.\n"
            "Install it with: pip install chromadb\n"
            "(or keep the default: dense_index: {impl: qdrant_local})"
        ) from err
    raise NotImplementedError("Chroma VectorIndex adapter is planned (rag_plan.md §2) but not implemented yet")


def _milvus_factory(**params: Any) -> Any:
    try:
        import pymilvus  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "dense_index impl 'milvus' requires the optional dependency "
            "'pymilvus', which is not installed.\n"
            "Install it with: pip install pymilvus\n"
            "(or keep the default: dense_index: {impl: qdrant_local})"
        ) from err
    raise NotImplementedError("Milvus VectorIndex adapter is planned (rag_plan.md §2) but not implemented yet")


DENSE_REGISTRY: dict[str, Callable[..., Any]] = {
    "qdrant_local": _qdrant_local_factory,
    "qdrant_server": _qdrant_server_factory,
    "chroma": _chroma_factory,
    "milvus": _milvus_factory,
}

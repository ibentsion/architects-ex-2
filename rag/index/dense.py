"""Dense (vector) index backends (rag_plan.md §5 stage 6).

Default: Qdrant local mode (``QdrantClient(path=…)`` — single-process lock,
ingest and query cannot open it concurrently). Server mode is the same code
path with ``url=``. Collection ``chunks``; payload = full chunk metadata with
a payload index on ``category`` (efficient filtered HNSW); cosine distance.
Chroma/Milvus are optional-dependency alternatives (ImportError with pip
hint when selected uninstalled).
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from rag.types import Chunk

COLLECTION = "chunks"


@runtime_checkable
class VectorIndex(Protocol):
    """Dense retrieval backend. Payload carries full chunk metadata; category
    is a filterable field."""

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self, vector: list[float], top_k: int, category: str | list[str] | None = None
    ) -> list[tuple[Chunk, float]]: ...


def point_id(chunk_id: str) -> str:
    """Qdrant point ids must be ints or UUIDs — derive a stable UUID5 from
    the chunk_id (the cross-index join key stays ``chunk_id`` in payload)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantIndex:
    """Qdrant adapter; local mode (``path=``) and server mode (``url=``) are
    the same code path (locked requirement). Collection is created lazily on
    first ``add`` (vector size inferred), cosine distance."""

    def __init__(
        self,
        path: str | Path | None = None,
        url: str | None = None,
        collection: str = COLLECTION,
        **params: Any,
    ) -> None:
        if (path is None) == (url is None):
            raise ValueError("QdrantIndex needs exactly one of path= (local) or url= (server)")
        if params:
            raise TypeError(f"Unknown qdrant params: {sorted(params)}")
        from qdrant_client import QdrantClient

        self.collection = collection
        self._client = (
            QdrantClient(path=str(path)) if path is not None else QdrantClient(url=url)
        )

    # ------------------------------------------------------------------ #

    def _ensure_collection(self, dimensions: int) -> None:
        from qdrant_client import models

        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dimensions, distance=models.Distance.COSINE
                ),
            )
            # Local mode warns that payload indexes are a no-op (filtering
            # still works, brute-force); server mode uses the index. Create it
            # unconditionally so the code path is identical (locked req).
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Payload indexes have no effect in the local Qdrant"
                )
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name="category",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client import models

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        self._ensure_collection(len(vectors[0]))
        self._client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=point_id(chunk.chunk_id),
                    vector=[float(x) for x in vector],
                    payload=chunk.model_dump(),  # full chunk metadata
                )
                for chunk, vector in zip(chunks, vectors)
            ],
        )

    def search(
        self, vector: list[float], top_k: int, category: str | list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """``category`` filters the payload index: one name, or a set of names
        (MatchAny) when the classifier can only narrow the query to a family
        rather than to a single corpus directory."""
        from qdrant_client import models

        query_filter = None
        if category:
            names = [category] if isinstance(category, str) else list(category)
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key="category", match=models.MatchAny(any=names))
                ]
            )
        response = self._client.query_points(
            collection_name=self.collection,
            query=[float(x) for x in vector],
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            (Chunk.model_validate(point.payload), point.score)
            for point in response.points
        ]

    def fetch(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """Fetch chunk metadata by chunk_id (the payload is the single chunk
        metadata store — the sparse index returns bare chunk_ids, and fusion
        survivors that came only from BM25 need their Chunk hydrated)."""
        if not chunk_ids:
            return {}
        records = self._client.retrieve(
            collection_name=self.collection,
            ids=[point_id(chunk_id) for chunk_id in chunk_ids],
            with_payload=True,
        )
        chunks = (Chunk.model_validate(record.payload) for record in records)
        return {chunk.chunk_id: chunk for chunk in chunks}

    def close(self) -> None:
        """Release the client (local mode: frees the single-process lock)."""
        self._client.close()


def _qdrant_local_factory(**params: Any) -> QdrantIndex:
    if "path" not in params:
        raise ValueError(
            "dense_index impl 'qdrant_local' needs a 'path' param "
            "(the ingest/query CLIs derive it as <index_dir>/qdrant)"
        )
    return QdrantIndex(**params)


def _qdrant_server_factory(**params: Any) -> QdrantIndex:
    if "url" not in params:
        raise ValueError("dense_index impl 'qdrant_server' needs a 'url' param")
    return QdrantIndex(**params)


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

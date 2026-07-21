"""Read-through per-doc embedding cache (rag_plan.md §5 stage 5).

``<cache_dir>/embeddings/<sha256>.<chunker_id>.<embedder_id>-<dims>.npz``
stores one document's chunk vectors (float32), keyed by file content hash +
chunker identity + embedder identity + dimensions. Unchanged files never
re-embed on later ingests (API or local); a genuine embedder/dims/chunker
change misses by key, which is exactly right.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from rag.report import EVENT_CACHE_CORRUPT

logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, cache_dir: Path) -> None:
        self.embeddings_dir = Path(cache_dir) / "embeddings"

    def path_for(self, sha256: str, chunker_id: str, embedder_key: str) -> Path:
        """``embedder_key`` is ``"<embedder_id>-<dims>"`` (see ``embedder_cache_key``)."""
        return self.embeddings_dir / f"{sha256}.{chunker_id}.{embedder_key}.npz"

    def load(self, sha256: str, chunker_id: str, embedder_key: str) -> np.ndarray | None:
        path = self.path_for(sha256, chunker_id, embedder_key)
        if not path.is_file():
            return None
        try:
            with np.load(path) as data:
                return data["vectors"]
        except (OSError, KeyError, ValueError) as exc:
            logger.warning(
                "Corrupt embedding-cache entry %s (%s) — re-embedding",
                path,
                exc,
                extra={"rag_event": EVENT_CACHE_CORRUPT, "rag_detail": {"path": str(path), "cache": "embeddings"}},
            )
            return None

    def store(
        self, sha256: str, chunker_id: str, embedder_key: str, vectors: np.ndarray
    ) -> None:
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(sha256, chunker_id, embedder_key)
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, vectors=vectors.astype(np.float32))
        os.replace(tmp, path)  # atomic: a crashed write never poisons the cache


def embedder_cache_key(embedder_id: str, dimensions: int) -> str:
    """Cache-key component per §5: ``<embedder_id>-<dims>``."""
    return f"{embedder_id}-{dimensions}"


def embed_doc(
    embedder: Any,
    cache: EmbeddingCache,
    *,
    sha256: str,
    chunker_id: str,
    embedder_key: str,
    texts: list[str],
) -> np.ndarray:
    """Read-through: cache hit (with matching row count) skips the embedder
    (and thus the API) entirely; on miss, embed and cache."""
    cached = cache.load(sha256, chunker_id, embedder_key)
    if cached is not None and cached.shape[0] == len(texts):
        logger.debug("Embedding-cache hit: %s.%s.%s", sha256[:12], chunker_id, embedder_key)
        return cached
    vectors = np.asarray(embedder.embed_docs(texts), dtype=np.float32)
    cache.store(sha256, chunker_id, embedder_key, vectors)
    return vectors

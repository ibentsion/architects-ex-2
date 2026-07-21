"""Retrieval orchestration (rag_plan.md §6 stages 2-4).

``Retriever.retrieve(question, category=None)``:
dense top-k (Qdrant payload filter on category) + sparse top-k (bm25s; with a
category filter, fetch 3x and post-filter) -> RRF fusion joined on chunk_id
-> CrossEncoder rerank -> relevance gate. Returns ``RetrievedChunk``s with
all per-stage scores populated; empty list on gate fail (the caller must
skip generation and return the Hebrew fallback).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag.retrieve.fusion import rrf
from rag.retrieve.rerank import apply_gate
from rag.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


def _category_of(chunk_id: str) -> str:
    """§8 invariant: ``chunk_id`` starts with ``"{category}/…"`` — the sparse
    post-filter needs no metadata fetch."""
    return chunk_id.split("/", 1)[0]


class Retriever:
    def __init__(
        self,
        *,
        embedder: Any,
        normalizer: Any,
        dense: Any,
        sparse: Any,
        reranker: Any,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        rrf_k: int = 60,
        gate_threshold: float = 0.35,
        top_n: int = 6,
    ) -> None:
        self.embedder = embedder
        self.normalizer = normalizer
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.gate_threshold = gate_threshold
        self.top_n = top_n

    # ------------------------------------------------------------------ #

    def _dense_search(
        self, question: str, category: str | None
    ) -> list[tuple[Chunk, float]]:
        vector = self.embedder.embed_query(question)
        return self.dense.search(vector, top_k=self.dense_top_k, category=category)

    def _sparse_search(self, question: str, category: str | None) -> list[tuple[str, float]]:
        """With a category filter, fetch 3x and post-filter by the chunk_id's
        category prefix (rag_plan.md §6 stage 2 — small corpus, cheap)."""
        query_tokens = self.normalizer.tokens(question)
        fetch_k = self.sparse_top_k * 3 if category else self.sparse_top_k
        hits = self.sparse.search(query_tokens, top_k=fetch_k)
        if category is not None:
            hits = [(cid, score) for cid, score in hits if _category_of(cid) == category]
        return hits[: self.sparse_top_k]

    def retrieve(self, question: str, category: str | None = None) -> list[RetrievedChunk]:
        # Stage 2 — dual search.
        dense_hits = self._dense_search(question, category)
        sparse_hits = self._sparse_search(question, category)
        if not dense_hits and not sparse_hits:
            return []
        dense_scores = {chunk.chunk_id: score for chunk, score in dense_hits}
        sparse_scores = dict(sparse_hits)
        chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c, _ in dense_hits}

        # Stage 3 — RRF fusion, joined on chunk_id; top max(k) candidates.
        fused = rrf(
            [
                [chunk.chunk_id for chunk, _ in dense_hits],
                [chunk_id for chunk_id, _ in sparse_hits],
            ],
            k=self.rrf_k,
        )[: max(self.dense_top_k, self.sparse_top_k)]

        # Hydrate sparse-only survivors from the dense payload store.
        missing = [cid for cid, _ in fused if cid not in chunks_by_id]
        if missing:
            chunks_by_id.update(self.dense.fetch(missing))
        candidates: list[RetrievedChunk] = []
        for chunk_id, rrf_score in fused:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:  # sparse id absent from dense store — index skew
                logger.warning("Fused chunk_id missing from dense payloads: %s", chunk_id)
                continue
            candidates.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=dense_scores.get(chunk_id),
                    sparse_score=sparse_scores.get(chunk_id),
                    rrf_score=rrf_score,
                )
            )

        # Stage 4 — rerank + relevance gate (empty list = gate fail).
        scored = self.reranker.score(question, candidates)
        return apply_gate(scored, self.gate_threshold, self.top_n)

    def close(self) -> None:
        """Release backend resources (Qdrant local single-process lock)."""
        close = getattr(self.dense, "close", None)
        if close is not None:
            close()


# --------------------------------------------------------------------------- #
# Assembly from a validated config + ingested index (query CLI entry, §6 st. 0)
# --------------------------------------------------------------------------- #


def load_retriever(config: Any) -> Retriever:
    """Open an ingested index for querying: verify the manifest (embedder/
    chunker/normalizer identity must match), then load all backends. Model
    loads are lazy — the first query is the slow one."""
    from rag.config import ConfigError, build, get_registry
    from rag.index.manifest import verify_manifest

    index_dir = Path(config.index_dir)
    verify_manifest(index_dir, config)  # raises with a re-ingest hint on mismatch

    dense_impl = config.dense_index.impl
    dense_registry = get_registry("dense_index")
    if dense_impl not in dense_registry:
        raise ConfigError(
            f"Unknown dense_index impl '{dense_impl}'. Available: {sorted(dense_registry)}."
        )
    dense_params = dict(config.dense_index.params)
    if dense_impl == "qdrant_local":
        dense_params.setdefault("path", str(index_dir / "qdrant"))
    dense = dense_registry[dense_impl](**dense_params)

    if config.sparse_index.impl == "bm25s":
        from rag.index.sparse import Bm25sIndex

        sparse = Bm25sIndex.load(index_dir / "bm25", mmap=True)
    else:  # server-backed sparse impls hold no local state to load
        sparse = build("sparse_index", config)

    retrieval = config.retrieval
    return Retriever(
        embedder=build("embedder", config),
        normalizer=build("normalizer", config),
        dense=dense,
        sparse=sparse,
        reranker=build("reranker", config),
        dense_top_k=retrieval.dense_top_k,
        sparse_top_k=retrieval.sparse_top_k,
        rrf_k=retrieval.rrf_k,
        gate_threshold=retrieval.rerank.gate_threshold,
        top_n=retrieval.rerank.top_n,
    )

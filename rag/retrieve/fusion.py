"""Reciprocal-rank fusion (rag_plan.md §6 stage 3).

``score(c) = Σ_r 1/(rrf_k + rank_r(c))`` over the dense and sparse rankings,
joined on ``chunk_id``. Rank-based — no cross-backend score calibration.
"""
from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse rankings of chunk_ids into ``[(chunk_id, rrf_score), …]``,
    best-first.

    ``rankings`` are best-first chunk_id lists (ranks are 1-based). A chunk
    appearing in several rankings sums its reciprocal ranks — an item ranked
    #1 in both lists always beats any single-list item. Deterministic
    tie-break: equal scores order by ``chunk_id`` ascending.
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

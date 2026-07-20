"""Reciprocal-rank fusion: ``score(c) = Σ_r 1/(rrf_k + rank_r(c))`` over the
dense and sparse rankings, joined on ``chunk_id`` (rag_plan.md §6 stage 3).
Implemented in wave E4 (T7)."""
from __future__ import annotations

from typing import Any


def rrf(rankings: list[list[Any]], k: int = 60) -> list[Any]:
    raise NotImplementedError("RRF fusion is implemented in wave E4 (rag_plan.md T7)")

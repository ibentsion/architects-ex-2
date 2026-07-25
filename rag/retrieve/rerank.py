"""CrossEncoder reranking + relevance gate (rag_plan.md §6 stage 4).

Runs locally on CPU — Token Factory serves no reranker model (verified
2026-07-20). ``max_length=512`` truncates query + chunk head first (Hebrew
page chunks are long). The sigmoid score doubles as the relevance-gate
signal: zero survivors above ``gate_threshold`` → the caller skips
generation entirely and returns the "not enough information" fallback.

CPU cost is ~1.7 s/pair (re-measured post-T7 on this machine -- the original
"~1-3s for 20 pairs" T7 estimate was stale/optimistic) -- ~18-34s for the
default 20-candidate pool, and it DOMINATES total query latency regardless
of chunker choice (candidate count is fixed by dense_top_k/sparse_top_k, not
by chunk granularity). No config fix available on this hardware short of
fewer candidates, a smaller/faster reranker, or GPU. The first ``score``
call lazily downloads + loads the model (~2.2 GB for bge-reranker-v2-m3,
one-time).
"""
from __future__ import annotations

import logging
from typing import Any

from rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class _CrossEncoderReranker:
    """Shared sentence-transformers CrossEncoder adapter: lazy model load,
    explicit sigmoid activation, score() returns candidates (input order)
    with ``rerank_score`` populated."""

    def __init__(
        self,
        model: str,
        max_length: int = 512,
        trust_remote_code: bool = False,
        **params: Any,
    ) -> None:
        if params:
            raise TypeError(f"Unknown reranker params: {sorted(params)}")
        self.model_name = model
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading CrossEncoder %s (max_length=%d) — first ever use downloads the model",
                self.model_name,
                self.max_length,
            )
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                trust_remote_code=self.trust_remote_code,
            )
        return self._model

    def score(
        self, question: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Score ``(question, chunk_text)`` pairs; sigmoid forced explicitly
        (bge-reranker outputs a single logit — sigmoid maps it to [0,1] so the
        gate threshold is calibration-free)."""
        if not candidates:
            return []
        import torch

        model = self._get_model()
        pairs = [(question, c.chunk.text) for c in candidates]
        scores = model.predict(
            pairs,
            activation_fn=torch.nn.Sigmoid(),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [
            candidate.model_copy(update={"rerank_score": float(score)})
            for candidate, score in zip(candidates, scores)
        ]


class BgeReranker(_CrossEncoderReranker):
    """BAAI/bge-reranker-v2-m3 CrossEncoder (Apache-2.0, local CPU)."""

    def __init__(
        self, model: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512, **params: Any
    ) -> None:
        super().__init__(model=model, max_length=max_length, **params)


class JinaReranker(_CrossEncoderReranker):
    """jina-reranker-v2 CrossEncoder — CC-BY-NC license, non-commercial only."""

    def __init__(
        self,
        model: str = "jinaai/jina-reranker-v2-base-multilingual",
        max_length: int = 512,
        **params: Any,
    ) -> None:
        logger.warning(
            "jina-reranker-v2 is CC-BY-NC licensed — NON-COMMERCIAL use only."
        )
        super().__init__(
            model=model, max_length=max_length, trust_remote_code=True, **params
        )


def apply_gate(
    candidates: list[RetrievedChunk], gate_threshold: float, top_n: int
) -> list[RetrievedChunk]:
    """Relevance gate: keep candidates with ``rerank_score >= gate_threshold``
    (unscored candidates fail), sorted by rerank score desc (tie-break
    chunk_id asc), truncated to ``top_n``. Empty list = gate fail — the
    caller must skip generation (rag_plan.md §6 stage 4)."""
    survivors = [
        c
        for c in candidates
        if c.rerank_score is not None and c.rerank_score >= gate_threshold
    ]
    survivors.sort(key=lambda c: (-(c.rerank_score or 0.0), c.chunk.chunk_id))
    return survivors[:top_n]

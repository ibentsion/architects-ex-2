"""Local sentence-transformers embedder (fully offline fallback —
rag_plan.md §5 stage 5).

Model-specific prefix logic lives HERE, keyed on model name — call sites
never know: e5-family models require ``query:`` / ``passage:`` prefixes
(retrieval quality collapses without them). The model loads lazily (first
call), keeping import + construction cheap on a RAM-tight machine.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower().split("/")[-1]


class SentenceTransformersEmbedder:
    """Satisfies the ``Embedder`` protocol. Vector-space identity =
    (provider='sentence_transformers', model, dimensions)."""

    provider = "sentence_transformers"

    def __init__(
        self,
        model: str,
        batch_size: int = 16,
        device: str | None = None,
        **params: Any,
    ) -> None:
        if params:
            raise TypeError(f"Unknown sentence_transformers embedder params: {sorted(params)}")
        self.model_name = model
        self.batch_size = batch_size
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model: Any = None
        # e5 prefix handling (rag_plan.md: lives inside this adapter)
        self._query_prefix = "query: " if _is_e5(model) else ""
        self._doc_prefix = "passage: " if _is_e5(model) else ""

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers model %s (lazy)", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimensions(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # cosine-ready unit vectors
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    # ------------------------------------------------------------------ #
    # Embedder protocol
    # ------------------------------------------------------------------ #

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"{self._doc_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self._query_prefix}{text}"])[0]

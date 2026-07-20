"""Token Factory embedder (DEFAULT — rag_plan.md §5 stage 5).

``litellm.embedding`` against the Nebius Token Factory /v1/embeddings
endpoint (same env vars as ``tf_client``): batching, exponential backoff on
429/5xx (3 attempts), query-side ``Instruct:``/``Query:`` framing, cumulative
input-token usage logging (shared-key etiquette), ``.env`` key loading via
python-dotenv. Documents are sent plain; the instruct prefix applies only to
``embed_query``.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import litellm
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

#: HTTP status codes worth a backoff-retry (rate limit + server-side).
_RETRYABLE = lambda status: status == 429 or (status is not None and status >= 500)  # noqa: E731


class TokenFactoryEmbedder:
    """Satisfies the ``Embedder`` protocol. Vector-space identity =
    (provider='tokenfactory', model, dimensions) — stamped into the index
    manifest; query must match ingest."""

    provider = "tokenfactory"

    def __init__(
        self,
        model: str,
        dimensions: int,
        batch_size: int = 64,
        query_instruct: str = "",
        api_base: str | None = None,
        max_attempts: int = 3,
        backoff_base: float = 1.0,
        **params: Any,
    ) -> None:
        if params:
            raise TypeError(f"Unknown tokenfactory embedder params: {sorted(params)}")
        load_dotenv()
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.query_instruct = query_instruct
        self.api_base = api_base or os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL)
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.total_input_tokens = 0  # cumulative usage this run (logged per call)

    # ------------------------------------------------------------------ #

    def _api_key(self) -> str:
        key = os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError(
                "NEBIUS_API_KEY not set — put the course key in .env "
                "(loaded via python-dotenv) or export it."
            )
        return key

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """One /v1/embeddings request with backoff-retry on 429/5xx."""
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = litellm.embedding(
                    model=f"openai/{self.model}",
                    input=texts,
                    api_base=self.api_base,
                    api_key=self._api_key(),
                    dimensions=self.dimensions,
                    # litellm gates `dimensions` to text-embedding-3* on the
                    # openai provider; Token Factory's Qwen3-Embedding honors
                    # it (MRL) — allow it through explicitly.
                    allowed_openai_params=["dimensions"],
                )
                usage = getattr(response, "usage", None)
                if usage is not None and getattr(usage, "prompt_tokens", None):
                    self.total_input_tokens += usage.prompt_tokens
                    logger.info(
                        "Embedded %d texts (%d input tokens; %d cumulative this run)",
                        len(texts), usage.prompt_tokens, self.total_input_tokens,
                    )
                rows = sorted(response.data, key=lambda item: item["index"])
                vectors = [row["embedding"] for row in rows]
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"Embedding API returned {len(vectors)} vectors for {len(texts)} texts"
                    )
                return vectors
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if not _RETRYABLE(status) or attempt == self.max_attempts - 1:
                    raise
                delay = self.backoff_base * (2 ** attempt)
                logger.warning(
                    "Embedding request failed (HTTP %s, attempt %d/%d) — retrying in %.1fs: %s",
                    status, attempt + 1, self.max_attempts, delay, exc,
                )
                last_exc = exc
                time.sleep(delay)
        raise last_exc  # pragma: no cover — loop always returns or raises

    # ------------------------------------------------------------------ #
    # Embedder protocol
    # ------------------------------------------------------------------ #

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        """Documents are sent PLAIN (no instruct prefix), batch_size per request."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Qwen3-Embedding is instruction-aware: queries are framed as
        ``Instruct: {query_instruct}\\nQuery: {q}``; skipping costs ~1-5%
        retrieval quality."""
        framed = (
            f"Instruct: {self.query_instruct}\nQuery: {text}"
            if self.query_instruct
            else text
        )
        return self._embed_batch([framed])[0]

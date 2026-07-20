"""Read-through per-doc token cache (rag_plan.md §5 stage 4).

``<cache_dir>/tokens/<sha256>.<chunker_id>.<normalizer_id>.json`` stores the
normalized token lists for one document's chunks, keyed by file content hash +
the identity of the chunker/normalizer config that produced them (ids from
``rag.config.impl_id``). An embedder swap re-ingests without paying the Stanza
pass again; a chunker or normalizer swap misses by key, which is exactly right.

Fallback-tokenized docs (Stanza error → whitespace tokens) are NOT cached, so
a fixed environment re-normalizes them on the next run.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TokenCache:
    def __init__(self, cache_dir: Path) -> None:
        self.tokens_dir = Path(cache_dir) / "tokens"

    def path_for(self, sha256: str, chunker_id: str, normalizer_id: str) -> Path:
        return self.tokens_dir / f"{sha256}.{chunker_id}.{normalizer_id}.json"

    def load(self, sha256: str, chunker_id: str, normalizer_id: str) -> list[list[str]] | None:
        path = self.path_for(sha256, chunker_id, normalizer_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt token-cache entry %s (%s) — re-normalizing", path, exc)
            return None
        if not isinstance(data, list):
            logger.warning("Malformed token-cache entry %s — re-normalizing", path)
            return None
        return data

    def store(
        self, sha256: str, chunker_id: str, normalizer_id: str, token_lists: list[list[str]]
    ) -> None:
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(sha256, chunker_id, normalizer_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(token_lists, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crashed write never poisons the cache


def _batch_ex(normalizer: Any, texts: list[str]) -> list[tuple[list[str], bool]]:
    """Use the normalizer's batch API when it has one (Stanza: bulk_process +
    per-chunk fallback flags); otherwise loop ``tokens`` with no fallback info."""
    if hasattr(normalizer, "tokens_batch_ex"):
        return normalizer.tokens_batch_ex(texts)
    return [(normalizer.tokens(text), False) for text in texts]


def tokens_for_doc(
    normalizer: Any,
    cache: TokenCache,
    *,
    sha256: str,
    chunker_id: str,
    normalizer_id: str,
    texts: list[str],
) -> list[list[str]]:
    """Read-through: cache hit (with matching chunk count) skips the
    normalizer entirely; on miss, normalize and cache — unless any chunk used
    the whitespace fallback."""
    cached = cache.load(sha256, chunker_id, normalizer_id)
    if cached is not None and len(cached) == len(texts):
        logger.debug("Token-cache hit: %s.%s.%s", sha256[:12], chunker_id, normalizer_id)
        return cached
    results = _batch_ex(normalizer, texts)
    token_lists = [tokens for tokens, _ in results]
    if any(fallback for _, fallback in results):
        logger.warning(
            "Doc %s… had fallback-tokenized chunks — NOT caching its tokens", sha256[:12]
        )
    else:
        cache.store(sha256, chunker_id, normalizer_id, token_lists)
    return token_lists

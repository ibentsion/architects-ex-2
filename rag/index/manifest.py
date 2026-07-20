"""Index manifest: write/verify ``<index_dir>/manifest.json`` (config identity
hash, resolved impl names, embedder/normalizer/chunker identity, chunk counts,
per-file sha256 + status, canary result, timestamps) — rag_plan.md §5 stage 7.
The query CLI refuses an index whose embedder/chunker/normalizer identity does
not match the active config. Implemented in wave E3 (T5)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def write_manifest(index_dir: Path, manifest: dict[str, Any]) -> None:
    raise NotImplementedError("Index manifest is implemented in wave E3 (rag_plan.md T5)")


def verify_manifest(index_dir: Path, config: Any) -> dict[str, Any]:
    raise NotImplementedError("Index manifest is implemented in wave E3 (rag_plan.md T5)")

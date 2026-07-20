"""Parse cache: ``<cache_dir>/parsed/<sha256>.json`` keyed by file content
hash (rag_plan.md §5 stage 2). Implemented in wave E2 (T2)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_parsed(cache_dir: Path, sha256: str) -> Any:
    raise NotImplementedError("Parse cache is implemented in wave E2 (rag_plan.md T2)")


def store_parsed(cache_dir: Path, sha256: str, doc_dict: dict[str, Any]) -> None:
    raise NotImplementedError("Parse cache is implemented in wave E2 (rag_plan.md T2)")

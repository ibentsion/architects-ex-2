"""Read-through parse cache (rag_plan.md §5 stage 2).

``<cache_dir>/parsed/<sha256>.json`` stores the ``DoclingDocument`` dict,
keyed by file content hash — a changed file misses by key, an unchanged file
never re-parses (Docling on CPU = hours for the full corpus; this cache is
what makes chunker/embedder experiments free).

TXT files bypass the cache: a UTF-8 read is cheaper than the JSON round-trip.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from rag.parsing import ParsedDoc, Parser, SourceFile

logger = logging.getLogger(__name__)


class ParseCache:
    def __init__(self, cache_dir: Path) -> None:
        self.parsed_dir = Path(cache_dir) / "parsed"

    def path_for(self, sha256: str) -> Path:
        return self.parsed_dir / f"{sha256}.json"

    def load(self, sha256: str) -> dict[str, Any] | None:
        path = self.path_for(sha256)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt parse-cache entry %s (%s) — re-parsing", path, exc)
            return None

    def store(self, sha256: str, doc_dict: dict[str, Any]) -> None:
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(sha256)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc_dict, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crashed write never poisons the cache


class CachedParser:
    """Wraps a PDF parser with the read-through cache; satisfies ``Parser``."""

    def __init__(self, parser: Parser, cache_dir: Path, force_reparse: bool = False) -> None:
        self._parser = parser
        self._cache = ParseCache(cache_dir)
        self._force_reparse = force_reparse

    def parse(self, source: SourceFile) -> ParsedDoc:
        if source.kind != "pdf":
            return self._parser.parse(source)
        if not self._force_reparse:
            cached = self._cache.load(source.sha256)
            if cached is not None:
                logger.debug("Parse-cache hit: %s", source.rel_path)
                return ParsedDoc(source=source, docling=cached)
        parsed = self._parser.parse(source)
        assert parsed.docling is not None
        self._cache.store(source.sha256, parsed.docling)
        return parsed

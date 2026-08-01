"""Bring an existing parse cache forward through the BiDi repair.

    python -m rag.cli.repair_cache --config configs/default.yaml [--dry-run]

``rag/parsing/docling_parser.py`` now applies ``rag.parsing.rtl_repair`` to
every parse, so fresh parses are already correct. Caches written before it
existed hold Docling's raw visual-order text, and re-parsing 350 PDFs is hours
of CPU — the repair is a pure function of the parsed document, so replaying it
over ``cache/parsed/`` reaches exactly the same state.

**The derived caches must be invalidated too.** ``cache/tokens/`` and
``cache/embeddings/`` are keyed by ``(file sha256, chunker id, ...)`` — none of
which changes when the parse output does, and the embedding cache only
re-checks the row count, which a re-ordering preserves. Left alone they would
silently serve vectors of reversed text forever. Entries for repaired files are
therefore deleted here, and the next ingest recomputes them (re-embedding those
files is the real cost of this migration).

Exit codes: 0 ok, 3 config error.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from rag.config import ConfigError, load_config
from rag.parsing import discover
from rag.parsing.cache import ParseCache
from rag.parsing.rtl_repair import RepairStats, repair_docling

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_CONFIG_ERROR = 3


def _derived_entries(cache_dir: Path, sha256: str) -> list[Path]:
    """Token/embedding cache files that were derived from this document."""
    entries: list[Path] = []
    for sub, suffix in (("tokens", ".json"), ("embeddings", ".npz")):
        directory = cache_dir / sub
        if directory.is_dir():
            entries += sorted(directory.glob(f"{sha256}.*{suffix}"))
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag.cli.repair_cache", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--config", required=True, help="YAML config (for corpus_dir/cache_dir)")
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change; write nothing"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log every repaired file")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    cache_dir = Path(config.cache_dir)
    cache = ParseCache(cache_dir)
    sources = {s.sha256: s for s in discover(config.corpus_dir) if s.kind == "pdf"}

    totals = RepairStats()
    changed = orphaned = invalidated = 0
    scanned = 0

    for path in sorted(cache.parsed_dir.glob("*.json")):
        sha256 = path.stem
        source = sources.get(sha256)
        if source is None:
            orphaned += 1  # cached parse whose PDF is no longer in the corpus
            continue
        scanned += 1
        try:
            doc_dict = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("skipping unreadable cache entry %s: %s", path, exc)
            continue

        stats = repair_docling(doc_dict, source.abs_path)
        totals.merge(stats)
        if not stats.repaired:
            continue
        changed += 1
        if args.verbose:
            logger.info(
                "  %s: %d cells, %d text items", source.rel_path, stats.cells_repaired, stats.texts_repaired
            )
        stale = _derived_entries(cache_dir, sha256)
        invalidated += len(stale)
        if args.dry_run:
            continue
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc_dict, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crashed write never poisons the cache
        for entry in stale:
            entry.unlink(missing_ok=True)

    verb = "would repair" if args.dry_run else "repaired"
    print(
        f"{verb} {changed}/{scanned} cached documents: "
        f"{totals.cells_repaired}/{totals.cells_total} table cells, "
        f"{totals.texts_repaired}/{totals.texts_total} text items "
        f"({totals.by_bbox} by bbox, {totals.by_segment} by segment)",
        file=sys.stderr,
    )
    verb = "would invalidate" if args.dry_run else "invalidated"
    print(f"{verb} {invalidated} derived token/embedding cache entries", file=sys.stderr)
    if orphaned:
        print(f"{orphaned} cached parses have no corpus file — left untouched", file=sys.stderr)
    if totals.pages_without_oracle:
        print(
            f"pages with no pdfium text (left as Docling produced them): "
            f"{len(totals.pages_without_oracle)}",
            file=sys.stderr,
        )
    if not args.dry_run and changed:
        print("re-run `python -m rag.cli.ingest --config ...` to rebuild the index", file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

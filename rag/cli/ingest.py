"""Ingestion CLI: ``python -m rag.cli.ingest --config configs/default.yaml``
(rag_plan.md §5 stages 1-7, §7).

Flags: --config (required), --categories, --force-reparse, --skip-canary,
--dry-run. Exit codes: 0 ok, 2 canary failure, 3 config error.

Incremental semantics — full rebuild, cache-hot (§5): every run logically
rebuilds the whole index, but parse/token/embedding stages read through the
shared per-file caches, so only what actually changed recomputes.

Atomicity: the index is built into ``<index_dir>.tmp-<pid>`` and swapped into
place on success via rename-old-away-then-rename-new-then-delete-old — a
crash at any point leaves either the previous or the new complete index at
``index_dir`` (never a partial one).
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from rag.config import (
    ConfigError,
    RagConfig,
    build,
    get_registry,
    impl_id,
    load_config,
)
from rag.embed.cache import EmbeddingCache, embed_doc, embedder_cache_key
from rag.index.manifest import build_manifest, write_manifest
from rag.normalize.cache import TokenCache, tokens_for_doc
from rag.parsing import ParsedDoc, SourceFile, discover
from rag.parsing.cache import CachedParser, ParseCache
from rag.parsing.canary import CanaryError, run_canary
from rag.parsing.docling_parser import ParseError
from rag.parsing.txt_parser import TxtParser
from rag.types import Chunk

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_CANARY_FAILURE = 2
EXIT_CONFIG_ERROR = 3

#: Ground-truth anchor PDF for the canary spot-check (rag_plan.md §5 stage 2).
ANCHOR_PDF_REL = unicodedata.normalize(
    "NFC", "apartment/files/הודעה-על-תקופת-התיישנות.pdf"
)


def _stage(msg: str) -> None:
    """Stage-level count+timing logs go to stderr (rag_plan.md §5)."""
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Stage 1 — Discover
# --------------------------------------------------------------------------- #


def _discover_sources(
    config: RagConfig, categories: list[str] | None
) -> list[SourceFile]:
    sources = discover(config.corpus_dir)
    if not categories:
        return sources
    available = {s.category for s in sources}
    unknown = [c for c in categories if c not in available]
    if unknown:
        raise ConfigError(
            f"--categories {' '.join(unknown)}: not found under {config.corpus_dir} "
            f"(available: {', '.join(sorted(available))})"
        )
    wanted = set(categories)
    return [s for s in sources if s.category in wanted]


# --------------------------------------------------------------------------- #
# Stage 2 — Parse (cache + RTL canary gate)
# --------------------------------------------------------------------------- #


def _canary_sample(pdfs: list[SourceFile], sample_size: int) -> list[SourceFile]:
    """Pick up to ``sample_size`` PDFs spread across categories (round-robin),
    force-including the ground-truth anchor PDF when present."""
    by_category: dict[str, list[SourceFile]] = {}
    for source in pdfs:
        by_category.setdefault(source.category, []).append(source)
    sample: list[SourceFile] = []
    anchor = next((s for s in pdfs if s.rel_path == ANCHOR_PDF_REL), None)
    if anchor is not None:
        sample.append(anchor)
    queues = [list(group) for _, group in sorted(by_category.items())]
    while len(sample) < min(sample_size, len(pdfs)) and any(queues):
        for queue in queues:
            while queue:
                candidate = queue.pop(0)
                if candidate not in sample:
                    sample.append(candidate)
                    break
            if len(sample) >= min(sample_size, len(pdfs)):
                break
    return sample


def _parse_batch(
    sources: list[SourceFile],
    pdf_parser: CachedParser,
    txt_parser: TxtParser,
    parse_cache: ParseCache,
    force_reparse: bool,
    files_meta: list[dict[str, Any]],
) -> list[ParsedDoc]:
    """Parse a batch; per-file failures are logged + recorded, not fatal
    (canary failure is the only hard stop)."""
    parsed: list[ParsedDoc] = []
    for source in sources:
        cache_hit = (
            source.kind == "pdf"
            and not force_reparse
            and parse_cache.path_for(source.sha256).is_file()
        )
        try:
            doc = (
                pdf_parser.parse(source)
                if source.kind == "pdf"
                else txt_parser.parse(source)
            )
        except ParseError as exc:
            logger.error("Parse FAILED (continuing): %s", exc)
            files_meta.append(
                {"file": source.rel_path, "sha256": source.sha256, "status": "failed"}
            )
            continue
        files_meta.append(
            {
                "file": source.rel_path,
                "sha256": source.sha256,
                "status": "cached" if cache_hit else "ok",
            }
        )
        parsed.append(doc)
    return parsed


# --------------------------------------------------------------------------- #
# Stage 6 — dense index construction (path derived by the CLI)
# --------------------------------------------------------------------------- #


def _build_dense_index(config: RagConfig, build_dir: Path) -> Any:
    """qdrant_local's path is derived (``<index_dir>/qdrant``) — the config
    params stay empty so the identity hash is location-independent."""
    impl = config.dense_index.impl
    registry = get_registry("dense_index")
    if impl not in registry:
        raise ConfigError(
            f"Unknown dense_index impl '{impl}'. Available: {sorted(registry)}."
        )
    params = dict(config.dense_index.params)
    if impl == "qdrant_local":
        params.setdefault("path", str(build_dir / "qdrant"))
    return registry[impl](**params)


def _swap_into_place(build_dir: Path, index_dir: Path) -> None:
    """Atomic activation: rename-old-away-then-rename-new-then-delete-old.

    ``os.rename`` is atomic on the same filesystem (build_dir is a sibling of
    index_dir). The live index is never deleted before the new one is in
    place, so a crash at any point leaves a complete index at ``index_dir``
    (plus at worst a leftover ``.old-<pid>``/``.tmp-<pid>`` dir to clean up).
    """
    old_dir = index_dir.with_name(f"{index_dir.name}.old-{os.getpid()}")
    if index_dir.exists():
        os.rename(index_dir, old_dir)
    os.rename(build_dir, index_dir)
    if old_dir.exists():
        shutil.rmtree(old_dir)


# --------------------------------------------------------------------------- #
# run_ingestion — stages 1-7 (rag_plan.md §5)
# --------------------------------------------------------------------------- #


def run_ingestion(
    config: RagConfig,
    categories: list[str] | None = None,
    force_reparse: bool = False,
    skip_canary: bool = False,
    dry_run: bool = False,
) -> int:
    run_start = time.monotonic()

    # ---- Stage 1: discover -------------------------------------------- #
    t0 = time.monotonic()
    sources = _discover_sources(config, categories)
    pdfs = [s for s in sources if s.kind == "pdf"]
    txts = [s for s in sources if s.kind == "txt"]
    n_categories = len({s.category for s in sources})
    _stage(
        f"[stage 1/7 discover] {len(sources)} files ({len(pdfs)} PDF, {len(txts)} TXT) "
        f"across {n_categories} categories ({time.monotonic() - t0:.1f}s)"
    )
    if not sources:
        raise ConfigError(f"No corpus files discovered under {config.corpus_dir}")

    parse_cache = ParseCache(config.cache_dir)
    cache_hits = sum(1 for s in pdfs if parse_cache.path_for(s.sha256).is_file())
    if dry_run:
        per_category = Counter(s.category for s in sources)
        _stage(
            f"[dry-run] parse cache: {cache_hits}/{len(pdfs)} PDFs cached, "
            f"{len(pdfs) - cache_hits} to parse; per category: "
            + ", ".join(f"{c}={n}" for c, n in sorted(per_category.items()))
        )
        return EXIT_OK

    # ---- Stages 2+3: parse (cache + RTL canary gate) + chunk ------------ #
    # Interleaved per doc: DoclingDocument dicts are LARGE (tens of MB for
    # policy books) — holding the whole corpus's parses in RAM alongside the
    # Docling models OOM-kills this ~4GB-free machine. Each ParsedDoc is
    # chunked immediately and discarded; only (source, chunks) survives.
    t0 = time.monotonic()
    parser = build("parser", config)
    cached_parser = CachedParser(parser, config.cache_dir, force_reparse=force_reparse)
    txt_parser = TxtParser()
    chunker = build("chunker", config)
    files_meta: list[dict[str, Any]] = []
    doc_chunks: list[tuple[SourceFile, list[Chunk]]] = []
    chunk_counts: Counter[str] = Counter()
    chunk_seconds = 0.0
    n_parsed = 0

    def _chunk_doc(doc: ParsedDoc) -> None:
        nonlocal chunk_seconds, n_parsed
        n_parsed += 1
        t_chunk = time.monotonic()
        chunks = chunker.chunk(doc)
        chunk_seconds += time.monotonic() - t_chunk
        if not chunks:
            logger.warning("Document produced no chunks: %s", doc.source.rel_path)
            return
        doc_chunks.append((doc.source, chunks))
        chunk_counts[doc.source.category] += len(chunks)

    do_canary = bool(getattr(parser, "rtl_canary", True)) and not skip_canary and pdfs
    canary_info: dict[str, Any] | None = {"skipped": True}
    if do_canary:
        sample = _canary_sample(pdfs, int(getattr(parser, "canary_sample", 10)))
        parsed_sample = _parse_batch(
            sample, cached_parser, txt_parser, parse_cache, force_reparse, files_meta
        )
        anchor_doc = next(
            (d for d in parsed_sample if d.source.rel_path == ANCHOR_PDF_REL), None
        )
        result = run_canary(parsed_sample, anchor_doc)  # raises CanaryError → exit 2
        canary_info = result.model_dump(mode="json")
        _stage(
            f"[stage 2/7 canary] PASSED on {len(parsed_sample)} PDFs"
            + (" (incl. ground-truth anchor)" if anchor_doc is not None else "")
        )
        while parsed_sample:
            _chunk_doc(parsed_sample.pop(0))  # pop: free each parse as we go
        remaining = [s for s in pdfs if s not in sample] + txts
    else:
        remaining = [*pdfs, *txts]
        if skip_canary:
            _stage("[stage 2/7 canary] SKIPPED (--skip-canary)")

    for source in remaining:
        for doc in _parse_batch(
            [source], cached_parser, txt_parser, parse_cache, force_reparse, files_meta
        ):
            _chunk_doc(doc)

    # Release the Docling converter + layout/table models BEFORE Stanza loads
    # (they cannot coexist within this machine's RAM budget).
    del cached_parser, parser
    gc.collect()

    n_failed = sum(1 for f in files_meta if f["status"] == "failed")
    n_cached = sum(1 for f in files_meta if f["status"] == "cached")
    _stage(
        f"[stage 2/7 parse] {n_parsed} parsed ({n_cached} from cache, "
        f"{n_failed} failed) ({time.monotonic() - t0 - chunk_seconds:.1f}s)"
    )
    total_chunks = sum(chunk_counts.values())
    _stage(
        f"[stage 3/7 chunk] {total_chunks} chunks from {len(doc_chunks)} docs "
        f"({', '.join(f'{c}={n}' for c, n in sorted(chunk_counts.items()))}) "
        f"({chunk_seconds:.1f}s)"
    )

    # ---- Stage 4: normalize (token cache) ------------------------------ #
    t0 = time.monotonic()
    normalizer = build("normalizer", config)
    token_cache = TokenCache(config.cache_dir)
    chunker_id = impl_id(config.chunker)
    normalizer_id = impl_id(config.normalizer)
    all_chunk_ids: list[str] = []
    all_token_lists: list[list[str]] = []
    for source, chunks in doc_chunks:
        token_lists = tokens_for_doc(
            normalizer,
            token_cache,
            sha256=source.sha256,
            chunker_id=chunker_id,
            normalizer_id=normalizer_id,
            texts=[c.text for c in chunks],
        )
        all_chunk_ids.extend(c.chunk_id for c in chunks)
        all_token_lists.extend(token_lists)
    _stage(
        f"[stage 4/7 normalize] {len(all_token_lists)} chunk token lists "
        f"({time.monotonic() - t0:.1f}s)"
    )

    # ---- Stages 5+6: embed (cache) + index dense/sparse ---------------- #
    embedder = build("embedder", config)
    embedding_cache = EmbeddingCache(config.cache_dir)
    dims = config.embedder.params.get("dimensions") or getattr(
        embedder, "dimensions", None
    )
    if not dims:
        raise ConfigError(
            "Cannot determine embedding dimensions — set embedder.params.dimensions"
        )
    embedder_key = embedder_cache_key(impl_id(config.embedder), int(dims))

    index_dir = Path(config.index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = index_dir.with_name(f"{index_dir.name}.tmp-{os.getpid()}")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    dense = None
    try:
        dense = _build_dense_index(config, build_dir)
        embed_seconds = 0.0
        dense_seconds = 0.0
        for source, chunks in doc_chunks:
            t0 = time.monotonic()
            vectors = embed_doc(
                embedder,
                embedding_cache,
                sha256=source.sha256,
                chunker_id=chunker_id,
                embedder_key=embedder_key,
                texts=[c.text for c in chunks],
            )
            embed_seconds += time.monotonic() - t0
            t0 = time.monotonic()
            dense.add(chunks, vectors.tolist())
            dense_seconds += time.monotonic() - t0
        usage = getattr(embedder, "total_input_tokens", 0)
        _stage(
            f"[stage 5/7 embed] {total_chunks} chunks embedded "
            f"({usage} API input tokens this run; cached docs cost 0) "
            f"({embed_seconds:.1f}s)"
        )

        t0 = time.monotonic()
        sparse = build("sparse_index", config)
        sparse.add(all_chunk_ids, all_token_lists)
        sparse.save(build_dir / "bm25")
        _stage(
            f"[stage 6/7 index] dense ({dense_seconds:.1f}s) + sparse "
            f"({time.monotonic() - t0:.1f}s) written to {build_dir.name}"
        )

        # ---- Stage 7: manifest ---------------------------------------- #
        manifest = build_manifest(
            config,
            chunk_counts=dict(chunk_counts),
            files=files_meta,
            canary=canary_info,
        )
        write_manifest(build_dir, manifest)
    except BaseException:
        if dense is not None:
            dense.close()
            dense = None
        shutil.rmtree(build_dir, ignore_errors=True)  # caches survive; index doesn't
        raise
    finally:
        if dense is not None:
            dense.close()  # release the Qdrant local lock BEFORE the rename

    _swap_into_place(build_dir, index_dir)
    _stage(
        f"[stage 7/7 manifest] index live at {index_dir} — {total_chunks} chunks, "
        f"{n_failed} failed files (total {time.monotonic() - run_start:.1f}s)"
    )
    return EXIT_OK


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag.cli.ingest",
        description="Build the hybrid RAG index from the corpus (rag_plan.md §5).",
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CAT",
        help="limit ingestion to the listed categories (subset ingest)",
    )
    parser.add_argument(
        "--force-reparse", action="store_true", help="ignore the parse cache"
    )
    parser.add_argument(
        "--skip-canary",
        action="store_true",
        help="skip the RTL gate (only after it has passed once for this corpus)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="discover + parse-cache stats only, no indexing",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
        return run_ingestion(
            config,
            categories=args.categories,
            force_reparse=args.force_reparse,
            skip_canary=args.skip_canary,
            dry_run=args.dry_run,
        )
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except CanaryError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return EXIT_CANARY_FAILURE
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_CONFIG_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

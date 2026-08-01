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
from rag.chunking.common import iter_reading_order, load_docling
from rag.embed.cache import EmbeddingCache, embed_doc, embedder_cache_key
from rag.index.manifest import build_manifest, write_manifest
from rag.normalize.cache import TokenCache, tokens_for_doc
from rag.parsing import ParseError, ParsedDoc, SourceFile, discover, doc_source_for
from rag.parsing.cache import CachedParser, ParseCache
from rag.parsing.canary import CanaryError, run_canary
from rag.parsing.txt_parser import TxtParser
from rag.report import ReportBuilder, current_source, load_previous
from rag.report import attach_collector, detach_collector
from rag.report import render_markdown, render_summary_line
from rag.report import save_report_history, write_report_files

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


def _rss_mb() -> float:
    """Current resident set size in MB — read directly from the kernel's
    per-process accounting (``/proc/self/status``), not just the historical
    peak, so growth *and* recovery (e.g. after ``gc.collect()``) are both
    visible. Falls back to ``resource.ru_maxrss`` (peak, not current) on
    non-Linux platforms."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except OSError:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _log_rss(tag: str) -> None:
    _stage(f"[mem] {tag}: RSS={_rss_mb():.0f}MB")


# --------------------------------------------------------------------------- #
# Stage 1 — Discover
# --------------------------------------------------------------------------- #


def _discover_sources(
    config: RagConfig, categories: list[str] | None
) -> list[SourceFile]:
    sources = discover(config.corpus_dir, doc_source_for(config.parser.impl))
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
            with current_source(source.rel_path, source.category):
                doc = (
                    txt_parser.parse(source)
                    if source.kind == "txt"
                    else pdf_parser.parse(source)  # pdf (docling) or md (markdown)
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


def _raw_chars(doc: ParsedDoc) -> int:
    """Raw extracted character count, for the content-loss reconciliation
    check (rag/report.py): "kept whole" and heading-prepend (per_table) only
    grow the chunked/raw ratio, so a drop below it means a chunk/item was
    actually dropped somewhere in the chunker."""
    if doc.kind == "txt":
        assert doc.text is not None
        return len(doc.text)
    dl_doc = load_docling(doc)
    return sum(len(text) for _item, _page, text in iter_reading_order(dl_doc))


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
    """Thin wrapper: owns the warning-collecting handler's lifetime so it is
    detached on every exit path (return, exception) without re-indenting the
    whole staged pipeline below."""
    raw_warnings: list[dict[str, Any]] = []
    handler = attach_collector(raw_warnings)
    try:
        return _run_ingestion(
            config, categories, force_reparse, skip_canary, dry_run, raw_warnings
        )
    finally:
        detach_collector(handler)


def _run_ingestion(
    config: RagConfig,
    categories: list[str] | None,
    force_reparse: bool,
    skip_canary: bool,
    dry_run: bool,
    raw_warnings: list[dict[str, Any]],
) -> int:
    run_start = time.monotonic()

    # ---- Stage 1: discover -------------------------------------------- #
    t0 = time.monotonic()
    sources = _discover_sources(config, categories)
    #: "documents" = the parser's rendering of the policy PDFs (Docling reads
    #: .pdf, the markdown parser reads .md); scraped TXT pages are separate.
    pdfs = [s for s in sources if s.kind != "txt"]
    txts = [s for s in sources if s.kind == "txt"]
    n_categories = len({s.category for s in sources})
    doc_label = "MD" if config.parser.impl == "markdown" else "PDF"
    _stage(
        f"[stage 1/7 discover] {len(sources)} files ({len(pdfs)} {doc_label}, {len(txts)} TXT) "
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

    # ---- Build every phase component BEFORE the per-doc loop ----------- #
    # Chunk -> normalize -> embed -> dense-index run in ONE PASS per
    # document (below), so at most one document's chunks/text are resident
    # at a time. The old design chunked the WHOLE corpus first, then
    # normalized the whole corpus, then embedded the whole corpus — holding
    # every document's chunk text in RAM across all three passes. That OOM-
    # killed a full 571-file/12-category run (silent SIGKILL right after
    # chunking finished, no traceback — see ingest_report.md history/commit
    # message for the incident). Only chunk_ids + token_lists now survive
    # per-doc processing (needed for the one unavoidable full-corpus call:
    # bm25s builds its scoring matrix in one shot; see rag/index/sparse.py).
    t0 = time.monotonic()
    parser = build("parser", config)
    cached_parser = CachedParser(parser, config.cache_dir, force_reparse=force_reparse)
    txt_parser = TxtParser()
    chunker = build("chunker", config)
    normalizer = build("normalizer", config)
    token_cache = TokenCache(config.cache_dir)
    chunker_id = impl_id(config.chunker)
    normalizer_id = impl_id(config.normalizer)
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

    files_meta: list[dict[str, Any]] = []
    chunk_counts: Counter[str] = Counter()
    all_chunk_ids: list[str] = []
    all_token_lists: list[list[str]] = []
    chunk_seconds = normalize_seconds = embed_seconds = dense_seconds = 0.0
    n_parsed = 0
    n_docs_with_chunks = 0
    report = ReportBuilder()
    _log_rss("before per-doc loop")

    dense = None
    try:
        dense = _build_dense_index(config, build_dir)

        def _process_doc(doc: ParsedDoc) -> None:
            """Chunk -> normalize -> embed -> dense.add() for ONE document;
            its chunks/text go out of scope when this call returns."""
            nonlocal chunk_seconds, normalize_seconds, embed_seconds, dense_seconds
            nonlocal n_parsed, n_docs_with_chunks
            n_parsed += 1
            raw_chars = _raw_chars(doc)

            t = time.monotonic()
            chunks = chunker.chunk(doc)
            chunk_seconds += time.monotonic() - t
            report.record_doc(
                file=doc.source.rel_path,
                category=doc.source.category,
                chunker=chunker.name,
                raw_chars=raw_chars,
                chunks=[c.text for c in chunks],
            )
            if not chunks:
                logger.warning("Document produced no chunks: %s", doc.source.rel_path)
                return
            chunk_counts[doc.source.category] += len(chunks)
            n_docs_with_chunks += 1

            with current_source(doc.source.rel_path, doc.source.category):
                t = time.monotonic()
                token_lists = tokens_for_doc(
                    normalizer,
                    token_cache,
                    sha256=doc.source.sha256,
                    chunker_id=chunker_id,
                    normalizer_id=normalizer_id,
                    texts=[c.text for c in chunks],
                )
                normalize_seconds += time.monotonic() - t
            all_chunk_ids.extend(c.chunk_id for c in chunks)
            all_token_lists.extend(token_lists)

            t = time.monotonic()
            vectors = embed_doc(
                embedder,
                embedding_cache,
                sha256=doc.source.sha256,
                chunker_id=chunker_id,
                embedder_key=embedder_key,
                texts=[c.text for c in chunks],
            )
            embed_seconds += time.monotonic() - t
            t = time.monotonic()
            dense.add(chunks, vectors.tolist())
            dense_seconds += time.monotonic() - t

            if n_parsed % 25 == 0:
                _log_rss(f"after {n_parsed} docs")

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
                f"[stage 2/6 canary] PASSED on {len(parsed_sample)} PDFs"
                + (" (incl. ground-truth anchor)" if anchor_doc is not None else "")
            )
            while parsed_sample:
                _process_doc(parsed_sample.pop(0))  # pop: free each parse as we go
            remaining = [s for s in pdfs if s not in sample] + txts
        else:
            remaining = [*pdfs, *txts]
            if skip_canary:
                _stage("[stage 2/6 canary] SKIPPED (--skip-canary)")

        for source in remaining:
            for doc in _parse_batch(
                [source], cached_parser, txt_parser, parse_cache, force_reparse, files_meta
            ):
                _process_doc(doc)

        # Release the Docling converter + layout/table models — the parse
        # cache means a re-run never needs them again this session.
        del cached_parser, parser
        gc.collect()
        _log_rss("after per-doc loop (parse+chunk+normalize+embed+dense-index)")

        n_failed = sum(1 for f in files_meta if f["status"] == "failed")
        n_cached = sum(1 for f in files_meta if f["status"] == "cached")
        total_chunks = sum(chunk_counts.values())
        usage = getattr(embedder, "total_input_tokens", 0)
        _stage(
            f"[stage 3/6 parse+chunk+normalize+embed+index] {n_parsed} parsed "
            f"({n_cached} from cache, {n_failed} failed), {total_chunks} chunks from "
            f"{n_docs_with_chunks} docs "
            f"({', '.join(f'{c}={n}' for c, n in sorted(chunk_counts.items()))}) "
            f"— parse+chunk {chunk_seconds:.1f}s, normalize {normalize_seconds:.1f}s, "
            f"embed {embed_seconds:.1f}s ({usage} API input tokens this run; cached "
            f"docs cost 0), dense-index {dense_seconds:.1f}s "
            f"(total {time.monotonic() - t0:.1f}s)"
        )

        t0 = time.monotonic()
        sparse = build("sparse_index", config)
        sparse.add(all_chunk_ids, all_token_lists)
        sparse.save(build_dir / "bm25")
        _stage(
            f"[stage 4/6 sparse-index] written to {build_dir.name} "
            f"({time.monotonic() - t0:.1f}s)"
        )
        _log_rss("after sparse index build")

        # ---- Stage 5/6: manifest ---------------------------------------- #
        manifest = build_manifest(
            config,
            chunk_counts=dict(chunk_counts),
            files=files_meta,
            canary=canary_info,
        )
        write_manifest(build_dir, manifest)

        # ---- Ingestion report (rag/report.py) -------------------------- #
        # Loaded BEFORE this run's history is saved, so the delta section
        # compares against the run before this one.
        previous_report = load_previous(config.cache_dir, config.index_dir)
        ingest_report = report.finalize(
            raw_warnings=raw_warnings,
            config_identity=manifest["config_identity"],
            impls=manifest["impls"],
            categories_filter=categories,
            files_meta=files_meta,
            chunk_counts=dict(chunk_counts),
            canary=canary_info,
            embedding_tokens=usage,
            wall_seconds=time.monotonic() - run_start,
            stage_seconds={
                "parse_chunk": chunk_seconds,
                "normalize": normalize_seconds,
                "embed": embed_seconds,
                "dense_index": dense_seconds,
            },
        )
        write_report_files(
            build_dir, ingest_report, render_markdown(ingest_report, previous_report)
        )
        save_report_history(config.cache_dir, config.index_dir, ingest_report)
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
        f"[stage 5/6 manifest] + [stage 6/6 swap] index live at {index_dir} — "
        f"{total_chunks} chunks, {n_failed} failed files "
        f"(total {time.monotonic() - run_start:.1f}s)"
    )
    _log_rss("run complete")
    _stage(f"[report] {render_summary_line(ingest_report)}")
    if previous_report is not None:
        _stage(
            f"[report] vs previous run ({previous_report.get('created_at', '?')}): "
            f"{previous_report.get('chunks', {}).get('total', '?')} -> {ingest_report['chunks']['total']} chunks"
        )
    _stage(f"[report] full report: {index_dir}/ingest_report.md")
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

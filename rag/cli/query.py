"""Query CLI: single question, --interactive REPL, or --questions batch mode
(rag_plan.md §7).

    python -m rag.cli.query --config configs/default.yaml "שאלה"
    python -m rag.cli.query --config configs/default.yaml --interactive
    python -m rag.cli.query --config configs/default.yaml \
        --questions reference_questions.json --out answers.jsonl

Flags: --config (required), --category, --interactive/-i, --show-chunks,
--json, --questions/--out (batch mode).

Retrieval-stage tools (no generation) — for wiring an external harness:

    python -m rag.cli.query dense "שאלה" --config C [--top-k N] [--category X]
    python -m rag.cli.query sparse "שאלה" --config C [--top-k N] [--category X]
    python -m rag.cli.query fuse "שאלה" --config C [--dense-top-k N] \
        [--sparse-top-k N] [--rrf-k N] [--category X]
    python -m rag.cli.query rerank "שאלה" --config C --candidates FILE|- \
        [--gate-threshold F] [--top-n N]
    python -m rag.cli.query retrieve "שאלה" --config C [fuse flags] \
        [--gate-threshold F] [--top-n N]

Tool commands are selected by the first positional argument; anything else is
the classic answer flow. Every tool prints one JSON object to stdout
(``results`` = RetrievedChunk dumps; ``stats`` for fuse/retrieve); parameter
flags default to the config values. ``rerank --candidates`` accepts a JSON
list of chunk_ids, of ``{chunk_id, ...scores}`` objects, or the output
envelope of ``fuse``/``retrieve`` — texts are hydrated from the dense
payload store.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rag.config import ConfigError, RagConfig, load_config
from rag.generate.generator import Generator
from rag.generate.prompts import FALLBACK_TEXT, SOURCES_HEADER
from rag.index.manifest import ManifestError, ManifestMismatchError
from rag.retrieve.retriever import Retriever, load_retriever
from rag.types import Answer, RetrievedChunk

EXIT_OK = 0
EXIT_CONFIG_ERROR = 3

INTERACTIVE_PROMPT = "שאלה> "


def _confidence(retrieved: list[RetrievedChunk]) -> float:
    """§6 stage 8: confidence = max rerank sigmoid (already sigmoid-scaled by
    the CrossEncoder's forced activation, rag/retrieve/rerank.py)."""
    scores = [r.rerank_score for r in retrieved if r.rerank_score is not None]
    return max(scores) if scores else 0.0


class QueryEngine:
    """Stage 0 load & validate (manifest compatibility, lazy/warm component
    loads), then stateless ``answer(question, category=None)`` per query
    (rag_plan.md §6). Each query is independent -- no conversation memory."""

    def __init__(self, config: RagConfig) -> None:
        self.config = config
        # §6 stage 0: verify_manifest (inside load_retriever) raises a clear
        # "re-ingest with this config" error on embedder/chunker/normalizer
        # mismatch. Model loads themselves are lazy (warm after 1st query).
        self.retriever: Retriever = load_retriever(config)
        self.generator = Generator(
            model=config.generation.model,
            prompt=config.generation.prompt,
            max_tokens=config.generation.max_tokens,
            temperature=config.generation.temperature,
            retry_on_citation_failure=config.generation.retry_on_citation_failure,
            extra_params=config.generation.extra_params,
        )

    def answer(self, question: str, category: str | None = None) -> Answer:
        return self._answer(question, category)[0]

    def _answer(
        self, question: str, category: str | None = None
    ) -> tuple[Answer, dict[str, int] | None]:
        """Returns (Answer, tokens) -- tokens is None on the gate-fail path
        (no LLM call) and exposed only for the evalharness-compatible batch
        writer; ``answer()`` is the public single-value API."""
        t0 = time.monotonic()
        retrieved = self.retriever.retrieve(question, category=category)
        retrieval_ms = (time.monotonic() - t0) * 1000
        retrieval_stats = self.retriever.last_stats or None
        if not retrieved:
            # Gate fail (rag_plan.md §6 stage 4): zero LLM cost.
            answer = Answer(
                text=FALLBACK_TEXT,
                citations=[],
                category=category,
                confidence=0.0,
                latency_ms=retrieval_ms,
                cost_estimate=0.0,
                retrieved=[],
                retrieval_stats=retrieval_stats,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
            )
            return answer, None

        t1 = time.monotonic()
        result = self.generator.generate(question, retrieved)
        generation_ms = (time.monotonic() - t1) * 1000
        answer = Answer(
            text=result.text,
            citations=result.citations,
            category=category,
            confidence=_confidence(retrieved),
            latency_ms=retrieval_ms + generation_ms,
            cost_estimate=result.cost_estimate,
            citation_fallback=result.citation_fallback,
            retrieved=retrieved,
            retrieval_stats=retrieval_stats,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            max_tokens_hit=result.max_tokens_hit,
            n_retries=result.n_retries,
        )
        return answer, result.tokens

    def close(self) -> None:
        self.retriever.close()


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #


def _print_chunks(retrieved: list[RetrievedChunk]) -> None:
    for r in retrieved:
        page = r.chunk.page if r.chunk.page is not None else "-"
        score = f"{r.rerank_score:.4f}" if r.rerank_score is not None else "?"
        print(f"  [{score}] {r.chunk.file} | page {page} | {r.chunk.category}", file=sys.stderr)
        snippet = r.chunk.text[:200].replace("\n", " ")
        print(f"      {snippet}...", file=sys.stderr)


def _strip_trailing_sources_block(text: str) -> str:
    """The raw LLM completion already ends with its own ``מקורות:`` block
    (that's what citations.py parses); strip it from the display text so the
    CLI shows exactly one sources block -- built from the VALIDATED
    citations, not the model's raw (possibly-corrected-away) one."""
    header_idx = text.rfind(SOURCES_HEADER)
    return text[:header_idx].rstrip() if header_idx != -1 else text


def _print_answer(answer: Answer, *, show_chunks: bool, as_json: bool) -> None:
    if show_chunks and answer.retrieved:
        _print_chunks(answer.retrieved)
    if as_json:
        print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    print(_strip_trailing_sources_block(answer.text))
    if answer.citations:
        print("\nמקורות:")
        for c in answer.citations:
            page = c.page if c.page is not None else "-"
            print(f"- file: {c.file} | page: {page}")


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


def _run_interactive(engine: QueryEngine, category: str | None, show_chunks: bool, as_json: bool) -> None:
    print("מצב אינטראקטיבי — הקלד/י 'exit' או 'quit' ליציאה (Ctrl-D גם עובד).", file=sys.stderr)
    while True:
        try:
            line = input(INTERACTIVE_PROMPT)
        except EOFError:
            print(file=sys.stderr)
            break
        question = line.strip()
        if question in ("exit", "quit"):
            break
        if not question:
            continue
        answer = engine.answer(question, category=category)
        _print_answer(answer, show_chunks=show_chunks, as_json=as_json)


def _run_batch(engine: QueryEngine, questions_path: Path, out_path: Path, category: str | None) -> None:
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):  # staff sets wrap the list in {"questions": [...]}
        data = data["questions"]
    n = len(data)
    with out_path.open("w", encoding="utf-8") as out:
        for i, q in enumerate(data, start=1):
            answer, tokens = engine._answer(q["question"], category=category)
            record: dict[str, Any] = {
                "id": q["id"],
                "answer": answer.text,
                "citations": [c.model_dump(mode="json") for c in answer.citations],
                "latency_ms": answer.latency_ms,
                "retrieval_ms": answer.retrieval_ms,
                "generation_ms": answer.generation_ms,
                "retrieval_stats": answer.retrieval_stats,
                "max_tokens_hit": answer.max_tokens_hit,
                "n_retries": answer.n_retries,
            }
            if tokens is not None:
                record["tokens"] = tokens
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            gated_n = (answer.retrieval_stats or {}).get("gated", {}).get("n_chunks", 0)
            flags = []
            if answer.max_tokens_hit:
                flags.append("max_tokens_hit")
            if answer.n_retries:
                flags.append(f"retries={answer.n_retries}")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(
                f"  [{i}/{n}] {q['id']}: {answer.text[:70]!r}... "
                f"({answer.latency_ms:.0f} ms, {gated_n} chunks){flag_str}",
                file=sys.stderr,
            )
    print(f"\nwrote {out_path}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Retrieval-stage tools (no generation) — external-harness entry points
# --------------------------------------------------------------------------- #

TOOL_COMMANDS = ("dense", "sparse", "fuse", "rerank", "retrieve")


def build_tools_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.cli.query",
        description="Retrieval-stage tools: JSON to stdout, no LLM generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_command(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("question", help="query text")
        p.add_argument("--config", required=True, help="YAML config path (must match the ingested index)")
        return p

    for name in ("dense", "sparse"):
        p = add_command(name, f"{name} search only")
        p.add_argument("--top-k", type=int, help=f"override retrieval.{name}_top_k")
        p.add_argument("--category", help="restrict to one category")

    fuse = add_command("fuse", "dense + sparse + RRF fusion (no rerank)")
    retrieve = add_command("retrieve", "full retrieval pipeline (fusion + rerank + gate)")
    for p in (fuse, retrieve):
        p.add_argument("--dense-top-k", type=int, help="override retrieval.dense_top_k")
        p.add_argument("--sparse-top-k", type=int, help="override retrieval.sparse_top_k")
        p.add_argument("--rrf-k", type=int, help="override retrieval.rrf_k")
        p.add_argument("--category", help="restrict to one category")

    rerank = add_command("rerank", "rerank + gate an arbitrary candidate list")
    rerank.add_argument(
        "--candidates",
        required=True,
        help="JSON file of candidates ('-' for stdin): chunk_ids, {chunk_id,...} objects, or fuse/retrieve output",
    )
    for p in (rerank, retrieve):
        p.add_argument("--gate-threshold", type=float, help="override retrieval.rerank.gate_threshold")
        p.add_argument("--top-n", type=int, help="override retrieval.rerank.top_n")
    return parser


def _load_candidates(retriever: Retriever, spec: str) -> list[RetrievedChunk]:
    """Parse a rerank candidate list; hydrate text-less entries by chunk_id
    from the dense payload store (skew warnings on stderr, entries skipped)."""
    raw = sys.stdin.read() if spec == "-" else Path(spec).read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, dict):  # fuse/retrieve output envelope
        data = data["results"]
    candidates: list[RetrievedChunk] = []
    to_hydrate: list[tuple[str, dict[str, float]]] = []
    for item in data:
        if isinstance(item, str):
            to_hydrate.append((item, {}))
        elif isinstance(item, dict) and isinstance(item.get("chunk"), dict):
            candidates.append(RetrievedChunk.model_validate(item))
        elif isinstance(item, dict) and "chunk_id" in item:
            scores = {
                k: item[k]
                for k in ("dense_score", "sparse_score", "rrf_score")
                if item.get(k) is not None
            }
            to_hydrate.append((item["chunk_id"], scores))
        else:
            raise ValueError(f"Unrecognized candidate entry: {item!r}")
    if to_hydrate:
        fetched = retriever.dense.fetch([cid for cid, _ in to_hydrate])
        for cid, scores in to_hydrate:
            chunk = fetched.get(cid)
            if chunk is None:
                print(f"warning: chunk_id not in dense payload store, skipped: {cid}", file=sys.stderr)
                continue
            candidates.append(RetrievedChunk(chunk=chunk, **scores))
    return candidates


def _run_tool(retriever: Retriever, args: argparse.Namespace) -> dict[str, Any]:
    def dumped(results: list[RetrievedChunk]) -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in results]

    if args.command == "dense":
        hits = retriever.dense_search(args.question, top_k=args.top_k, category=args.category)
        return {"results": dumped([RetrievedChunk(chunk=c, dense_score=s) for c, s in hits])}
    if args.command == "sparse":
        hits = retriever.sparse_search(args.question, top_k=args.top_k, category=args.category)
        fetched = retriever.dense.fetch([cid for cid, _ in hits])
        results = []
        for cid, score in hits:
            chunk = fetched.get(cid)
            if chunk is None:
                print(f"warning: chunk_id not in dense payload store, skipped: {cid}", file=sys.stderr)
                continue
            results.append(RetrievedChunk(chunk=chunk, sparse_score=score))
        return {"results": dumped(results)}
    if args.command == "fuse":
        candidates = retriever.fuse(
            args.question,
            dense_top_k=args.dense_top_k,
            sparse_top_k=args.sparse_top_k,
            rrf_k=args.rrf_k,
            category=args.category,
        )
        return {"results": dumped(candidates), "stats": retriever.last_stats}
    if args.command == "rerank":
        candidates = _load_candidates(retriever, args.candidates)
        gated = retriever.rerank_candidates(
            args.question, candidates, gate_threshold=args.gate_threshold, top_n=args.top_n
        )
        return {"results": dumped(gated)}
    if args.command == "retrieve":
        gated = retriever.retrieve(
            args.question,
            category=args.category,
            dense_top_k=args.dense_top_k,
            sparse_top_k=args.sparse_top_k,
            rrf_k=args.rrf_k,
            gate_threshold=args.gate_threshold,
            top_n=args.top_n,
        )
        return {"results": dumped(gated), "stats": retriever.last_stats}
    raise ValueError(f"Unknown tool command: {args.command}")


def tools_main(argv: list[str]) -> int:
    args = build_tools_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        retriever = load_retriever(config)
    except (ConfigError, ManifestError, ManifestMismatchError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    try:
        result = _run_tool(retriever, args)
    finally:
        retriever.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return EXIT_OK


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.cli.query",
        description="Answer a question against an ingested RAG index (rag_plan.md §6-7).",
    )
    parser.add_argument("question", nargs="?", help="single question (omit for --interactive / --questions)")
    parser.add_argument("--config", required=True, help="YAML config path (must match the ingested index)")
    parser.add_argument("--category", help="restrict retrieval to one category")
    parser.add_argument("--interactive", "-i", action="store_true", help="REPL mode")
    parser.add_argument("--show-chunks", action="store_true", help="print retrieved chunks + rerank scores")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit the full Answer as JSON")
    parser.add_argument("--questions", help="reference_questions.json for batch mode")
    parser.add_argument("--out", help="output answers.jsonl path (batch mode, requires --questions)")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in TOOL_COMMANDS:
        return tools_main(argv)
    args = build_arg_parser().parse_args(argv)

    if bool(args.questions) != bool(args.out):
        print("--questions and --out must be given together (batch mode)", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    if not args.interactive and not args.questions and not args.question:
        print("Provide a question, --interactive/-i, or --questions/--out for batch mode", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        config = load_config(args.config)
        engine = QueryEngine(config)
    except (ConfigError, ManifestError, ManifestMismatchError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        if args.questions:
            _run_batch(engine, Path(args.questions), Path(args.out), args.category)
        elif args.interactive:
            _run_interactive(engine, args.category, args.show_chunks, args.as_json)
        else:
            answer = engine.answer(args.question, category=args.category)
            _print_answer(answer, show_chunks=args.show_chunks, as_json=args.as_json)
    finally:
        engine.close()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

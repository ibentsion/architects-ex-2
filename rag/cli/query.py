"""Query CLI: single question, --interactive REPL, or --questions batch mode
(rag_plan.md §7).

    python -m rag.cli.query --config configs/default.yaml "שאלה"
    python -m rag.cli.query --config configs/default.yaml --interactive
    python -m rag.cli.query --config configs/default.yaml \
        --questions reference_questions.json --out answers.jsonl

Flags: --config (required), --category, --interactive/-i, --show-chunks,
--json, --questions/--out (batch mode).
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
        if not retrieved:
            # Gate fail (rag_plan.md §6 stage 4): zero LLM cost.
            answer = Answer(
                text=FALLBACK_TEXT,
                citations=[],
                category=category,
                confidence=0.0,
                latency_ms=(time.monotonic() - t0) * 1000,
                cost_estimate=0.0,
                retrieved=[],
            )
            return answer, None

        result = self.generator.generate(question, retrieved)
        answer = Answer(
            text=result.text,
            citations=result.citations,
            category=category,
            confidence=_confidence(retrieved),
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_estimate=result.cost_estimate,
            citation_fallback=result.citation_fallback,
            retrieved=retrieved,
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
            }
            if tokens is not None:
                record["tokens"] = tokens
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [{i}/{n}] {q['id']}: {answer.text[:70]!r}... ({answer.latency_ms:.0f} ms)", file=sys.stderr)
    print(f"\nwrote {out_path}", file=sys.stderr)


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

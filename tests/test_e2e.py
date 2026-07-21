"""E2E system test (rag_plan.md §9 test_e2e.py, T10): full query pipeline
against a real, already-built index -- real dense/sparse retrieval, real
CPU reranker, real Token Factory generation call. Marked ``slow``+``llm``.

Reuses ``rag_index/default`` (built via ``configs/default.yaml``, cache-hot
incremental ingest -- see rag_plan.md §5) rather than re-ingesting the mini
fixture corpus: the index already covers at least apartment+travel (it may
have been extended to the full 12-category corpus by T10's Part B), so the
dev-02-apartment-easy ground-truth anchor is always present. Pipelines run
sequentially (Qdrant local single-process lock, rag_plan.md §6 stage 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config import load_config
from rag.cli.query import QueryEngine

ANCHOR_FILE = "apartment/files/הודעה-על-תקופת-התיישנות.pdf"
ANCHOR_QUESTION_ID = "dev-02-apartment-easy"
OUT_OF_CORPUS_QUESTION = "מה מזג האוויר היום?"


def _dev_question(question_id: str, repo_root: Path) -> dict:
    data = json.loads((repo_root / "reference_questions.json").read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data["questions"]
    matches = [q for q in data if q["id"] == question_id]
    assert matches, f"question {question_id} not found in reference_questions.json"
    return matches[0]


@pytest.fixture(scope="module")
def engine(repo_root: Path):
    config = load_config(repo_root / "configs" / "default.yaml")
    manifest_path = Path(config.index_dir) / "manifest.json"
    assert manifest_path.is_file(), (
        f"no index at {config.index_dir} -- run "
        "`python -m rag.cli.ingest --config configs/default.yaml --categories apartment travel` first"
    )
    eng = QueryEngine(config)
    try:
        yield eng
    finally:
        eng.close()


@pytest.mark.slow
@pytest.mark.llm
def test_anchor_question_answered_with_correct_citation(engine, repo_root):
    """dev-02-apartment-easy through the full pipeline: real retrieval,
    real rerank+gate, real generation -- must ground the answer in the
    ground-truth anchor {file, page:1}."""
    question = _dev_question(ANCHOR_QUESTION_ID, repo_root)

    answer = engine.answer(question["question"])

    assert answer.text.strip(), "answer text must be non-empty"
    assert any("֐" <= ch <= "׿" for ch in answer.text), "answer must be Hebrew"
    assert len(answer.citations) >= 1
    assert any(
        c.file == ANCHOR_FILE and c.page == 1 for c in answer.citations
    ), f"expected a citation matching {{file: {ANCHOR_FILE!r}, page: 1}}, got {answer.citations}"
    assert answer.latency_ms is not None and answer.latency_ms > 0


@pytest.mark.slow
@pytest.mark.llm
def test_out_of_corpus_question_hits_relevance_gate(engine):
    """A question with no relevant corpus content must trip the relevance
    gate (rag_plan.md §6 stage 4): zero LLM cost, empty citations, zero
    confidence -- the retriever contract from E4/E5, not a literal string
    match on the fallback sentence (which is only emitted on THIS path)."""
    answer = engine.answer(OUT_OF_CORPUS_QUESTION)

    assert answer.citations == []
    assert answer.confidence == 0.0

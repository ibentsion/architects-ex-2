"""Query classification + --route answering tests. tf_client mocked — no LLM.

Covers: strict-JSON parsing (fences/prose tolerated), category validation
against the closed 12-list, derived mode/categories, graceful fallback on
any failure, the routed answer flow (per-sub-question retrieval, category
filters, pool/dedupe, single generation, stats summing), and the classify
CLI subcommand.
"""
from __future__ import annotations

import json

import pytest

import rag.classify as classify_mod
from rag.classify import CATEGORIES, QueryClassifier, _system_prompt
from rag.cli import query as query_cli
from rag.generate.generator import GenerationResult
from rag.generate.prompts import FALLBACK_TEXT
from rag.types import Classification, SubQuestion

from tests.test_retrieve import APT1, APT2, TRV1, make_candidate


def fake_chat(reply: str, cost: float = 0.001):
    def _chat(messages, **kwargs):
        return reply, {"prompt": 100, "completion": 50, "finish_reason": "stop"}, cost

    return _chat


QUESTION = "מה מכסה ביטוח הדירה שלי במקרה של הצפה, ואיך מגישים תביעת רכב?"


# --------------------------------------------------------------------------- #
# QueryClassifier parsing
# --------------------------------------------------------------------------- #


def test_classify_multi_category_decomposition(monkeypatch):
    reply = json.dumps(
        {
            "sub_questions": [
                {"question": "מה מכסה ביטוח הדירה במקרה של הצפה?", "categories": ["apartment"]},
                {"question": "איך מגישים תביעת רכב?", "categories": ["car", "bogus-cat"]},
            ]
        },
        ensure_ascii=False,
    )
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(reply, cost=0.002))
    result = QueryClassifier("m").classify(QUESTION)
    assert result.mode == "multi"
    assert result.categories == ["apartment", "car"]  # union, unknown dropped
    assert [sq.categories for sq in result.sub_questions] == [["apartment"], ["car"]]
    assert result.cost_estimate == 0.002


def test_classify_single_category_single_subquestion(monkeypatch):
    reply = json.dumps(
        {"sub_questions": [{"question": "מה תקופת ההתיישנות?", "categories": ["apartment"]}]},
        ensure_ascii=False,
    )
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(reply))
    result = QueryClassifier("m").classify("מה תקופת ההתיישנות?")
    assert result.mode == "single"
    assert result.categories == ["apartment"]


def test_classify_tolerates_code_fences_and_prose(monkeypatch):
    reply = 'הנה הסיווג:\n```json\n{"sub_questions": [{"question": "ש", "categories": ["travel"]}]}\n```'
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(reply))
    assert QueryClassifier("m").classify("ש").categories == ["travel"]


@pytest.mark.parametrize(
    "reply",
    [
        "אין לי מושג",  # no JSON at all
        '{"sub_questions": []}',  # empty decomposition
        '{"sub_questions": [{"question": "", "categories": ["car"]}]}',  # blank question
        '{"sub_questions": "oops"}',  # wrong type
    ],
)
def test_classify_falls_back_to_single_no_filter(monkeypatch, reply):
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(reply, cost=0.003))
    result = QueryClassifier("m").classify(QUESTION)
    assert result.mode == "single"
    assert result.categories == []
    assert [sq.question for sq in result.sub_questions] == [QUESTION]
    assert result.cost_estimate == 0.003


def test_classify_falls_back_when_chat_raises(monkeypatch):
    def boom(messages, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(classify_mod, "tf_chat", boom)
    result = QueryClassifier("m").classify(QUESTION)
    assert result.mode == "single" and result.cost_estimate == 0.0
    assert result.sub_questions[0].question == QUESTION


def test_system_prompt_lists_all_corpus_categories():
    prompt = _system_prompt()
    assert len(CATEGORIES) == 12
    assert all(cid in prompt for cid in CATEGORIES)


# --------------------------------------------------------------------------- #
# Routed answering (QueryEngine._answer_routed) — all components faked
# --------------------------------------------------------------------------- #


class FakeRoutedRetriever:
    def __init__(self, results_by_question):
        self.results_by_question = results_by_question
        self.calls = []
        self.last_stats = {}

    def retrieve(self, question, category=None, **overrides):
        self.calls.append((question, category))
        results = self.results_by_question.get(question, [])
        self.last_stats = {
            "gated": {"n_chunks": len(results), "n_documents": len(results)}
        }
        return results


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, retrieved):
        self.calls.append((question, retrieved))
        return GenerationResult(
            text="תשובה",
            citations=[],
            citation_fallback=False,
            cost_estimate=0.01,
            tokens={"prompt": 10, "completion": 5},
        )


class FakeClassifier:
    def __init__(self, classification):
        self.classification = classification

    def classify(self, question):
        return self.classification


def make_engine(classification, results_by_question):
    engine = query_cli.QueryEngine.__new__(query_cli.QueryEngine)
    engine.route = True
    engine.classifier = FakeClassifier(classification)
    engine.retriever = FakeRoutedRetriever(results_by_question)
    engine.generator = FakeGenerator()
    return engine


def multi_classification():
    return Classification(
        mode="multi",
        categories=["apartment", "car"],
        sub_questions=[
            SubQuestion(question="שאלת דירה", categories=["apartment"]),
            SubQuestion(question="שאלת רכב", categories=["car", "travel"]),
        ],
        cost_estimate=0.002,
    )


def test_routed_answer_filters_pools_and_generates_once():
    shared_low = make_candidate(APT2, rerank_score=0.5)
    shared_high = make_candidate(APT2, rerank_score=0.7)
    engine = make_engine(
        multi_classification(),
        {
            "שאלת דירה": [make_candidate(APT1, rerank_score=0.9), shared_low],
            "שאלת רכב": [shared_high, make_candidate(TRV1, rerank_score=0.6)],
        },
    )
    answer, tokens = engine._answer("שאלה מקורית")
    # Single-category sub-question filtered; two-category one unfiltered.
    assert engine.retriever.calls == [("שאלת דירה", "apartment"), ("שאלת רכב", None)]
    # Pool: deduped on chunk_id (max rerank kept), sorted desc.
    assert [(r.chunk.chunk_id, r.rerank_score) for r in answer.retrieved] == [
        (APT1, 0.9),
        (APT2, 0.7),
        (TRV1, 0.6),
    ]
    # ONE generation call, with the ORIGINAL question over the pooled chunks.
    assert len(engine.generator.calls) == 1
    assert engine.generator.calls[0][0] == "שאלה מקורית"
    assert answer.classification == engine.classifier.classification
    assert answer.category is None  # multi-category -> no single routed category
    assert answer.cost_estimate == pytest.approx(0.01 + 0.002)
    assert answer.retrieval_stats == {"gated": {"n_chunks": 4, "n_documents": 4}}  # summed
    assert tokens == {"prompt": 10, "completion": 5}


def test_routed_single_category_sets_answer_category():
    classification = Classification(
        mode="single",
        categories=["apartment"],
        sub_questions=[SubQuestion(question="שאלת דירה", categories=["apartment"])],
        cost_estimate=0.001,
    )
    engine = make_engine(classification, {"שאלת דירה": [make_candidate(APT1, rerank_score=0.8)]})
    answer, _ = engine._answer("שאלת דירה")
    assert answer.category == "apartment"
    assert answer.confidence == 0.8


def test_routed_empty_pool_falls_back_without_generation():
    engine = make_engine(multi_classification(), {})
    answer, tokens = engine._answer("שאלה")
    assert answer.text == FALLBACK_TEXT
    assert engine.generator.calls == []
    assert tokens is None
    assert answer.confidence == 0.0
    assert answer.cost_estimate == pytest.approx(0.002)  # classification cost only
    assert answer.classification is not None


# --------------------------------------------------------------------------- #
# classify CLI subcommand + --route/--category exclusivity
# --------------------------------------------------------------------------- #


def test_classify_subcommand_prints_classification_json(monkeypatch, repo_root, capsys):
    classification = multi_classification()
    monkeypatch.setattr(query_cli, "load_config", lambda path: {"path": path})
    monkeypatch.setattr(
        query_cli, "build_classifier", lambda config, model=None: FakeClassifier(classification)
    )
    config = str(repo_root / "configs" / "default.yaml")
    assert query_cli.main(["classify", QUESTION, "--config", config]) == query_cli.EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "multi"
    assert out["categories"] == ["apartment", "car"]
    assert len(out["sub_questions"]) == 2


def test_route_and_category_are_mutually_exclusive(repo_root, capsys):
    config = str(repo_root / "configs" / "default.yaml")
    argv = ["--config", config, "--route", "--category", "car", "שאלה"]
    assert query_cli.main(argv) == query_cli.EXIT_CONFIG_ERROR
    assert "mutually exclusive" in capsys.readouterr().err

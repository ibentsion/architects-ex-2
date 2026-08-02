"""Query classification tests. tf_client mocked — no LLM.

Covers: strict-JSON parsing (fences/prose tolerated), category validation
against the closed 12-list, derived mode/categories, needs_calculation/
dependent flags, graceful fallback on any failure, and the classify CLI
subcommand. The agent answer flow lives in tests/test_agent.py.
"""
from __future__ import annotations

import json

import pytest

import rag.classify as classify_mod
from rag.classify import CATEGORIES, QueryClassifier, _system_prompt
from rag.cli import query as query_cli
from rag.types import Classification, SubQuestion


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


def test_classify_parses_calculation_and_dependency_flags(monkeypatch):
    reply = json.dumps(
        {
            "sub_questions": [{"question": "כמה זה 15% מההשתתפות העצמית?", "categories": ["car"]}],
            "needs_calculation": True,
            "dependent": True,
        },
        ensure_ascii=False,
    )
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(reply))
    result = QueryClassifier("m").classify("ש")
    assert result.needs_calculation is True
    assert result.dependent is True


def test_classify_flags_default_false(monkeypatch):
    reply = '{"sub_questions": [{"question": "ש", "categories": ["car"]}]}'
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(reply))
    result = QueryClassifier("m").classify("ש")
    assert result.needs_calculation is False and result.dependent is False
    assert result.estimated_difficulty == "medium"  # unstated -> the safe middle


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_classify_parses_difficulty(monkeypatch, difficulty):
    reply = json.dumps(
        {"sub_questions": [{"question": "ש", "categories": ["car"]}], "difficulty": difficulty},
        ensure_ascii=False,
    )
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(reply))
    assert QueryClassifier("m").classify("ש").estimated_difficulty == difficulty


@pytest.mark.parametrize("value", ["trivial", "", 3, None])
def test_classify_unknown_difficulty_falls_back_to_medium(monkeypatch, value):
    reply = json.dumps(
        {"sub_questions": [{"question": "ש", "categories": ["car"]}], "difficulty": value},
        ensure_ascii=False,
    )
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(reply))
    assert QueryClassifier("m").classify("ש").estimated_difficulty == "medium"


def test_classify_fallback_difficulty_is_medium(monkeypatch):
    def boom(messages, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(classify_mod, "tf_chat", boom)
    assert QueryClassifier("m").classify(QUESTION).estimated_difficulty == "medium"


def test_system_prompt_lists_all_corpus_categories():
    prompt = _system_prompt()
    assert len(CATEGORIES) == 12
    assert all(cid in prompt for cid in CATEGORIES)


# --------------------------------------------------------------------------- #
# classify CLI subcommand + --engine agent/--category exclusivity
# --------------------------------------------------------------------------- #


class FakeClassifier:
    def __init__(self, classification):
        self.classification = classification

    def classify(self, question):
        return self.classification


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


def test_engine_agent_and_category_are_mutually_exclusive(repo_root, capsys):
    config = str(repo_root / "configs" / "default.yaml")
    argv = ["--config", config, "--engine", "agent", "--category", "car", "שאלה"]
    assert query_cli.main(argv) == query_cli.EXIT_CONFIG_ERROR
    assert "mutually exclusive" in capsys.readouterr().err

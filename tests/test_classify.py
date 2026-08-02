"""Query classification tests. tf_client mocked — no LLM.

Covers: strict-JSON parsing (fences/prose tolerated), category validation
against the closed 12-list, derived mode/categories, needs_calculation/
dependent flags, graceful fallback on any failure, and the classify CLI
subcommand. The agent answer flow lives in tests/test_agent.py.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

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


# --------------------------------------------------------------------------- #
# Malformed-JSON repair
#
# Every reply below is a verbatim gpt-oss-120b emission recorded in
# eval_results/classify-sweep-20260802T043108Z. Between them the two defects
# accounted for 60 of the sweep's 64 broken replies (4-12% per arm), each of
# which used to throw away a usable classification.
# --------------------------------------------------------------------------- #

#: The model closes `sub_questions` twice: `...}]], "needs_calculation"`.
DUPLICATED_BRACKET = (
    '{"sub_questions": [{"question": "מי נחשב לאדם עם מוגבלות מקצרת חיים לצורך '
    'נספח משכנתא מיוחד?", "categories": ["mortgage"]}]], "needs_calculation": '
    'false, "dependent": false, "difficulty": "medium"}'
)

#: Hebrew gershayim written as a bare ASCII quote inside the question string
#: (בחו"ל), pretty-printed across several lines.
STRAY_QUOTE = (
    '{\n  "sub_questions": [\n    {\n      "question": "אם חלילה אמות בתאונה '
    'בחו"ל ויש לי את הרחבת תאונות אישיות בדרכון פרימיום, כמה יקבלו היורשים '
    'שלי?",\n      "categories": ["personal-accident"]\n    }\n  ],\n  '
    '"needs_calculation": true,\n  "dependent": false,\n  "difficulty": "medium"\n}'
)

#: The same reply escapes one gershayim and not the other — the repair has to
#: leave the already-correct one alone.
HALF_ESCAPED_QUOTES = (
    '{"sub_questions": [{"question": "האם ההוצאות הרפואיות שלי עקב ההיריון שזוהה '
    'בחו"ל עד לשבוע 12 יכוסו בביטוח נסיעות לחו\\"ל?", "categories": ["travel"]}], '
    '"needs_calculation": false, "dependent": false, "difficulty": "medium"}'
)


def test_repair_recovers_a_duplicated_closing_bracket(monkeypatch):
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(DUPLICATED_BRACKET))
    result = QueryClassifier("m").classify("ש")
    assert result.categories == ["mortgage"]
    assert result.estimated_difficulty == "medium"


def test_repair_recovers_an_unescaped_quote_inside_a_hebrew_string(monkeypatch):
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(STRAY_QUOTE))
    result = QueryClassifier("m").classify("ש")
    assert result.categories == ["personal-accident"]
    assert result.needs_calculation is True
    # The gershayim survives as a literal character rather than being deleted.
    assert 'בחו"ל' in result.sub_questions[0].question


def test_repair_leaves_an_already_escaped_quote_alone(monkeypatch):
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(HALF_ESCAPED_QUOTES))
    question = QueryClassifier("m").classify("ש").sub_questions[0].question
    assert question.count('"') == 2 and 'לחו"ל' in question and 'בחו"ל' in question


@pytest.mark.parametrize(
    "reply",
    [
        '{"sub_questions": [{"question": "ש", "categories": ["car"]}]}',
        '{"a": [[1, 2], [3]], "sub_questions": [{"question": "ש", "categories": []}]}',
        '{"sub_questions": [{"question": "ש", "categories": ["car", "travel"]}]}',
    ],
)
def test_repair_is_a_noop_on_well_formed_json(reply):
    """Valid replies must not go anywhere near the repair, and must survive it
    byte-for-byte if they ever do."""
    assert classify_mod._repair_json(reply) == reply
    assert classify_mod._extract_json(reply) == json.loads(reply)


def test_a_truncated_reply_is_not_repaired_into_an_invented_classification(monkeypatch):
    """Closing a cut-off object would fabricate tags the model never finished
    choosing — a truncation still falls back."""
    truncated = ('{"sub_questions": [{"question": "ש", "categories": ["car"]},\n'
                 '    {"question": "ש2", "categories": ["travel"]}')
    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat(truncated))
    result = QueryClassifier("m").classify(QUESTION)
    assert result.categories == [] and result.sub_questions[0].question == QUESTION


def test_system_prompt_lists_all_corpus_categories():
    prompt = _system_prompt()
    assert len(CATEGORIES) == 12
    assert all(cid in prompt for cid in CATEGORIES)


def test_system_prompt_carries_both_landed_treatments():
    """The shipped prompt is the task definition plus the two treatments that
    won the 260802-003 sweep, composed in that order."""
    prompt = _system_prompt()
    assert prompt.startswith(classify_mod._base_prompt())
    assert classify_mod.ABSTAIN_RULE in prompt
    assert prompt.endswith(classify_mod.DECISION_RULES)


# --------------------------------------------------------------------------- #
# Retrieval evidence (the hint)
# --------------------------------------------------------------------------- #


def fake_hit(category, file, text="טקסט כלשהו"):
    return SimpleNamespace(chunk=SimpleNamespace(category=category, file=file, text=text))


def test_hint_summarises_the_category_distribution_of_the_hits():
    hits = [fake_hit("apartment", "apartment/a.pdf"), fake_hit("apartment", "apartment/b.pdf"),
            fake_hit("mortgage", "mortgage/c.pdf")]
    summary = classify_mod.hint_from_hits(hits)
    assert summary["n_hits"] == 3
    assert summary["top_category"] == "apartment"
    assert summary["top_share"] == pytest.approx(2 / 3, abs=1e-4)
    # Ordered by frequency, so the block reads as a ranking.
    assert list(summary["histogram"]) == ["apartment", "mortgage"]
    assert len(summary["snippets"]) == 3


def test_hint_snippets_are_capped_and_whitespace_collapsed():
    hits = [fake_hit("car", "car/a.pdf", "שורה\n\n   אחת   " + "א" * 500)]
    snippet = classify_mod.hint_from_hits(hits)["snippets"][0]
    assert len(snippet["text"]) == classify_mod.HINT_SNIPPET_CHARS
    assert "\n" not in snippet["text"] and "   " not in snippet["text"]


def test_render_hint_says_so_when_the_index_returns_nothing():
    assert classify_mod.render_hint(classify_mod.hint_from_hits([])) == \
        "החיפוש באינדקס לא החזיר תוצאות."
    assert classify_mod.render_hint(None) == "החיפוש באינדקס לא החזיר תוצאות."


def test_build_hint_uses_fuse_depth_and_never_reranks():
    calls = []

    class Retriever:
        def fuse(self, question):
            calls.append(question)
            return [fake_hit("travel", "travel/x.pdf")] * 25

        def retrieve(self, *a, **kw):  # pragma: no cover - must never be called
            raise AssertionError("build_hint must not run the reranker")

    rendered, summary = classify_mod.build_hint(Retriever(), "שאלה")
    assert calls == ["שאלה"]
    assert summary["n_hits"] == classify_mod.HINT_TOP_K
    assert f"ב-{classify_mod.HINT_TOP_K} התוצאות המובילות" in rendered


def test_hint_reaches_the_model_as_a_delimited_block_below_the_question(monkeypatch):
    seen = {}

    def _chat(messages, **kwargs):
        seen["user"] = messages[1]["content"]
        return json.dumps({"sub_questions": [{"question": "ש", "categories": ["car"]}]}), \
            {"prompt": 1, "completion": 1, "finish_reason": "stop"}, 0.0

    monkeypatch.setattr(classify_mod, "tf_chat", _chat)
    QueryClassifier("m").classify("שאלה", hint="התפלגות")

    assert seen["user"].startswith("שאלה")
    assert "התפלגות" in seen["user"]
    # The question keeps the last word: the block is advisory, and says so.
    assert "השאלה עצמה קובעת" in seen["user"]


# --------------------------------------------------------------------------- #
# classify CLI subcommand + --engine agent/--category exclusivity
# --------------------------------------------------------------------------- #


class FakeClassifier:
    def __init__(self, classification):
        self.classification = classification
        self.hint = None

    def classify(self, question, hint=None):
        self.hint = hint
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
    fake = FakeClassifier(classification)
    monkeypatch.setattr(query_cli, "load_config", lambda path: {"path": path})
    monkeypatch.setattr(query_cli, "build_classifier", lambda config, model=None: fake)
    monkeypatch.setattr(query_cli, "load_retriever", lambda config: object())
    monkeypatch.setattr(query_cli, "build_hint", lambda retriever, question: ("ראיות", {}))
    config = str(repo_root / "configs" / "default.yaml")
    assert query_cli.main(["classify", QUESTION, "--config", config]) == query_cli.EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "multi"
    assert out["categories"] == ["apartment", "car"]
    assert len(out["sub_questions"]) == 2
    # The tool mirrors production: same evidence pass, or it reports something
    # the agent never does.
    assert fake.hint == "ראיות"


def test_engine_agent_and_category_are_mutually_exclusive(repo_root, capsys):
    config = str(repo_root / "configs" / "default.yaml")
    argv = ["--config", config, "--engine", "agent", "--category", "car", "שאלה"]
    assert query_cli.main(argv) == query_cli.EXIT_CONFIG_ERROR
    assert "mutually exclusive" in capsys.readouterr().err

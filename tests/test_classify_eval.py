"""Classifier arm-sweep tests. tf_client and the retriever are mocked — no
network, no index.

Covers the parts that decide what the sweep concludes: the useful/harmful/none
verdict for every prediction shape, the metric arithmetic, the paired
fixed/broken counting and its exact McNemar p, hint rendering, self-consistency
voting, the verify stage's retraction-only contract, and the --limit /
--max-cost guards. Also pins the prompt registry: `baseline` must be the
production prompt, and no example question may be lifted from a reference set.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

import rag.classify as classify_mod
from evalharness import classify_eval as ce
from evalharness.classify_arms import get_arm
from evalharness.classify_prompts import (
    EXAMPLE_QUESTIONS,
    PROMPT_VARIANTS,
    build_prompt,
    build_verify_user_message,
)
from rag.classify import CATEGORIES, _system_prompt

# --------------------------------------------------------------------------- #
# Verdicts — the engine's filter semantics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("categories", "expected_filter", "expected_verdict"),
    [
        (["car"], "car", "useful"),                 # exactly gold
        (["apartment"], "apartment", "harmful"),    # exactly one wrong tag
        ([], None, "none"),                         # abstained
        (["car", "apartment"], None, "none"),       # 2+ tags => engine filters nothing
        (["apartment", "travel"], None, "none"),    # 2+ wrong tags are still idle
    ],
)
def test_verdict_covers_every_prediction_shape(categories, expected_filter, expected_verdict):
    assert ce.effective_filter(categories) == expected_filter
    assert ce.verdict_for(categories, "car") == expected_verdict


def test_applied_filters_records_what_the_engine_really_filters():
    """A 2-sub-question query filters twice even though its union has 2 tags —
    the headline verdict calls that 'none', so the detail is kept separately."""
    sub_questions = [
        {"question": "a", "categories": ["apartment"]},
        {"question": "b", "categories": ["car"]},
        {"question": "c", "categories": ["car", "travel"]},  # 2 tags: no filter
        {"question": "d", "categories": []},
    ]
    assert ce.applied_filters(sub_questions) == ["apartment", "car"]
    assert ce.derive_categories(sub_questions) == ["apartment", "car", "travel"]
    assert ce.verdict_for(ce.derive_categories(sub_questions), "car") == "none"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _record(qid, gold, categories, *, set_name="v2", latency=100, cost=0.001,
            parse_failed=False, finish_reason="stop", subs=None):
    sub_questions = subs if subs is not None else [{"question": qid, "categories": categories}]
    derived = ce.derive_categories(sub_questions)
    return {
        "id": qid,
        "set": set_name,
        "gold": gold,
        "question": qid,
        "sub_questions": sub_questions,
        "categories": derived,
        "effective_filter": ce.effective_filter(derived),
        "verdict": ce.verdict_for(derived, gold),
        "applied_filters": ce.applied_filters(sub_questions),
        "recall_any": gold in derived,
        "parse_failed": parse_failed,
        "finish_reason": finish_reason,
        "n_calls": 1,
        "latency_ms": latency,
        "cost_usd": cost,
        "raw_reply": "{}",
    }


def test_summarize_metric_arithmetic():
    records = [
        _record("q1", "car", ["car"]),                        # useful
        _record("q2", "car", ["apartment"]),                  # harmful
        _record("q3", "health", []),                          # none, no recall
        _record("q4", "life", ["life", "mortgage"],           # none, recall yes
                latency=300, finish_reason="length"),
        _record("q5", "travel", ["apartment"], parse_failed=True, latency=500),
    ]
    summary = ce.summarize(records)
    assert summary["n"] == 5
    assert summary["harmful_rate"] == 0.4          # q2, q5
    assert summary["filter_correct_rate"] == 0.2   # q1
    assert summary["no_filter_rate"] == 0.4        # q3, q4
    assert summary["recall_any"] == 0.4            # q1, q4
    assert summary["parse_fail_rate"] == 0.2
    assert summary["truncation_rate"] == 0.2
    assert summary["mean_sub_questions"] == 1.0
    assert summary["mean_tags_per_sub"] == 1.0     # (1+1+0+2+1)/5
    assert summary["p50_ms"] == 100                # 100,100,100,300,500
    assert summary["total_cost_usd"] == 0.005
    # q4's two tags mean no filter is applied at all, so it contributes nothing
    # to the sub-level diagnostic: 3 filters applied, 2 of them wrong.
    assert summary["sub_filter_count"] == 3
    assert summary["sub_harmful_rate"] == pytest.approx(0.6667, abs=1e-4)


def test_summarize_recall_any_counts_gold_anywhere_in_the_union():
    records = [
        _record("q1", "car", ["car"]),                   # gold, alone
        _record("q2", "life", ["life", "mortgage"]),     # gold, among others
        _record("q3", "health", ["apartment"]),          # gold absent
        _record("q4", "health", []),                     # gold absent
    ]
    assert ce.summarize(records)["recall_any"] == 0.5


def test_summarize_empty_bucket_is_not_a_crash():
    assert ce.summarize([]) == {"n": 0}


def test_confusion_table_is_gold_by_predicted_filter():
    records = [
        _record("q1", "car", ["car"]),
        _record("q2", "car", ["apartment"]),
        _record("q3", "car", []),
        _record("q4", "health", ["long-term-care"]),
    ]
    table = ce.confusion(records)
    assert table["car"] == {"apartment": 1, "car": 1, ce.NO_FILTER: 1}
    assert table["health"] == {"long-term-care": 1}


# --------------------------------------------------------------------------- #
# Paired stats
# --------------------------------------------------------------------------- #


def test_paired_counts_fixed_and_broken_on_shared_ids():
    baseline = [
        _record("q1", "car", ["apartment"]),   # harmful
        _record("q2", "car", ["apartment"]),   # harmful
        _record("q3", "car", ["car"]),         # fine
        _record("q4", "car", ["car"]),         # fine
        _record("q5", "car", ["car"]),         # arm never ran this one
    ]
    arm = [
        _record("q1", "car", ["car"]),         # fixed
        _record("q2", "car", []),              # fixed (no filter is not harmful)
        _record("q3", "car", ["travel"]),      # broken
        _record("q4", "car", ["car"]),         # unchanged
    ]
    stats = ce.paired(baseline, arm)
    assert stats["n_pairs"] == 4               # q5 has no counterpart
    assert stats["n_fixed"] == 2
    assert stats["n_broken"] == 1
    assert stats["fixed_ids"] == ["q1", "q2"]
    assert stats["broken_ids"] == ["q3"]


@pytest.mark.parametrize(
    ("n_fixed", "n_broken", "expected"),
    [
        (0, 0, 1.0),        # no discordant pairs — nothing was shown
        (3, 3, 1.0),        # symmetric, capped at 1
        (5, 0, 0.0625),     # 2 * (1/32)
        (6, 0, 0.03125),    # 2 * (1/64) — the first count that clears 0.05
        (0, 6, 0.03125),    # direction-free: the p-value alone never says who won
        (8, 2, 0.109375),   # 2 * (1 + 10 + 45) / 1024
    ],
)
def test_mcnemar_exact_two_sided(n_fixed, n_broken, expected):
    assert ce.mcnemar_exact(n_fixed, n_broken) == pytest.approx(expected)


def test_mcnemar_flags_a_small_win_as_not_significant():
    """The guard the whole sweep rests on: 4 fixed vs 1 broken looks like a
    win and is not one."""
    assert ce.mcnemar_exact(4, 1) > 0.05


# --------------------------------------------------------------------------- #
# Hints
# --------------------------------------------------------------------------- #


HINT = {
    "n_hits": 10,
    "histogram": {"apartment": 6, "car": 4},
    "top_category": "apartment",
    "top_share": 0.6,
    "snippets": [
        {"rank": 1, "file": "apartment/files/a.pdf", "category": "apartment",
         "text": "נזקי מים בדירה"},
        {"rank": 2, "file": "car/files/b.pdf", "category": "car", "text": "השתתפות עצמית"},
    ],
}


def test_render_hint_shows_histogram_shares_and_snippets():
    rendered = ce.render_hint(HINT)
    assert "apartment: 6 מתוך 10 (60%)" in rendered
    assert "car: 4 מתוך 10 (40%)" in rendered
    assert "1. [apartment] apartment/files/a.pdf" in rendered
    assert "נזקי מים בדירה" in rendered


def test_render_hint_handles_an_empty_retrieval():
    for empty in (None, {}, {"n_hits": 0, "histogram": {}, "snippets": []}):
        assert ce.render_hint(empty) == "החיפוש באינדקס לא החזיר תוצאות."


def test_hint_is_appended_to_the_user_turn_not_the_system_prompt(monkeypatch):
    seen = {}

    def fake_chat(messages, **kwargs):
        seen["messages"] = messages
        return '{"sub_questions": [{"question": "q", "categories": ["car"]}]}', \
               {"finish_reason": "stop"}, 0.001

    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat)
    classify_mod.QueryClassifier("m").classify("שאלה", hint=ce.render_hint(HINT))
    assert seen["messages"][0]["content"] == _system_prompt()   # untouched
    user = seen["messages"][1]["content"]
    assert user.startswith("שאלה")
    assert "ראיות מהאינדקס" in user and "apartment: 6" in user


def test_hint_defaults_to_the_bare_question(monkeypatch):
    seen = {}

    def fake_chat(messages, **kwargs):
        seen["messages"] = messages
        return '{"sub_questions": [{"question": "q", "categories": []}]}', \
               {"finish_reason": "stop"}, 0.0

    monkeypatch.setattr(classify_mod, "tf_chat", fake_chat)
    classify_mod.QueryClassifier("m").classify("שאלה")
    assert seen["messages"][1]["content"] == "שאלה"


def test_build_hints_summarizes_the_top_hits(monkeypatch):
    def chunk(category, file, text):
        return SimpleNamespace(chunk=SimpleNamespace(
            category=category, file=file, text=text))

    hits = ([chunk("apartment", "apartment/f.pdf", "נזק   מים\nבדירה")] * 7 +
            [chunk("car", "car/f.pdf", "רכב")] * 5)
    retriever = SimpleNamespace(fuse=lambda q: hits)
    questions = [{"id": "q1", "question": "שאלה"}]

    hints = ce.build_hints(retriever, questions, workers=1)
    assert hints["q1"]["n_hits"] == ce.HINT_TOP_K        # top 10 of the 12 fused
    assert hints["q1"]["histogram"] == {"apartment": 7, "car": 3}
    assert hints["q1"]["top_category"] == "apartment"
    assert hints["q1"]["top_share"] == 0.7
    assert len(hints["q1"]["snippets"]) == ce.HINT_SNIPPETS
    assert hints["q1"]["snippets"][0]["text"] == "נזק מים בדירה"   # whitespace collapsed


# --------------------------------------------------------------------------- #
# Self-consistency voting
# --------------------------------------------------------------------------- #


def test_vote_categories_keeps_only_categories_with_enough_votes():
    samples = [["car", "travel"], ["car"], ["car", "apartment"]]
    assert ce.vote_categories(samples, min_votes=2) == {"car"}
    assert ce.vote_categories(samples, min_votes=1) == {"car", "travel", "apartment"}
    assert ce.vote_categories(samples, min_votes=4) == set()


def test_vote_categories_counts_a_repeated_category_once_per_sample():
    assert ce.vote_categories([["car", "car"], ["travel"]], min_votes=2) == set()


def test_selfcons_arm_drops_the_minority_tag(monkeypatch, tiny_config):
    """3 samples, 2 of which say car and 1 of which also says travel: travel
    has one vote and must not survive."""
    replies = iter([
        '{"sub_questions": [{"question": "a", "categories": ["car"]}, '
        '{"question": "b", "categories": ["travel"]}]}',
        '{"sub_questions": [{"question": "a", "categories": ["car"]}]}',
        '{"sub_questions": [{"question": "a", "categories": ["car"]}]}',
    ])

    def fake_chat(messages, **kwargs):
        return next(replies), {"finish_reason": "stop"}, 0.001

    with _installed(fake_chat) as (recorder, capture):
        runner = ce.ArmRunner(get_arm("selfcons-3"), tiny_config, {}, recorder, capture)
        record = runner.predict({"id": "q1", "set": "v2", "domain": "car",
                                 "question": "שאלה"})
    assert record["categories"] == ["car"]
    assert record["verdict"] == "useful"
    assert record["n_calls"] == 3
    assert record["cost_usd"] == pytest.approx(0.003)


# --------------------------------------------------------------------------- #
# Arm execution
# --------------------------------------------------------------------------- #


class _Installed:
    """Install a fake chat under rag.classify plus the run's warning capture,
    exactly as `main` does."""

    def __init__(self, chat, max_cost=10.0):
        self.budget = ce.Budget(max_cost)
        self.recorder = ce.ChatRecorder(self.budget, chat=chat, max_retries=1,
                                        base_delay=0.0)
        self.capture = ce.WarningCapture()

    def __enter__(self):
        self._original = classify_mod.tf_chat
        classify_mod.tf_chat = self.recorder
        logging.getLogger("rag.classify").addHandler(self.capture)
        return self.recorder, self.capture

    def __exit__(self, *exc):
        classify_mod.tf_chat = self._original
        logging.getLogger("rag.classify").removeHandler(self.capture)
        return False


def _installed(chat, max_cost=10.0):
    return _Installed(chat, max_cost)


@pytest.fixture
def tiny_config():
    return SimpleNamespace(harness=SimpleNamespace(
        orchestrator_model="openai/gpt-oss-120b",
        orchestrator_extra_params={"reasoning_effort": "low",
                                   "allowed_openai_params": ["reasoning_effort"]}))


QUESTIONS = [
    {"id": "q1", "set": "v2", "domain": "car", "question": "שאלת רכב"},
    {"id": "q2", "set": "v2", "domain": "apartment", "question": "שאלת דירה"},
]


def test_run_arm_produces_a_full_record_per_question(tiny_config):
    def fake_chat(messages, **kwargs):
        question = messages[1]["content"]
        category = "car" if "רכב" in question else "travel"
        return (json.dumps({"sub_questions": [{"question": question,
                                               "categories": [category]}]},
                           ensure_ascii=False),
                {"finish_reason": "stop"}, 0.0005)

    with _installed(fake_chat) as (recorder, capture):
        records, aborted = ce.run_arm(get_arm("baseline"), QUESTIONS, tiny_config,
                                      {}, recorder, capture, concurrency=2)
    assert not aborted
    assert [r["id"] for r in records] == ["q1", "q2"]          # input order
    assert [r["verdict"] for r in records] == ["useful", "harmful"]
    assert records[1]["effective_filter"] == "travel"
    for record in records:
        assert set(record) >= {"id", "set", "gold", "question", "sub_questions",
                               "categories", "effective_filter", "verdict",
                               "recall_any", "parse_failed", "finish_reason",
                               "latency_ms", "cost_usd", "raw_reply"}
        assert record["parse_failed"] is False
        assert record["cost_usd"] == pytest.approx(0.0005)


def test_parse_failure_is_recorded_not_silently_scored_as_an_abstention(tiny_config):
    """rag.classify swallows a bad reply into a no-filter fallback; the eval
    must still be able to tell the two apart."""
    def fake_chat(messages, **kwargs):
        return "אין לי מושג", {"finish_reason": "stop"}, 0.0005

    with _installed(fake_chat) as (recorder, capture):
        records, _ = ce.run_arm(get_arm("baseline"), QUESTIONS, tiny_config,
                                {}, recorder, capture, concurrency=1)
    assert all(r["parse_failed"] for r in records)
    assert all(r["verdict"] == "none" for r in records)


def test_a_multi_sample_arm_counts_each_failed_sample(tiny_config):
    """selfcons-3 has three chances to fall back, so `parse_failed` alone would
    not be comparable with a single-call arm's."""
    replies = iter([
        "אין לי מושג",                                                  # sample 1 fails
        '{"sub_questions": [{"question": "q", "categories": ["car"]}]}',
        '{"sub_questions": [{"question": "q", "categories": ["car"]}]}',
    ])

    def fake_chat(messages, **kwargs):
        return next(replies), {"finish_reason": "stop"}, 0.001

    with _installed(fake_chat) as (recorder, capture):
        runner = ce.ArmRunner(get_arm("selfcons-3"), tiny_config, {}, recorder, capture)
        record = runner.predict(QUESTIONS[0])
    assert record["parse_failed"] is True
    assert record["n_parse_failures"] == 1
    assert record["n_calls"] == 3
    # Sample 1 fell back to a single no-category sub-question, and the voted
    # categories have nowhere to attach — a real cost of the strategy.
    assert record["categories"] == []
    assert ce.summarize([record])["calls_per_question"] == 3.0


def test_truncated_reply_is_flagged(tiny_config):
    def fake_chat(messages, **kwargs):
        return '{"sub_questions": [{"question": "q", "categories": ["car"]}]}', \
               {"finish_reason": "length"}, 0.0005

    with _installed(fake_chat) as (recorder, capture):
        records, _ = ce.run_arm(get_arm("baseline"), QUESTIONS[:1], tiny_config,
                                {}, recorder, capture, concurrency=1)
    assert ce.summarize(records)["truncation_rate"] == 1.0


def test_retryable_error_is_retried_then_succeeds(tiny_config):
    calls = {"n": 0}

    def flaky_chat(messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 rate limit exceeded")
        return '{"sub_questions": [{"question": "q", "categories": ["car"]}]}', \
               {"finish_reason": "stop"}, 0.0005

    with _installed(flaky_chat) as (recorder, capture):
        records, _ = ce.run_arm(get_arm("baseline"), QUESTIONS[:1], tiny_config,
                                {}, recorder, capture, concurrency=1)
    assert calls["n"] == 2
    assert records[0]["verdict"] == "useful"
    assert records[0]["parse_failed"] is False


def test_non_retryable_error_is_not_retried(tiny_config):
    calls = {"n": 0}

    def bad_chat(messages, **kwargs):
        calls["n"] += 1
        raise ValueError("invalid model id")

    with _installed(bad_chat) as (recorder, capture):
        records, _ = ce.run_arm(get_arm("baseline"), QUESTIONS[:1], tiny_config,
                                {}, recorder, capture, concurrency=1)
    assert calls["n"] == 1
    assert records[0]["parse_failed"] is True


def test_hint_vote_arm_uses_the_index_alone(tiny_config):
    def never_called(messages, **kwargs):
        raise AssertionError("hint-vote must not call the LLM")

    hints = {
        "q1": {**HINT, "top_category": "car", "top_share": 0.7},
        "q2": {**HINT, "top_category": "car", "top_share": 0.4},   # below threshold
    }
    with _installed(never_called) as (recorder, capture):
        records, _ = ce.run_arm(get_arm("hint-vote"), QUESTIONS, tiny_config,
                                hints, recorder, capture, concurrency=1)
    assert records[0]["categories"] == ["car"] and records[0]["verdict"] == "useful"
    assert records[1]["categories"] == [] and records[1]["verdict"] == "none"
    assert all(r["cost_usd"] == 0.0 and r["n_calls"] == 0 for r in records)


# --------------------------------------------------------------------------- #
# verify-2stage — retraction only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"keep": ["car"]}', ["car"]),
        ('{"keep": []}', []),
        ('{"keep": ["car", "travel"]}', ["car"]),      # travel was never proposed
        ('{"keep": ["not-a-category"]}', []),          # outside the closed list
    ],
)
def test_parse_verify_reply_can_only_remove_tags(reply, expected):
    assert ce.parse_verify_reply(reply, ["car"]) == expected


def test_parse_verify_reply_rejects_a_reply_with_no_keep_list():
    with pytest.raises(Exception):
        ce.parse_verify_reply('{"verdict": "ok"}', ["car"])


def test_verify_2stage_retracts_a_tag(tiny_config):
    replies = iter([
        '{"sub_questions": [{"question": "q", "categories": ["apartment"]}]}',
        '{"keep": []}',
    ])

    def fake_chat(messages, **kwargs):
        return next(replies), {"finish_reason": "stop"}, 0.0005

    with _installed(fake_chat) as (recorder, capture):
        runner = ce.ArmRunner(get_arm("verify-2stage"), tiny_config,
                              {"q1": HINT}, recorder, capture)
        record = runner.predict(QUESTIONS[0])
    assert record["categories"] == []
    assert record["verdict"] == "none"          # was harmful before the retraction
    assert record["n_calls"] == 2


def test_verify_2stage_keeps_the_first_stage_tags_when_the_check_fails(tiny_config):
    replies = iter([
        '{"sub_questions": [{"question": "q", "categories": ["car"]}]}',
        "not json at all",
    ])

    def fake_chat(messages, **kwargs):
        return next(replies), {"finish_reason": "stop"}, 0.0005

    with _installed(fake_chat) as (recorder, capture):
        runner = ce.ArmRunner(get_arm("verify-2stage"), tiny_config,
                              {"q1": HINT}, recorder, capture)
        record = runner.predict(QUESTIONS[0])
    assert record["categories"] == ["car"]


def test_verify_2stage_skips_the_second_call_when_nothing_was_proposed(tiny_config):
    def fake_chat(messages, **kwargs):
        return '{"sub_questions": [{"question": "q", "categories": []}]}', \
               {"finish_reason": "stop"}, 0.0005

    with _installed(fake_chat) as (recorder, capture):
        runner = ce.ArmRunner(get_arm("verify-2stage"), tiny_config,
                              {"q1": HINT}, recorder, capture)
        record = runner.predict(QUESTIONS[0])
    assert record["n_calls"] == 1


def test_verify_user_message_carries_the_question_tags_and_evidence():
    message = build_verify_user_message("שאלה", ["car"], ce.render_hint(HINT))
    assert "שאלה" in message and "car" in message and "apartment: 6" in message


# --------------------------------------------------------------------------- #
# Guards: --limit and --max-cost
# --------------------------------------------------------------------------- #


def test_limit_takes_the_first_n_of_each_set(tmp_path):
    v1 = tmp_path / "questions.json"
    v2 = tmp_path / "questions_v2.json"
    v1.write_text(json.dumps([{"id": f"a{i}", "question": "q", "domain": "car"}
                              for i in range(5)]), encoding="utf-8")
    v2.write_text(json.dumps([{"id": f"b{i}", "question": "q", "domain": "car"}
                              for i in range(5)]), encoding="utf-8")

    questions = ce.load_questions([str(v1), str(v2)], limit=2)
    assert [q["id"] for q in questions] == ["a0", "a1", "b0", "b1"]
    assert [q["set"] for q in questions] == ["v1", "v1", "v2", "v2"]
    assert len(ce.load_questions([str(v1), str(v2)])) == 10


def test_colliding_question_ids_across_sets_are_refused(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([{"id": "dup", "question": "q", "domain": "car"}]),
                    encoding="utf-8")
    with pytest.raises(SystemExit):
        ce.load_questions([str(path), str(path)])


def test_max_cost_aborts_the_arm_and_keeps_what_finished(tiny_config):
    def fake_chat(messages, **kwargs):
        return '{"sub_questions": [{"question": "q", "categories": ["car"]}]}', \
               {"finish_reason": "stop"}, 0.004

    questions = [{"id": f"q{i}", "set": "v2", "domain": "car", "question": "שאלה"}
                 for i in range(6)]
    with _installed(fake_chat, max_cost=0.01) as (recorder, capture):
        records, aborted = ce.run_arm(get_arm("baseline"), questions, tiny_config,
                                      {}, recorder, capture, concurrency=1)
    assert aborted
    assert 0 < len(records) < len(questions)


def test_cost_ceiling_is_not_swallowed_by_the_classifier_fallback(tiny_config):
    """CostExceeded has to escape rag.classify's blanket except — otherwise the
    abort would look like a run of legitimate no-filter predictions."""
    budget = ce.Budget(0.0)
    recorder = ce.ChatRecorder(budget, chat=lambda *a, **k: ("", {}, 0.0))
    original = classify_mod.tf_chat
    classify_mod.tf_chat = recorder
    try:
        with pytest.raises(ce.CostExceeded):
            classify_mod.QueryClassifier("m").classify("שאלה")
    finally:
        classify_mod.tf_chat = original


def test_budget_tracks_the_running_total():
    budget = ce.Budget(1.0)
    budget.add(0.4)
    budget.check()
    budget.add(0.7)
    with pytest.raises(ce.CostExceeded):
        budget.check()
    assert budget.total == pytest.approx(1.1)


# --------------------------------------------------------------------------- #
# Arm wiring
# --------------------------------------------------------------------------- #


def test_baseline_arm_reproduces_the_production_classifier(tiny_config):
    classifier = ce.build_arm_classifier(get_arm("baseline"), tiny_config)
    assert classifier.model == "openai/gpt-oss-120b"
    assert classifier.system_prompt == _system_prompt()
    assert classifier.extra_params == tiny_config.harness.orchestrator_extra_params
    assert classifier.temperature == 0.0


def test_model_arms_drop_the_orchestrator_reasoning_knobs(tiny_config):
    """Same rule as rag.classify.build_classifier: the reasoning knobs are
    gpt-oss-specific and would be rejected by another model."""
    classifier = ce.build_arm_classifier(get_arm("model-qwen"), tiny_config)
    assert classifier.model == "Qwen/Qwen3-235B-A22B-Instruct-2507"
    assert classifier.extra_params == {}


def test_effort_medium_changes_only_the_reasoning_effort(tiny_config):
    classifier = ce.build_arm_classifier(get_arm("effort-medium"), tiny_config)
    assert classifier.model == "openai/gpt-oss-120b"
    assert classifier.extra_params["reasoning_effort"] == "medium"
    assert classifier.system_prompt == _system_prompt()


def test_every_arm_builds_and_only_hint_arms_ask_for_hints(tiny_config):
    from evalharness.classify_arms import ARMS

    assert len(ARMS) == 12
    assert len({a.id for a in ARMS}) == 12
    needs = {a.id for a in ARMS if a.needs_hints}
    assert needs == {"hint-sparse", "hint-vote", "verify-2stage"}
    for arm in ARMS:
        if arm.strategy != "hint_vote":
            ce.build_arm_classifier(arm, tiny_config)


# --------------------------------------------------------------------------- #
# Prompt registry
# --------------------------------------------------------------------------- #


def test_baseline_variant_is_the_production_prompt():
    assert build_prompt("baseline") == _system_prompt()


@pytest.mark.parametrize("name", sorted(set(PROMPT_VARIANTS) - {"baseline"}))
def test_each_variant_differs_from_baseline_and_keeps_the_closed_list(name):
    prompt = build_prompt(name)
    assert prompt != _system_prompt()
    assert all(cid in prompt for cid in CATEGORIES)
    assert '"sub_questions"' in prompt          # the output contract survives


def test_rich_desc_replaces_the_category_descriptions():
    prompt = build_prompt("rich-desc")
    plain = "\n".join(f"- {cid}: {desc}" for cid, desc in CATEGORIES.items())
    assert plain not in prompt
    assert "לא כולל" in prompt and "הבחנות בין משפחות תחומים חופפות" in prompt


def test_additive_variants_contain_the_baseline_verbatim():
    for name in ("abstain", "examples", "decision-rules"):
        assert _system_prompt() in build_prompt(name)


def test_decision_rules_is_not_also_an_abstain_arm():
    """The two treatments must stay separable: only `abstain` may argue that a
    wrong tag costs more than no tag."""
    assert "גרוע יותר מהיעדר תג" in build_prompt("abstain")
    assert "גרוע יותר מהיעדר תג" not in build_prompt("decision-rules")


def test_examples_cover_every_category_twice():
    assert set(EXAMPLE_QUESTIONS) == set(CATEGORIES)
    assert all(len(v) == 2 for v in EXAMPLE_QUESTIONS.values())
    prompt = build_prompt("examples")
    for examples in EXAMPLE_QUESTIONS.values():
        assert all(example in prompt for example in examples)


def test_no_example_question_is_lifted_from_a_reference_set(repo_root):
    """A prompt example drawn from an evaluation question would leak the answer
    key into the arm. Shared *vocabulary* is unavoidable in one domain; a shared
    run of 20+ characters is a paraphrase."""
    references = []
    for name in ("reference_questions.json", "reference_questions_v2.json"):
        path = repo_root / name
        references += [q["question"] for q in
                       json.loads(path.read_text(encoding="utf-8"))]
    assert len(references) == 169

    def longest_common_run(a: str, b: str) -> int:
        previous = [0] * (len(b) + 1)
        best = 0
        for i in range(1, len(a) + 1):
            current = [0] * (len(b) + 1)
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    current[j] = previous[j - 1] + 1
                    best = max(best, current[j])
            previous = current
        return best

    for category, examples in EXAMPLE_QUESTIONS.items():
        for example in examples:
            assert example not in references
            worst = max(longest_common_run(example, ref) for ref in references)
            assert worst < 20, f"{category} example overlaps a reference question " \
                               f"by {worst} characters: {example}"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_report_renders_with_paired_stats_and_confusion(tmp_path):
    results = {
        "baseline": [_record("q1", "car", ["apartment"]), _record("q2", "life", ["life"])],
        "abstain": [_record("q1", "car", []), _record("q2", "life", ["life"])],
    }
    arms = {arm_id: get_arm(arm_id) for arm_id in results}
    agg = ce.aggregate(results, arms)
    assert agg["abstain"]["paired_vs_baseline"]["pooled"]["n_fixed"] == 1
    assert "paired_vs_baseline" not in agg["baseline"]

    report = ce.render_report(agg, {"run_name": "t", "date": "2026-08-02",
                                    "questions_files": ["ref.json"],
                                    "n_questions": 2, "total_cost_usd": 0.01})
    assert "`abstain`" in report
    assert "McNemar" in report
    assert "gold \\ predicted" in report

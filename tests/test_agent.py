"""Agent harness tests: calculator safety, engine routing (fast paths vs
tool loop), concurrency plumbing, and degradation. All backends/LLMs faked.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import rag.agent.engine as engine_mod
from rag.agent.calculator import CalculationError, calculate
from rag.agent.engine import AgentEngine
from rag.generate.generator import GenerationResult
from rag.generate.prompts import FALLBACK_TEXT
from rag.types import Classification, SubQuestion

from tests.test_retrieve import APT1, APT2, TRV1, make_candidate


# --------------------------------------------------------------------------- #
# Calculator — tool computes, LLM never does arithmetic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("0.15 * 2340", 351.0),
        ("(1500 + 300) / 2", 900.0),
        ("2 ** 10", 1024.0),
        ("17 // 5", 3.0),
        ("17 % 5", 2.0),
        ("-5 + +3", -2.0),
        ("round(3.14159, 2)", 3.14),
        ("min(3, 1, 2)", 1.0),
        ("max(3, 1, 2)", 3.0),
        ("abs(-7.5)", 7.5),
    ],
)
def test_calculate_supported_arithmetic(expression, expected):
    assert calculate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('rm -rf /')",  # call of a name
        "().__class__",  # attribute access
        "x + 1",  # free variable
        "[1,2][0]",  # subscript/list
        "'a' * 3",  # string operand
        "1 if True else 2",  # conditional
        "lambda: 1",  # lambda
        "1; 2",  # not an expression
        "round(1.5, ndigits=0)",  # keyword args rejected
        "9 ** 999",  # exponent bomb
        "1 / 0",  # division by zero
        "True + 1",  # bool literal
    ],
)
def test_calculate_rejects_unsafe_or_invalid(expression):
    with pytest.raises(CalculationError):
        calculate(expression)


def test_calculate_rejects_overlong_expression():
    with pytest.raises(CalculationError, match="longer"):
        calculate("1+" * 300 + "1")


# --------------------------------------------------------------------------- #
# AgentEngine — routing, concurrency plumbing, loop, degradation
# --------------------------------------------------------------------------- #


class FakeStatsRetriever:
    def __init__(self, results_by_question, empty_for_categories=()):
        self.results_by_question = results_by_question
        self.empty_for_categories = set(empty_for_categories)
        self.calls = []

    def retrieve_with_stats(self, question, category=None, **overrides):
        self.calls.append((question, category))
        if category in self.empty_for_categories:
            results = []  # wrong category tag -> filtered pool is empty
        else:
            results = self.results_by_question.get(question, [])
        return results, {"gated": {"n_chunks": len(results), "n_documents": len(results)}}


class FakeGenerator:
    def __init__(self, model="fake-synth"):
        self.model = model
        self.calls = []

    def generate(self, question, retrieved, system_addendum=None):
        self.calls.append((question, retrieved))
        self.addenda = getattr(self, "addenda", []) + [system_addendum]
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


def make_engine(
    classification,
    results_by_question,
    max_hops=4,
    empty_for_categories=(),
    fast_generator=None,
):
    engine = AgentEngine.__new__(AgentEngine)
    engine.harness = SimpleNamespace(
        orchestrator_model="fake-orch",
        orchestrator_max_tokens=512,
        orchestrator_extra_params={},
        max_hops=max_hops,
        max_workers=4,
    )
    engine.classifier = FakeClassifier(classification)
    engine.retriever = FakeStatsRetriever(results_by_question, empty_for_categories)
    engine.generator = FakeGenerator()
    engine.fast_generator = fast_generator
    return engine


def simple_classification(**overrides):
    fields = dict(
        mode="multi",
        categories=["apartment", "car"],
        sub_questions=[
            SubQuestion(question="שאלת דירה", categories=["apartment"]),
            SubQuestion(question="שאלת רכב", categories=["car", "travel"]),
        ],
        cost_estimate=0.002,
    )
    fields.update(overrides)
    return Classification(**fields)


def single_classification(**overrides):
    fields = dict(
        mode="single",
        categories=["apartment"],
        sub_questions=[SubQuestion(question="שאלת דירה", categories=["apartment"])],
        cost_estimate=0.002,
    )
    fields.update(overrides)
    return Classification(**fields)


def tool_call(name, arguments, call_id="tc-1"):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


def orchestrator_turns(monkeypatch, turns):
    """Fake tf_chat inside the agent loop: pops one scripted (content,
    tool_calls) turn per call."""
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append((list(messages), kwargs))
        content, tool_calls = turns.pop(0)
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return message, {"prompt": 100, "completion": 20, "finish_reason": "stop"}, 0.0005

    monkeypatch.setattr(engine_mod, "tf_chat", fake_chat)
    return calls


def test_fast_path_no_loop_pools_and_generates_once():
    engine = make_engine(
        simple_classification(),
        {
            "שאלת דירה": [make_candidate(APT1, rerank_score=0.9), make_candidate(APT2, rerank_score=0.5)],
            "שאלת רכב": [make_candidate(APT2, rerank_score=0.7), make_candidate(TRV1, rerank_score=0.6)],
        },
    )
    answer, tokens = engine._answer("שאלה מקורית")
    assert engine.retriever.calls == [("שאלת דירה", "apartment"), ("שאלת רכב", None)]
    assert [(r.chunk.chunk_id, r.rerank_score) for r in answer.retrieved] == [
        (APT1, 0.9), (APT2, 0.7), (TRV1, 0.6),
    ]
    assert len(engine.generator.calls) == 1
    assert engine.generator.calls[0][0] == "שאלה מקורית"  # no calc block appended
    assert engine.generator.addenda == [None]  # no calculation -> untouched prompt
    assert answer.cost_estimate == pytest.approx(0.01 + 0.002)
    assert answer.retrieval_stats == {"gated": {"n_chunks": 4, "n_documents": 4}}
    assert tokens == {"prompt": 10, "completion": 5}
    assert [t["step"] for t in answer.trace] == ["classify", "retrieve", "retrieve", "synthesize"]


def test_empty_pool_falls_back_without_generation():
    engine = make_engine(simple_classification(), {})
    answer, tokens = engine._answer("שאלה")
    assert answer.text == FALLBACK_TEXT
    assert engine.generator.calls == []
    assert tokens is None
    assert answer.cost_estimate == pytest.approx(0.002)


def test_calculation_loop_executes_tool_and_appends_results(monkeypatch):
    classification = simple_classification(needs_calculation=True)
    engine = make_engine(
        classification,
        {"שאלת דירה": [make_candidate(APT1, rerank_score=0.9)], "שאלת רכב": []},
    )
    calls = orchestrator_turns(
        monkeypatch,
        [
            (None, [tool_call("calculate", {"expression": "0.15 * 2340"})]),
            ("DONE", None),
        ],
    )
    answer, tokens = engine._answer("כמה זה 15% מ-2340?")
    # Tool computed the number (not the LLM); result appended to synthesis question.
    synth_question = engine.generator.calls[0][0]
    assert "351" in synth_question and "תוצאות חישוב" in synth_question
    assert engine.generator.addenda == [engine_mod.CALCULATION_ADDENDUM]
    # Tool-result message fed back to the orchestrator on the second turn.
    second_turn_messages = calls[1][0]
    assert second_turn_messages[-1]["role"] == "tool"
    assert second_turn_messages[-1]["content"] == "351.0"
    assert calls[0][1]["tools"] is engine_mod.TOOLS
    # Orchestrator tokens accounted on top of synthesis tokens.
    assert tokens == {"prompt": 10 + 200, "completion": 5 + 40}
    steps = [t["step"] for t in answer.trace]
    assert steps == ["classify", "retrieve", "retrieve", "orchestrator", "calculate", "orchestrator", "synthesize"]


def test_loop_retrieve_tool_merges_into_pool(monkeypatch):
    classification = simple_classification(dependent=True)
    engine = make_engine(
        classification,
        {
            "שאלת דירה": [make_candidate(APT1, rerank_score=0.9)],
            "שאלת רכב": [],
            "מה ההשתתפות העצמית": [make_candidate(TRV1, rerank_score=0.8)],
        },
    )
    orchestrator_turns(
        monkeypatch,
        [
            (None, [tool_call("retrieve", {"query": "מה ההשתתפות העצמית", "category": "travel"})]),
            ("DONE", None),
        ],
    )
    answer, _ = engine._answer("שאלה")
    assert {r.chunk.chunk_id for r in answer.retrieved} == {APT1, TRV1}
    assert ("מה ההשתתפות העצמית", "travel") in engine.retriever.calls


def test_loop_respects_max_hops(monkeypatch):
    classification = simple_classification(needs_calculation=True)
    engine = make_engine(
        classification, {"שאלת דירה": [make_candidate(APT1, rerank_score=0.9)], "שאלת רכב": []},
        max_hops=2,
    )
    endless = [
        (None, [tool_call("calculate", {"expression": "1 + 1"})]),
        (None, [tool_call("calculate", {"expression": "2 + 2"})]),
        (None, [tool_call("calculate", {"expression": "3 + 3"})]),  # never reached
    ]
    orchestrator_turns(monkeypatch, endless)
    answer, _ = engine._answer("שאלה")
    assert len(endless) == 1  # exactly max_hops=2 turns consumed
    assert answer.text == "תשובה"  # still synthesized


def test_loop_failure_degrades_to_prefetched_pool(monkeypatch):
    classification = simple_classification(needs_calculation=True)
    engine = make_engine(
        classification, {"שאלת דירה": [make_candidate(APT1, rerank_score=0.9)], "שאלת רכב": []}
    )

    def boom(messages, **kwargs):
        raise RuntimeError("TF down")

    monkeypatch.setattr(engine_mod, "tf_chat", boom)
    answer, _ = engine._answer("שאלה")
    assert answer.text == "תשובה"  # degraded but answered from prefetched evidence
    assert [r.chunk.chunk_id for r in answer.retrieved] == [APT1]
    assert any(t["step"] == "orchestrator" and "error" in t for t in answer.trace)


def test_bad_tool_args_and_calc_errors_reported_not_raised(monkeypatch):
    classification = simple_classification(needs_calculation=True)
    engine = make_engine(
        classification, {"שאלת דירה": [make_candidate(APT1, rerank_score=0.9)], "שאלת רכב": []}
    )
    bad_calc = SimpleNamespace(
        id="tc-2", function=SimpleNamespace(name="calculate", arguments='{"expression": "x+1"}')
    )
    calls = orchestrator_turns(
        monkeypatch,
        [(None, [bad_calc]), ("DONE", None)],
    )
    answer, _ = engine._answer("שאלה")
    assert "error" in calls[1][0][-1]["content"]  # error fed back to the LLM
    assert answer.text == "תשובה"
    assert not any("value" in t for t in answer.trace if t["step"] == "calculate")


# --------------------------------------------------------------------------- #
# Unfiltered retry — a wrong category tag must never be the reason for a refusal
# --------------------------------------------------------------------------- #


def test_prefetch_retries_unfiltered_when_category_gates_to_zero():
    engine = make_engine(
        single_classification(),
        {"שאלת דירה": [make_candidate(APT1, rerank_score=0.9)]},
        empty_for_categories={"apartment"},
    )
    answer, _ = engine._answer("שאלה")
    assert engine.retriever.calls == [("שאלת דירה", "apartment"), ("שאלת דירה", None)]
    assert [r.chunk.chunk_id for r in answer.retrieved] == [APT1]
    prefetch = [t for t in answer.trace if t["step"] == "retrieve"]
    assert prefetch[0]["retried_unfiltered"] is True


def test_no_unfiltered_retry_when_filtered_retrieval_hits():
    engine = make_engine(
        single_classification(), {"שאלת דירה": [make_candidate(APT1, rerank_score=0.9)]}
    )
    answer, _ = engine._answer("שאלה")
    assert engine.retriever.calls == [("שאלת דירה", "apartment")]  # one call, no retry
    assert "retried_unfiltered" not in [t for t in answer.trace if t["step"] == "retrieve"][0]


def test_unfiltered_retry_does_not_loop_when_corpus_has_nothing():
    engine = make_engine(single_classification(), {})
    answer, _ = engine._answer("שאלה")
    assert engine.retriever.calls == [("שאלת דירה", "apartment"), ("שאלת דירה", None)]
    assert answer.text == FALLBACK_TEXT  # genuinely empty pool still refuses


def test_loop_retrieve_tool_retries_unfiltered(monkeypatch):
    engine = make_engine(
        single_classification(dependent=True),
        {
            "שאלת דירה": [make_candidate(APT1, rerank_score=0.9)],
            "מה ההשתתפות העצמית": [make_candidate(TRV1, rerank_score=0.8)],
        },
        empty_for_categories={"travel"},
    )
    orchestrator_turns(
        monkeypatch,
        [
            (None, [tool_call("retrieve", {"query": "מה ההשתתפות העצמית", "category": "travel"})]),
            ("DONE", None),
        ],
    )
    answer, _ = engine._answer("שאלה")
    assert ("מה ההשתתפות העצמית", None) in engine.retriever.calls
    assert {r.chunk.chunk_id for r in answer.retrieved} == {APT1, TRV1}
    loop_retrieve = [t for t in answer.trace if t["step"] == "retrieve" and t.get("phase") == "loop"]
    assert loop_retrieve[0]["retried_unfiltered"] is True


# --------------------------------------------------------------------------- #
# Difficulty-aware synthesis routing
# --------------------------------------------------------------------------- #


def routing_engine(classification):
    return make_engine(
        classification,
        {
            "שאלת דירה": [make_candidate(APT1, rerank_score=0.9)],
            "שאלת רכב": [make_candidate(TRV1, rerank_score=0.6)],
        },
        fast_generator=FakeGenerator(model="fake-fast"),
    )


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_easy_single_question_synthesizes_on_the_fast_model(difficulty):
    engine = routing_engine(single_classification(estimated_difficulty=difficulty))
    answer, _ = engine._answer("שאלה")
    assert engine.fast_generator.calls and engine.generator.calls == []
    synth = [t for t in answer.trace if t["step"] == "synthesize"][0]
    assert (synth["model"], synth["fast_synthesis"]) == ("fake-fast", True)


@pytest.mark.parametrize(
    "classification",
    [
        pytest.param(single_classification(estimated_difficulty="hard"), id="hard"),
        pytest.param(simple_classification(estimated_difficulty="easy"), id="multi-category"),
        pytest.param(
            single_classification(estimated_difficulty="easy", needs_calculation=True),
            id="needs-calculation",
        ),
        pytest.param(
            single_classification(estimated_difficulty="easy", dependent=True), id="dependent"
        ),
    ],
)
def test_hard_multi_or_agentic_questions_synthesize_on_the_strong_model(
    classification, monkeypatch
):
    orchestrator_turns(monkeypatch, [("DONE", None)])  # only used by the loop paths
    engine = routing_engine(classification)
    answer, _ = engine._answer("שאלה")
    assert engine.generator.calls and engine.fast_generator.calls == []
    synth = [t for t in answer.trace if t["step"] == "synthesize"][0]
    assert (synth["model"], synth["fast_synthesis"]) == ("fake-synth", False)


def test_routing_disabled_keeps_everything_on_the_strong_model():
    engine = routing_engine(single_classification(estimated_difficulty="easy"))
    engine.fast_generator = None
    answer, _ = engine._answer("שאלה")
    assert engine.generator.calls
    assert [t for t in answer.trace if t["step"] == "synthesize"][0]["fast_synthesis"] is False


def test_build_fast_generator_inherits_the_generation_answer_contract():
    config = SimpleNamespace(
        generation=SimpleNamespace(
            model="strong",
            prompt="grounded_cite",
            max_tokens=4096,
            temperature=0.3,
            retry_on_citation_failure=False,
            extra_params={"reasoning_effort": "high"},
        ),
        harness=SimpleNamespace(
            fast_synthesis_model="fast",
            fast_synthesis_max_tokens=777,
            fast_synthesis_extra_params={"reasoning_effort": "low"},
        ),
    )
    generator = engine_mod.build_fast_generator(config)
    assert (generator.model, generator.max_tokens) == ("fast", 777)
    assert generator.extra_params == {"reasoning_effort": "low"}
    # Answer contract (prompt variant, temperature, citation retry) is shared.
    assert generator.prompt_name == "grounded_cite"
    assert generator.temperature == 0.3
    assert generator.retry_on_citation_failure is False

    config.harness.fast_synthesis_model = None
    assert engine_mod.build_fast_generator(config) is None

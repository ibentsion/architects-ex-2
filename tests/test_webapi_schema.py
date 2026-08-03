"""webapi wire schema, dataset discovery and path confinement.

Every filesystem read the bridge does is driven by a request parameter, so the
confinement tests here are the security gate for the whole webapi package
(threat T-260803-01/02).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.types import Answer, Citation
from webapi import datasets as datasets_mod
from webapi import paths
from webapi.datasets import QuestionIndex, UnknownDataset, discover_datasets, load_pairs
from webapi.paths import PathEscape, resolve_in_repo
from webapi.schema import SupportPair, answer_to_pair, record_to_pair


# --------------------------------------------------------------------------- #
# Path confinement — the single chokepoint
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "attack",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "corpus/../../secrets",
        "../.env",
        "corpus/./../../root/.ssh/id_rsa",
    ],
)
def test_resolve_in_repo_rejects_traversal(attack):
    with pytest.raises(PathEscape):
        resolve_in_repo(attack)


def test_resolve_in_repo_accepts_an_in_repo_path():
    resolved = resolve_in_repo("corpus/apartment/files/x.pdf")
    assert resolved == paths.REPO_ROOT / "corpus/apartment/files/x.pdf"


def test_resolve_in_repo_rejects_a_symlink_pointing_out_of_the_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "corpus").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "corpus" / "escape.txt").symlink_to(outside)
    monkeypatch.setattr(paths, "REPO_ROOT", root)

    with pytest.raises(PathEscape):
        resolve_in_repo("corpus/escape.txt")
    # ...while a symlink that stays inside is fine.
    (root / "corpus" / "inside.txt").write_text("ok", encoding="utf-8")
    (root / "link.txt").symlink_to(root / "corpus" / "inside.txt")
    assert resolve_in_repo("link.txt").read_text(encoding="utf-8") == "ok"


# --------------------------------------------------------------------------- #
# Adapters — one wire shape out of four observed answer-file shapes
# --------------------------------------------------------------------------- #


def test_answer_to_pair_maps_the_live_engine_answer():
    answer = Answer(
        text="תשובה",
        citations=[
            Citation(file="apartment/files/a.pdf", page=3, quote="ציטוט"),
            Citation(file="car/pages/car.txt", page=None, quote=None),
        ],
        category="apartment",
        confidence=0.91,
        latency_ms=1234.5,
        cost_estimate=0.004,
        trace=[{"step": "classify"}],
    )
    pair = answer_to_pair(answer, pair_id="live-1", question="שאלה")

    assert (pair.id, pair.question, pair.answer) == ("live-1", "שאלה", "תשובה")
    assert (pair.domain, pair.confidence, pair.cost_usd) == ("apartment", 0.91, 0.004)
    assert pair.trace == [{"step": "classify"}]
    pdf, txt = pair.citations
    assert (pdf.id, pdf.file_name, pdf.page_number) == ("live-1:0", "apartment/files/a.pdf", 3)
    assert pdf.content_preview == "ציטוט"
    assert pdf.thumbnail_url == (
        "/api/citation/thumbnail?file=apartment%2Ffiles%2Fa.pdf&page=3"
    )
    # TXT sources have no page, therefore no page image.
    assert (txt.page_number, txt.thumbnail_url) == (None, None)


def test_record_to_pair_absorbs_the_cli_answer_shape():
    """rag_answers_*.jsonl / baseline_answers.jsonl: no domain, no confidence."""
    pair = record_to_pair(
        {"id": "q1", "answer": "תשובה", "citations": [{"file": "car/pages/car.txt", "page": None}],
         "latency_ms": 900.0, "tokens": {"prompt": 1}},
        pair_id="q1",
        question_record={"question": "שאלה", "domain": "car", "difficulty": "easy",
                         "ground_truth_answer": "אמת"},
        judgment=None,
    )
    assert (pair.question, pair.reference_answer, pair.difficulty) == ("שאלה", "אמת", "easy")
    assert pair.domain == "car"  # fell through to the question's domain
    assert (pair.confidence, pair.cost_usd, pair.trace) == (None, None, None)
    assert pair.judgment is None


def test_record_to_pair_absorbs_the_contract_shape_domain_and_cost_usd():
    """team_1_results.jsonl speaks contract.py's field names."""
    pair = record_to_pair(
        {"id": "q1", "answer": "a", "citations": [], "domain": "travel",
         "confidence": 0.5, "latency_ms": 10.0, "cost_usd": 0.007},
        pair_id="q1", question_record=None, judgment=None,
    )
    assert (pair.domain, pair.cost_usd, pair.confidence) == ("travel", 0.007, 0.5)
    assert pair.question is None  # never fabricated


def test_record_to_pair_absorbs_the_agent_eval_shape_with_category_and_trace():
    pair = record_to_pair(
        {"id": "q1", "answer": "a", "citations": [], "category": "health",
         "confidence": 0.8, "cost_estimate": 0.002,
         "classification": {"mode": "single"}, "trace": [{"step": "hint"}]},
        pair_id="q1", question_record={"question": "שאלה", "domain": "car"}, judgment=None,
    )
    # `category` wins over the question's domain; `cost_estimate` is the CLI's
    # name for `cost_usd`.
    assert (pair.domain, pair.cost_usd) == ("health", 0.002)
    assert pair.classification == {"mode": "single"} and pair.trace == [{"step": "hint"}]


def test_record_to_pair_keeps_a_zero_cost_instead_of_dropping_it():
    pair = record_to_pair({"id": "q", "answer": "a", "cost_usd": 0.0, "confidence": 0.0},
                          pair_id="q", question_record=None, judgment=None)
    assert (pair.cost_usd, pair.confidence) == (0.0, 0.0)


def test_record_to_pair_carries_the_judge_grades():
    judgment = {
        "id": "q1", "domain": "life", "difficulty": "hard",
        "judgment": {"correctness": 8, "completeness": 7, "conversational_quality": 9,
                     "verdict": "partial", "hallucination": False,
                     "reasoning": {"judge-a": "כמעט"}},
    }
    pair = record_to_pair({"id": "q1", "answer": "a"}, pair_id="q1",
                          question_record=None, judgment=judgment)
    assert (pair.domain, pair.difficulty) == ("life", "hard")
    assert pair.judgment.correctness == 8
    assert pair.judgment.verdict == "partial"
    assert pair.judgment.hallucination is False
    assert pair.judgment.reasoning == {"judge-a": "כמעט"}


def test_record_to_pair_survives_a_judgments_only_dataset():
    pair = record_to_pair({}, pair_id="run#3", question_record=None,
                          judgment={"id": "q1", "judgment": {"verdict": "correct"}})
    assert pair.id == "run#3"
    assert pair.answer is None  # no answers file resolved — not an empty string
    assert pair.judgment.verdict == "correct"


# --------------------------------------------------------------------------- #
# Question index
# --------------------------------------------------------------------------- #


def write_repo(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(paths, "REPO_ROOT", root)
    return root


def jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def test_question_index_reads_wrapped_bare_and_jsonl_sources(tmp_path, monkeypatch):
    root = write_repo(tmp_path, monkeypatch)
    (root / "reference_questions.json").write_text(
        json.dumps({"questions": [{"id": "v1-1", "question": "שאלה 1", "domain": "car"}]}),
        encoding="utf-8",
    )
    (root / "reference_questions_v3.json").write_text(
        json.dumps([{"id": "v3-1", "question": "שאלה 3", "difficulty": "hard"}]), encoding="utf-8"
    )
    jsonl(root / "ref_q_validation_set_v1.jsonl",
          [{"id": "v1-1", "question": "גרסה אחרת", "_set": "validation"},
           {"id": "split-1", "question": "שאלה 4"}])

    index = QuestionIndex.load()

    assert set(index.by_id) == {"v1-1", "v3-1", "split-1"}
    assert index.get("v1-1")["question"] == "שאלה 1"  # first source wins the re-split
    assert index.get("v3-1")["difficulty"] == "hard"
    assert index.get("nope") is None


def test_question_index_reads_the_blind_submission_questions(tmp_path, monkeypatch):
    """blind_questions.json names the ids in team_1_results.jsonl — the graded
    submission. Without it that dataset renders with no questions at all."""
    root = write_repo(tmp_path, monkeypatch)
    (root / "blind_questions.json").write_text(
        json.dumps({"questions": [{"id": "int-001-apartment-easy", "question": "שאלה עיוורת"}]}),
        encoding="utf-8",
    )
    jsonl(root / "team_1_results.jsonl",
          [{"id": "int-001-apartment-easy", "answer": "תשובה", "domain": "apartment"}])

    _total, pairs = load_pairs("team_1_results.jsonl")
    assert pairs[0].question == "שאלה עיוורת"


def test_a_missing_question_source_is_silent_not_an_error(tmp_path, monkeypatch):
    """blind_questions.json is not in git — it only exists after a fetch. Its
    absence must not break the index (nor must any other source's)."""
    root = write_repo(tmp_path, monkeypatch)
    (root / "reference_questions.json").write_text(
        json.dumps([{"id": "dev-01", "question": "שאלה"}]), encoding="utf-8")
    assert not (root / "blind_questions.json").exists()

    index = QuestionIndex.load()
    assert set(index.by_id) == {"dev-01"}


def test_question_index_covers_every_real_question_source():
    """Regression guard: if a question file's shape changes, the History view
    silently loses its question text. v1/v2/v3 ids are pairwise disjoint."""
    index = QuestionIndex.load()
    per_file = {}
    for name in ("reference_questions.json", "reference_questions_v2.json",
                 "reference_questions_v3.json"):
        raw = json.loads((paths.REPO_ROOT / name).read_text(encoding="utf-8"))
        records = raw["questions"] if isinstance(raw, dict) else raw
        per_file[name] = {r["id"] for r in records}

    v1, v2, v3 = per_file.values()
    assert not (v1 & v2) and not (v1 & v3) and not (v2 & v3)  # 0 collisions
    assert len(index.by_id) >= 180
    assert (v1 | v2 | v3) <= set(index.by_id)
    assert all(index.get(qid)["question"] for qid in v1 | v2 | v3)


@pytest.mark.skipif(
    not (paths.REPO_ROOT / "blind_questions.json").is_file(),
    reason="blind_questions.json is fetched from the artifacts dataset, not in git",
)
def test_the_graded_submission_joins_to_its_questions_in_the_real_repo():
    """The regression this guards: team_1_results.jsonl IS the graded run, and
    it was rendering 390 pairs with question=None."""
    raw = json.loads((paths.REPO_ROOT / "blind_questions.json").read_text(encoding="utf-8"))
    blind_ids = {q["id"] for q in (raw["questions"] if isinstance(raw, dict) else raw)}

    index = QuestionIndex.load()
    assert blind_ids <= set(index.by_id)

    # The blind namespace must not shadow (or be shadowed by) the dev sets.
    for name in ("reference_questions.json", "reference_questions_v2.json",
                 "reference_questions_v3.json"):
        other = json.loads((paths.REPO_ROOT / name).read_text(encoding="utf-8"))
        other_ids = {r["id"] for r in (other["questions"] if isinstance(other, dict) else other)}
        assert not (blind_ids & other_ids), f"id collision with {name}"

    if (paths.REPO_ROOT / "team_1_results.jsonl").is_file():
        _total, pairs = load_pairs("team_1_results.jsonl", limit=25)
        assert pairs and all(p.question for p in pairs)


# --------------------------------------------------------------------------- #
# Dataset discovery and loading
# --------------------------------------------------------------------------- #


def test_discovers_root_answer_files_and_eval_answers(tmp_path, monkeypatch):
    root = write_repo(tmp_path, monkeypatch)
    jsonl(root / "rag_answers_full.jsonl", [{"id": "q1", "answer": "a"}])
    jsonl(root / "rag_answers.jsonl", [{"id": "q1", "answer": "a"}])
    jsonl(root / "team_1_results.jsonl", [{"id": "q1", "answer": "a"}])
    jsonl(root / "baseline_answers.jsonl", [{"id": "q1", "answer": "a"}])
    jsonl(root / "eval_results/run-x/answers/v1.jsonl",
          [{"id": "q1", "answer": "a", "trace": [{"step": "hint"}]}])
    jsonl(root / "eval_results/run-x/notes.jsonl", [{"id": "q1"}])  # not an answers dir

    found = {d.id: d for d in discover_datasets()}

    assert set(found) == {
        "rag_answers_full.jsonl", "rag_answers.jsonl", "team_1_results.jsonl",
        "baseline_answers.jsonl", "eval_results/run-x/answers/v1.jsonl",
    }
    assert found["eval_results/run-x/answers/v1.jsonl"].has_trace is True
    assert found["rag_answers_full.jsonl"].has_trace is False
    assert all(d.kind == "answers" and d.n_pairs == 1 for d in found.values())


def test_discovers_nested_judged_runs_and_pairs_them_by_basename(tmp_path, monkeypatch):
    root = write_repo(tmp_path, monkeypatch)
    jsonl(root / "eval_results/agent-eval/answers/v1.jsonl",
          [{"id": "q1", "answer": "תשובה", "trace": [{"step": "hint"}]}])
    run = root / "eval_results/agent-eval/v1-deepseek"
    jsonl(run / "judgments.jsonl", [{"id": "q1", "judgment": {"verdict": "correct"}}])
    # An ABSOLUTE, out-of-repo answers_file: joined by basename, never opened.
    (run / "metrics.json").write_text(json.dumps(
        {"meta": {"answers_file": "/home/someone/else/answers/v1.jsonl",
                  "questions_file": "reference_questions.json"}}), encoding="utf-8")

    found = {d.id: d for d in discover_datasets()}
    judged = found["eval_results/agent-eval/v1-deepseek/judgments.jsonl"]

    assert (judged.kind, judged.n_pairs, judged.has_judgment) == ("judged", 1, True)
    assert judged.has_trace is True  # from the paired answers file
    assert judged.questions_file == "reference_questions.json"

    total, pairs = load_pairs(judged.id)
    assert total == 1
    assert pairs[0].answer == "תשובה"  # answers joined in by basename
    assert pairs[0].judgment.verdict == "correct"


def test_judged_run_whose_answers_file_is_unresolvable_degrades_to_judgments_only(
    tmp_path, monkeypatch
):
    root = write_repo(tmp_path, monkeypatch)
    run = root / "eval_results/rag-default"
    jsonl(run / "judgments.jsonl",
          [{"id": "q1", "domain": "car", "judgment": {"verdict": "incorrect"}}])
    (run / "metrics.json").write_text(
        json.dumps({"meta": {"answers_file": "rag_answers.jsonl"}}), encoding="utf-8")

    info = {d.id: d for d in discover_datasets()}["eval_results/rag-default/judgments.jsonl"]
    assert info.kind == "judged" and info.has_trace is False

    total, pairs = load_pairs(info.id)
    assert total == 1
    assert pairs[0].answer is None
    assert (pairs[0].domain, pairs[0].judgment.verdict) == ("car", "incorrect")


def test_load_pairs_joins_questions_and_paginates(tmp_path, monkeypatch):
    root = write_repo(tmp_path, monkeypatch)
    (root / "reference_questions.json").write_text(json.dumps(
        [{"id": f"q{i}", "question": f"שאלה {i}", "domain": "car",
          "ground_truth_answer": f"אמת {i}"} for i in range(5)]), encoding="utf-8")
    jsonl(root / "rag_answers_full.jsonl",
          [{"id": f"q{i}", "answer": f"תשובה {i}", "citations": []} for i in range(5)])

    total, pairs = load_pairs("rag_answers_full.jsonl", limit=2, offset=1)

    assert total == 5
    assert [p.id for p in pairs] == ["q1", "q2"]
    assert [p.question for p in pairs] == ["שאלה 1", "שאלה 2"]
    assert pairs[0].reference_answer == "אמת 1"


def test_load_pairs_synthesizes_ids_and_never_fabricates_questions(tmp_path, monkeypatch):
    root = write_repo(tmp_path, monkeypatch)
    jsonl(root / "rag_answers_x.jsonl",
          [{"answer": "בלי מזהה"}, {"id": "unknown-id", "answer": "לא בשום מאגר"}])

    _total, pairs = load_pairs("rag_answers_x.jsonl")

    assert pairs[0].id == "rag_answers_x#1"  # 1-based line number
    assert pairs[0].question is None
    assert pairs[1].id == "unknown-id" and pairs[1].question is None


def test_load_pairs_skips_malformed_lines(tmp_path, monkeypatch, caplog):
    root = write_repo(tmp_path, monkeypatch)
    (root / "rag_answers_x.jsonl").write_text(
        '{"id": "q1", "answer": "טוב"}\nnot json at all\n\n{"id": "q2", "answer": "גם טוב"}\n',
        encoding="utf-8",
    )

    total, pairs = load_pairs("rag_answers_x.jsonl")

    assert total == 2
    assert [p.id for p in pairs] == ["q1", "q2"]


def test_load_pairs_refuses_an_undiscovered_dataset(tmp_path, monkeypatch):
    root = write_repo(tmp_path, monkeypatch)
    (root / ".env").write_text("NEBIUS_API_KEY=secret", encoding="utf-8")

    with pytest.raises(UnknownDataset):
        load_pairs(".env")
    with pytest.raises(PathEscape):
        load_pairs("../../etc/passwd")


def test_datasets_are_newest_first(tmp_path, monkeypatch):
    root = write_repo(tmp_path, monkeypatch)
    jsonl(root / "rag_answers_old.jsonl", [{"id": "q1"}])
    jsonl(root / "rag_answers_new.jsonl", [{"id": "q1"}])
    import os
    os.utime(root / "rag_answers_old.jsonl", (1_000_000, 1_000_000))
    os.utime(root / "rag_answers_new.jsonl", (2_000_000, 2_000_000))

    assert [d.id for d in discover_datasets()] == [
        "rag_answers_new.jsonl", "rag_answers_old.jsonl"
    ]


def test_discovery_on_a_repo_with_nothing_to_show(tmp_path, monkeypatch):
    write_repo(tmp_path, monkeypatch)
    assert discover_datasets() == []
    assert QuestionIndex.load().by_id == {}


def test_discovery_against_the_real_repo_finds_the_agent_eval_runs():
    found = {d.id: d for d in discover_datasets()}
    assert "eval_results/agent-eval-20260801T193537Z/v1-agent-deepseek/judgments.jsonl" in found
    judged = found["eval_results/agent-eval-20260801T193537Z/v1-agent-deepseek/judgments.jsonl"]
    assert (judged.kind, judged.has_judgment, judged.has_trace) == ("judged", True, True)

    total, pairs = load_pairs(judged.id, limit=3)
    assert total == 48 and len(pairs) == 3
    first = pairs[0]
    assert isinstance(first, SupportPair)
    assert first.question and first.answer and first.reference_answer
    assert first.judgment is not None and first.trace

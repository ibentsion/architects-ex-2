"""Evaluation-harness tests — citation accuracy is LLM-judged against the real
cited page, so the pieces under test are: page resolution, the score formula,
committee aggregation, prompt assembly, and the end-to-end run wiring (judge
LLM mocked throughout).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalharness import citations, judge, metrics, pages, prompts, run
from evalharness.pages import PageStore

ANCHOR = "apartment/files/הודעה-על-תקופת-התיישנות.pdf"
PAGE1 = "תקופת ההתיישנות היא שלוש שנים."
PAGE2 = "רק הגשת תביעה לבית משפט עוצרת את מרוץ ההתיישנות."


def _docling_dict(page_texts: dict[int, str], n_pages: int) -> dict:
    """A minimal but schema-valid DoclingDocument dict, as the parse cache
    stores it. Pages with no text exercise the `empty_page` path."""
    from docling_core.types.doc import (
        BoundingBox,
        CoordOrigin,
        DocItemLabel,
        DoclingDocument,
        ProvenanceItem,
        Size,
    )

    doc = DoclingDocument(name="doc")
    for page_no in range(1, n_pages + 1):
        doc.add_page(page_no=page_no, size=Size(width=612, height=792))
    bbox = BoundingBox(l=0, t=0, r=100, b=20, coord_origin=CoordOrigin.TOPLEFT)
    for page_no, text in sorted(page_texts.items()):
        doc.add_text(label=DocItemLabel.TEXT, text=text,
                     prov=ProvenanceItem(page_no=page_no, bbox=bbox,
                                         charspan=(0, len(text))))
    return doc.export_to_dict()


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny corpus + populated parse cache: one 3-page PDF (page 3 parses to
    nothing), one TXT page, and a same-basename PDF in another category so
    bare-filename citations are genuinely ambiguous."""
    from rag.parsing import discover
    from rag.parsing.cache import ParseCache

    corpus_dir, cache_dir = tmp_path / "corpus", tmp_path / "cache"
    for rel, content in (
        (ANCHOR, b"%PDF-1.4 apartment"),
        ("travel/files/הודעה-על-תקופת-התיישנות.pdf", b"%PDF-1.4 travel"),
        ("travel/files/unique-name.pdf", b"%PDF-1.4 unique"),
        ("apartment/pages/faq.txt", "שאלות ותשובות בנושא התיישנות".encode()),
    ):
        path = corpus_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (corpus_dir / "manifest.json").write_text("{}", encoding="utf-8")

    cache = ParseCache(cache_dir)
    for source in discover(corpus_dir):
        if source.kind == "pdf":
            cache.store(source.sha256, _docling_dict({1: PAGE1, 2: PAGE2}, n_pages=3))
    return corpus_dir, cache_dir


@pytest.fixture
def store(corpus) -> PageStore:
    return PageStore(*corpus)


# -- page resolution --------------------------------------------------------


def test_resolves_cited_page_to_its_real_text(store: PageStore):
    assert store.resolve(ANCHOR, 1) == (PAGE1, None)
    assert store.resolve(ANCHOR, 2) == (PAGE2, None)


def test_resolves_txt_page_as_whole_file(store: PageStore):
    text, reason = store.resolve("apartment/pages/faq.txt", None)
    assert reason is None and "התיישנות" in text


@pytest.mark.parametrize("file,page,reason", [
    ("apartment/files/does-not-exist.pdf", 1, pages.UNKNOWN_FILE),
    (ANCHOR, 99, pages.PAGE_OUT_OF_RANGE),
    (ANCHOR, None, pages.MISSING_PAGE),
    (ANCHOR, 3, pages.EMPTY_PAGE),  # page exists in the PDF, parsed to nothing
    ("apartment/pages/faq.txt", 2, pages.PAGE_ON_TXT),
    ("הודעה-על-תקופת-התיישנות.pdf", 1, pages.AMBIGUOUS_FILE),  # two categories
])
def test_unresolvable_citations_report_why(store: PageStore, file, page, reason):
    text, got = store.resolve(file, page)
    assert (text, got) == (None, reason)


def test_bare_filename_resolves_when_unambiguous(store: PageStore):
    assert store.resolve("unique-name.pdf", 1) == (PAGE1, None)


def test_missing_parse_cache_entry_is_loud(corpus):
    corpus_dir, _ = corpus
    with pytest.raises(FileNotFoundError, match="parse-cache"):
        PageStore(corpus_dir, corpus_dir / "empty-cache").resolve(ANCHOR, 1)


# -- scoring ----------------------------------------------------------------


def _resolved(*reasons):
    return [{"file": ANCHOR, "page": i, "text": None if r else PAGE1,
             "invalid_reason": r} for i, r in enumerate(reasons, start=1)]


def _judgment(support, labels):
    return {"citation_support": support, "citation_labels": labels}


def test_full_credit_when_every_citation_resolves_and_establishes():
    score = citations.score_citations(_resolved(None, None),
                                      _judgment("fully", ["establishes", "partial"]))
    assert score["accuracy"] == 1.0
    assert score["labels"] == ["establishes", "partial"]


def test_partial_support_halves_the_credit():
    score = citations.score_citations(_resolved(None), _judgment("partially", ["partial"]))
    assert score["accuracy"] == 0.5


def test_invalid_citation_dilutes_the_credit():
    score = citations.score_citations(
        _resolved(None, None, pages.UNKNOWN_FILE), _judgment("fully", ["establishes", "partial"]))
    assert score["accuracy"] == pytest.approx(2 / 3)
    assert score["invalid_reasons"] == [pages.UNKNOWN_FILE]
    # A label per citation made, with None where the citation resolved to nothing.
    assert score["labels"] == ["establishes", "partial", None]


def test_citing_nothing_scores_zero():
    score = citations.score_citations([], None)
    assert score["accuracy"] == 0.0 and score["cited_count"] == 0
    assert score["judge_failed"] is False


def test_all_citations_invalid_scores_zero_without_a_judge_call():
    score = citations.score_citations(_resolved(pages.PAGE_OUT_OF_RANGE), None)
    assert score["accuracy"] == 0.0 and score["support"] is None
    assert score["judge_failed"] is False  # nothing to judge is not a failure


def test_judge_failure_is_flagged_not_blamed_on_the_system():
    score = citations.score_citations(_resolved(None), {"error": "all judges failed"})
    assert score["accuracy"] == 0.0 and score["judge_failed"] is True


def test_gt_source_hit_is_a_diagnostic_over_any_of_groups():
    groups = [{"any_of": [{"file": ANCHOR, "page": 1},
                          {"file": "travel/files/unique-name.pdf", "page": 4}]},
              {"any_of": [{"file": "travel/files/unique-name.pdf", "page": 9}]}]
    hit = citations.gt_source_hit([{"file": "unique-name.pdf", "page": 4}], groups)
    assert hit == {"groups_total": 2, "groups_hit": 1, "hit_rate": 0.5}


# -- judge normalization + committee ----------------------------------------


def test_citation_judgment_normalizes_and_defaults_missing_labels():
    normalized = judge._normalize_citation(
        {"citation_support": "FULLY ", "reasoning": " ok ",
         "per_citation": [{"idx": 1, "label": "Establishes"}, {"idx": 7, "label": "partial"}]}, 3)
    assert normalized["citation_support"] == "fully"
    assert normalized["citation_labels"] == ["unrelated", "establishes", "unrelated"]
    assert normalized["reasoning"] == "ok"


def test_citation_judgment_rejects_an_unknown_support_level():
    with pytest.raises(ValueError, match="citation_support"):
        judge._normalize_citation({"citation_support": "maybe"}, 1)


def test_citation_committee_takes_the_majority():
    votes = [{"citation_support": s, "citation_labels": [lbl], "reasoning": "r",
              "judge_model": m}
             for s, lbl, m in (("fully", "establishes", "a"),
                               ("fully", "establishes", "b"),
                               ("not_at_all", "unrelated", "c"))]
    agg = judge.aggregate_citation_committee(votes)
    assert agg["citation_support"] == "fully"
    assert agg["citation_labels"] == ["establishes"]
    assert agg["disagreement"] is True
    assert set(agg["reasoning"]) == {"a", "b", "c"}


def test_citation_committee_ties_break_pessimistically():
    votes = [{"citation_support": s, "citation_labels": [], "reasoning": "r",
              "judge_model": m} for s, m in (("fully", "a"), ("partially", "b"))]
    assert judge.aggregate_citation_committee(votes)["citation_support"] == "partially"


def test_citation_committee_with_no_valid_judges_errors():
    agg = judge.aggregate_citation_committee([{"judge_model": "a", "error": "boom"}])
    assert agg["error"] and agg["judges_failed"] == 1


# -- prompt -----------------------------------------------------------------


def test_citation_prompt_shows_the_real_page_text_per_citation():
    question = {"question": "כמה שנים?", "ground_truth_answer": "שלוש שנים."}
    messages = prompts.build_citation_messages(question, [
        {"file": ANCHOR, "page": 1, "text": PAGE1},
        {"file": "apartment/pages/faq.txt", "page": None, "text": PAGE2},
    ])
    user = messages[1]["content"]
    assert "CITATION 0" in user and "CITATION 1" in user
    assert PAGE1 in user and PAGE2 in user
    assert "page: 1" in user and "page: n/a" in user
    # The "any page earns credit" rule is load-bearing: without it the judge
    # reverts to checking citations against an expected source.
    assert "ANY page that genuinely establishes" in messages[0]["content"]


def test_citation_prompt_truncates_an_over_long_page():
    long_page = "א" * (prompts.MAX_PAGE_CHARS + 500)
    user = prompts.build_citation_messages(
        {"question": "q", "ground_truth_answer": "gt"},
        [{"file": ANCHOR, "page": 1, "text": long_page}])[1]["content"]
    assert "[TRUNCATED]" in user
    assert len(user) < len(long_page) + 1000


# -- end-to-end run ---------------------------------------------------------


def test_run_scores_citations_against_the_corpus(tmp_path, corpus, monkeypatch, capsys):
    """Whole harness, LLM mocked: one answer citing a page that establishes the
    ground truth plus a page that does not exist."""
    corpus_dir, cache_dir = corpus
    questions = [{
        "id": "dev-01", "domain": "apartment", "difficulty": "easy",
        "question": "כמה שנים?", "ground_truth_answer": "שלוש שנים.",
        "ground_truth_sources": [{"any_of": [{"file": ANCHOR, "page": 1}]}],
    }]
    answers = [{
        "id": "dev-01", "answer": "שלוש שנים.", "latency_ms": 1200,
        "citations": [{"file": ANCHOR, "page": 1},
                      {"file": ANCHOR, "page": 99}],
    }]
    q_path, a_path = tmp_path / "q.json", tmp_path / "a.jsonl"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    a_path.write_text(json.dumps(answers[0], ensure_ascii=False), encoding="utf-8")

    seen = []

    def fake_chat(messages, model, **kw):
        seen.append(messages[0]["content"])
        if "citation_support" in messages[0]["content"]:
            assert PAGE1 in messages[1]["content"], "judge must see the real page"
            assert "page: 99" not in messages[1]["content"], "invalid page is not shown"
            return json.dumps({"citation_support": "fully",
                               "per_citation": [{"idx": 0, "label": "establishes"}],
                               "reasoning": "The page states three years."})
        return json.dumps({"verdict": "correct", "hallucination": False,
                           "correctness": 10, "completeness": 9,
                           "conversational_quality": 8, "reasoning": "ok"})

    monkeypatch.setattr(judge, "chat", fake_chat)
    out_dir = tmp_path / "out"
    assert run.main(["--questions", str(q_path), "--answers", str(a_path),
                     "--out", str(out_dir), "--corpus", str(corpus_dir),
                     "--cache-dir", str(cache_dir), "--workers", "1"]) == 0

    assert len(seen) == 2, "one answer-quality call + one citation call"
    result = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    overall = result["metrics"]["overall"]
    assert overall["citation_accuracy"] == 0.5  # fully x 1 of 2 citations valid
    assert overall["invalid_citation_rate"] == 0.5
    assert overall["uncited_rate"] == 0.0
    assert overall["gt_source_hit_rate"] == 1.0
    assert result["metrics"]["invalid_citation_reasons"] == {pages.PAGE_OUT_OF_RANGE: 1}

    record = json.loads((out_dir / "judgments.jsonl").read_text(encoding="utf-8"))
    assert record["citations"]["support"] == "fully"
    assert record["citations"]["labels"] == ["establishes", None]
    assert "three years" in record["citation_judgment"]["reasoning"]
    assert "Citation accuracy" in (out_dir / "report.md").read_text(encoding="utf-8")


def test_run_skips_the_citation_judge_when_nothing_resolves(tmp_path, corpus, monkeypatch):
    corpus_dir, cache_dir = corpus
    questions = [{
        "id": "dev-01", "domain": "apartment", "difficulty": "easy",
        "question": "q", "ground_truth_answer": "gt",
        "ground_truth_sources": [{"any_of": [{"file": ANCHOR, "page": 1}]}],
    }]
    q_path, a_path = tmp_path / "q.json", tmp_path / "a.jsonl"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    a_path.write_text(json.dumps({"id": "dev-01", "answer": "איני יודע",
                                  "citations": []}), encoding="utf-8")

    calls = []

    def fake_chat(messages, model, **kw):
        calls.append(messages[0]["content"])
        return json.dumps({"verdict": "refusal", "hallucination": False,
                           "correctness": 0, "completeness": 0,
                           "conversational_quality": 5, "reasoning": "refused"})

    monkeypatch.setattr(judge, "chat", fake_chat)
    out_dir = tmp_path / "out"
    run.main(["--questions", str(q_path), "--answers", str(a_path),
              "--out", str(out_dir), "--corpus", str(corpus_dir),
              "--cache-dir", str(cache_dir), "--workers", "1"])

    assert len(calls) == 1, "no citations to read — no citation judge call"
    overall = json.loads((out_dir / "metrics.json").read_text())["metrics"]["overall"]
    assert overall["citation_accuracy"] == 0.0, "a refusal establishes nothing"
    assert overall["uncited_rate"] == 1.0


def test_unanswerable_question_is_judged_on_abstention(tmp_path, corpus, monkeypatch):
    """A question the corpus cannot answer is scored on whether the system
    declined — and its citations are not judged at all."""
    corpus_dir, cache_dir = corpus
    questions = [{
        "id": "v2-001-apartment-medium", "domain": "apartment", "difficulty": "medium",
        "kind": "unanswerable", "answerable": False,
        "question": "כמה זמן לוקח לטפל בתביעה הספציפית שלי?",
        "ground_truth_answer": "המסמכים אינם כוללים מידע על משך הטיפול בתביעה פרטנית.",
        "ground_truth_sources": [],
    }]
    q_path, a_path = tmp_path / "q.json", tmp_path / "a.jsonl"
    q_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    # The system answered anyway, and cited a page to back it up.
    a_path.write_text(json.dumps({"id": "v2-001-apartment-medium",
                                  "answer": "הטיפול אורך 14 ימים.",
                                  "citations": [{"file": ANCHOR, "page": 1}]}),
                      encoding="utf-8")

    seen = []

    def fake_chat(messages, model, **kw):
        seen.append(messages[0]["content"])
        assert "CANNOT answer" in messages[0]["content"], \
            "the unanswerable prompt must be used, not the rubric"
        return json.dumps({"verdict": "incorrect", "hallucination": True,
                           "correctness": 0, "completeness": 0,
                           "conversational_quality": 7,
                           "reasoning": "invented a duration"})

    monkeypatch.setattr(judge, "chat", fake_chat)
    out_dir = tmp_path / "out"
    run.main(["--questions", str(q_path), "--answers", str(a_path),
              "--out", str(out_dir), "--corpus", str(corpus_dir),
              "--cache-dir", str(cache_dir), "--workers", "1"])

    assert len(seen) == 1, "citations of an unanswerable question are never judged"
    overall = json.loads((out_dir / "metrics.json").read_text())["metrics"]["overall"]
    assert overall["citation_accuracy"] is None, "excluded, not scored zero"
    assert overall["abstention_rate"] == 0.0, "it answered instead of abstaining"
    assert overall["unanswerable_citation_rate"] == 1.0
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## Abstention (unanswerable questions)" in report


def test_abstention_counts_a_refusal_as_correct():
    def record(answerable, verdict, cited):
        return {"id": "x", "domain": "d", "difficulty": "easy", "kind": "unanswerable",
                "answerable": answerable, "n_source_groups": 0,
                "judgment": {"correctness": 10, "completeness": 5,
                             "conversational_quality": 5, "verdict": verdict,
                             "hallucination": False, "reasoning": ""},
                "citations": {"accuracy": None if not answerable else 1.0,
                              "cited_count": cited, "invalid_count": 0,
                              "invalid_reasons": []},
                "gt_source_hit": {"hit_rate": None}, "latency_ms": 10}

    agg = metrics.aggregate([record(False, "correct", 0), record(False, "incorrect", 2)])
    assert agg["overall"]["abstention_rate"] == 0.5
    assert agg["overall"]["unanswerable_citation_rate"] == 0.5
    assert agg["overall"]["citation_accuracy"] is None
    assert set(agg["by_kind"]) == {"unanswerable"}


def test_metrics_average_citation_accuracy_over_every_answer():
    def record(accuracy, cited, invalid):
        return {"id": "x", "domain": "d", "difficulty": "easy", "n_source_groups": 1,
                "judgment": {"correctness": 5, "completeness": 5,
                             "conversational_quality": 5, "verdict": "correct",
                             "hallucination": False, "reasoning": ""},
                "citations": {"accuracy": accuracy, "cited_count": cited,
                              "invalid_count": invalid, "invalid_reasons": []},
                "gt_source_hit": {"hit_rate": 1.0}, "latency_ms": 10}

    agg = metrics.aggregate([record(1.0, 2, 0), record(0.0, 0, 0)])
    assert agg["overall"]["citation_accuracy"] == 0.5
    assert agg["overall"]["uncited_rate"] == 0.5
    assert agg["overall"]["full_citation_credit_rate"] == 0.5


# -- retrieval hit-rate audit ------------------------------------------------


class FakeAuditRetriever:
    """Returns scripted candidates/gated chunks by question, in the shape the
    audit reads (chunk_id only)."""

    def __init__(self, candidates, gated):
        self.candidates, self.gated = candidates, gated

    @staticmethod
    def _wrap(chunk_ids):
        from types import SimpleNamespace

        return [SimpleNamespace(chunk=SimpleNamespace(chunk_id=cid), rerank_score=0.9)
                for cid in chunk_ids]

    def fuse(self, question):
        return self._wrap(self.candidates)

    def rerank_candidates(self, question, candidates):
        return self._wrap(self.gated)


def _group(*sources):
    return {"any_of": [{"file": f, "page": p} for f, p in sources]}


def test_chunk_id_parses_back_into_file_and_page():
    from evalharness.retrieval_audit import parse_chunk_id

    assert parse_chunk_id("dental/files/x.pdf#p3#c7") == ("dental/files/x.pdf", 3)
    assert parse_chunk_id("dental/pages/claim.txt#pnull#c0") == ("dental/pages/claim.txt", None)


def test_a_pdf_page_and_its_markdown_twin_are_the_same_page():
    from evalharness.retrieval_audit import page_key

    assert page_key("dental/files/x.pdf", 3) == page_key("dental/markdown-files/x.md", 3)
    # Category stays in the key: every category has its own pages/claim.txt.
    assert page_key("dental/pages/claim.txt", None) != page_key("car/pages/claim.txt", None)
    # Page numbers still separate pages of one document.
    assert page_key("dental/files/x.pdf", 3) != page_key("dental/files/x.pdf", 4)


def test_each_source_group_is_filed_under_the_furthest_stage_it_reached():
    from evalharness.retrieval_audit import classify_group, page_key

    indexed = {page_key("dental/files/a.pdf", 1), page_key("dental/files/b.pdf", 1),
               page_key("dental/files/c.pdf", 1)}
    candidates = {page_key("dental/files/a.pdf", 1), page_key("dental/files/b.pdf", 1)}
    gated = {page_key("dental/files/a.pdf", 1)}

    def stage(*sources):
        return classify_group(_group(*sources), gated, candidates, indexed)

    assert stage(("dental/files/a.pdf", 1)) == "gated"
    assert stage(("dental/files/b.pdf", 1)) == "not_gated"
    assert stage(("dental/files/c.pdf", 1)) == "not_retrieved"
    assert stage(("dental/files/d.pdf", 1)) == "missing_from_index"
    # any_of is satisfied by its best member — these are interchangeable
    # sources for one fact, not a list of pages all of which must be found.
    assert stage(("dental/files/d.pdf", 1), ("dental/files/a.pdf", 1)) == "gated"


def test_audit_question_classifies_every_group_of_one_question():
    from evalharness.retrieval_audit import audit_question, page_key

    retriever = FakeAuditRetriever(
        candidates=["dental/files/a.pdf#p1#c0", "dental/files/b.pdf#p2#c3"],
        gated=["dental/files/a.pdf#p1#c0"])
    question = {
        "id": "v3-001-dental-hard", "domain": "dental", "difficulty": "hard",
        "kind": "multi_source", "question": "ש",
        "ground_truth_sources": [_group(("dental/files/a.pdf", 1)),
                                 _group(("dental/files/b.pdf", 2)),
                                 _group(("dental/files/z.pdf", 9))],
    }
    record = audit_question(retriever, question, {page_key("dental/files/z.pdf", 9)})
    assert [g["stage"] for g in record["groups"]] == ["gated", "not_gated", "not_retrieved"]
    assert (record["n_candidates"], record["n_gated"]) == (2, 1)


def test_summary_counts_groups_and_fully_covered_questions():
    from evalharness.retrieval_audit import summarize

    records = [
        {"id": "a", "domain": "dental", "difficulty": "hard", "kind": "multi_source",
         "groups": [{"stage": "gated"}, {"stage": "not_gated"}]},
        {"id": "b", "domain": "car", "difficulty": "easy", "kind": "standard",
         "groups": [{"stage": "gated"}]},
    ]
    summary = summarize("pdf-per_table", "v2", records)
    assert summary["by_stage"] == {"gated": 2, "not_gated": 1, "not_retrieved": 0,
                                   "missing_from_index": 0}
    assert summary["group_hit_rate"] == 2 / 3
    # A multi-source question is only covered when EVERY group it needs is.
    assert summary["questions_fully_covered"] == 1
    assert summary["question_hit_rate"] == 0.5
    assert summary["by_difficulty"]["hard"]["not_gated"] == 1


def test_indexed_pages_reads_what_ingest_actually_produced(tmp_path):
    from evalharness.retrieval_audit import indexed_pages, page_key

    bm25 = tmp_path / "bm25"
    bm25.mkdir()
    (bm25 / "chunk_ids.json").write_text(
        json.dumps(["dental/files/a.pdf#p1#c0", "dental/files/a.pdf#p1#c1",
                    "dental/pages/claim.txt#pnull#c0"]), encoding="utf-8")
    assert indexed_pages(tmp_path) == {page_key("dental/files/a.pdf", 1),
                                       page_key("dental/pages/claim.txt", None)}


def test_an_arm_without_an_index_is_skipped_with_a_reason(tmp_path):
    from evalharness.retrieval_audit import arm_unavailable

    assert "not found" in arm_unavailable(str(tmp_path / "absent.yaml"))
    config = tmp_path / "arm.yaml"
    config.write_text(f"extends: configs/default.yaml\nindex_dir: {tmp_path / 'nope'}\n",
                      encoding="utf-8")
    assert "no ingested index" in arm_unavailable(str(config))


def test_unanswerable_questions_are_not_audited(tmp_path):
    from evalharness.retrieval_audit import load_questions

    path = tmp_path / "questions.json"
    path.write_text(json.dumps([
        {"id": "a", "ground_truth_sources": [_group(("dental/files/a.pdf", 1))]},
        {"id": "b", "ground_truth_sources": []},  # unanswerable: nothing to find
    ]), encoding="utf-8")
    assert [q["id"] for q in load_questions(str(path), None)] == ["a"]

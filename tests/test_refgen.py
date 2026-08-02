"""Reference-dataset generation tests.

The point of `refgen` is that a question is admitted only when gates say so, so
these tests are mostly about refusal: every gate must be able to reject, a
rejection must reach the generator as a retry reason, and nothing rejected may
reach the dataset. All LLM calls are mocked.
"""
from __future__ import annotations

import collections
import json

import pytest

from refgen import audit, generate, prompts, schema, verify
from refgen.inventory import Page, Sampler, bm25_search, build_inventory
from refgen.schema import RefQuestion

CATEGORY = "dental"
EXAMPLES = [{"domain": "car", "difficulty": "hard", "question": "שאלה לדוגמה מלקוח",
             "ground_truth_answer": "תשובה לדוגמה"}]
QUESTION = "כואבת לי שן ואני רוצה לדעת אם הטיפול הזה מכוסה בביטוח שלי?"
ANSWER = "כן, הטיפול מכוסה עד תקרה של 500 ש\"ח לשנה."


def make_pages(n: int = 6, category: str = CATEGORY) -> list[Page]:
    return [Page(f"{category}/files/f{i}.pdf", 1, f"טקסט מספר {i} " + "מידע " * 200)
            for i in range(n)]


def item_dict(**overrides) -> dict:
    base = {
        "id": "v2-001-dental-easy", "domain": CATEGORY, "difficulty": "easy",
        "kind": "standard", "answerable": True, "question": QUESTION,
        "ground_truth_answer": ANSWER,
        "ground_truth_sources": [{"any_of": [{"file": "dental/files/f0.pdf", "page": 1}]}],
        "provenance": {"generator_model": "a", "verifier_models": ["b", "c"],
                       "attempts": 1, "gates": {"form": "pass"}},
    }
    return {**base, **overrides}


@pytest.fixture
def llm(monkeypatch):
    """Mocks every model call. Tests tweak `replies` to make a gate reject."""
    from evalharness import judge

    state = {
        "support": {1: "fully", 2: "fully"},  # by number of pages shown
        "difficulty": "easy",
        "form": "pass",
        "topicality": "pass",
        "generation": {"question": QUESTION, "ground_truth_answer": ANSWER,
                       "rationale": "r"},
        "calls": [],
    }

    def fake_chat(messages, model, **kw):
        system, user = messages[0]["content"], messages[1]["content"]
        # The last message, not messages[1]: a retry appends the rejection
        # reason after the original request.
        state["calls"].append({"model": model, "system": system,
                               "last": messages[-1]["content"], "user": user})
        if "citation_support" in system:
            n = user.count("--- CITATION")
            support = state["support"].get(n, state["support"][1])
            return json.dumps({"citation_support": support,
                               "per_citation": [{"idx": i, "label": "establishes"}
                                                for i in range(n)],
                               "reasoning": "ok"})
        if "classify how hard" in system:
            return json.dumps({"difficulty": state["difficulty"], "reason": "because"})
        if "right form" in system:
            return json.dumps({"verdict": state["form"], "reason": "voice"})
        if "belongs to an insurance category" in system:
            return json.dumps({"verdict": state["topicality"], "reason": "topic"})
        return json.dumps(state["generation"])

    monkeypatch.setattr(judge, "chat", fake_chat)
    return state


def build(kind="standard", difficulty="easy", pages=None, **kwargs):
    pages = pages or make_pages()
    defaults = dict(category=CATEGORY, sampler=Sampler(pages, seed=0),
                    category_pages=pages, examples=EXAMPLES,
                    model=generate.GENERATOR_MODELS[0], item_id=f"v2-001-{CATEGORY}-{difficulty}",
                    cell_used=set(), existing_questions=[], wanted=None)
    return generate.build_item(kind, difficulty, **{**defaults, **kwargs})


# -- inventory + sampler ----------------------------------------------------


def test_sampler_spreads_draws_across_files():
    sampler = Sampler(make_pages(6), seed=0)
    drawn = [sampler.draw() for _ in range(6)]
    assert len({p.file for p in drawn}) == 6, "each draw should come from a fresh file"
    assert sampler.draw() is None, "pool exhausted"


def test_sampler_never_offers_a_held_out_page():
    pages = make_pages(3)
    sampler = Sampler(pages, seed=0, excluded={pages[1].key})
    assert pages[1] not in [sampler.draw() for _ in range(2)]


def test_sampler_falls_back_to_the_cell_constraint_when_the_pool_runs_out():
    """A thin category may have fewer pages than questions; the spec only
    forbids reuse inside one category x difficulty cell."""
    pages = make_pages(2)
    sampler = Sampler(pages, seed=0)
    [sampler.draw() for _ in range(2)]
    assert sampler.draw() is None
    assert sampler.draw(cell_used=set()) is not None


def test_sampler_pairs_two_different_files():
    first, second = Sampler(make_pages(6), seed=0).draw_pair()
    assert first.file != second.file


def test_released_pages_return_to_the_pool():
    sampler = Sampler(make_pages(1), seed=0)
    page = sampler.draw()
    assert sampler.draw() is None
    sampler.release(page)
    assert sampler.draw() == page


def test_inventory_reads_pages_through_the_page_store(tmp_path):
    """build_inventory must see exactly what the citation judge will later
    resolve, so it goes through PageStore rather than the corpus directly."""
    from tests.test_evalharness import ANCHOR, PAGE1  # shared synthetic corpus

    class FakeStore:
        _pages = {ANCHOR: {1: PAGE1 * 40, 2: "short"}}
        _page_ids = {ANCHOR: {1, 2}}

        def _load_sources(self):
            source = type("S", (), {"category": "apartment", "kind": "pdf"})()
            return {ANCHOR: source}

        def _extract(self, rel_path):  # already populated
            pass

    pages = build_inventory("apartment", FakeStore(), verify_order=False)
    assert [p.page for p in pages] == [1], "the 'short' page is below the threshold"


def test_scrambled_pages_are_kept_out_of_the_inventory():
    """Docling leaves some Hebrew in visual word order. A question written from
    a scrambled page gets a scrambled ground truth, and the derivability judge
    reads that same page and agrees — so the only defence is to never write
    from those pages."""
    from refgen.inventory import order_agreement

    oracle = "הננו להביא לתשומת לבך כי תקופת ההתיישנות היא שלוש שנים"
    good = "הננו להביא לתשומת לבך כי תקופת\nההתיישנות היא שלוש שנים"
    scrambled = "שנים שלוש היא ההתיישנות תקופת כי לבך לתשומת להביא הננו"
    assert order_agreement(good, oracle) == 1.0
    assert order_agreement(scrambled, oracle) == 0.0
    # Nothing checkable is not evidence of a problem.
    assert order_agreement("שתי מילים", oracle) == 1.0


def test_the_page_oracle_reads_a_real_pdf_per_page(anchor_pdf):
    """poppler's pdftotext, one subprocess per document, split on form feeds."""
    from refgen.inventory import _oracle_pages, oracle_available

    if not oracle_available():
        pytest.skip("pdftotext not installed")
    pages = _oracle_pages(anchor_pdf)
    assert pages and 1 in pages
    assert "התיישנות" in pages[1], "the anchor PDF's page 1 states the limitation period"


def test_a_missing_page_oracle_is_an_error_not_a_silently_skipped_check(monkeypatch):
    """Dropping the check silently would let scrambled pages become questions."""
    import refgen.inventory as inventory

    monkeypatch.setattr(inventory.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="poppler-utils"):
        inventory.build_inventory("dental", object())


def test_bm25_finds_the_page_that_shares_the_query_terms():
    pages = [Page("a.pdf", 1, "ביטוח שיניים כתר וסתימות " * 20),
             Page("b.pdf", 1, "ביטוח נסיעות לחו\"ל מזוודה " * 20)]
    assert bm25_search("כמה עולה כתר", pages, k=1)[0].file == "a.pdf"


# -- gates ------------------------------------------------------------------


def test_accepted_item_records_its_gates_and_verifiers(llm):
    outcome = build()
    assert outcome.item is not None
    assert outcome.item.provenance.gates == {"form": "pass", "derivable": "fully",
                                             "difficulty": "easy"}
    assert generate.GENERATOR_MODELS[0] not in outcome.item.provenance.verifier_models


def test_partly_supported_answer_is_rejected(llm):
    """An item whose own sources only half-establish its answer would punish a
    system that retrieved them perfectly."""
    llm["support"] = {1: "partially", 2: "partially"}
    outcome = build()
    assert outcome.item is None
    assert {a.gate for a in outcome.attempts} == {"derivable"}


def test_multi_source_item_needs_both_pages(llm):
    llm["support"] = {1: "partially", 2: "fully"}  # one page alone is not enough
    llm["difficulty"] = "hard"
    outcome = build(kind="multi_source", difficulty="hard")
    assert outcome.item is not None
    assert len(outcome.item.ground_truth_sources) == 2
    assert outcome.item.provenance.gates["needs_both"]


def test_multi_source_rejected_when_one_page_answers_alone(llm):
    llm["support"] = {1: "fully", 2: "fully"}
    llm["difficulty"] = "hard"
    outcome = build(kind="multi_source", difficulty="hard", pages=make_pages(20))
    assert outcome.item is None
    assert {a.gate for a in outcome.attempts} == {"needs_both"}


def test_multi_source_spends_its_budget_on_pairs_not_on_retries(llm):
    """Whether a question can need both pages is decided by the pair, so a
    rejected pair is usually unusable however well the reason is explained."""
    llm["support"] = {1: "fully", 2: "fully"}
    llm["difficulty"] = "hard"
    outcome = build(kind="multi_source", difficulty="hard", pages=make_pages(30))
    assert len(outcome.attempts) == generate.MAX_PAIRS * (generate.MAX_RETRIES_MULTI + 1)
    assert generate.MAX_PAIRS > generate.MAX_PAGES


def test_difficulty_mismatch_is_rejected_when_no_cell_is_open(llm):
    """multi_source and unanswerable items pass wanted=None: their difficulty
    is fixed by kind, so a mismatch is a rejection."""
    llm["difficulty"] = "easy"
    outcome = build(difficulty="hard")
    assert outcome.item is None
    assert "rates this easy, not hard" in outcome.attempts[0].reason


def test_item_is_filed_under_the_difficulty_the_judge_gives_it(llm):
    """Classify and place: the label is the independent judge's verdict, not
    the generator's claim, so asking for easy and earning medium is fine as
    long as the medium cell has room."""
    llm["difficulty"] = "medium"
    outcome = build(difficulty="easy", wanted={"easy", "medium", "hard"})
    assert outcome.item is not None
    assert outcome.item.difficulty == "medium"
    assert outcome.item.id.endswith("-medium"), "the id must follow the earned label"


def test_item_is_rejected_when_its_earned_cell_is_already_full(llm):
    llm["difficulty"] = "medium"
    outcome = build(difficulty="easy", wanted={"easy"})
    assert outcome.item is None
    assert "still needs easy questions" in outcome.attempts[0].reason


def test_form_failure_is_rejected(llm):
    llm["form"] = "fail"
    outcome = build()
    assert outcome.item is None
    assert {a.gate for a in outcome.attempts} == {"form"}


def test_unanswerable_item_is_accepted_only_when_the_corpus_stays_silent(llm):
    llm["support"] = {n: "not_at_all" for n in range(1, 30)}
    outcome = build(kind="unanswerable", difficulty="medium")
    assert outcome.item is not None
    assert outcome.item.answerable is False
    assert outcome.item.ground_truth_sources == []
    assert "not_at_all" in outcome.item.provenance.gates["unanswerable"]


def test_unanswerable_item_rejected_when_the_corpus_does_answer_it(llm):
    llm["support"] = {n: "fully" for n in range(1, 30)}
    outcome = build(kind="unanswerable", difficulty="medium")
    assert outcome.item is None
    assert {a.gate for a in outcome.attempts} == {"unanswerable"}


def test_off_topic_unanswerable_item_is_rejected(llm):
    """Unanswerable but off-topic tests nothing: the corpus is silent about
    the weather too. Only the topicality gate may fail here."""
    llm["topicality"] = "fail"
    llm["support"] = {n: "not_at_all" for n in range(1, 30)}
    outcome = build(kind="unanswerable", difficulty="easy")
    assert outcome.item is None
    assert {a.gate for a in outcome.attempts} == {"topicality"}


def test_the_most_actionable_failure_is_the_one_reported(llm):
    """Gates run concurrently, so several can fail at once. A question its own
    page does not support needs rewriting before its wording matters."""
    llm["form"] = "fail"
    llm["support"] = {1: "not_at_all", 2: "not_at_all"}
    outcome = build()
    assert {a.gate for a in outcome.attempts} == {"derivable"}


def test_duplicate_question_is_rejected_before_any_gate_runs(llm):
    outcome = build(existing_questions=[QUESTION])
    assert outcome.item is None
    assert {a.gate for a in outcome.attempts} == {"duplicate"}
    assert not any("citation_support" in c["system"] for c in llm["calls"]), \
        "a duplicate must not cost a judge call"


def test_rejection_reason_is_fed_back_to_the_generator(llm):
    llm["form"] = "fail"
    build()
    retries = [c for c in llm["calls"] if "That question was rejected" in c["last"]]
    assert retries, "the generator must be told why, not blindly re-rolled"
    assert "voice" in retries[0]["last"], "the reason must be the gate's own"


def test_a_schema_violation_is_a_rejection_not_a_crash(llm):
    """The LLM gates do let things past: a non-Hebrew question survived the
    form gate in the full run and the schema raised, which propagated out of
    the worker and killed all twelve categories."""
    llm["generation"] = {"question": "Does my policy cover this dental treatment?",
                         "ground_truth_answer": "Yes, up to 500 NIS.", "rationale": "r"}
    outcome = build()  # must not raise
    assert outcome.item is None
    assert {a.gate for a in outcome.attempts} == {"schema"}
    assert "Hebrew" in outcome.attempts[0].reason


def test_a_non_answer_ground_truth_is_treated_as_a_skip(llm):
    """Handed a thin page, generators write "the page does not state X" rather
    than declining. That is not a ground truth — and arguing with it wastes
    three gate calls to be told what we already know."""
    llm["generation"] = {"question": QUESTION,
                         "ground_truth_answer": "הדף אינו מציין את גובה ההשתתפות העצמית.",
                         "rationale": "r"}
    outcome = build()
    assert outcome.item is None
    assert [a.gate for a in outcome.attempts] == ["skip"] * generate.MAX_PAGES
    assert not any("citation_support" in c["system"] for c in llm["calls"]), \
        "a disguised skip must not reach the derivability judge"


def test_an_unanswerable_item_may_say_the_corpus_lacks_it(llm):
    """The same wording is the whole point for an unanswerable item."""
    llm["support"] = {n: "not_at_all" for n in range(1, 30)}
    llm["generation"] = {"question": QUESTION,
                         "ground_truth_answer": "המסמכים אינם מציינים מידע זה; יש לפנות לסוכן.",
                         "rationale": "r"}
    assert build(kind="unanswerable", difficulty="easy").item is not None


def test_a_skipped_page_is_not_argued_with(llm):
    """A model that declines a page should get a different page, not two more
    attempts at the same one."""
    llm["generation"] = {"skip": "this page is a form with no substantive content"}
    outcome = build()
    assert outcome.item is None
    assert [a.gate for a in outcome.attempts] == ["skip"] * generate.MAX_PAGES


def test_gates_never_run_on_the_generating_model(llm):
    build()
    gate_calls = [c["model"] for c in llm["calls"]
                  if "You write evaluation questions" not in c["system"]]
    assert gate_calls, "gates did run"
    assert generate.GENERATOR_MODELS[0] not in gate_calls


# -- schema + audit ---------------------------------------------------------


def test_unanswerable_item_may_not_carry_sources():
    with pytest.raises(ValueError, match="answerable=false with no sources"):
        RefQuestion(**item_dict(kind="unanswerable", answerable=False))


def test_multi_source_item_must_cite_two_different_files():
    same_file = [{"any_of": [{"file": "dental/files/f0.pdf", "page": 1}]},
                 {"any_of": [{"file": "dental/files/f0.pdf", "page": 2}]}]
    with pytest.raises(ValueError, match="different files"):
        RefQuestion(**item_dict(id="v2-001-dental-hard", difficulty="hard",
                                kind="multi_source", ground_truth_sources=same_file))


def test_sources_must_belong_to_the_items_own_category():
    with pytest.raises(ValueError, match="own category"):
        RefQuestion(**item_dict(ground_truth_sources=[
            {"any_of": [{"file": "travel/files/f0.pdf", "page": 1}]}]))


def test_a_model_may_not_verify_its_own_item():
    with pytest.raises(ValueError, match="verified its own item"):
        RefQuestion(**item_dict(provenance={"generator_model": "a",
                                            "verifier_models": ["a", "b"]}))


def test_dataset_check_flags_page_reuse_inside_a_cell():
    items = [RefQuestion(**item_dict(id="v2-001-dental-easy")),
             RefQuestion(**item_dict(id="v2-002-dental-easy", question=QUESTION + " ועוד"))]
    problems = schema.check_dataset(items, strict_counts=False)
    assert any("reuses" in p for p in problems)


def test_dataset_check_flags_leakage_from_the_held_out_set():
    items = [RefQuestion(**item_dict())]
    problems = schema.check_dataset(items, v1_pages={("dental/files/f0.pdf", 1)},
                                    strict_counts=False)
    assert any("held-out v1 page" in p for p in problems)


def test_dataset_check_flags_a_near_duplicate_of_a_v1_question():
    items = [RefQuestion(**item_dict())]
    problems = schema.check_dataset(items, v1_questions=[QUESTION + " בבקשה"],
                                    strict_counts=False)
    assert any("near-duplicates" in p for p in problems)


def test_dataset_check_demands_every_category_and_count():
    problems = schema.check_dataset([RefQuestion(**item_dict())])
    assert any("categories with no questions" in p for p in problems)
    assert any("dental/easy: 1 standard items, expected 3" in p for p in problems)


def test_full_size_dataset_passes_every_structural_check():
    """The shape the real run must produce: 12 categories x 11 items."""
    items, n = [], 0
    for index, category in enumerate(sorted(schema.KNOWN_CATEGORIES)):
        for kind, difficulty in generate.category_plan(
                unanswerable_at=generate.unanswerable_difficulty(index)):
            n += 1
            sources = [{"any_of": [{"file": f"{category}/files/f{n}.pdf", "page": 1}]}]
            if kind == "multi_source":
                sources.append({"any_of": [{"file": f"{category}/files/g{n}.pdf", "page": 1}]})
            items.append(RefQuestion(**item_dict(
                id=f"v2-{n:03d}-{category}-{difficulty}", domain=category,
                difficulty=difficulty, kind=kind,
                answerable=kind != "unanswerable",
                # Genuinely different wording per item — questions that differ
                # only by a serial number are near-duplicates, and rightly
                # rejected by check_dataset.
                question=f"שאלה {n} בנושא {category} " + " ".join(
                    f"מילה{n}{w}" for w in range(6)),
                ground_truth_sources=[] if kind == "unanswerable" else sources)))
    assert len(items) == 132
    assert schema.check_dataset(items) == []
    assert len(schema.coverage(items)) == 12


def test_category_plan_asks_for_everything_when_nothing_exists():
    slots = generate.category_plan()
    assert len(slots) == schema.ITEMS_PER_CATEGORY
    assert collections.Counter(k for k, _ in slots) == {
        "standard": 9, "multi_source": 1, "unanswerable": 1}


def test_category_plan_asks_only_for_the_shortfall():
    """What --fill generates: a category with 3 easy, 1 medium and its
    unanswerable item already done needs 2 medium, 3 hard and a multi-source."""
    have = [RefQuestion(**item_dict(id=f"v2-{n:03d}-dental-{d}", difficulty=d,
                                    kind=k, answerable=k != "unanswerable",
                                    question=f"שאלה {n} בנושא ביטוח " + " ".join(
                                        f"מילה{n}{w}" for w in range(6)),
                                    ground_truth_sources=[] if k == "unanswerable" else
                                    [{"any_of": [{"file": f"dental/files/f{n}.pdf", "page": 1}]}]))
            for n, (k, d) in enumerate([("standard", "easy")] * 3
                                       + [("standard", "medium")]
                                       + [("unanswerable", "easy")], start=1)]
    assert collections.Counter(generate.category_plan(have)) == {
        ("standard", "medium"): 2, ("standard", "hard"): 3, ("multi_source", "hard"): 1}


def test_a_full_category_needs_nothing_more():
    def sources(kind, n):
        if kind == "unanswerable":
            return []
        groups = [{"any_of": [{"file": f"dental/files/f{n}.pdf", "page": 1}]}]
        if kind == "multi_source":
            groups.append({"any_of": [{"file": f"dental/files/g{n}.pdf", "page": 1}]})
        return groups

    have = [RefQuestion(**item_dict(id=f"v2-{n:03d}-dental-{d}", difficulty=d, kind=k,
                                    answerable=k != "unanswerable",
                                    question=f"שאלה {n} בנושא ביטוח " + " ".join(
                                        f"מילה{n}{w}" for w in range(6)),
                                    ground_truth_sources=sources(k, n)))
            for n, (k, d) in enumerate(
                [("standard", d) for d in ("easy", "medium", "hard") for _ in range(3)]
                + [("multi_source", "hard"), ("unanswerable", "easy")], start=1)]
    assert generate.category_plan(have) == []


# -- v3 profile: multi-source/calculation top-up -----------------------------


def v3_item(n: int, category: str = CATEGORY) -> dict:
    return item_dict(
        id=f"v3-{n:03d}-{category}-hard", domain=category, difficulty="hard",
        kind="multi_source",
        question=f"שאלה {n} בנושא {category} " + " ".join(f"מילה{n}{w}" for w in range(6)),
        ground_truth_sources=[
            {"any_of": [{"file": f"{category}/files/f{n}.pdf", "page": 1}]},
            {"any_of": [{"file": f"{category}/files/g{n}.pdf", "page": 1}]},
        ])


def test_v3_plan_is_three_multi_source_items_per_category():
    assert generate.plan_for("v3") == [("multi_source", "hard")] * 3


def test_v3_plan_asks_only_for_the_shortfall():
    have = [RefQuestion(**v3_item(1))]
    assert generate.plan_for("v3", have) == [("multi_source", "hard")] * 2


def test_v3_ids_are_accepted_and_recognised_as_a_profile():
    items = [RefQuestion(**v3_item(n)) for n in range(1, 4)]
    assert schema.profile_of(items) == "v3"
    assert schema.profile_of([RefQuestion(**item_dict())]) == "v2"
    assert schema.profile_of(items + [RefQuestion(**item_dict())]) == "v2"


def test_v3_dataset_passes_without_v2s_per_cell_shape():
    """v3 tops up one shape; the counts check is v2's and would fail it."""
    items = [RefQuestion(**v3_item(n)) for n in range(1, 4)]
    assert schema.check_dataset(items, strict_counts=False) == []
    assert schema.check_dataset(items) != []  # the v2 shape check still bites


def test_holdout_excludes_the_pages_and_questions_of_every_held_out_file(tmp_path):
    v1 = tmp_path / "v1.json"
    v2 = tmp_path / "v2.json"
    v1.write_text(json.dumps([item_dict()], ensure_ascii=False), encoding="utf-8")
    v2.write_text(json.dumps([v3_item(9)], ensure_ascii=False), encoding="utf-8")
    pages, questions = schema.load_exclusions([v1, v2])
    assert pages == {("dental/files/f0.pdf", 1), ("dental/files/f9.pdf", 1),
                     ("dental/files/g9.pdf", 1)}
    assert len(questions) == 2
    # A single path is accepted too (the v2 run passes one file).
    assert schema.load_exclusions(v1)[0] == {("dental/files/f0.pdf", 1)}


def test_v3_audit_reports_no_shape_problems(tmp_path, capsys):
    path = tmp_path / "reference_questions_v3.json"
    path.write_text(json.dumps([v3_item(n) for n in range(1, 4)], ensure_ascii=False),
                    encoding="utf-8")
    assert audit.main([str(path), "--no-sources", "--holdout", str(tmp_path / "absent.json")]) == 0


def test_v3_multi_source_prompt_asks_for_combined_figures():
    pages = make_pages(2)
    calc = prompts.build_generation_messages(
        "multi_source", "hard", pages, EXAMPLES, CATEGORY, "calc")[0]["content"]
    mixed = prompts.build_generation_messages(
        "multi_source", "hard", pages, EXAMPLES, CATEGORY)[0]["content"]
    assert "RULE PLUS FIGURE (best)" in calc and "TWO FIGURES COMBINED" in calc
    assert "show its working" in calc  # operands and result must be in the ground truth
    assert "TWO PRODUCTS" in mixed and "RULE PLUS FIGURE (best)" not in mixed
    # Both keep the two-page requirement the leave-one-out gate enforces.
    for system in (calc, mixed):
        assert "each page ALONE" in system


def test_the_calc_variant_reaches_the_generator(llm):
    llm["support"] = {1: "partially", 2: "fully"}  # neither page answers alone
    llm["difficulty"] = "hard"
    outcome = build("multi_source", "hard", pages=make_pages(2), variant="calc",
                    item_id=f"v3-001-{CATEGORY}-hard")
    assert outcome.item is not None and outcome.item.id.startswith("v3-")
    generation = [c for c in llm["calls"] if "TWO FIGURES COMBINED" in c["system"]]
    assert generation, "the calculation-biased prompt was not the one sent"


def test_audit_reports_every_malformed_item_at_once(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([item_dict(id="nope"), item_dict(difficulty="trivial")],
                               ensure_ascii=False), encoding="utf-8")
    with pytest.raises(audit.MalformedDataset) as error:
        audit.load(path)
    assert "2 malformed item(s)" in str(error.value)


def test_audit_passes_a_conforming_file(tmp_path, capsys):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps([item_dict()], ensure_ascii=False), encoding="utf-8")
    assert audit.main([str(path), "--no-sources", "--holdout", str(tmp_path / "absent.json")]) == 1
    assert "1 items" in capsys.readouterr().err  # structure fine, counts incomplete


# -- prompts ----------------------------------------------------------------


def test_form_anchors_never_come_from_the_target_category():
    import random

    from refgen.run import _examples_for

    v1 = json.load(open("reference_questions.json", encoding="utf-8"))
    examples = _examples_for(v1, "easy", "apartment", random.Random(0))
    assert examples and all(e["domain"] != "apartment" for e in examples)
    assert all(e["difficulty"] == "easy" for e in examples)


def test_generation_prompt_shows_the_page_and_forbids_referring_to_it():
    pages = make_pages(1)
    messages = prompts.build_generation_messages("standard", "easy", pages, EXAMPLES, CATEGORY)
    assert pages[0].text in messages[1]["content"]
    assert "never mention the document" in messages[0]["content"].lower()
    assert "skip" in messages[0]["content"].lower()


def test_a_skip_is_a_valid_reply_not_a_parse_error():
    assert prompts.parse_generation({"skip": "form with no content"})["skip"]


def test_generation_reply_must_carry_both_fields():
    with pytest.raises(ValueError, match="ground_truth_answer too short"):
        prompts.parse_generation({"question": QUESTION, "ground_truth_answer": ""})

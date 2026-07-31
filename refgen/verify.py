"""The gates a candidate question must pass before it enters the dataset.

Derivability reuses the citation judge from `evalharness` verbatim: that judge
answers "do these pages establish this ground-truth answer, fully / partially /
not at all", which is exactly the question being asked of a candidate item —
and it means a v2 item is accepted under the same rule the eval harness will
later score systems by.

Every gate runs on a model that did not write the item.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from evalharness.judge import _call_judge, judge_citations_one

from . import prompts


class Rejected(Exception):
    """A gate refused the item. `gate` names it; the message is fed back to the
    generator as the retry reason."""

    def __init__(self, gate: str, reason: str):
        super().__init__(f"{gate}: {reason}")
        self.gate = gate
        self.reason = reason


def _as_pages(pages: list) -> list:
    """Inventory pages in the shape the citation judge expects."""
    return [{"file": p.file, "page": p.page, "text": p.text} for p in pages]


def _support(item: dict, pages: list, model: str) -> str:
    """The citation judge's ruling for `pages` against the item's answer."""
    question = {"question": item["question"],
                "ground_truth_answer": item["ground_truth_answer"]}
    judgment = judge_citations_one(question, _as_pages(pages), model)
    if "error" in judgment:
        raise Rejected("judge_error", judgment["error"])
    return judgment["citation_support"]


def gate_derivable(item: dict, pages: list, model: str) -> str:
    """The cited pages must fully establish the answer — no half credit. An
    item whose own sources only partly support it would punish a system that
    retrieved them perfectly."""
    support = _support(item, pages, model)
    if support != "fully":
        raise Rejected("derivable",
                       f"the source page(s) only support the answer '{support}' — "
                       "the answer must be fully established by the page(s) shown")
    return support


def gate_needs_both(item: dict, pages: list, model: str) -> str:
    """Leave-one-out: this is what "the answer spans two citations" means.

    Both pages together must establish the answer (checked by `gate_derivable`);
    each page alone must not. A model that quietly wrote a single-page question
    fails here.
    """
    for index, page in enumerate(pages):
        support = _support(item, [page], model)
        if support == "fully":
            raise Rejected("needs_both",
                           f"page {index + 1} ({page.file} p{page.page}) answers the "
                           "question on its own — the question must need both pages")
    return "neither page suffices alone"


def gate_unanswerable(item: dict, category_pages: list, model: str, k: int = 20) -> str:
    """Nothing in the category may answer it.

    BM25 over the whole category is the recall net; the citation judge reads the
    top-k and must rule `not_at_all`. Weak Hebrew morphology in the plain-token
    BM25 costs some recall, so this proves "no closely-matching page answers
    it", not a corpus-wide impossibility proof — recorded as such in the item's
    provenance.
    """
    from .inventory import bm25_search

    hits = bm25_search(item["question"], category_pages, k=k)
    if not hits:
        return "no candidate pages retrieved"
    support = _support(item, hits, model)
    if support != "not_at_all":
        raise Rejected("unanswerable",
                       f"the corpus does answer this ({support}) — e.g. "
                       f"{hits[0].file} p{hits[0].page}. Ask about something the "
                       "documents genuinely do not state")
    return f"not_at_all over BM25 top-{len(hits)}"


def gate_form(item: dict, model: str) -> str:
    """Customer voice, self-contained, decisive answer."""
    verdict = _call_judge(prompts.build_form_messages(item), model, prompts.parse_verdict)
    if "error" in verdict:
        raise Rejected("judge_error", verdict["error"])
    if verdict["verdict"] != "pass":
        raise Rejected("form", verdict["reason"] or "does not read like a customer question")
    return "pass"


def gate_difficulty(item: dict, difficulty: str, n_sources: int, model: str) -> str:
    """An independent model must agree with the requested difficulty label."""
    got = _call_judge(prompts.build_difficulty_messages(item, n_sources), model,
                      prompts.parse_difficulty)
    if "error" in got:
        raise Rejected("judge_error", got["error"])
    if got["difficulty"] != difficulty:
        raise Rejected("difficulty",
                       f"an independent judge rates this {got['difficulty']}, not "
                       f"{difficulty} ({got['reason']})")
    return difficulty


def gate_topicality(item: dict, category: str, category_pages: list, model: str) -> str:
    """An unanswerable question still has to be a question this category's
    customers would ask — otherwise it tests nothing."""
    verdict = _call_judge(
        prompts.build_topicality_messages(item["question"], category, category_pages),
        model, prompts.parse_verdict)
    if "error" in verdict:
        raise Rejected("judge_error", verdict["error"])
    if verdict["verdict"] != "pass":
        raise Rejected("topicality", verdict["reason"] or "not a question this category's customers would ask")
    return "pass"


def run_gates(kind: str, item: dict, difficulty: str, pages: list, category: str,
              category_pages: list, verifiers: list) -> dict:
    """Every gate for one candidate, run concurrently.

    The gates are independent, and each is a slow reasoning-model call — run
    sequentially they dominate the wall clock (~10 minutes per accepted item in
    the first trial). Concurrency costs nothing extra: a rejected item spends
    the same calls either way, since the retry needs the reason regardless.

    `verifiers` are models other than the generator; gates are spread over them
    round-robin, so one model's blind spot cannot admit an item on its own.
    Returns {gate: verdict}; raises the most important `Rejected` on failure.
    """
    pick = lambda i: verifiers[i % len(verifiers)]
    if kind == "unanswerable":
        jobs = {
            "form": lambda: gate_form(item, pick(0)),
            "topicality": lambda: gate_topicality(item, category, category_pages, pick(1)),
            "unanswerable": lambda: gate_unanswerable(item, category_pages, pick(2)),
        }
    else:
        jobs = {
            "form": lambda: gate_form(item, pick(0)),
            "derivable": lambda: gate_derivable(item, pages, pick(1)),
            "difficulty": lambda: gate_difficulty(item, difficulty, len(pages), pick(3)),
        }
        if kind == "multi_source":
            jobs["needs_both"] = lambda: gate_needs_both(item, pages, pick(2))

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        gates, failures = {}, {}
        for name, future in futures.items():
            try:
                gates[name] = future.result()
            except Rejected as rejection:
                failures[name] = rejection

    if failures:
        # Report the most actionable failure: a question that is not supported
        # by its page needs rewriting before its form or difficulty matter.
        order = ("derivable", "unanswerable", "needs_both", "topicality", "form",
                 "difficulty", "judge_error")
        first = min(failures.values(), key=lambda r: order.index(r.gate)
                    if r.gate in order else len(order))
        raise first
    return gates

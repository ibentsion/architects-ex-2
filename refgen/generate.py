"""Building one item, and building the whole dataset.

Per item: draw pages -> generate -> run the gates -> accept, or feed the
rejection reason back and try again. A generator that skips, or fails its
retries, gets a fresh page rather than another attempt at the same one.
Nothing that failed a gate is ever emitted; exhausted items are reported.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from evalharness.judge import _call_judge

from . import prompts, schema, verify
from .inventory import Sampler
from .schema import DIFFICULTIES, RefQuestion

logger = logging.getLogger(__name__)

#: Question writers. Round-robin per item so no single model's habits shape the
#: dataset; every item's gates run on the other two.
GENERATOR_MODELS = (
    "google/gemma-3-27b-it",
    "zai-org/GLM-5.1",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
)
#: Re-prompts with the rejection reason before giving up on a page.
MAX_RETRIES = 2
#: Pages tried before giving up on an item.
MAX_PAGES = 3
#: Pages shown to an unanswerable-question writer as category context.
UNANSWERABLE_CONTEXT_PAGES = 8
#: Generation prompts carry whole corpus pages, and the writers are reasoning
#: models — at the judges' 4096 budget GLM-5.1 spent it all thinking and
#: returned nothing on two-page multi-source prompts.
GENERATION_MAX_TOKENS = 12288


@dataclass
class Attempt:
    """One rejected candidate, kept for the run report."""

    kind: str
    difficulty: str
    model: str
    gate: str
    reason: str
    question: str = ""


@dataclass
class Outcome:
    """What building one item produced."""

    item: RefQuestion | None = None
    attempts: list[Attempt] = field(default_factory=list)
    failure: str = ""


class Skipped(Exception):
    """The generator declined this page — draw another rather than argue."""


def _generate_candidate(kind: str, difficulty: str, pages: list, examples: list,
                        category: str, model: str, reason: str | None) -> dict:
    """One generator call. `reason` re-prompts after a rejection."""
    messages = prompts.build_generation_messages(kind, difficulty, pages, examples, category)
    if reason:
        messages = messages + [prompts.retry_message(reason)]
    result = _call_judge(messages, model, prompts.parse_generation,
                         max_tokens=GENERATION_MAX_TOKENS)
    if "error" in result:
        raise verify.Rejected("generation", result["error"])
    if "skip" in result:
        raise Skipped(result["skip"])
    return result


def build_item(kind: str, difficulty: str, category: str, sampler: Sampler,
               category_pages: list, examples: list, model: str, item_id: str,
               cell_used: set, existing_questions: list) -> Outcome:
    """Draw pages and write one accepted item, or report why it could not be."""
    verifiers = [m for m in GENERATOR_MODELS if m != model]
    outcome = Outcome()

    for _page_attempt in range(MAX_PAGES):
        if kind == "unanswerable":
            pages, drawn = category_pages[:UNANSWERABLE_CONTEXT_PAGES], []
        elif kind == "multi_source":
            pair = sampler.draw_pair()
            if pair is None:
                outcome.failure = "no page pair available"
                return outcome
            pages = drawn = list(pair)
        else:
            page = sampler.draw(prefer_numbers=difficulty in ("medium", "hard"),
                                cell_used=cell_used)
            if page is None:
                outcome.failure = "no page available"
                return outcome
            pages = drawn = [page]

        reason = None
        for _retry in range(MAX_RETRIES + 1):
            candidate = None
            try:
                candidate = _generate_candidate(kind, difficulty, pages, examples,
                                                category, model, reason)
                duplicate = schema.is_duplicate(candidate["question"], existing_questions)
                if duplicate is not None:
                    raise verify.Rejected("duplicate",
                                          "this repeats an existing question in the "
                                          f"dataset: {duplicate[:80]}")
                gates = verify.run_gates(kind, candidate, difficulty, drawn, category,
                                         category_pages, verifiers)
            except Skipped as skip:
                outcome.attempts.append(Attempt(kind, difficulty, model, "skip", str(skip)))
                break  # the page is unusable — draw another
            except verify.Rejected as rejection:
                logger.info("%s rejected at %s: %s", item_id, rejection.gate, rejection.reason)
                outcome.attempts.append(Attempt(kind, difficulty, model, rejection.gate,
                                                rejection.reason,
                                                (candidate or {}).get("question", "")))
                reason = rejection.reason
                continue

            sources = [{"any_of": [{"file": p.file, "page": p.page}]} for p in drawn]
            outcome.item = RefQuestion(
                id=item_id, domain=category, difficulty=difficulty, kind=kind,
                answerable=kind != "unanswerable",
                question=candidate["question"],
                ground_truth_answer=candidate["ground_truth_answer"],
                ground_truth_sources=sources,
                provenance={"generator_model": model, "verifier_models": verifiers,
                            "attempts": len(outcome.attempts) + 1, "gates": gates},
            )
            return outcome

        sampler.release(*drawn)  # this page did not work out; try a different one

    outcome.failure = f"exhausted {MAX_PAGES} pages"
    return outcome


def category_plan() -> list[tuple[str, str]]:
    """The (kind, difficulty) slots one category must fill.

    Three standard questions at each difficulty, one multi-source (hard by
    definition), one unanswerable (its difficulty is assigned per category by
    the caller).
    """
    slots = [("standard", difficulty)
             for difficulty in DIFFICULTIES
             for _ in range(schema.STANDARD_PER_CELL)]
    return slots + [("multi_source", "hard")]


def unanswerable_difficulty(category_index: int) -> str:
    """Rotate the 12 unanswerable items across difficulties (4 each): for these,
    difficulty is how tempting the gap is, not how hard the lookup is."""
    return DIFFICULTIES[category_index % len(DIFFICULTIES)]

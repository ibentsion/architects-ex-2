"""Building one item, and building the whole dataset.

Per item: draw pages -> generate -> run the gates -> accept, or feed the
rejection reason back and try again. A generator that skips, or fails its
retries, gets a fresh page rather than another attempt at the same one.
Nothing that failed a gate is ever emitted; exhausted items are reported.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import ValidationError

from evalharness.judge import _call_judge

from . import prompts, schema, verify
from .inventory import Sampler
from .schema import DIFFICULTIES, RefQuestion

logger = logging.getLogger(__name__)

#: Question writers: three labs, round-robin per item so no single model's
#: habits shape the dataset; every item's gates run on the other two.
#:
#: Measured on a real two-page generation prompt (11k chars): gemma 5.8 s,
#: Qwen 12.5 s, Nemotron 25.0 s. GLM-5.1 was the original third member and
#: took 537 s on the same call — it is a reasoning model that spent the whole
#: budget thinking, which is what made the first trial run unusable.
#: gpt-oss-120b is faster still (3.6 s) but is deliberately absent: it
#: generates the RAG system's answers and sits on the judge committee, so
#: letting it also write the questions would tune the dataset to one model.
GENERATOR_MODELS = (
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
)
#: Re-prompts with the rejection reason before giving up on a page.
MAX_RETRIES = 2
#: Pages tried before giving up on an item.
MAX_PAGES = 3
#: Multi-source is pair-limited, not prompt-limited: whether a question can
#: need both pages is decided by the pair, so a rejected pair is usually
#: unusable however well the reason is explained. Spend the budget on more
#: pairs and less arguing — 1 of 3 categories succeeded at 3 pairs x 3 tries.
MAX_PAIRS = 8
MAX_RETRIES_MULTI = 1
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
    if kind != "unanswerable" and prompts.is_non_answer(result["ground_truth_answer"]):
        # A disguised skip: the model was handed a page too thin to support a
        # question and wrote "the page does not state X" rather than declining.
        raise Skipped(f"non-answer ground truth: {result['ground_truth_answer'][:80]}")
    return result


def build_item(kind: str, difficulty: str, category: str, sampler: Sampler,
               category_pages: list, examples: list, model: str, item_id: str,
               cell_used: set, existing_questions: list,
               wanted: set | None = None) -> Outcome:
    """Draw pages and write one accepted item, or report why it could not be.

    `difficulty` steers the generator; for standard items the accepted item's
    difficulty is the one an independent judge assigns (see
    `verify.gate_difficulty`), and `wanted` is the set of difficulties this
    category still has room for.
    """
    verifiers = [m for m in GENERATOR_MODELS if m != model]
    outcome = Outcome()
    draws = MAX_PAIRS if kind == "multi_source" else MAX_PAGES
    retries = MAX_RETRIES_MULTI if kind == "multi_source" else MAX_RETRIES

    for _page_attempt in range(draws):
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
        for _retry in range(retries + 1):
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
                                         category_pages, verifiers, wanted)
            except Skipped as skip:
                outcome.attempts.append(Attempt(kind, difficulty, model, "skip", str(skip)))
                # Keep the page out of the pool: a model that called it
                # unusable was not making a mistake about it, and releasing it
                # only spends later items' page budget on the same dead end.
                drawn = []
                break
            except verify.Rejected as rejection:
                logger.info("%s rejected at %s: %s", item_id, rejection.gate, rejection.reason)
                outcome.attempts.append(Attempt(kind, difficulty, model, rejection.gate,
                                                rejection.reason,
                                                (candidate or {}).get("question", "")))
                reason = rejection.reason
                continue

            sources = [{"any_of": [{"file": p.file, "page": p.page}]} for p in drawn]
            # The schema is the last gate, and the LLM gates do let things past
            # it (a non-Hebrew question survived the form gate once). Treat a
            # violation as one more rejection to retry against, never as an
            # exception — it would take down every other category's thread.
            # The judge's classification wins for standard items; multi-source
            # items are hard by definition and unanswerable ones carry the
            # difficulty of the gap, which no page-based rubric can rate.
            earned = gates.get("difficulty", difficulty) if kind == "standard" else difficulty
            try:
                outcome.item = RefQuestion(
                    id=item_id.rsplit("-", 1)[0] + f"-{earned}",
                    domain=category, difficulty=earned, kind=kind,
                    answerable=kind != "unanswerable",
                    question=candidate["question"],
                    ground_truth_answer=candidate["ground_truth_answer"],
                    ground_truth_sources=sources,
                    provenance={"generator_model": model, "verifier_models": verifiers,
                                "attempts": len(outcome.attempts) + 1, "gates": gates},
                )
            except ValidationError as invalid:
                messages = "; ".join(error["msg"] for error in invalid.errors())
                logger.info("%s rejected at schema: %s", item_id, messages)
                outcome.attempts.append(Attempt(kind, difficulty, model, "schema",
                                                messages, candidate["question"]))
                reason = messages
                continue
            return outcome

        sampler.release(*drawn)  # this page did not work out; try a different one

    outcome.failure = f"exhausted {draws} {'pairs' if kind == 'multi_source' else 'pages'}"
    return outcome


def category_plan(existing: list | None = None,
                  unanswerable_at: str = "medium") -> list[tuple[str, str]]:
    """The (kind, difficulty) slots one category still needs.

    Three standard questions at each difficulty, one multi-source (hard by
    definition), one unanswerable. With `existing` — items already accepted for
    this category, from a previous run — only the shortfall is returned, which
    is what `--fill` generates.
    """
    existing = existing or []
    slots = []
    for difficulty in DIFFICULTIES:
        have = sum(1 for item in existing
                   if item.kind == "standard" and item.difficulty == difficulty)
        slots += [("standard", difficulty)] * max(0, schema.STANDARD_PER_CELL - have)
    if not any(item.kind == "multi_source" for item in existing):
        slots.append(("multi_source", "hard"))
    if not any(item.kind == "unanswerable" for item in existing):
        slots.append(("unanswerable", unanswerable_at))
    return slots


def unanswerable_difficulty(category_index: int) -> str:
    """Rotate the 12 unanswerable items across difficulties (4 each): for these,
    difficulty is how tempting the gap is, not how hard the lookup is."""
    return DIFFICULTIES[category_index % len(DIFFICULTIES)]

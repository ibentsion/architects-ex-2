"""CLI: A/B the query classifier's domain tags across arms and reference sets.

    python -m evalharness.classify_eval \
        --config configs/default.yaml \
        --questions reference_questions.json reference_questions_v2.json \
        --all --build-hints \
        --out eval_results/classify-sweep-$(date -u +%Y%m%dT%H%M%SZ)

What is being measured
----------------------
The agent engine (``rag/agent/engine.py``) filters retrieval for a sub-question
**only when that sub-question carries exactly one category**; zero or two-plus
categories mean no filter. So a predicted tag set maps onto three outcomes:

    exactly [gold]        -> filters to the right slice   -> useful
    exactly one wrong tag -> filters to the wrong slice   -> harmful
    empty, or 2+ tags     -> no filter at all             -> none

``harmful_rate`` is the primary metric (lower is better) and
``filter_correct_rate`` the tie-break: a wrong filter costs a whole retrieval
round and a degraded pool, a missing one only costs precision.

The question-level ``verdict`` is computed from the derived union of the
sub-question categories, exactly as the plan defines it. That collapses one
real case: a two-sub-question query whose sub-questions carry one *different*
category each does filter, twice, even though the union has two entries. That
detail is not discarded — ``applied_filters`` and the ``sub_*`` metrics record
the per-sub-question truth as a diagnostic — it is just kept out of the headline
number so the metric stays the one the plan specifies.

Reading the numbers
-------------------
n is 169 across both sets, so arm-to-arm gaps of a few points are noise. Every
arm is therefore also scored *paired* against ``baseline`` on the same question
ids: ``n_fixed`` (baseline harmful, arm not), ``n_broken`` (the reverse), and an
exact two-sided McNemar p-value over those two counts. An arm with a better
harmful_rate and p > 0.05 has not been shown to be better.

Instrumentation note
--------------------
``rag.classify`` swallows every exception into its no-filter fallback, by
design — classification must never block answering. That leaves an evaluator
with no way to see a 429, a truncation, or the raw reply from outside. So the
run installs :class:`ChatRecorder` over ``rag.classify.tf_chat`` for its
duration: retry/backoff, cost accounting and per-call capture all live *below*
the swallow, and production keeps its behaviour untouched.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import rag.classify as classify_mod
from rag.classify import CATEGORIES, QueryClassifier, _extract_json
from rag.config import load_config

from .classify_arms import ARMS, BASELINE_ARM, Arm, get_arm
from .classify_prompts import (
    VERIFY_SYSTEM_PROMPT,
    build_prompt,
    build_verify_user_message,
)

#: Confusion-table / report label for "predicted no filter at all".
NO_FILTER = "(none)"

#: Hint pass: top-10 fused hits, no rerank. The cross-encoder is the expensive
#: CPU stage and a category prior does not need its precision.
HINT_TOP_K = 10
HINT_SNIPPETS = 3
HINT_SNIPPET_CHARS = 200

#: Warnings rag.classify logs on the two paths that end in the fallback. Any
#: other warning it emits (unknown category dropped) is not a parse failure.
FALLBACK_MARKERS = ("Query classification failed", "no usable sub-questions")

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_TEXT = ("rate limit", "ratelimit", "too many requests", "timeout",
                  "timed out", "overloaded", "temporarily", "bad gateway",
                  "service unavailable", "connection reset", "connection error")


class CostExceeded(BaseException):
    """The run's --max-cost ceiling was hit; no further calls are made.

    Deliberately not an ``Exception``: it is raised underneath
    ``rag.classify.classify``, whose blanket ``except Exception`` would
    otherwise turn a budget abort into a silent no-filter prediction."""


# --------------------------------------------------------------------------- #
# Verdicts — the engine's filter semantics, in one place
# --------------------------------------------------------------------------- #


def effective_filter(categories: list[str]) -> str | None:
    """The category retrieval would actually be filtered to, or None."""
    return categories[0] if len(categories) == 1 else None


def verdict_for(categories: list[str], gold: str) -> str:
    """useful / harmful / none for one prediction against its gold domain."""
    chosen = effective_filter(categories)
    if chosen is None:
        return "none"
    return "useful" if chosen == gold else "harmful"


def applied_filters(sub_questions: list[dict]) -> list[str]:
    """The filters the engine would really apply, one per sub-question that
    carries exactly one category (diagnostic — see the module docstring)."""
    return [sq["categories"][0] for sq in sub_questions if len(sq["categories"]) == 1]


def derive_categories(sub_questions: list[dict]) -> list[str]:
    """Ordered union, the same derivation ``rag.classify`` applies."""
    return list(dict.fromkeys(c for sq in sub_questions for c in sq["categories"]))


def vote_categories(samples: list[list[str]], min_votes: int) -> set[str]:
    """Self-consistency: categories carried by at least ``min_votes`` samples.

    Votes are counted over each sample's question-level category union, so a
    category proposed twice inside one sample still counts once."""
    counts = Counter(c for sample in samples for c in dict.fromkeys(sample))
    return {c for c, n in counts.items() if n >= min_votes}


# --------------------------------------------------------------------------- #
# Budget + instrumented chat
# --------------------------------------------------------------------------- #


class Budget:
    """Shared running cost with a hard ceiling."""

    def __init__(self, max_cost: float) -> None:
        self.max_cost = max_cost
        self.total = 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        if self.total >= self.max_cost:
            raise CostExceeded(
                f"spent ${self.total:.2f} of the ${self.max_cost:.2f} ceiling")

    def add(self, cost: float) -> None:
        with self._lock:
            self.total += cost


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS
    text = str(exc).lower()
    return any(marker in text for marker in RETRYABLE_TEXT)


class ChatRecorder:
    """Stand-in for ``tf_client.chat`` that retries, meters and records.

    Installed over ``rag.classify.tf_chat`` for the duration of a run (see the
    module docstring for why it has to sit below the classifier). Per-call
    capture is thread-local, so questions can run concurrently; the cost total
    is shared and guarded.
    """

    def __init__(self, budget: Budget, chat=None, max_retries: int = 4,
                 base_delay: float = 2.0) -> None:
        self.budget = budget
        self._chat = chat
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._local = threading.local()

    @property
    def _real_chat(self):
        if self._chat is None:
            from tf_client import chat

            self._chat = chat
        return self._chat

    # -- per-question capture ------------------------------------------- #

    def begin(self) -> None:
        self._local.calls = []

    @property
    def calls(self) -> list[dict]:
        return getattr(self._local, "calls", [])

    def __call__(self, messages, **kwargs):
        self.budget.check()
        kwargs = {**kwargs, "return_usage": True}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                text, usage, cost = self._real_chat(messages, **kwargs)
                self.budget.add(cost)
                self.calls.append({
                    "reply": text,
                    "finish_reason": usage.get("finish_reason"),
                    "cost": cost,
                    "attempts": attempt + 1,
                })
                return text, usage, cost
            except Exception as exc:  # CostExceeded is a BaseException and passes through
                last_exc = exc
                if attempt >= self.max_retries or not _is_retryable(exc):
                    break
                time.sleep(self.base_delay * (2 ** attempt))
        self.calls.append({"reply": None, "finish_reason": "error", "cost": 0.0,
                           "attempts": self.max_retries + 1,
                           "error": f"{type(last_exc).__name__}: {last_exc}"})
        raise last_exc  # rag.classify turns this into its no-filter fallback


class WarningCapture(logging.Handler):
    """Per-thread capture of ``rag.classify``'s warnings — the only signal that
    a call fell back rather than parsed."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self._local = threading.local()

    def begin(self) -> None:
        self._local.messages = []

    @property
    def messages(self) -> list[str]:
        return getattr(self._local, "messages", [])

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    @property
    def fallback_count(self) -> int:
        """Fallbacks seen since ``begin()`` — one per failed call, so a
        multi-sample arm can report more than one."""
        return sum(1 for msg in self.messages
                   if any(marker in msg for marker in FALLBACK_MARKERS))

    @property
    def saw_fallback(self) -> bool:
        return self.fallback_count > 0


# --------------------------------------------------------------------------- #
# Hints — one retrieval pass, reused by every hint arm
# --------------------------------------------------------------------------- #


def render_hint(hint: dict | None) -> str:
    """The evidence block a hint arm shows the classifier."""
    if not hint or not hint.get("n_hits"):
        return "החיפוש באינדקס לא החזיר תוצאות."
    n_hits = hint["n_hits"]
    lines = [f"התפלגות תחומים ב-{n_hits} התוצאות המובילות מהאינדקס:"]
    for category, count in hint["histogram"].items():
        lines.append(f"- {category}: {count} מתוך {n_hits} ({count / n_hits:.0%})")
    if hint.get("snippets"):
        lines.append("")
        lines.append("הקטעים המדורגים ראשונים:")
        for snippet in hint["snippets"]:
            lines.append(f"{snippet['rank']}. [{snippet['category']}] {snippet['file']}")
            lines.append(f"   {snippet['text']}")
    return "\n".join(lines)


def _hint_from_hits(hits) -> dict:
    histogram = Counter(hit.chunk.category for hit in hits)
    ordered = dict(histogram.most_common())
    top_category, top_count = next(iter(ordered.items()), (None, 0))
    return {
        "n_hits": len(hits),
        "histogram": ordered,
        "top_category": top_category,
        "top_share": round(top_count / len(hits), 4) if hits else 0.0,
        "snippets": [
            {
                "rank": rank,
                "file": hit.chunk.file,
                "category": hit.chunk.category,
                "text": " ".join(hit.chunk.text.split())[:HINT_SNIPPET_CHARS],
            }
            for rank, hit in enumerate(hits[:HINT_SNIPPETS], start=1)
        ],
    }


def build_hints(retriever, questions: list[dict], workers: int = 4,
                progress=None) -> dict:
    """Dense + sparse -> RRF fuse -> top 10, no rerank, once per question.

    Local qdrant is a single-process lock and stanza is not thread-safe (the
    retriever serializes the sparse stage internally), so this stays a modest
    thread pool and its own pass."""

    def one(question: dict) -> tuple[str, dict]:
        hits = retriever.fuse(question["question"])[:HINT_TOP_K]
        return question["id"], _hint_from_hits(hits)

    hints: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (qid, hint) in enumerate(pool.map(one, questions), start=1):
            hints[qid] = hint
            if progress:
                progress(done, len(questions), qid)
    return hints


# --------------------------------------------------------------------------- #
# Arm execution
# --------------------------------------------------------------------------- #


def build_arm_classifier(arm: Arm, config) -> QueryClassifier:
    """Instantiate the classifier an arm describes.

    Mirrors ``rag.classify.build_classifier``: the orchestrator's reasoning
    knobs only apply while the model is the orchestrator model. ``baseline``
    passes ``system_prompt=None`` so it is the production path itself, not a
    reconstruction of it."""
    model = arm.model or config.harness.orchestrator_model
    if arm.extra_params is not None:
        extra = dict(arm.extra_params)
    elif model == config.harness.orchestrator_model:
        extra = dict(config.harness.orchestrator_extra_params)
    else:
        extra = {}
    return QueryClassifier(
        model,
        extra_params=extra,
        temperature=arm.temperature,
        system_prompt=None if arm.prompt == "baseline" else build_prompt(arm.prompt),
    )


def _sub_questions_of(classification) -> list[dict]:
    return [{"question": sq.question, "categories": list(sq.categories)}
            for sq in classification.sub_questions]


def _keep_only(sub_questions: list[dict], keep: set[str]) -> list[dict]:
    return [{"question": sq["question"],
             "categories": [c for c in sq["categories"] if c in keep]}
            for sq in sub_questions]


def parse_verify_reply(text: str, proposed: list[str]) -> list[str]:
    """``{"keep": [...]}`` -> the surviving subset of ``proposed``.

    Retraction only: ids the first stage never proposed are ignored, so the
    verify arm can only ever remove a tag. Raises on a malformed reply — the
    caller then keeps the first-stage tags, because an LLM hiccup must not
    become a silent tag drop."""
    data = _extract_json(text or "")
    keep = data.get("keep")
    if not isinstance(keep, list):
        raise ValueError(f"verify reply has no keep list: {text!r}")
    return [c for c in proposed if c in set(keep) & set(CATEGORIES)]


class ArmRunner:
    """Produces one prediction per question for a single arm."""

    def __init__(self, arm: Arm, config, hints: dict, recorder: ChatRecorder,
                 capture: WarningCapture) -> None:
        self.arm = arm
        self.config = config
        self.hints = hints
        self.recorder = recorder
        self.capture = capture
        self.classifier = (None if arm.strategy == "hint_vote"
                           else build_arm_classifier(arm, config))

    def predict(self, question: dict) -> dict:
        self.recorder.begin()
        self.capture.begin()
        hint = self.hints.get(question["id"])
        started = time.monotonic()
        if self.arm.strategy == "hint_vote":
            sub_questions = self._hint_vote(question, hint)
        elif self.arm.strategy == "verify_2stage":
            sub_questions = self._verify_2stage(question, hint)
        else:
            sub_questions = self._llm(question, hint)
        latency_ms = (time.monotonic() - started) * 1000

        categories = derive_categories(sub_questions)
        calls = self.recorder.calls
        return {
            "id": question["id"],
            "set": question["set"],
            "gold": question["domain"],
            "question": question["question"],
            "sub_questions": sub_questions,
            "categories": categories,
            "effective_filter": effective_filter(categories),
            "verdict": verdict_for(categories, question["domain"]),
            "applied_filters": applied_filters(sub_questions),
            "recall_any": question["domain"] in categories,
            "parse_failed": self.capture.saw_fallback,
            "n_parse_failures": self.capture.fallback_count,
            "finish_reason": self._finish_reason(calls),
            "n_calls": len(calls),
            "latency_ms": round(latency_ms),
            "cost_usd": round(sum(c["cost"] for c in calls), 6),
            "raw_reply": calls[0]["reply"] if calls else None,
        }

    @staticmethod
    def _finish_reason(calls: list[dict]) -> str | None:
        """Truncation anywhere in a multi-call arm is truncation for the arm."""
        reasons = [c["finish_reason"] for c in calls]
        if "length" in reasons:
            return "length"
        return reasons[0] if reasons else None

    # -- strategies ------------------------------------------------------ #

    def _hint_vote(self, question: dict, hint: dict | None) -> list[dict]:
        """No LLM: the index's own top category, when it is confident enough."""
        categories: list[str] = []
        if hint and hint.get("top_category") and \
                hint.get("top_share", 0.0) >= self.arm.hint_vote_threshold:
            categories = [hint["top_category"]]
        return [{"question": question["question"], "categories": categories}]

    def _llm(self, question: dict, hint: dict | None) -> list[dict]:
        rendered = render_hint(hint) if self.arm.use_hint else None
        samples = [self.classifier.classify(question["question"], hint=rendered)
                   for _ in range(self.arm.samples)]
        sub_questions = _sub_questions_of(samples[0])
        if self.arm.samples == 1:
            return sub_questions
        # Sub-questions come from sample 1; a category survives only with
        # enough votes across samples. A voted-in category that sample 1 never
        # proposed has no sub-question to attach to and is dropped.
        keep = vote_categories([list(dict.fromkeys(c for sq in _sub_questions_of(s)
                                                   for c in sq["categories"]))
                                for s in samples], self.arm.min_votes)
        return _keep_only(sub_questions, keep)

    def _verify_2stage(self, question: dict, hint: dict | None) -> list[dict]:
        classification = self.classifier.classify(question["question"])
        sub_questions = _sub_questions_of(classification)
        proposed = derive_categories(sub_questions)
        if not proposed:
            return sub_questions
        messages = [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": build_verify_user_message(
                question["question"], proposed, render_hint(hint))},
        ]
        try:
            text, _usage, _cost = self.recorder(
                messages,
                model=self.classifier.model,
                max_tokens=256,
                temperature=0.0,
                quiet=True,
                **self.classifier.extra_params,
            )
            keep = parse_verify_reply(text, proposed)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "verify stage failed (%s: %s) — keeping the first-stage tags",
                type(exc).__name__, exc)
            return sub_questions
        return _keep_only(sub_questions, set(keep))


def run_arm(arm: Arm, questions: list[dict], config, hints: dict,
            recorder: ChatRecorder, capture: WarningCapture,
            concurrency: int = 6, progress=None) -> tuple[list[dict], bool]:
    """Run one arm over all questions. Returns (records, aborted).

    Aborted means the cost ceiling was reached: whatever completed is kept and
    written, and the caller stops the sweep."""
    runner = ArmRunner(arm, config, hints, recorder, capture)
    by_id: dict[str, dict] = {}
    aborted = False
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(runner.predict, q): q for q in questions}
        for done, future in enumerate(futures, start=1):
            question = futures[future]
            if aborted:
                # Already over budget: cancel what has not started and never
                # wait on what has — the in-flight calls fail fast on check().
                future.cancel()
                continue
            try:
                by_id[question["id"]] = future.result()
            except CostExceeded as exc:
                print(f"  cost ceiling reached ({exc}) — stopping", file=sys.stderr)
                aborted = True
                continue
            if progress:
                progress(done, len(questions), question["id"])
    # Emit in input order, not completion order, so reruns diff cleanly.
    return [by_id[q["id"]] for q in questions if q["id"] in by_id], aborted


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _rate(values) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def summarize(records: list[dict]) -> dict:
    """Metrics for one (arm, set) bucket."""
    if not records:
        return {"n": 0}
    verdicts = [r["verdict"] for r in records]
    latencies = [r["latency_ms"] for r in records]
    sub_counts = [len(r["sub_questions"]) for r in records]
    tags_per_sub = [len(sq["categories"]) for r in records for sq in r["sub_questions"]]
    # Diagnostic: over the sub-questions that really do filter, how many filter
    # to the wrong slice (see the module docstring on 2+-tag questions).
    sub_filters = [(f, r["gold"]) for r in records for f in r["applied_filters"]]
    return {
        "n": len(records),
        "harmful_rate": _rate([v == "harmful" for v in verdicts]),
        "filter_correct_rate": _rate([v == "useful" for v in verdicts]),
        "no_filter_rate": _rate([v == "none" for v in verdicts]),
        "recall_any": _rate([r["recall_any"] for r in records]),
        "mean_tags_per_sub": _rate(tags_per_sub),
        "mean_sub_questions": _rate(sub_counts),
        # A multi-sample arm gets one chance to fall back per sample, so read
        # its parse_fail_rate against calls_per_question, not against 1.
        "parse_fail_rate": _rate([r["parse_failed"] for r in records]),
        "calls_per_question": _rate([r["n_calls"] for r in records]),
        "truncation_rate": _rate([r["finish_reason"] == "length" for r in records]),
        "p50_ms": round(statistics.median(latencies)),
        "mean_cost_usd": round(statistics.mean(r["cost_usd"] for r in records), 6),
        "total_cost_usd": round(sum(r["cost_usd"] for r in records), 4),
        "sub_filter_count": len(sub_filters),
        "sub_harmful_rate": _rate([f != gold for f, gold in sub_filters]),
    }


def confusion(records: list[dict]) -> dict:
    """gold domain x predicted filter (``(none)`` for the no-filter outcome)."""
    table: dict[str, dict[str, int]] = {}
    for record in records:
        row = table.setdefault(record["gold"], {})
        predicted = record["effective_filter"] or NO_FILTER
        row[predicted] = row.get(predicted, 0) + 1
    return {gold: dict(sorted(row.items(), key=lambda kv: (-kv[1], kv[0])))
            for gold, row in sorted(table.items())}


def mcnemar_exact(n_fixed: int, n_broken: int) -> float:
    """Two-sided exact McNemar p over the two discordant counts.

    Binomial(n_fixed + n_broken, 0.5); n is small enough here that the
    chi-square approximation is not trustworthy."""
    n = n_fixed + n_broken
    if n == 0:
        return 1.0
    smaller = min(n_fixed, n_broken)
    tail = sum(math.comb(n, i) for i in range(smaller + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 6)


def paired(baseline_records: list[dict], arm_records: list[dict]) -> dict:
    """Fixed/broken counts and McNemar p for an arm against baseline, over the
    question ids both actually produced."""
    base = {r["id"]: r["verdict"] == "harmful" for r in baseline_records}
    arm = {r["id"]: r["verdict"] == "harmful" for r in arm_records}
    shared = sorted(set(base) & set(arm))
    n_fixed = sum(1 for qid in shared if base[qid] and not arm[qid])
    n_broken = sum(1 for qid in shared if not base[qid] and arm[qid])
    return {
        "n_pairs": len(shared),
        "n_fixed": n_fixed,
        "n_broken": n_broken,
        "mcnemar_p": mcnemar_exact(n_fixed, n_broken),
        "fixed_ids": [qid for qid in shared if base[qid] and not arm[qid]],
        "broken_ids": [qid for qid in shared if not base[qid] and arm[qid]],
    }


def _by_set(records: list[dict]) -> dict:
    sets = sorted({r["set"] for r in records})
    buckets = {name: [r for r in records if r["set"] == name] for name in sets}
    summary = {name: summarize(rows) for name, rows in buckets.items()}
    summary["pooled"] = summarize(records)
    return summary


def aggregate(results: dict[str, list[dict]], arms: dict[str, Arm]) -> dict:
    """Per-arm metrics, confusion tables and paired stats vs baseline."""
    baseline_records = results.get(BASELINE_ARM, [])
    out = {}
    for arm_id, records in results.items():
        entry = {
            "what": arms[arm_id].what,
            "arm": asdict(arms[arm_id]),
            "sets": _by_set(records),
            "confusion": confusion(records),
        }
        if baseline_records and arm_id != BASELINE_ARM:
            entry["paired_vs_baseline"] = {
                "pooled": paired(baseline_records, records),
                **{name: paired([r for r in baseline_records if r["set"] == name],
                                [r for r in records if r["set"] == name])
                   for name in sorted({r["set"] for r in records})},
            }
        out[arm_id] = entry
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

_REPORT_COLS = [
    ("harmful_rate", "Harmful"),
    ("filter_correct_rate", "Correct"),
    ("no_filter_rate", "No filter"),
    ("recall_any", "Recall any"),
    ("mean_tags_per_sub", "Tags/sub"),
    ("mean_sub_questions", "Subs/q"),
    ("parse_fail_rate", "Parse fail"),
    ("truncation_rate", "Trunc."),
    ("p50_ms", "p50 ms"),
    ("mean_cost_usd", "$/q"),
]


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if value < 0.01 else f"{value:.3f}"
    return str(value)


def _harmful(entry: dict, set_name: str = "pooled") -> float:
    """Sort key: an arm with no usable rows sorts last, not first."""
    value = entry["sets"].get(set_name, {}).get("harmful_rate")
    return 1.0 if value is None else value


def _metric_table(agg: dict, set_name: str) -> str:
    header = "| Arm | n | " + " | ".join(name for _, name in _REPORT_COLS) + " |"
    sep = "|" + "---|" * (len(_REPORT_COLS) + 2)
    rows = []
    for arm_id, entry in sorted(agg.items(), key=lambda kv: _harmful(kv[1], set_name)):
        bucket = entry["sets"].get(set_name)
        if not bucket or not bucket.get("n"):
            continue
        rows.append(f"| `{arm_id}` | {bucket['n']} | "
                    + " | ".join(_fmt(bucket.get(f)) for f, _ in _REPORT_COLS) + " |")
    return "\n".join([header, sep] + rows)


def _paired_table(agg: dict) -> str:
    header = "| Arm | pooled harmful | fixed | broken | McNemar p | verdict |"
    sep = "|---|---|---|---|---|---|"
    rows = []
    base = _harmful(agg[BASELINE_ARM])
    for arm_id, entry in sorted(agg.items(), key=lambda kv: _harmful(kv[1])):
        if arm_id == BASELINE_ARM or "paired_vs_baseline" not in entry:
            continue
        stats = entry["paired_vs_baseline"]["pooled"]
        harmful = entry["sets"]["pooled"].get("harmful_rate")
        if harmful is None:
            continue
        if stats["mcnemar_p"] > 0.05:
            call = "tied with baseline"
        elif harmful < base:
            call = "**better**"
        else:
            call = "worse"
        rows.append(f"| `{arm_id}` | {_fmt(harmful)} | {stats['n_fixed']} | "
                    f"{stats['n_broken']} | {stats['mcnemar_p']:.4f} | {call} |")
    return "\n".join([header, sep] + rows)


def _confusion_table(table: dict) -> str:
    predicted = sorted({p for row in table.values() for p in row},
                       key=lambda p: (p == NO_FILTER, p))
    header = "| gold \\ predicted | " + " | ".join(predicted) + " |"
    sep = "|" + "---|" * (len(predicted) + 1)
    rows = []
    for gold, row in table.items():
        cells = []
        for p in predicted:
            count = row.get(p, 0)
            cells.append(f"**{count}**" if count and p == gold else str(count or ""))
        rows.append(f"| {gold} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _worst_pairs(table: dict, limit: int = 8) -> list[str]:
    pairs = [(count, gold, predicted)
             for gold, row in table.items()
             for predicted, count in row.items()
             if predicted not in (gold, NO_FILTER)]
    pairs.sort(reverse=True)
    return [f"`{gold}` tagged `{predicted}` — {count}x" for count, gold, predicted
            in pairs[:limit]]


def render_report(agg: dict, meta: dict) -> str:
    sets = [s for s in sorted({s for e in agg.values() for s in e["sets"]})
            if s != "pooled"]
    best_id = min(agg, key=lambda arm_id: _harmful(agg[arm_id]))
    lines = [
        f"# Classifier arm sweep — {meta['run_name']}",
        "",
        f"- **Date:** {meta['date']}",
        f"- **Question sets:** {', '.join(meta['questions_files'])} "
        f"({meta['n_questions']} questions)",
        f"- **Arms:** {', '.join(f'`{a}`' for a in agg)}",
        f"- **Total cost:** ~${meta['total_cost_usd']:.2f}"
        + (" (**aborted on the cost ceiling**)" if meta.get("aborted") else ""),
        "",
        "The agent filters retrieval for a sub-question only when it carries "
        "exactly one category, so `harmful` = filtered to the wrong domain, "
        "`correct` = filtered to the gold domain, `no filter` = zero or 2+ tags. "
        "**Harmful is the primary metric** (lower is better); a wrong filter "
        "costs a retrieval round, a missing one only costs precision.",
        "",
        f"## Pooled ({' + '.join(sets)}), ranked by harmful_rate",
        "",
        _metric_table(agg, "pooled"),
        "",
    ]
    for set_name in sets:
        lines += [f"## {set_name}", "", _metric_table(agg, set_name), ""]

    if len(agg) > 1 and BASELINE_ARM in agg:
        lines += [
            "## Paired against baseline (same question ids)",
            "",
            "`fixed` = baseline was harmful and this arm is not; `broken` = the "
            "reverse. p is an exact two-sided McNemar over those two counts — at "
            "n=169 an arm that is not significant here has not been shown to "
            "differ from baseline at all.",
            "",
            _paired_table(agg),
            "",
        ]

    for arm_id in dict.fromkeys([BASELINE_ARM, best_id]):
        if arm_id not in agg:
            continue
        label = "baseline" if arm_id == BASELINE_ARM else f"best arm (`{arm_id}`)"
        lines += [
            f"## Confusion — {label}",
            "",
            _confusion_table(agg[arm_id]["confusion"]),
            "",
            "Most frequent wrong filters: "
            + ("; ".join(_worst_pairs(agg[arm_id]["confusion"])) or "none") + ".",
            "",
        ]

    lines += [
        "## Arms",
        "",
        *[f"- `{arm_id}` — {entry['what']}" for arm_id, entry in agg.items()],
        "",
        "---",
        "*Generated by `evalharness.classify_eval`; per-question records are in "
        "`<arm>/predictions.jsonl` and every metric in `summary.json`.*",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _set_name(path: str) -> str:
    """v1 / v2 from the questions filename — the label every metric is sliced
    by, so it has to be stable across reruns."""
    stem = Path(path).stem
    return "v2" if stem.endswith("_v2") or stem.endswith("-v2") else "v1"


def load_questions(paths: list[str], limit: int | None = None) -> list[dict]:
    questions = []
    for path in paths:
        name = _set_name(path)
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if limit:
            loaded = loaded[:limit]
        for q in loaded:
            questions.append({**q, "set": name})
    ids = Counter(q["id"] for q in questions)
    duplicates = [qid for qid, n in ids.items() if n > 1]
    if duplicates:
        raise SystemExit(f"question ids collide across sets: {duplicates}")
    return questions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--questions", nargs="+",
                    default=["reference_questions.json", "reference_questions_v2.json"])
    ap.add_argument("--arms", nargs="+", help="arm ids to run")
    ap.add_argument("--all", action="store_true", help="run every defined arm")
    ap.add_argument("--out", required=True, help="output directory for this sweep")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="parallel classify calls (shared API key — keep <=6)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N questions of each set (smoke tests)")
    ap.add_argument("--hints", default=None, help="existing hints.json to reuse")
    ap.add_argument("--build-hints", action="store_true",
                    help="run the retrieval pass and write hints.json first")
    ap.add_argument("--hint-workers", type=int, default=4,
                    help="parallel retrievals in the hint pass (qdrant is a "
                         "single-process lock — keep <=4)")
    ap.add_argument("--max-cost", type=float, default=5.0,
                    help="abort the sweep once the running cost estimate hits this")
    args = ap.parse_args(argv)

    if not args.arms and not args.all and not args.build_hints:
        ap.error("pass --arms, --all, or --build-hints")
    arm_ids = [a.id for a in ARMS] if args.all else (args.arms or [])
    arms = {arm_id: get_arm(arm_id) for arm_id in arm_ids}
    if arms and BASELINE_ARM not in arms:
        print(f"WARNING: '{BASELINE_ARM}' is not in this run — no paired stats "
              "will be computed", file=sys.stderr)

    config = load_config(args.config)
    questions = load_questions(args.questions, args.limit)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(questions)} questions, {len(arms)} arm(s)", file=sys.stderr)

    def progress(done, total, qid):
        if done % 10 == 0 or done == total:
            print(f"    [{done}/{total}] {qid}", file=sys.stderr)

    hints: dict = {}
    if args.hints:
        hints = json.loads(Path(args.hints).read_text(encoding="utf-8"))
    if args.build_hints:
        from rag.retrieve.retriever import load_retriever

        print(f"Building hints for {len(questions)} questions "
              f"(dense+sparse, no rerank)...", file=sys.stderr)
        started = time.monotonic()
        retriever = load_retriever(config)
        try:
            hints = build_hints(retriever, questions, args.hint_workers, progress)
        finally:
            retriever.close()
        (out_dir / "hints.json").write_text(
            json.dumps(hints, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  hints built in {time.monotonic() - started:.0f}s -> "
              f"{out_dir / 'hints.json'}", file=sys.stderr)

    missing_hints = [arm_id for arm_id, arm in arms.items() if arm.needs_hints and not hints]
    if missing_hints:
        raise SystemExit(f"arms {missing_hints} need hints — pass --build-hints or "
                         "--hints path/to/hints.json")
    if not arms:
        return 0

    budget = Budget(args.max_cost)
    recorder = ChatRecorder(budget)
    capture = WarningCapture()
    classify_logger = logging.getLogger("rag.classify")
    original_chat = classify_mod.tf_chat
    classify_mod.tf_chat = recorder
    classify_logger.addHandler(capture)
    # parse_fail_rate is read off those warnings, so they must not be filtered
    # out by whatever level the ambient logging config left behind.
    if not classify_logger.isEnabledFor(logging.WARNING):
        classify_logger.setLevel(logging.WARNING)
    results: dict[str, list[dict]] = {}
    aborted = False
    try:
        # Baseline first: every other arm is scored paired against it, so a
        # cost abort should never be the reason it is missing.
        for arm_id in sorted(arms, key=lambda a: a != BASELINE_ARM):
            print(f"\nArm '{arm_id}' — {arms[arm_id].what}", file=sys.stderr)
            records, arm_aborted = run_arm(arms[arm_id], questions, config, hints,
                                           recorder, capture, args.concurrency,
                                           progress)
            results[arm_id] = records
            arm_dir = out_dir / arm_id
            arm_dir.mkdir(parents=True, exist_ok=True)
            with open(arm_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            pooled = summarize(records)
            print(f"  harmful={_fmt(pooled.get('harmful_rate'))} "
                  f"correct={_fmt(pooled.get('filter_correct_rate'))} "
                  f"(${budget.total:.2f} spent)", file=sys.stderr)
            if arm_aborted:
                aborted = True
                break
    finally:
        classify_mod.tf_chat = original_chat
        classify_logger.removeHandler(capture)

    agg = aggregate(results, arms)
    meta = {
        "run_name": out_dir.name,
        "date": datetime.date.today().isoformat(),
        "config": args.config,
        "questions_files": args.questions,
        "n_questions": len(questions),
        "arms": list(results),
        "concurrency": args.concurrency,
        "total_cost_usd": round(budget.total, 4),
        "max_cost_usd": args.max_cost,
        "aborted": aborted,
    }
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "arms": agg}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(agg, meta), encoding="utf-8")
    print(f"\nReport: {out_dir / 'report.md'} (${budget.total:.2f})", file=sys.stderr)
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())

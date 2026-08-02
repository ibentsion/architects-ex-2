"""CLI: re-check an emitted dataset against the spec.

    python -m refgen.audit reference_questions_v2.json            # structure only
    python -m refgen.audit reference_questions_v2.json --gates    # + re-run the LLM gates

Structural auditing is free and runs over every item: schema, per-cell counts,
distinct pages, resolvable sources, no leakage from the held-out v1 set. The
`--gates` pass re-runs the LLM acceptance gates on a sample, with a fresh model
assignment — a dataset whose items only pass when re-checked by the same model
that admitted them is not verified, it is agreed with.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from evalharness.pages import PageStore
from rag.parsing import KNOWN_CATEGORIES

from . import generate, schema, verify
from .inventory import build_inventory


class MalformedDataset(Exception):
    """One or more items violate the item schema, listed per item."""


def load(path) -> list[schema.RefQuestion]:
    """Parse a dataset file into validated items.

    Reports every malformed item at once rather than dying on the first —
    auditing a freshly generated file should tell you everything wrong with it.
    """
    from pydantic import ValidationError

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items, problems = [], []
    for index, entry in enumerate(raw):
        try:
            items.append(schema.RefQuestion(**entry))
        except ValidationError as error:
            item_id = entry.get("id", f"#{index}")
            reasons = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                                for e in error.errors())
            problems.append(f"{item_id}: {reasons}")
    if problems:
        raise MalformedDataset("\n  - ".join([f"{len(problems)} malformed item(s):"] + problems))
    return items


def _pages_of(item: schema.RefQuestion, store: PageStore) -> list:
    """The item's cited pages, as inventory Pages the gates can read."""
    from .inventory import Page

    pages = []
    for file, page in item.pages():
        text, reason = store.resolve(file, page)
        if text is None:
            raise ValueError(f"{item.id}: {file} p{page} unresolvable ({reason})")
        pages.append(Page(file, page, text))
    return pages


def audit_gates(items: list, store: PageStore, sample: int, seed: int) -> list[str]:
    """Re-run the LLM gates on a sample of items. Returns failures."""
    rng = random.Random(seed)
    chosen = rng.sample(items, min(sample, len(items)))
    failures = []
    inventories: dict[str, list] = {}
    for item in chosen:
        # A fresh verifier assignment: rotate past whoever checked it the first
        # time, so re-auditing is a second opinion rather than an echo.
        generator = item.provenance.generator_model if item.provenance else ""
        verifiers = [m for m in generate.GENERATOR_MODELS if m != generator]
        rng.shuffle(verifiers)
        candidate = {"question": item.question,
                     "ground_truth_answer": item.ground_truth_answer}
        if item.domain not in inventories:
            inventories[item.domain] = build_inventory(item.domain, store)
        try:
            verify.run_gates(item.kind, candidate, item.difficulty,
                             _pages_of(item, store), item.domain,
                             inventories[item.domain], verifiers)
            print(f"  ok   {item.id} ({item.kind}/{item.difficulty})", file=sys.stderr)
        except verify.Rejected as rejection:
            failures.append(f"{item.id}: {rejection.gate} — {rejection.reason}")
            print(f"  FAIL {item.id}: {rejection.gate}", file=sys.stderr)
        except ValueError as error:
            failures.append(str(error))
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--holdout", nargs="+", default=["reference_questions.json"],
                    help="held-out dataset file(s) whose pages and questions this "
                         "one must not reuse (v3 holds out both v1 and v2)")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--gates", action="store_true", help="re-run the LLM gates on a sample")
    ap.add_argument("--sample", type=int, default=12, help="items to re-gate (default 12)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sources", action="store_true",
                    help="skip resolving cited pages (no corpus needed)")
    args = ap.parse_args(argv)

    try:
        items = load(args.dataset)
    except MalformedDataset as error:
        print(f"{args.dataset}: {error}", file=sys.stderr)
        return 1
    store = None if args.no_sources else PageStore(args.corpus, args.cache_dir)
    present = [p for p in args.holdout if Path(p).is_file()]
    held_pages, held_questions = schema.load_exclusions(present) if present else ((), ())

    # v3 is a multi-source top-up, not a full dataset: the per-cell counts and
    # every-category shape checks are v2's, and would fail it by design.
    problems = schema.check_dataset(items, held_pages, held_questions, store,
                                    strict_counts=schema.profile_of(items) == "v2")

    print(f"{args.dataset}: {len(items)} items, "
          f"{len({i.domain for i in items})}/{len(KNOWN_CATEGORIES)} categories",
          file=sys.stderr)
    print(f"  by kind: {dict(Counter(i.kind for i in items))}", file=sys.stderr)
    print(f"  by difficulty: {dict(Counter(i.difficulty for i in items))}", file=sys.stderr)
    print(f"  by generator: {dict(Counter(i.provenance.generator_model for i in items if i.provenance))}",
          file=sys.stderr)

    print("\ncoverage (distinct files / pages per category):", file=sys.stderr)
    for category, stats in schema.coverage(items).items():
        overused = f"  overused: {', '.join(stats['overused_files'])}" if stats["overused_files"] else ""
        print(f"  {category:24} {stats['items']:3} items  "
              f"{stats['distinct_files']:3} files  {stats['distinct_pages']:3} pages{overused}",
              file=sys.stderr)

    if args.gates:
        print(f"\nre-running LLM gates on {min(args.sample, len(items))} items:", file=sys.stderr)
        problems += audit_gates(items, store, args.sample, args.seed)

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\nOK — every checked invariant holds.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

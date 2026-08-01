"""CLI: build the v2 reference dataset.

    python -m refgen.run --out reference_questions_v2.json

Writes the dataset, a run report (what was rejected and why, per gate), and a
coverage summary. Categories are independent, so they run in parallel; within
a category items are sequential because each one's pages depend on what the
previous ones took.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from evalharness.pages import PageStore
from rag.parsing import KNOWN_CATEGORIES

from . import generate, schema
from .inventory import Sampler, build_inventory

#: v1 items shown to a generator as form anchors, drawn from other categories.
N_EXAMPLES = 2


def _examples_for(v1_items: list, difficulty: str, category: str, rng: random.Random) -> list:
    """Form anchors: same difficulty, deliberately different category."""
    pool = [q for q in v1_items
            if q["difficulty"] == difficulty and q["domain"] != category]
    return rng.sample(pool, min(N_EXAMPLES, len(pool)))


def build_category(category: str, index: int, store: PageStore, v1_items: list,
                   v1_pages: set, existing: list, seed: int, progress,
                   checkpoint=None) -> tuple:
    """Every item for one category. Returns (items, attempts, failures)."""
    rng = random.Random(seed + index)
    pages = build_inventory(category, store)
    sampler = Sampler(pages, seed=seed + index, excluded=set(v1_pages))

    slots = generate.category_plan()
    slots.append(("unanswerable", generate.unanswerable_difficulty(index)))

    items, attempts, failures = [], [], []
    cell_used: dict[str, set] = {}
    for slot_index, (kind, difficulty) in enumerate(slots):
        model = generate.GENERATOR_MODELS[(index + slot_index) % len(generate.GENERATOR_MODELS)]
        item_id = f"v2-{index * len(slots) + slot_index + 1:03d}-{category}-{difficulty}"
        # Standard items are placed by the difficulty they earn, so aim at
        # whatever this category still has room for — with a preference for the
        # slot's own difficulty, to keep the mix spread across the corpus.
        wanted = None
        if kind == "standard":
            filled = Counter(i.difficulty for i in items if i.kind == "standard")
            wanted = {d for d in schema.DIFFICULTIES
                      if filled[d] < schema.STANDARD_PER_CELL}
            if not wanted:
                continue
            if difficulty not in wanted:
                difficulty = sorted(wanted)[0]
        used = cell_used.setdefault(difficulty, set())
        outcome = generate.build_item(
            kind, difficulty, category, sampler, pages,
            _examples_for(v1_items, difficulty, category, rng), model, item_id,
            used, existing + [i.question for i in items], wanted)
        attempts += outcome.attempts
        if outcome.item is None:
            failures.append(f"{item_id} ({kind}/{difficulty}): {outcome.failure}")
        else:
            items.append(outcome.item)
            # File the pages under the difficulty the item actually earned.
            cell_used.setdefault(outcome.item.difficulty, set()).update(outcome.item.pages())
            if checkpoint:
                checkpoint(outcome.item)
        progress(category, len(items), len(slots))
    return items, attempts, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reference_questions_v2.json")
    ap.add_argument("--v1", default="reference_questions.json",
                    help="held-out set: its pages and questions are excluded")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--categories", nargs="+", default=sorted(KNOWN_CATEGORIES))
    ap.add_argument("--workers", type=int, default=4, help="categories in parallel")
    ap.add_argument("--seed", type=int, default=20260731)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    v1_items = json.loads(Path(args.v1).read_text(encoding="utf-8"))
    v1_pages, v1_questions = schema.load_v1_exclusions(args.v1)
    store = PageStore(args.corpus, args.cache_dir)
    store._load_sources()  # warm the corpus walk once, before the threads start

    print(f"Building {len(args.categories)} categories x "
          f"{schema.ITEMS_PER_CATEGORY} items "
          f"({len(args.categories) * schema.ITEMS_PER_CATEGORY} questions); "
          f"excluding {len(v1_pages)} held-out v1 pages", file=sys.stderr)

    def progress(category, done, total):
        print(f"  [{category}] {done}/{total}", file=sys.stderr)

    # Every accepted item is appended to a JSONL checkpoint as it is produced.
    # Generation is slow and paid for; a run that is killed at minute 29 must
    # not throw away what it already verified.
    checkpoint_path = Path(args.out).with_suffix(".partial.jsonl")
    checkpoint_lock = Lock()

    def checkpoint(item):
        with checkpoint_lock, open(checkpoint_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_category, category, index, store, v1_items,
                               v1_pages, v1_questions, args.seed, progress, checkpoint)
                   for index, category in enumerate(args.categories)]
        results = [f.result() for f in futures]

    items = [item for r in results for item in r[0]]
    attempts = [a for r in results for a in r[1]]
    failures = [f for r in results for f in r[2]]

    problems = schema.check_dataset(items, v1_pages, v1_questions, store,
                                    strict_counts=len(args.categories) == len(KNOWN_CATEGORIES))
    out_path = Path(args.out)
    out_path.write_text(
        json.dumps([item.model_dump() for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8")

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": args.seed,
        "n_items": len(items),
        "by_kind": dict(Counter(item.kind for item in items)),
        "by_generator": dict(Counter(item.provenance.generator_model for item in items)),
        "by_difficulty": dict(Counter(item.difficulty for item in items)),
        "rejections_by_gate": dict(Counter(a.gate for a in attempts)),
        "rejections": [vars(a) for a in attempts],
        "failures": failures,
        "schema_problems": problems,
        "coverage": schema.coverage(items),
    }
    report_path = out_path.with_name(out_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {len(items)} items to {out_path} (report: {report_path})", file=sys.stderr)
    print(f"  by kind: {report['by_kind']}", file=sys.stderr)
    print(f"  by generator: {report['by_generator']}", file=sys.stderr)
    print(f"  rejections: {report['rejections_by_gate']}", file=sys.stderr)
    for failure in failures:
        print(f"  FAILED {failure}", file=sys.stderr)
    for problem in problems:
        print(f"  SCHEMA {problem}", file=sys.stderr)
    return 1 if (failures or problems) else 0


if __name__ == "__main__":
    sys.exit(main())

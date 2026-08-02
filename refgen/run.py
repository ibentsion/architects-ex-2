"""CLI: build a reference dataset.

    python -m refgen.run --profile v2 --out reference_questions_v2.json
    python -m refgen.run --profile v3 --holdout reference_questions.json \
                         reference_questions_v2.json

Two profiles. `v2` is the full spec: 9 standard + 1 multi-source + 1
unanswerable item per category. `v3` is a targeted top-up — 3 multi-source
items per category, written from the calculation-biased prompt, because that
is the shape the harness eval separated systems on and the one v1+v2 are
thinnest in.

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


def build_category(category: str, index: int, pages: list, v1_items: list,
                   v1_pages: set, existing: list, seed: int, progress,
                   checkpoint=None, have: list | None = None,
                   id_from: int = 1, profile: str = "v2") -> tuple:
    """Every item one category still needs. Returns (items, attempts, failures).

    `pages` is prepared by the caller, in the main thread: building it touches
    pdfium, which is not thread-safe even under a lock (its finalizers run on
    whatever thread collects them).

    `have` is what a previous run already produced for this category; only the
    shortfall is generated, and its pages and questions are held back so a fill
    run cannot duplicate what it is topping up.
    """
    have = have or []
    rng = random.Random(seed + index)
    already_used = {page for item in have for page in item.pages()}
    sampler = Sampler(pages, seed=seed + index,
                      excluded=set(v1_pages) | already_used)

    slots = generate.plan_for(profile, have, index)
    variant = generate.PROFILE_VARIANT[profile]

    items, attempts, failures = [], [], []
    cell_used: dict[str, set] = {}
    for item in have:  # pages of existing items are spoken for
        cell_used.setdefault(item.difficulty, set()).update(item.pages())

    for slot_index, (kind, difficulty) in enumerate(slots):
        model = generate.GENERATOR_MODELS[(index + slot_index) % len(generate.GENERATOR_MODELS)]
        item_id = f"{profile}-{id_from + slot_index:03d}-{category}-{difficulty}"
        # Standard items are placed by the difficulty they earn, so aim at
        # whatever this category still has room for — with a preference for the
        # slot's own difficulty, to keep the mix spread across the corpus.
        wanted = None
        if kind == "standard":
            filled = Counter(i.difficulty for i in have + items if i.kind == "standard")
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
            used, existing + [i.question for i in have + items], wanted, variant)
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
    ap.add_argument("--profile", choices=generate.PROFILES, default="v2",
                    help="v2: the full per-category spec. v3: 3 multi-source "
                         "items per category from the calculation-biased prompt")
    ap.add_argument("--out", default=None,
                    help="default: reference_questions_<profile>.json")
    ap.add_argument("--fill", action="store_true",
                    help="top up an existing --out file: generate only the slots "
                         "it is short of, holding back the pages and questions "
                         "it already uses, then merge")
    ap.add_argument("--holdout", nargs="+", default=["reference_questions.json"],
                    help="held-out set(s): their pages and questions are excluded "
                         "(v3 holds out both v1 and v2)")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--categories", nargs="+", default=sorted(KNOWN_CATEGORIES))
    ap.add_argument("--workers", type=int, default=4, help="categories in parallel")
    ap.add_argument("--seed", type=int, default=20260731)
    args = ap.parse_args(argv)
    out_path = Path(args.out or f"reference_questions_{args.profile}.json")
    per_category = len(generate.plan_for(args.profile))

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    # Every held-out item is also a form anchor: examples are drawn from a
    # different category than the one being written, so they carry voice only.
    held_items = [item for path in args.holdout
                  for item in json.loads(Path(path).read_text(encoding="utf-8"))]
    held_pages, held_questions = schema.load_exclusions(args.holdout)

    existing_items: list = []
    if args.fill:
        from .audit import load

        existing_items = load(out_path)
        short = {c: len(generate.plan_for(
                    args.profile, [i for i in existing_items if i.domain == c], n))
                 for n, c in enumerate(args.categories)}
        print(f"Filling {out_path.name}: {len(existing_items)} items present, "
              f"{sum(short.values())} slots short "
              f"({', '.join(f'{c}:{n}' for c, n in short.items() if n)})",
              file=sys.stderr)
    have_by_category: dict[str, list] = {}
    for item in existing_items:
        have_by_category.setdefault(item.domain, []).append(item)
    # New ids continue past the highest already used, so a fill never collides.
    next_id = 1 + max((int(i.id.split("-")[1]) for i in existing_items), default=0)
    store = PageStore(args.corpus, args.cache_dir)
    store._load_sources()  # warm the corpus walk once, before the threads start

    # Inventories are built here, serially, because building one calls pdfium
    # to check page word order and pdfium is not thread-safe.
    print(f"Reading corpus pages for {len(args.categories)} categories...", file=sys.stderr)
    inventories = {}
    for category in args.categories:
        inventories[category] = build_inventory(category, store)
        print(f"  {category:26}{len(inventories[category]):5} usable pages", file=sys.stderr)

    print(f"Building {args.profile}: {len(args.categories)} categories x "
          f"{per_category} items "
          f"({len(args.categories) * per_category} questions); "
          f"excluding {len(held_pages)} held-out pages from "
          f"{', '.join(Path(p).name for p in args.holdout)}", file=sys.stderr)

    def progress(category, done, total):
        print(f"  [{category}] {done}/{total}", file=sys.stderr)

    # Every accepted item is appended to a JSONL checkpoint as it is produced.
    # Generation is slow and paid for; a run that is killed at minute 29 must
    # not throw away what it already verified.
    checkpoint_path = out_path.with_suffix(".partial.jsonl")
    checkpoint_lock = Lock()

    def checkpoint(item):
        with checkpoint_lock, open(checkpoint_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_category, category, index, inventories[category],
                               held_items, held_pages, held_questions, args.seed, progress,
                               checkpoint, have_by_category.get(category),
                               next_id + index * per_category, args.profile)
                   for index, category in enumerate(args.categories)]
        results = [f.result() for f in futures]

    items = existing_items + [item for r in results for item in r[0]]
    attempts = [a for r in results for a in r[1]]
    failures = [f for r in results for f in r[2]]

    # v3 has no per-cell shape to check: it is a multi-source top-up, not a
    # dataset that must cover every category x difficulty cell.
    problems = schema.check_dataset(
        items, held_pages, held_questions, store,
        strict_counts=(args.profile == "v2"
                       and len(args.categories) == len(KNOWN_CATEGORIES)))
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

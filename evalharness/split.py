"""CLI: split the reference sets into a validation half and a holdout half.

    python -m evalharness.split

Retrieval knobs (top_k, the gate threshold, whether to filter by category) have
to be tuned against *something*, and tuning them against every question we have
leaves nothing to check the result on. This writes a seeded 50% sample as
`ref_q_validation_set_v1.jsonl` and its complement as
`ref_q_holdout_set_v1.jsonl`, so a policy chosen on the first can be confirmed
on the second.

The sample is stratified by (set, domain, difficulty, kind): the corpus's tail
categories carry 4-6 questions each, and an unstratified coin flip routinely
empties one. Strata are filled by largest remainder — a stratum of 3 at 50%
contributes 1 or 2 depending on the fractional debt carried from the strata
before it, so the totals come out exact instead of rounding 12 singleton
categories all the same way.

Unanswerable items stay in the split. They have no ground-truth page and every
retrieval analysis skips them, but a retrieval policy has to be judged on them
too: "retrieve more when the evidence looks thin" is precisely the rule that
turns a correct refusal into a fabricated answer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

#: The reference sets to split, as name=path. v3 is deliberately absent: it was
#: generated for a different purpose (multi-source/calculation stress) and is
#: small enough that halving it would leave neither half usable.
DEFAULT_SETS = {"v1": "reference_questions.json", "v2": "reference_questions_v2.json"}


def load_reference(path: str | Path) -> list[dict]:
    """Read a reference set from `.json` (one array) or `.jsonl` (one item per
    line). The split writes JSONL; every consumer of a reference set has to
    accept both or the split is unusable."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def stratum_of(item: dict) -> tuple[str, str, str, str]:
    return (item["_set"], item["domain"], item["difficulty"], item.get("kind", "standard"))


def split(items: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified split by largest remainder. Deterministic in `seed`: strata
    are visited in sorted order and each is shuffled with its own derived seed,
    so adding a question to one stratum cannot reshuffle another."""
    strata: dict[tuple, list[dict]] = {}
    for item in items:
        strata.setdefault(stratum_of(item), []).append(item)

    validation, holdout, debt = [], [], 0.0
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda i: i["id"])
        random.Random(f"{seed}:{key}").shuffle(group)
        exact = len(group) * fraction + debt
        take = int(exact)
        debt = exact - take
        validation += group[:take]
        holdout += group[take:]
    return (sorted(validation, key=lambda i: i["id"]),
            sorted(holdout, key=lambda i: i["id"]))


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8")


def describe(name: str, items: list[dict]) -> str:
    kinds = Counter(i.get("kind", "standard") for i in items)
    sets = Counter(i["_set"] for i in items)
    return (f"{name}: {len(items)} items "
            f"({', '.join(f'{k} {n}' for k, n in sorted(sets.items()))}; "
            f"{', '.join(f'{k} {n}' for k, n in sorted(kinds.items()))}; "
            f"{len({i['domain'] for i in items})} categories)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", metavar="NAME=PATH", default=None,
                    help="repeatable; default: " + ", ".join(DEFAULT_SETS))
    ap.add_argument("--out", default="ref_q_validation_set_v1.jsonl")
    ap.add_argument("--holdout-out", default="ref_q_holdout_set_v1.jsonl")
    ap.add_argument("--fraction", type=float, default=0.5,
                    help="share of each stratum that goes to the validation half")
    ap.add_argument("--seed", type=int, default=20260802)
    args = ap.parse_args(argv)

    sets = dict(s.split("=", 1) for s in args.set) if args.set else dict(DEFAULT_SETS)
    items: list[dict] = []
    for name, path in sets.items():
        loaded = load_reference(path)
        # The source set travels with the item: it is a stratum key, and every
        # downstream report slices v1 against v2.
        for item in loaded:
            items.append({**item, "_set": name})
        print(f"read {len(loaded)} items from {path} ({name})", file=sys.stderr)

    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate question ids across the input sets")

    validation, holdout = split(items, args.fraction, args.seed)
    write_jsonl(Path(args.out), validation)
    write_jsonl(Path(args.holdout_out), holdout)

    print(describe(f"\n{args.out}", validation), file=sys.stderr)
    print(describe(args.holdout_out, holdout), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

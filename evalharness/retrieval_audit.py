"""CLI: where does retrieval lose the ground-truth page?

    python -m evalharness.retrieval_audit --out eval_results/retrieval-audit

Answer quality tells you a system missed a fact; it does not tell you whether
the fact was never in the index, never retrieved, or retrieved and then thrown
away by the rerank gate. This walks every reference question through the real
retriever (fuse, then rerank+gate — no generation, so the whole matrix is
free) and files each ground-truth source group under the furthest stage it
reached:

  gated              a chunk from that page survived the rerank gate/top-n
  not_gated          it was a fused candidate, killed by the gate or top-n
  not_retrieved      it is in the index, but never entered the candidate pool
  missing_from_index no chunk exists for that page at all (corpus/parse gap)

`missing_from_index` is read from the sparse index's chunk_ids.json, so it is
a statement about what ingest actually produced, not about what the corpus
directory contains.

Pages are matched by path stem + page number, which is the identity the pdf
and markdown corpora share (the same document is `<stem>.pdf` under files/ and
`<stem>.md` under markdown-files/), so their arms are directly comparable.

Arms whose config or index is absent on this node are skipped with a reason
rather than failing the run.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from .pages import norm_file

#: Furthest stage a ground-truth source group reached, worst last.
STAGES = ("gated", "not_gated", "not_retrieved", "missing_from_index")

#: The comparison matrix: {pdf, markdown} x {per_table, per_page}. Everything
#: but the parse and the chunker is held constant (bge-m3 + stanza), so a
#: delta between two arms is a delta of those two axes.
DEFAULT_ARMS = {
    "pdf-per_table": "configs/final-per_table-bgem3-gptoss-low.yaml",
    "pdf-per_page": "configs/final-per_page-bgem3-gptoss-low.yaml",
    "markdown-per_table": "configs/markdown-per_table-bgem3.yaml",
    "markdown-per_page": "configs/markdown-per_page-bgem3.yaml",
}
DEFAULT_DATASETS = {
    "v1": "reference_questions.json",
    "v2": "reference_questions_v2.json",
}


def page_key(file: str, page: int | None) -> tuple[str, str, int | None]:
    """The identity a pdf arm and a markdown arm share: category, document
    stem (extension and sub-directory dropped — `dental/files/x.pdf` and
    `dental/markdown-files/x.md` are the same document), page. The category
    stays in the key because filenames repeat across categories (every
    category has a `pages/claim.txt`), and both sides of every comparison are
    category-relative paths (§8's chunk_id invariant on one side, the
    reference schema's own-category rule on the other)."""
    path = Path(norm_file(file))
    category = path.parts[0] if len(path.parts) > 1 else ""
    return category, path.stem, page


def parse_chunk_id(chunk_id: str) -> tuple[str, int | None]:
    """`{file}#p{page-or-null}#c{n}` -> (file, page) — §8 invariant, see
    rag.chunking.common.chunk_id_for."""
    head, _, _n = chunk_id.rpartition("#c")
    file, _, page = head.rpartition("#p")
    return file, None if page == "null" else int(page)


def indexed_pages(index_dir: Path) -> set[tuple[str, str, int | None]]:
    """Every page ingest actually produced a chunk for, from the sparse
    index's chunk_ids.json (the one artifact that lists the whole index
    without loading a model or opening Qdrant's single-process lock)."""
    ids = json.loads((index_dir / "bm25" / "chunk_ids.json").read_text(encoding="utf-8"))
    return {page_key(*parse_chunk_id(chunk_id)) for chunk_id in ids}


def classify_group(group: dict, gated: set, candidates: set, indexed: set) -> str:
    """The furthest stage any source in the group reached. A group is
    satisfied by ANY of its members (`any_of` = interchangeable sources for
    one fact), so the best member's stage is the group's."""
    keys = [page_key(s["file"], s.get("page")) for s in group.get("any_of", [])]
    for stage, reached in (("gated", gated), ("not_gated", candidates),
                           ("not_retrieved", indexed)):
        if any(key in reached for key in keys):
            return stage
    return "missing_from_index"


def audit_question(retriever, question: dict, indexed: set) -> dict:
    """One question through fuse -> rerank+gate, with no category filter (the
    reference set carries no predicted category, and this measures retrieval
    itself). Returns the per-question detail record.

    Thread-safe for the parts that matter: the sparse stage self-serializes,
    dense and CrossEncoder reads are concurrent-safe, and the only shared
    mutable state `fuse` touches is `last_stats`, which nothing here reads.
    """
    candidates = retriever.fuse(question["question"])
    gated = retriever.rerank_candidates(question["question"], candidates)
    candidate_keys = {page_key(*parse_chunk_id(c.chunk.chunk_id)) for c in candidates}
    gated_keys = {page_key(*parse_chunk_id(c.chunk.chunk_id)) for c in gated}
    groups = [
        {"sources": group.get("any_of", []),
         "stage": classify_group(group, gated_keys, candidate_keys, indexed)}
        for group in question["ground_truth_sources"]
    ]
    return {
        "id": question["id"],
        "domain": question["domain"],
        "difficulty": question["difficulty"],
        "kind": question.get("kind", "standard"),
        "n_candidates": len(candidates),
        "n_gated": len(gated),
        "best_rerank_score": max((c.rerank_score or 0.0 for c in gated), default=None),
        "groups": groups,
    }


def summarize(arm: str, dataset: str, records: list[dict], **extra) -> dict:
    """Stage counts over all groups, plus the two rates worth quoting: how
    many ground-truth pages reach generation, and how many questions have
    EVERY page they need (a multi-source question needs both)."""
    groups = [g for r in records for g in r["groups"]]
    by_stage = {stage: sum(1 for g in groups if g["stage"] == stage) for stage in STAGES}
    complete = sum(1 for r in records
                   if r["groups"] and all(g["stage"] == "gated" for g in r["groups"]))

    def slice_by(key: str) -> dict:
        out: dict[str, dict[str, int]] = {}
        for record in records:
            bucket = out.setdefault(str(record[key]), {stage: 0 for stage in STAGES})
            for group in record["groups"]:
                bucket[group["stage"]] += 1
        return dict(sorted(out.items()))

    n_groups = len(groups)
    return {
        "arm": arm,
        "dataset": dataset,
        "n_questions": len(records),
        "n_groups": n_groups,
        "by_stage": by_stage,
        "stage_rates": {stage: (count / n_groups if n_groups else None)
                        for stage, count in by_stage.items()},
        "group_hit_rate": by_stage["gated"] / n_groups if n_groups else None,
        "questions_fully_covered": complete,
        "question_hit_rate": complete / len(records) if records else None,
        "by_difficulty": slice_by("difficulty"),
        "by_kind": slice_by("kind"),
        "by_domain": slice_by("domain"),
        **extra,
    }


def load_questions(path: str, limit: int | None) -> list[dict]:
    """Answerable items only: an unanswerable question has no ground-truth
    page, so there is no retrieval to audit."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [q for q in items if q.get("ground_truth_sources")]
    return items[:limit] if limit else items


def arm_unavailable(config_path: str) -> str | None:
    """Why this arm cannot run here, or None. Configs and indexes differ
    between the laptop and the GPU node; a missing arm is a skip, not a
    failure of the run."""
    if not Path(config_path).is_file():
        return f"config {config_path} not found"
    from rag.config import load_config

    index_dir = Path(load_config(config_path).index_dir)
    if not (index_dir / "manifest.json").is_file():
        return f"no ingested index at {index_dir}"
    if not (index_dir / "bm25" / "chunk_ids.json").is_file():
        return f"no sparse chunk_ids.json under {index_dir}"
    return None


class _Progress:
    """Counted progress lines from the retrieval threads."""

    def __init__(self, label: str, total: int) -> None:
        self.label, self.total, self.done = label, total, 0
        self.lock = Lock()

    def tick(self, item_id: str) -> None:
        with self.lock:
            self.done += 1
            print(f"  [{self.label}] {self.done}/{self.total} {item_id}", file=sys.stderr)


def run_arm(arm: str, config_path: str, datasets: dict[str, list[dict]],
            workers: int, out_dir: Path) -> list[dict]:
    """One arm against every dataset — the index is opened once (Qdrant local
    is a single-process lock) and closed before the next arm."""
    from rag.config import load_config
    from rag.retrieve.retriever import load_retriever

    config = load_config(config_path)
    index_dir = Path(config.index_dir)
    indexed = indexed_pages(index_dir)
    print(f"[{arm}] {len(indexed)} indexed pages in {index_dir}", file=sys.stderr)

    retriever = load_retriever(config)
    summaries = []
    try:
        for dataset, questions in datasets.items():
            progress = _Progress(f"{arm}/{dataset}", len(questions))

            def one(question: dict) -> dict:
                record = audit_question(retriever, question, indexed)
                progress.tick(question["id"])
                return record

            with ThreadPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(one, questions))
            detail_path = out_dir / f"{arm}__{dataset}.jsonl"
            detail_path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                encoding="utf-8")
            summaries.append(summarize(arm, dataset, records, config=config_path,
                                       index_dir=str(index_dir),
                                       n_indexed_pages=len(indexed),
                                       detail=detail_path.name))
    finally:
        retriever.close()
    return summaries


def report_md(summaries: list[dict], skipped: list[dict]) -> str:
    """One table: every arm x dataset, worst stage first — the point of the
    audit is which stage is losing the pages, not the headline hit rate."""
    lines = [
        "# Retrieval hit-rate audit",
        "",
        "Where each ground-truth source group ends up. `missing_from_index` is a "
        "corpus/parse gap, `not_retrieved` a recall failure, `not_gated` a "
        "ranking/threshold failure.",
        "",
        "| arm | set | groups | gated | not_gated | not_retrieved | missing_from_index | "
        "group hit | question hit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        stages = s["by_stage"]
        lines.append(
            f"| {s['arm']} | {s['dataset']} | {s['n_groups']} | {stages['gated']} | "
            f"{stages['not_gated']} | {stages['not_retrieved']} | "
            f"{stages['missing_from_index']} | {s['group_hit_rate']:.1%} | "
            f"{s['question_hit_rate']:.1%} |")
    if skipped:
        lines += ["", "## Skipped arms", ""]
        lines += [f"- **{s['arm']}**: {s['reason']}" for s in skipped]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", metavar="NAME=CONFIG", default=None,
                    help="repeatable; default: " + ", ".join(DEFAULT_ARMS))
    ap.add_argument("--dataset", action="append", metavar="NAME=PATH", default=None,
                    help="repeatable; default: " + ", ".join(DEFAULT_DATASETS))
    ap.add_argument("--out", required=True, help="output directory for this run")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent retrievals (the sparse stage self-serializes)")
    ap.add_argument("--limit", type=int, default=None,
                    help="audit only the first N questions of each set (smoke tests)")
    args = ap.parse_args(argv)

    def pairs(values, default):
        if not values:
            return dict(default)
        return dict(v.split("=", 1) for v in values)

    arms = pairs(args.arm, DEFAULT_ARMS)
    dataset_paths = pairs(args.dataset, DEFAULT_DATASETS)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {name: load_questions(path, args.limit)
                for name, path in dataset_paths.items() if Path(path).is_file()}
    for name, path in dataset_paths.items():
        if name not in datasets:
            print(f"WARNING: dataset {name} ({path}) not found — skipped", file=sys.stderr)
    if not datasets:
        print("No datasets found — nothing to audit.", file=sys.stderr)
        return 1
    print(f"Auditing {len(arms)} arm(s) x "
          f"{', '.join(f'{n}:{len(q)}q' for n, q in datasets.items())}", file=sys.stderr)

    summaries, skipped = [], []
    for arm, config_path in arms.items():
        reason = arm_unavailable(config_path)
        if reason is not None:
            print(f"SKIP {arm}: {reason}", file=sys.stderr)
            skipped.append({"arm": arm, "config": config_path, "reason": reason})
            continue
        summaries += run_arm(arm, config_path, datasets, args.workers, out_dir)

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arms": arms,
        "datasets": dataset_paths,
        "summaries": summaries,
        "skipped": skipped,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report_md(summaries, skipped), encoding="utf-8")

    print(f"\nWrote {len(summaries)} arm x dataset summaries to {out_dir}", file=sys.stderr)
    for s in summaries:
        print(f"  {s['arm']:22} {s['dataset']:3} group hit {s['group_hit_rate']:.1%}  "
              f"{s['by_stage']}", file=sys.stderr)
    return 0 if summaries else 1


if __name__ == "__main__":
    sys.exit(main())

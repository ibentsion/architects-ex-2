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

Two knobs turn the audit from a diagnosis into a tuning instrument:

* `--deep-k N` fuses a pool of N instead of the configured 20 and records where
  each ground-truth page sits in it, so the report answers "what would top_k=50
  have covered?" instead of only "did top_k=20 cover it?".
* `--filter-mode none|gold|predicted` sets the category filter. `none` is the
  unfiltered path the agent's retry converges to, `gold` the ceiling a perfect
  classifier would buy, and `predicted` replays a `classify_eval` arm's own
  filters from its predictions.jsonl — no classification calls, and the exact
  tags production would have used.

Arms whose config or index is absent on this node are skipped with a reason
rather than failing the run.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from .pages import norm_file
from .split import load_reference

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


def _ranks(candidates: list) -> tuple[dict, dict, dict]:
    """page_key -> (best 1-based rank, chunks contributed, best rerank score)
    over an ordered candidate list."""
    ranks: dict = {}
    counts: dict = {}
    scores: dict = {}
    for position, candidate in enumerate(candidates, start=1):
        key = page_key(*parse_chunk_id(candidate.chunk.chunk_id))
        ranks.setdefault(key, position)
        counts[key] = counts.get(key, 0) + 1
        score = candidate.rerank_score
        if score is not None and score > scores.get(key, -1.0):
            scores[key] = score
    return ranks, counts, scores


def _best(keys: list, table: dict):
    """The group's best member under `table` — `any_of` members are
    interchangeable, so the group is as deep as its shallowest source."""
    found = [(table[key], key) for key in keys if key in table]
    return min(found) if found else (None, None)


def audit_question(retriever, question: dict, indexed: set, *,
                   category: str | None = None, deep_k: int | None = None) -> dict:
    """One question through fuse -> rerank -> gate. Returns the per-question
    detail record.

    `deep_k` overrides both top_k's so the candidate pool is deeper than
    production's 20: the whole point is to see how deep a ground-truth page
    actually sits, which a pool truncated at the production value cannot show.
    The gate and top_n stay at their configured values, so `stage` still
    reports what production would have kept.

    Thread-safe for the parts that matter: the sparse stage self-serializes,
    dense and CrossEncoder reads are concurrent-safe, and the only shared
    mutable state `fuse` touches is `last_stats`, which nothing here reads.
    """
    from rag.retrieve.rerank import apply_gate

    text = question["question"]
    candidates = retriever.fuse(text, dense_top_k=deep_k, sparse_top_k=deep_k,
                                category=category)
    scored = retriever.reranker.score(text, candidates) if candidates else []
    ranked = sorted(scored, key=lambda c: (-(c.rerank_score or 0.0), c.chunk.chunk_id))
    gated = apply_gate(ranked, retriever.gate_threshold, retriever.top_n)

    fused_ranks, fused_counts, _ = _ranks(candidates)
    rerank_ranks, _, rerank_scores = _ranks(ranked)
    gated_keys = {page_key(*parse_chunk_id(c.chunk.chunk_id)) for c in gated}
    candidate_keys = set(fused_ranks)

    groups = []
    for group in question["ground_truth_sources"]:
        keys = [page_key(s["file"], s.get("page")) for s in group.get("any_of", [])]
        fused_rank, fused_key = _best(keys, fused_ranks)
        rerank_rank, rerank_key = _best(keys, rerank_ranks)
        groups.append({
            "sources": group.get("any_of", []),
            "stage": classify_group(group, gated_keys, candidate_keys, indexed),
            # How deep the pool had to be to hold this page at all...
            "fused_rank": fused_rank,
            # ...and where the cross-encoder then put it.
            "rerank_rank": rerank_rank,
            "rerank_score": rerank_scores.get(rerank_key),
            "n_chunks": fused_counts.get(fused_key, 0),
        })
    return {
        "id": question["id"],
        "domain": question["domain"],
        "difficulty": question["difficulty"],
        "kind": question.get("kind", "standard"),
        "category_filter": category,
        "n_candidates": len(candidates),
        "n_gated": len(gated),
        "best_rerank_score": max((c.rerank_score or 0.0 for c in ranked), default=None),
        "groups": groups,
    }


#: Candidate-pool depths the coverage curve is reported at (production fuses
#: dense 20 + sparse 20) and post-rerank cut-offs (production keeps top_n 6).
DEPTH_TOP_K = (5, 10, 20, 30, 50, 75, 100)
DEPTH_TOP_N = (1, 3, 6, 10, 20)


def coverage_curves(groups: list[dict], gate_threshold: float) -> dict:
    """How many source groups a given depth would have covered.

    `by_top_k` is pre-rerank: the share whose page is somewhere in a pool of
    that size — the ceiling any reranker could work with. `by_top_n` is
    post-rerank AND post-gate, so it is what generation would actually see.
    The gap between the two at production's (20, 6) is the reranker's loss;
    the distance from `by_top_k[100]` to 1.0 is first-stage recall."""
    n = len(groups)
    if not n:
        return {"by_top_k": {}, "by_top_n": {}, "depth_percentiles": {}}
    ranks = sorted(g["fused_rank"] for g in groups if g["fused_rank"] is not None)

    def percentile(p: float) -> int | None:
        """Pool depth covering at least p of ALL groups — None when even the
        deepest pool measured cannot (those groups are not in the index, or
        not retrieved at any depth)."""
        index = math.ceil(p * n) - 1
        return ranks[index] if 0 <= index < len(ranks) else None

    return {
        "by_top_k": {k: sum(1 for g in groups
                            if g["fused_rank"] is not None and g["fused_rank"] <= k) / n
                     for k in DEPTH_TOP_K},
        "by_top_n": {t: sum(1 for g in groups
                            if g["rerank_rank"] is not None and g["rerank_rank"] <= t
                            and (g["rerank_score"] or 0.0) >= gate_threshold) / n
                     for t in DEPTH_TOP_N},
        "depth_percentiles": {f"p{int(p * 100)}": percentile(p) for p in (0.5, 0.8, 0.9)},
        "mean_chunks_per_page": (sum(g["n_chunks"] for g in groups) / n),
    }


def summarize(arm: str, dataset: str, records: list[dict], gate_threshold: float,
              **extra) -> dict:
    """Stage counts over all groups, the two rates worth quoting (how many
    ground-truth pages reach generation, and how many questions have EVERY
    page they need), and the depth curves that say what top_k/top_n would."""
    groups = [g for r in records for g in r["groups"]]
    by_stage = {stage: sum(1 for g in groups if g["stage"] == stage) for stage in STAGES}
    complete = sum(1 for r in records
                   if r["groups"] and all(g["stage"] == "gated" for g in r["groups"]))
    # Retrieved, ranked inside production's top_n, and still dropped: the gate
    # threshold alone is responsible for these.
    gate_blocked = sum(1 for g in groups
                       if g["rerank_rank"] is not None and g["rerank_rank"] <= 6
                       and (g["rerank_score"] or 0.0) < gate_threshold)

    def slice_by(key: str) -> dict:
        out: dict[str, dict] = {}
        for record in records:
            bucket = out.setdefault(str(record[key]), [])
            bucket.extend(record["groups"])
        return {
            name: {**{stage: sum(1 for g in gs if g["stage"] == stage) for stage in STAGES},
                   **coverage_curves(gs, gate_threshold)}
            for name, gs in sorted(out.items())
        }

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
        "gate_blocked_groups": gate_blocked,
        "questions_gated_to_nothing": sum(1 for r in records if r["n_gated"] == 0),
        "coverage": coverage_curves(groups, gate_threshold),
        "by_difficulty": slice_by("difficulty"),
        "by_kind": slice_by("kind"),
        "by_domain": slice_by("domain"),
        **extra,
    }


def load_questions(path: str, limit: int | None) -> list[dict]:
    """Answerable items only: an unanswerable question has no ground-truth
    page, so there is no retrieval to audit."""
    items = [q for q in load_reference(path) if q.get("ground_truth_sources")]
    return items[:limit] if limit else items


#: How the category filter is chosen for a question.
#:   none              no filter — what the unfiltered-retry fix converges to
#:   gold              the item's own domain — the ceiling a perfect tag buys
#:   gold-family       the right tag, widened to its family — the cost of widening
#:                     when the tag is already right
#:   predicted         the classifier's tag, filtering only when it gives exactly
#:                     one (production's `single` policy)
#:   predicted-set     whatever tags it gives, however many (`set`)
#:   predicted-family  those tags widened to their families (`family`)
FILTER_MODES = ("none", "gold", "gold-family", "predicted", "predicted-set",
                "predicted-family")


def load_predicted_filters(path: str) -> dict[str, list[str]]:
    """`{question id: predicted categories}` from a classify_eval arm's
    predictions.jsonl — the derived union the engine reads, so replaying it
    reproduces exactly what production would have filtered by, without
    spending a single classification call."""
    predictions = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            predictions[record["id"]] = list(record.get("categories") or [])
    return predictions


def filter_for(mode: str, question: dict, predicted: dict[str, list[str]]
               ) -> str | list[str] | None:
    from rag.classify import expand_families

    if mode == "gold":
        return question["domain"]
    if mode == "gold-family":
        return expand_families([question["domain"]])
    if mode.startswith("predicted"):
        categories = predicted.get(question["id"]) or []
        if not categories:
            return None
        if mode == "predicted":  # production today: one tag or no filter at all
            return categories[0] if len(categories) == 1 else None
        if mode == "predicted-family":
            return expand_families(categories)
        return categories
    return None


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
            workers: int, out_dir: Path, filter_modes: tuple[str, ...] = ("none",),
            predicted: dict[str, str | None] | None = None,
            deep_k: int | None = None) -> list[dict]:
    """One arm against every dataset x filter mode — the index is opened once
    (Qdrant local is a single-process lock) and closed before the next arm."""
    from rag.config import load_config
    from rag.retrieve.retriever import load_retriever

    config = load_config(config_path)
    index_dir = Path(config.index_dir)
    indexed = indexed_pages(index_dir)
    print(f"[{arm}] {len(indexed)} indexed pages in {index_dir}", file=sys.stderr)

    retriever = load_retriever(config)
    predicted = predicted or {}
    summaries = []
    try:
        for dataset, questions in datasets.items():
            for mode in filter_modes:
                label = f"{arm}/{dataset}/{mode}"
                progress = _Progress(label, len(questions))

                def one(question: dict, mode: str = mode) -> dict:
                    record = audit_question(
                        retriever, question, indexed, deep_k=deep_k,
                        category=filter_for(mode, question, predicted))
                    progress.tick(question["id"])
                    return record

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    records = list(pool.map(one, questions))
                stem = f"{arm}__{dataset}__{mode}"
                detail_path = out_dir / f"{stem}.jsonl"
                detail_path.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")
                summaries.append(summarize(
                    arm, dataset, records, config.retrieval.rerank.gate_threshold,
                    filter_mode=mode, config=config_path, index_dir=str(index_dir),
                    n_indexed_pages=len(indexed), deep_k=deep_k,
                    top_n=config.retrieval.rerank.top_n, detail=detail_path.name))
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
        "| arm | set | filter | groups | gated | not_gated | not_retrieved | "
        "missing_from_index | group hit | question hit |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        stages = s["by_stage"]
        lines.append(
            f"| {s['arm']} | {s['dataset']} | {s.get('filter_mode', 'none')} | "
            f"{s['n_groups']} | {stages['gated']} | "
            f"{stages['not_gated']} | {stages['not_retrieved']} | "
            f"{stages['missing_from_index']} | {s['group_hit_rate']:.1%} | "
            f"{s['question_hit_rate']:.1%} |")

    lines += [
        "",
        "## Depth to cover",
        "",
        "Share of ground-truth source groups whose page is inside a candidate pool "
        "of size K (pre-rerank, the ceiling), and inside the top N after rerank AND "
        "the gate (what generation sees). Production fuses K=20 and keeps N=6.",
        "",
        "| arm | set | filter | " + " | ".join(f"K={k}" for k in DEPTH_TOP_K)
        + " | " + " | ".join(f"N={n}" for n in DEPTH_TOP_N) + " | p50 | p80 | p90 |",
        "|---" * (3 + len(DEPTH_TOP_K) + len(DEPTH_TOP_N) + 3) + "|",
    ]
    for s in summaries:
        cov = s["coverage"]
        depths = cov["depth_percentiles"]
        lines.append(
            f"| {s['arm']} | {s['dataset']} | {s.get('filter_mode', 'none')} | "
            + " | ".join(f"{cov['by_top_k'][k]:.0%}" for k in DEPTH_TOP_K) + " | "
            + " | ".join(f"{cov['by_top_n'][n]:.0%}" for n in DEPTH_TOP_N) + " | "
            + " | ".join(str(depths.get(p) or "—") for p in ("p50", "p80", "p90")) + " |")
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
    ap.add_argument("--deep-k", type=int, default=None,
                    help="override dense/sparse top_k so the depth curve is not "
                         "truncated at the production pool size (e.g. 100)")
    ap.add_argument("--filter-mode", nargs="+", choices=FILTER_MODES, default=["none"],
                    help="category filter per question: none | gold | predicted")
    ap.add_argument("--predictions", default=None,
                    help="classify_eval predictions.jsonl, required by "
                         "--filter-mode predicted")
    args = ap.parse_args(argv)

    predicted: dict[str, str | None] = {}
    if "predicted" in args.filter_mode:
        if not args.predictions:
            ap.error("--filter-mode predicted needs --predictions")
        predicted = load_predicted_filters(args.predictions)
        print(f"replaying {len(predicted)} classifier filters from {args.predictions}",
              file=sys.stderr)

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
        summaries += run_arm(arm, config_path, datasets, args.workers, out_dir,
                             tuple(args.filter_mode), predicted, args.deep_k)

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arms": arms,
        "datasets": dataset_paths,
        "filter_modes": args.filter_mode,
        "predictions": args.predictions,
        "deep_k": args.deep_k,
        "summaries": summaries,
        "skipped": skipped,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report_md(summaries, skipped), encoding="utf-8")

    print(f"\nWrote {len(summaries)} summaries to {out_dir}", file=sys.stderr)
    for s in summaries:
        cov = s["coverage"]["by_top_k"]
        print(f"  {s['arm']:18} {s['dataset']:4} {s.get('filter_mode', 'none'):9} "
              f"group hit {s['group_hit_rate']:.1%}  "
              f"pool@20 {cov.get(20, 0):.0%} pool@100 {cov.get(100, 0):.0%}  "
              f"{s['by_stage']}", file=sys.stderr)
    return 0 if summaries else 1


if __name__ == "__main__":
    sys.exit(main())

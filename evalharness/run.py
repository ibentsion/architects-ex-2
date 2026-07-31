"""CLI: score an answers JSONL against the dev questions and write
judgments.jsonl + metrics.json + report.md into the output directory.

    python -m evalharness.run \
        --questions reference_questions.json \
        --answers baseline_answers.jsonl \
        --out eval_results/baseline

Committee mode: pass --judges with several models, e.g.
    --judges deepseek-ai/DeepSeek-V4-Pro Qwen/Qwen3-235B-A22B zai-org/GLM-4.5
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

from . import citations, judge, metrics, report
from .pages import PageStore
from .prompts import SYSTEM_PROMPTS

DEFAULT_JUDGE = "deepseek-ai/DeepSeek-V4-Pro"
# Rough cost estimate per judge call (prompt ~1.5k tok in, ~0.2k out at
# tf_client's DeepSeek-class price estimate). Only used for reporting.
EST_COST_PER_CALL = 0.0015


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="reference questions JSON")
    ap.add_argument("--answers", required=True, help="answers JSONL to evaluate")
    ap.add_argument("--out", required=True, help="output directory for this run")
    ap.add_argument("--judges", nargs="+", default=[DEFAULT_JUDGE],
                    help=f"judge model(s); several = committee (default: {DEFAULT_JUDGE})")
    ap.add_argument("--prompt", choices=sorted(SYSTEM_PROMPTS), default="rubric",
                    help="judge prompt variant (default: rubric)")
    ap.add_argument("--corpus", default="corpus",
                    help="corpus directory, for resolving cited pages")
    ap.add_argument("--cache-dir", default="cache",
                    help="parse cache holding the corpus's Docling parses")
    ap.add_argument("--workers", type=int, default=8, help="parallel judge calls")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N questions (smoke tests)")
    args = ap.parse_args(argv)

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    answers = [json.loads(line) for line in
               Path(args.answers).read_text(encoding="utf-8").splitlines() if line.strip()]
    answers_by_id = {a["id"]: a for a in answers}

    missing = [q["id"] for q in questions if q["id"] not in answers_by_id]
    if missing:
        print(f"WARNING: {len(missing)} questions have no answer and are skipped: "
              f"{', '.join(missing)}", file=sys.stderr)
    questions = [q for q in questions if q["id"] in answers_by_id]
    if args.limit:
        questions = questions[:args.limit]

    # Resolve every cited {file, page} to its real corpus page before judging:
    # the citation judge reads the actual document, never a search result.
    store = PageStore(args.corpus, args.cache_dir)
    resolved_by_id = {q["id"]: citations.resolve_citations(
        answers_by_id[q["id"]].get("citations"), store) for q in questions}
    judgeable = {qid: [c for c in r if c["invalid_reason"] is None]
                 for qid, r in resolved_by_id.items()}
    n_pages = sum(len(v) for v in judgeable.values())
    n_invalid = sum(len(r) - len(judgeable[qid]) for qid, r in resolved_by_id.items())
    print(f"Resolved {n_pages} cited pages ({n_invalid} citation(s) point at no "
          f"real page)", file=sys.stderr)

    n_calls = (len(questions) + sum(1 for v in judgeable.values() if v)) * len(args.judges)
    print(f"Judging {len(questions)} answers with {len(args.judges)} judge(s) "
          f"({n_calls} calls incl. citations, ~${n_calls * EST_COST_PER_CALL:.2f} est.) "
          f"— prompt variant '{args.prompt}'", file=sys.stderr)

    def progress(done, total, qid):
        print(f"  [{done}/{total}] {qid}", file=sys.stderr)

    judged = judge.judge_all(questions, answers_by_id, args.judges,
                             args.prompt, args.workers, progress)
    print("Judging citations...", file=sys.stderr)
    cite_judged = judge.judge_citations_all(questions, judgeable, args.judges,
                                            args.workers, progress)

    records = []
    for q in questions:
        answer = answers_by_id[q["id"]]
        result = judged[q["id"]]
        cite_result = cite_judged.get(q["id"])
        records.append({
            "id": q["id"],
            "domain": q["domain"],
            "difficulty": q["difficulty"],
            "n_source_groups": len(q["ground_truth_sources"]),
            "judgment": result["aggregate"],
            "judges": result["judgments"],
            "citations": citations.score_citations(
                resolved_by_id[q["id"]],
                cite_result["aggregate"] if cite_result else None),
            "citation_judgment": cite_result["aggregate"] if cite_result else None,
            "citation_judges": cite_result["judgments"] if cite_result else [],
            "gt_source_hit": citations.gt_source_hit(answer.get("citations"),
                                                     q["ground_truth_sources"]),
            "latency_ms": answer.get("latency_ms"),
            "tokens": answer.get("tokens"),
        })

    agg = metrics.aggregate(records)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_name": out_dir.name,
        "date": datetime.date.today().isoformat(),
        "questions_file": args.questions,
        "answers_file": args.answers,
        "judges": args.judges,
        "prompt_variant": args.prompt,
        "est_cost_usd": n_calls * EST_COST_PER_CALL,
    }

    with open(out_dir / "judgments.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "metrics.json").write_text(
        json.dumps({"meta": meta, "metrics": agg}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / "report.md").write_text(report.render(agg, records, meta),
                                       encoding="utf-8")

    overall = agg["overall"]
    print(f"\nDone. correctness={overall['correctness']} "
          f"hallucination_rate={overall['hallucination_rate']} "
          f"citation_accuracy={overall['citation_accuracy']}", file=sys.stderr)
    print(f"Report: {out_dir / 'report.md'}", file=sys.stderr)
    if agg["judge_failures"]:
        print(f"Judge failures: {agg['judge_failures']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Aggregate per-question records into overall and sliced metrics.

A record (built in run.py) carries: id, domain, difficulty, n_source_groups,
the aggregated answer judgment, the citation scores, the ground-truth-source
diagnostic, and latency/token info from the answers file.
"""
import collections
import statistics


def _pct(values, p):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[idx]


def _mean(values):
    return round(statistics.mean(values), 2) if values else None


def summarize(records: list) -> dict:
    """Metrics for one bucket of records."""
    judged = [r for r in records if "error" not in r["judgment"]]
    scores = lambda f: [r["judgment"][f] for r in judged]
    verdicts = [r["judgment"]["verdict"] for r in judged]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    # Citation accuracy counts every answer, uncited and refusals included:
    # an answer that cites nothing establishes nothing.
    accuracies = [r["citations"]["accuracy"] for r in records]
    cited = sum(r["citations"]["cited_count"] for r in records)
    invalid = sum(r["citations"]["invalid_count"] for r in records)
    gt_hits = [r["gt_source_hit"]["hit_rate"] for r in records
               if r["gt_source_hit"]["hit_rate"] is not None]
    return {
        "n": len(records),
        "judged": len(judged),
        "correctness": _mean(scores("correctness")),
        "completeness": _mean(scores("completeness")),
        "conversational_quality": _mean(scores("conversational_quality")),
        "hallucination_rate": _mean([r["judgment"]["hallucination"] for r in judged]),
        "refusal_rate": _mean([v == "refusal" for v in verdicts]),
        "verdicts": {v: verdicts.count(v) for v in
                     ("correct", "partially_correct", "incorrect", "refusal")},
        "citation_accuracy": _mean(accuracies),
        "full_citation_credit_rate": _mean([a == 1.0 for a in accuracies]),
        "uncited_rate": _mean([r["citations"]["cited_count"] == 0 for r in records]),
        "invalid_citation_rate": round(invalid / cited, 2) if cited else None,
        "gt_source_hit_rate": _mean(gt_hits),
        "latency_ms_p50": round(_pct(latencies, 50)) if latencies else None,
        "latency_ms_p95": round(_pct(latencies, 95)) if latencies else None,
        "latency_ms_mean": round(statistics.mean(latencies)) if latencies else None,
    }


def _by(records, key):
    buckets = {}
    for r in records:
        buckets.setdefault(r[key], []).append(r)
    return {k: summarize(v) for k, v in sorted(buckets.items())}


def aggregate(records: list) -> dict:
    """Overall metrics plus the three requested slices."""
    judged = [r for r in records if "error" not in r["judgment"]]
    by_difficulty = _by(records, "difficulty")
    return {
        "overall": summarize(records),
        "by_difficulty": {k: by_difficulty[k] for k in ("easy", "medium", "hard")
                          if k in by_difficulty},
        "by_domain": _by(records, "domain"),
        "by_source_groups": _by(records, "n_source_groups"),
        "disagreement_rate": _mean([bool(r["judgment"].get("disagreement"))
                                    for r in judged]),
        "judge_failures": [r["id"] for r in records if "error" in r["judgment"]],
        # Why citations failed to resolve — separates fabricated references
        # from corpus/parse gaps (empty_page).
        "invalid_citation_reasons": dict(sorted(collections.Counter(
            reason for r in records
            for reason in r["citations"]["invalid_reasons"]).items())),
        "citation_judge_failures": [r["id"] for r in records
                                    if r["citations"].get("judge_failed")],
    }

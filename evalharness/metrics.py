"""Aggregate per-question records into overall and sliced metrics.

A record (built in run.py) carries: id, domain, difficulty, n_source_groups,
the aggregated judgment, the citation scores, and latency/token info from the
answers file.
"""
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
    recalls = [r["citations"]["recall"] for r in records
               if r["citations"]["recall"] is not None]
    precisions = [r["citations"]["precision"] for r in records
                  if r["citations"]["precision"] is not None]
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
        "citation_recall": _mean(recalls),
        "citation_precision": _mean(precisions) if precisions else None,
        "full_citation_credit_rate": _mean([r == 1.0 for r in recalls]),
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
    }

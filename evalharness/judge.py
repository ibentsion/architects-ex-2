"""LLM-judge invocation and committee aggregation.

Two judgments per answer, both through the same machinery (same models, same
retry-on-bad-JSON loop, same committee aggregation): answer quality, and
citation accuracy against the real cited corpus pages.

Single judge by default; pass several models to run a committee. Numeric
scores aggregate by median, categorical fields by majority vote, and
substantial disagreement is flagged per question so the report can surface it.
"""
import json
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from tf_client import chat

from .prompts import build_citation_messages, build_messages

SCORE_FIELDS = ("correctness", "completeness", "conversational_quality")
VERDICTS = ("correct", "partially_correct", "incorrect", "refusal")
# Worst-first, for pessimistic tie-breaking (mirrors VERDICTS' ordering).
SUPPORT_LEVELS = ("fully", "partially", "not_at_all")
CITATION_LABELS = ("establishes", "partial", "unrelated")
# Numeric spread across committee members that counts as disagreement.
DISAGREEMENT_SPREAD = 3


def _extract_json(text: str) -> dict:
    """Parse the judge reply, tolerating code fences and surrounding prose."""
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize(raw: dict) -> dict:
    """Validate and clamp a parsed judgment to the schema."""
    out = {}
    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"bad verdict: {verdict!r}")
    out["verdict"] = verdict
    out["hallucination"] = bool(raw.get("hallucination")) and verdict != "refusal"
    for field in SCORE_FIELDS:
        value = raw.get(field)
        if not isinstance(value, (int, float)):
            raise ValueError(f"non-numeric {field}: {value!r}")
        out[field] = max(0, min(10, round(float(value))))
    out["reasoning"] = str(raw.get("reasoning", "")).strip()
    if isinstance(raw.get("claims"), dict):
        out["claims"] = raw["claims"]
    return out


def _normalize_citation(raw: dict, n_citations: int) -> dict:
    """Validate and clamp a parsed citation judgment to the schema."""
    support = str(raw.get("citation_support", "")).strip().lower()
    if support not in SUPPORT_LEVELS:
        raise ValueError(f"bad citation_support: {support!r}")
    labels = ["unrelated"] * n_citations
    for entry in raw.get("per_citation") or []:
        if not isinstance(entry, dict):
            continue
        idx, label = entry.get("idx"), str(entry.get("label", "")).strip().lower()
        if isinstance(idx, int) and 0 <= idx < n_citations and label in CITATION_LABELS:
            labels[idx] = label
    return {
        "citation_support": support,
        "citation_labels": labels,
        "reasoning": str(raw.get("reasoning", "")).strip(),
    }


def _call_judge(messages: list, model: str, normalize, max_retries: int = 2) -> dict:
    """Call one judge model until it returns JSON that `normalize` accepts.

    Returns the normalized judgment tagged with `judge_model`, or
    {"error": ...} if the model never produced a valid reply.
    """
    last_err = None
    reply = ""
    for attempt in range(max_retries + 1):
        try:
            # 4096 tokens: reasoning-model judges (e.g. Nemotron) spend most of
            # the budget thinking and return content=None when it runs out.
            reply = chat(messages, model=model, max_tokens=4096,
                         temperature=0.0, quiet=True)
            if not reply:
                reply = ""
                raise ValueError("empty reply (reasoning budget exhausted?)")
            judgment = normalize(_extract_json(reply))
            judgment["judge_model"] = model
            return judgment
        except Exception as err:  # bad JSON, schema violation, or API error
            last_err = err
            if attempt < max_retries:
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content":
                        f"Your previous reply was invalid ({err}). "
                        "Reply again with ONLY the JSON object, exactly per the schema."},
                ]
    return {"judge_model": model, "error": f"{type(last_err).__name__}: {last_err}"}


def judge_one(question: dict, answer: dict, model: str, variant: str,
              max_retries: int = 2) -> dict:
    """Run one judge model on one answer. Returns a normalized judgment,
    or {"error": ...} if the model never produced valid JSON."""
    return _call_judge(build_messages(question, answer, variant), model,
                       _normalize, max_retries)


def judge_citations_one(question: dict, resolved: list, model: str,
                        max_retries: int = 2) -> dict:
    """Run one judge model on one answer's resolved citations."""
    return _call_judge(build_citation_messages(question, resolved), model,
                       lambda raw: _normalize_citation(raw, len(resolved)),
                       max_retries)


def _majority(values, order):
    """Majority vote over a categorical field; ties break pessimistically
    (the worst of the tied values, per `order`)."""
    counts = Counter(values).most_common()
    tied = [v for v, c in counts if c == counts[0][1]]
    return max(tied, key=order.index)


def aggregate_committee(judgments: list) -> dict:
    """Combine per-judge judgments into one verdict per question.

    Median for numeric scores, majority for verdict/hallucination (ties break
    pessimistically: worse verdict, hallucination=true). Flags disagreement
    when the committee splits on verdict or spreads wide on a numeric score.
    """
    valid = [j for j in judgments if "error" not in j]
    if not valid:
        return {"error": "all judges failed", "judges_failed": len(judgments)}

    agg = {}
    for field in SCORE_FIELDS:
        values = [j[field] for j in valid]
        agg[field] = statistics.median(values)

    verdicts = [j["verdict"] for j in valid]
    agg["verdict"] = _majority(verdicts, VERDICTS)

    halluc_votes = [j["hallucination"] for j in valid]
    agg["hallucination"] = halluc_votes.count(True) * 2 >= len(halluc_votes)

    agg["disagreement"] = (
        len(set(verdicts)) > 1
        or any(max(j[f] for j in valid) - min(j[f] for j in valid) >= DISAGREEMENT_SPREAD
               for f in SCORE_FIELDS)
    )
    agg["judges_valid"] = len(valid)
    agg["judges_failed"] = len(judgments) - len(valid)
    # Single judge: keep its reasoning; committee: keep all, keyed by model.
    if len(valid) == 1:
        agg["reasoning"] = valid[0]["reasoning"]
        if "claims" in valid[0]:
            agg["claims"] = valid[0]["claims"]
    else:
        agg["reasoning"] = {j["judge_model"]: j["reasoning"] for j in valid}
    return agg


def aggregate_citation_committee(judgments: list) -> dict:
    """Combine per-judge citation judgments into one per question."""
    valid = [j for j in judgments if "error" not in j]
    if not valid:
        return {"error": "all judges failed", "judges_failed": len(judgments)}

    supports = [j["citation_support"] for j in valid]
    n_labels = max(len(j["citation_labels"]) for j in valid)
    agg = {
        "citation_support": _majority(supports, SUPPORT_LEVELS),
        "citation_labels": [
            _majority([j["citation_labels"][i] for j in valid
                       if i < len(j["citation_labels"])], CITATION_LABELS)
            for i in range(n_labels)
        ],
        "disagreement": len(set(supports)) > 1,
        "judges_valid": len(valid),
        "judges_failed": len(judgments) - len(valid),
    }
    if len(valid) == 1:
        agg["reasoning"] = valid[0]["reasoning"]
    else:
        agg["reasoning"] = {j["judge_model"]: j["reasoning"] for j in valid}
    return agg


def _run_jobs(fn, jobs: list, workers: int, progress=None) -> dict:
    """Run `fn(*args)` for every (qid, *args) job in a thread pool, collecting
    results per question id."""
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): qid for qid, *args in jobs}
        for done, (future, qid) in enumerate(futures.items(), start=1):
            results.setdefault(qid, []).append(future.result())
            if progress:
                progress(done, len(jobs), qid)
    return results


def judge_all(questions: list, answers_by_id: dict, models: list, variant: str,
              workers: int = 8, progress=None) -> dict:
    """Judge every question that has an answer. Returns {question_id: record}
    where record = {"judgments": [per-judge...], "aggregate": {...}}."""
    jobs = [(q["id"], q, answers_by_id[q["id"]], m, variant)
            for q in questions if q["id"] in answers_by_id
            for m in models]
    results = _run_jobs(judge_one, jobs, workers, progress)
    return {qid: {"judgments": js, "aggregate": aggregate_committee(js)}
            for qid, js in results.items()}


def judge_citations_all(questions: list, resolved_by_id: dict, models: list,
                        workers: int = 8, progress=None) -> dict:
    """Judge the resolved citations of every question that has at least one.

    Questions whose citations all failed to resolve (or that cited nothing)
    are skipped — there is no evidence to read, so no judge call is spent."""
    jobs = [(q["id"], q, resolved_by_id[q["id"]], m)
            for q in questions if resolved_by_id.get(q["id"])
            for m in models]
    results = _run_jobs(judge_citations_one, jobs, workers, progress)
    return {qid: {"judgments": js, "aggregate": aggregate_citation_committee(js)}
            for qid, js in results.items()}

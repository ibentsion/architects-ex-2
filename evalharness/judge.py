"""LLM-judge invocation and committee aggregation.

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

from .prompts import build_messages

SCORE_FIELDS = ("correctness", "completeness", "conversational_quality")
VERDICTS = ("correct", "partially_correct", "incorrect", "refusal")
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


def judge_one(question: dict, answer: dict, model: str, variant: str,
              max_retries: int = 2) -> dict:
    """Run one judge model on one answer. Returns a normalized judgment,
    or {"error": ...} if the model never produced valid JSON."""
    messages = build_messages(question, answer, variant)
    last_err = None
    reply = ""
    for attempt in range(max_retries + 1):
        try:
            reply = chat(messages, model=model, max_tokens=1024,
                         temperature=0.0, quiet=True)
            judgment = _normalize(_extract_json(reply))
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

    verdict_counts = Counter(j["verdict"] for j in valid)
    top = verdict_counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:  # tie → worst of the tied verdicts
        tied = [v for v, c in top if c == top[0][1]]
        agg["verdict"] = max(tied, key=VERDICTS.index)
    else:
        agg["verdict"] = top[0][0]

    halluc_votes = [j["hallucination"] for j in valid]
    agg["hallucination"] = halluc_votes.count(True) * 2 >= len(halluc_votes)

    agg["disagreement"] = (
        len(verdict_counts) > 1
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


def judge_all(questions: list, answers_by_id: dict, models: list, variant: str,
              workers: int = 8, progress=None) -> dict:
    """Judge every question that has an answer. Returns {question_id: record}
    where record = {"judgments": [per-judge...], "aggregate": {...}}."""
    jobs = [(q, answers_by_id[q["id"]], m)
            for q in questions if q["id"] in answers_by_id
            for m in models]
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge_one, q, a, m, variant): (q["id"], m)
                   for q, a, m in jobs}
        done = 0
        for future, (qid, model) in futures.items():
            results.setdefault(qid, []).append(future.result())
            done += 1
            if progress:
                progress(done, len(jobs), qid)
    return {qid: {"judgments": js, "aggregate": aggregate_committee(js)}
            for qid, js in results.items()}

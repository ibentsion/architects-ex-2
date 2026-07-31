"""Render the markdown evaluation report (English) from aggregated metrics,
including rule-based, prioritized improvement suggestions derived from the
numbers — the "what should the system focus on next" section.
"""

_METRIC_COLS = [
    ("correctness", "Correctness"),
    ("completeness", "Completeness"),
    ("conversational_quality", "Conv. quality"),
    ("hallucination_rate", "Halluc. rate"),
    ("refusal_rate", "Refusal rate"),
    ("citation_accuracy", "Cite accuracy"),
    ("latency_ms_p50", "Latency p50 (ms)"),
]


def _fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _table(buckets: dict, label: str) -> str:
    header = f"| {label} | n | " + " | ".join(name for _, name in _METRIC_COLS) + " |"
    sep = "|" + "---|" * (len(_METRIC_COLS) + 2)
    rows = [
        f"| {key} | {m['n']} | " + " | ".join(_fmt(m[f]) for f, _ in _METRIC_COLS) + " |"
        for key, m in buckets.items()
    ]
    return "\n".join([header, sep] + rows)


def suggestions(metrics: dict) -> list:
    """Rule-based improvement suggestions, ordered by priority.

    Each rule fires from the aggregated numbers, so re-running the harness
    after a system change automatically re-derives the focus list.
    """
    out = []  # (priority, text); priority 1 = highest
    overall = metrics["overall"]
    by_diff = metrics["by_difficulty"]
    by_domain = metrics["by_domain"]
    by_groups = metrics["by_source_groups"]

    accuracy = overall["citation_accuracy"]
    uncited = overall["uncited_rate"] or 0
    if accuracy is not None and accuracy < 0.3:
        cause = ("most answers cite nothing at all"
                 if uncited >= 0.5 else
                 "the cited pages do not establish the ground-truth answer")
        out.append((1, f"**Grounding is absent or broken** (citation accuracy "
                       f"{accuracy:.2f} — {cause}). Highest-leverage fix: build "
                       "retrieval that returns file + page metadata and require "
                       "the generator to cite the page each factual claim came "
                       "from. Citation accuracy is a graded criterion and "
                       "currently earns ~nothing."))
    elif accuracy is not None and accuracy < 0.7:
        out.append((2, f"**Citations under-support the answers** (citation "
                       f"accuracy {accuracy:.2f}). The system cites real pages "
                       "that only partly establish the answer — tighten the "
                       "generator prompt so it cites the page carrying the "
                       "decisive number/condition rather than the topically "
                       "nearest chunk."))

    invalid = overall["invalid_citation_rate"]
    if invalid is not None and invalid >= 0.05:
        reasons = ", ".join(f"{k}={v}" for k, v in
                            metrics["invalid_citation_reasons"].items())
        out.append((1, f"**Citations point at pages that do not exist** "
                       f"({invalid:.2f} of all citations; {reasons}). Every one "
                       "dilutes the score. `unknown_file`/`page_out_of_range` "
                       "means the generator invents references — emit citations "
                       "only from retrieved chunk metadata; `empty_page` means "
                       "the page parsed to nothing and is a corpus/parsing "
                       "problem, not a generation one."))

    gt_hit = overall["gt_source_hit_rate"]
    if (accuracy is not None and gt_hit is not None
            and accuracy - gt_hit >= 0.3):
        out.append((3, f"**Retrieval finds the fact elsewhere** (citation "
                       f"accuracy {accuracy:.2f} vs ground-truth-source hit "
                       f"rate {gt_hit:.2f}). Not a defect — the corpus repeats "
                       "facts across documents and the judge credits any page "
                       "that establishes the answer. Noted only as a retrieval "
                       "debugging signal."))

    halluc = overall["hallucination_rate"]
    if halluc is not None and halluc >= 0.25:
        worst = [d for d, m in by_domain.items()
                 if m["hallucination_rate"] is not None
                 and m["hallucination_rate"] >= halluc + 0.1]
        domains_note = f" Worst domains: {', '.join(worst)}." if worst else ""
        out.append((1, f"**High hallucination rate** ({halluc:.2f} of answers "
                       "confidently contradict the ground truth). Add an "
                       "evidence-or-abstain policy: answer only from retrieved "
                       "context and fall back to \"I don't have enough "
                       f"information\" when evidence is missing.{domains_note}"))

    refusal = overall["refusal_rate"] or 0
    if refusal >= 0.2:
        out.append((2, f"**High refusal rate** ({refusal:.2f}). The system "
                       "abstains often — if retrieval exists, raise recall (more "
                       "candidates, query rewriting); if this is the bare "
                       "baseline, retrieval will convert refusals to grounded "
                       "answers."))

    easy = by_diff.get("easy", {}).get("correctness")
    hard = by_diff.get("hard", {}).get("correctness")
    if easy is not None and hard is not None and easy - hard >= 2:
        out.append((2, f"**Hard questions lag easy ones** (correctness "
                       f"{easy:.1f} easy vs {hard:.1f} hard). Hard questions "
                       "combine several documents and/or a calculation — add "
                       "multi-hop retrieval (retrieve per sub-question) and let "
                       "the model do explicit arithmetic on retrieved numbers."))

    single = by_groups.get(1, {}).get("correctness")
    multi = by_groups.get(2, {}).get("correctness")
    if single is not None and multi is not None and single - multi >= 1.5:
        out.append((2, f"**Cross-document questions underperform** (correctness "
                       f"{single:.1f} with one required source vs {multi:.1f} "
                       "with two). Retrieval must surface passages from multiple "
                       "documents per question — decompose the question or "
                       "retrieve per detected sub-topic."))

    overall_corr = overall["correctness"]
    if overall_corr is not None:
        weak = [f"{d} ({m['correctness']:.1f})" for d, m in by_domain.items()
                if m["correctness"] is not None
                and m["correctness"] <= overall_corr - 1.5]
        if weak:
            out.append((3, f"**Weak domains** (mean correctness "
                           f"{overall_corr:.1f} overall): {', '.join(weak)}. "
                           "Prioritize corpus parsing/coverage checks for these "
                           "domains — inspect whether their documents parse "
                           "cleanly (tables, scanned PDFs) before tuning prompts."))

    comp = overall["completeness"]
    if overall_corr is not None and comp is not None and overall_corr - comp >= 1.5:
        out.append((3, f"**Answers are right but partial** (correctness "
                       f"{overall_corr:.1f} vs completeness {comp:.1f}). "
                       "Ground truths bundle numbers + conditions + exceptions; "
                       "retrieve more context per question (higher k or larger "
                       "chunks) and instruct the model to enumerate conditions."))

    p95 = overall["latency_ms_p95"]
    if p95 is not None and p95 > 15000:
        out.append((3, f"**Latency tail is heavy** (p95 {p95/1000:.1f}s). "
                       "Efficiency is graded — cap generation length, and check "
                       "whether slow answers correlate with long ramble rather "
                       "than genuine retrieval work."))

    disagreement = metrics.get("disagreement_rate")
    if disagreement is not None and disagreement >= 0.25:
        out.append((3, f"**Judge committee disagrees often** ({disagreement:.2f} "
                       "of questions). Treat single-judge scores near decision "
                       "boundaries with skepticism; consider tightening the "
                       "rubric or manually auditing flagged questions."))

    if not out:
        out.append((3, "No rule-based issues fired — inspect the lowest-scoring "
                       "questions below for qualitative failure patterns."))
    return [text for _, text in sorted(out, key=lambda t: t[0])]


def render(metrics: dict, records: list, meta: dict) -> str:
    """Full markdown report."""
    overall = metrics["overall"]
    judged = [r for r in records if "error" not in r["judgment"]]
    worst = sorted(judged, key=lambda r: (r["judgment"]["correctness"],
                                          r["judgment"]["completeness"]))[:5]

    lines = [
        f"# Evaluation Report — {meta['run_name']}",
        "",
        f"- **Date:** {meta['date']}",
        f"- **Answers file:** `{meta['answers_file']}`",
        f"- **Questions file:** `{meta['questions_file']}` ({overall['n']} questions evaluated)",
        f"- **Judge(s):** {', '.join(meta['judges'])} (prompt variant: `{meta['prompt_variant']}`, temperature 0)",
        f"- **Estimated judge cost:** ~${meta['est_cost_usd']:.2f}",
        "",
        "## Executive summary",
        "",
        f"Mean correctness **{_fmt(overall['correctness'])}/10**, "
        f"completeness **{_fmt(overall['completeness'])}/10**, "
        f"conversational quality **{_fmt(overall['conversational_quality'])}/10**. "
        f"Hallucination rate **{_fmt(overall['hallucination_rate'])}**, "
        f"refusal rate **{_fmt(overall['refusal_rate'])}**, "
        f"citation accuracy **{_fmt(overall['citation_accuracy'])}** "
        f"(full-credit rate {_fmt(overall['full_citation_credit_rate'])}, "
        f"{_fmt(overall['invalid_citation_rate'])} of citations point at no real "
        f"page, {_fmt(overall['uncited_rate'])} of answers cite nothing). "
        f"Latency p50 {_fmt(overall['latency_ms_p50'])} ms / "
        f"p95 {_fmt(overall['latency_ms_p95'])} ms.",
        "",
        "Verdicts: " + ", ".join(f"{k.replace('_', ' ')} = {v}"
                                 for k, v in overall["verdicts"].items()) + ".",
        "",
        "## Overall metrics",
        "",
        _table({"all": overall}, "Slice"),
        "",
        "## By difficulty",
        "",
        _table(metrics["by_difficulty"], "Difficulty"),
        "",
        "## By domain",
        "",
        _table(metrics["by_domain"], "Domain"),
        "",
        "## By number of required source groups",
        "",
        _table({f"{k} group(s)": v for k, v in metrics["by_source_groups"].items()},
               "Sources"),
        "",
    ]

    if len(metrics["by_kind"]) > 1:
        lines += ["## By question kind", "", _table(metrics["by_kind"], "Kind"), ""]

    if overall["abstention_rate"] is not None:
        unanswerable = [r for r in records if r.get("answerable") is False]
        answered = [r for r in unanswerable if r["citations"]["cited_count"] > 0]
        lines += [
            "## Abstention (unanswerable questions)",
            "",
            f"{len(unanswerable)} questions in this set have no answer anywhere in "
            "the corpus. The only correct behaviour is to say so.",
            "",
            f"- **Abstained correctly:** {_fmt(overall['abstention_rate'])}",
            f"- **Cited evidence anyway:** {_fmt(overall['unanswerable_citation_rate'])} "
            f"({len(answered)} of {len(unanswerable)}) — a system citing pages for a "
            "question the corpus cannot answer is retrieving-and-answering regardless "
            "of relevance.",
            "",
        ]

    supports = [r["citations"]["support"] for r in records]
    lines += [
        "## Citation accuracy",
        "",
        "Each cited `{file, page}` is resolved to the real corpus page and an "
        "LLM judge rules whether the cited pages establish the ground-truth "
        "answer. Any page that establishes the fact earns credit — the corpus "
        "repeats facts across documents, so there is no fixed list of correct "
        "sources. Score = credit (1.0 fully / 0.5 partially / 0 not at all) × "
        "(citations resolving to a real page ÷ citations made).",
        "",
        f"- **Mean accuracy:** {_fmt(overall['citation_accuracy'])} "
        f"(full credit on {_fmt(overall['full_citation_credit_rate'])} of questions)",
        "- **Support ruling:** " + ", ".join(
            f"{level} = {supports.count(level)}"
            for level in ("fully", "partially", "not_at_all"))
        + f", not judged (no resolvable citation) = {supports.count(None)}",
        f"- **Invalid citations:** {_fmt(overall['invalid_citation_rate'])} of all "
        "citations" + (f" ({', '.join(f'{k}={v}' for k, v in metrics['invalid_citation_reasons'].items())})"
                       if metrics["invalid_citation_reasons"] else ""),
        f"- **Answers citing nothing:** {_fmt(overall['uncited_rate'])}",
        f"- **Ground-truth-source hit rate (diagnostic, not scored):** "
        f"{_fmt(overall['gt_source_hit_rate'])} — how often the answer cited the "
        "sources the reference answer was authored from.",
        "",
    ]

    if metrics["citation_judge_failures"]:
        lines += [
            "The citation judge failed on: "
            + ", ".join(f"`{i}`" for i in metrics["citation_judge_failures"])
            + " (scored 0.0 — treat as missing data, not as a system failure).",
            "",
        ]

    if len(meta["judges"]) > 1:
        flagged = [r["id"] for r in judged if r["judgment"].get("disagreement")]
        lines += [
            "## Committee disagreement",
            "",
            f"Disagreement rate: {_fmt(metrics['disagreement_rate'])} "
            f"({len(flagged)} questions flagged).",
            "Flagged: " + (", ".join(f"`{i}`" for i in flagged) if flagged else "none") + ".",
            "",
        ]

    if metrics["judge_failures"]:
        lines += [
            "## Judge failures",
            "",
            "These questions could not be judged (excluded from judged metrics): "
            + ", ".join(f"`{i}`" for i in metrics["judge_failures"]) + ".",
            "",
        ]

    lines += ["## Lowest-scoring questions", ""]
    for r in worst:
        j = r["judgment"]
        reasoning = j["reasoning"] if isinstance(j["reasoning"], str) \
            else next(iter(j["reasoning"].values()))
        cite = r["citations"]
        lines += [
            f"- **`{r['id']}`** ({r['domain']}, {r['difficulty']}) — "
            f"correctness {j['correctness']}, verdict {j['verdict']}, "
            f"hallucination {str(j['hallucination']).lower()}, "
            f"citation accuracy {cite['accuracy']:.2f} "
            f"({cite['support'] or 'no resolvable citation'}). Judge: {reasoning}",
        ]

    lines += ["", "## Improvement suggestions (prioritized)", ""]
    lines += [f"{i}. {text}" for i, text in enumerate(suggestions(metrics), 1)]
    lines += [
        "",
        "---",
        "*Generated by `evalharness` — numeric grades per category are in "
        "`judgments.jsonl` (per question) and `metrics.json` (aggregates) for "
        "automated comparison across runs.*",
        "",
    ]
    return "\n".join(lines)

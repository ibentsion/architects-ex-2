"""Citation accuracy — does the cited evidence actually establish the answer?

Each cited {file, page} is resolved to the real corpus page (see
`evalharness.pages`) and an LLM judge rules whether the cited pages establish
the ground-truth answer. The same fact appears in several corpus documents, so
there is no fixed list of "correct" sources: any page that truly establishes
the fact earns credit.

    citation_accuracy = credit x (resolved / cited)

    credit  = 1.0 fully / 0.5 partially / 0.0 not_at_all (judge's ruling)
    cited   = citations the answer made; resolved = those that point at a real
              corpus page. A citation pointing at a nonexistent file or page
              earns nothing and dilutes the rest.

An answer that cites nothing scores 0.0 — including a refusal, which
establishes nothing either.

`ground_truth_sources` is NOT the scoring target. It records where each
reference answer was authored from, and is kept here only as an unscored
retrieval diagnostic (`gt_source_hit`).
"""
from .pages import files_match

CREDIT = {"fully": 1.0, "partially": 0.5, "not_at_all": 0.0}


def resolve_citations(citations: list, store) -> list:
    """Resolve an answer's citations against the corpus.

    Returns one entry per citation, in order: {file, page, text, invalid_reason}
    where exactly one of `text` / `invalid_reason` is set.
    """
    out = []
    for citation in citations or []:
        if not isinstance(citation, dict):
            continue
        file, page = citation.get("file", ""), citation.get("page")
        text, reason = store.resolve(file, page)
        out.append({"file": file, "page": page, "text": text,
                    "invalid_reason": reason})
    return out


def score_citations(resolved: list, judgment: dict | None) -> dict:
    """Score one answer's citations from its resolution + the judge's ruling.

    `resolved` is the full list from `resolve_citations`; `judgment` is the
    aggregated citation judgment over the *valid* subset (None when nothing
    resolved, so no judge ran).
    """
    valid = [c for c in resolved if c["invalid_reason"] is None]
    invalid_reasons = [c["invalid_reason"] for c in resolved if c["invalid_reason"]]
    score = {
        "cited_count": len(resolved),
        "valid_count": len(valid),
        "invalid_count": len(invalid_reasons),
        "invalid_reasons": invalid_reasons,
        "support": None,
        "credit": None,
        "accuracy": 0.0,
        "labels": [],
    }
    if judgment is None or "error" in judgment:
        # Nothing cited, nothing resolvable, or every judge failed. The first
        # two are honest zeros; a judge failure is flagged so it can't be read
        # as the system's fault.
        score["judge_failed"] = bool(valid)
        return score

    credit = CREDIT[judgment["citation_support"]]
    score["support"] = judgment["citation_support"]
    score["credit"] = credit
    score["accuracy"] = credit * len(valid) / len(resolved)
    # Labels come back per *valid* citation; map them onto the cited list so a
    # label lines up with the citation the system actually made.
    labels = list(judgment["citation_labels"])
    score["labels"] = [
        None if c["invalid_reason"] else (labels.pop(0) if labels else None)
        for c in resolved
    ]
    return score


def gt_source_hit(citations: list, source_groups: list) -> dict:
    """Unscored diagnostic: did the answer cite the sources the reference
    answer was authored from? Useful for debugging retrieval — a low hit rate
    with high `accuracy` just means the system found the fact elsewhere."""
    citations = [c for c in (citations or []) if isinstance(c, dict)]

    def hits(source):
        return any(files_match(c.get("file", ""), source.get("file", ""))
                   and c.get("page") == source.get("page") for c in citations)

    total = len(source_groups)
    satisfied = sum(1 for group in source_groups
                    if any(hits(s) for s in group.get("any_of", [])))
    return {
        "groups_total": total,
        "groups_hit": satisfied,
        "hit_rate": satisfied / total if total else None,
    }

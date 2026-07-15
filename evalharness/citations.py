"""Deterministic citation scoring — no LLM involved.

Ground truth per question is a list of source *groups*; each group is an
`any_of` list of acceptable {file, page} sources. Citing any member of a group
satisfies that group (recall); a cited item that matches no acceptable source
in any group hurts precision.
"""
import unicodedata


def _norm_file(path: str) -> str:
    """Normalize a file reference for comparison: NFC unicode, lowercase,
    forward slashes, no leading './'."""
    path = unicodedata.normalize("NFC", str(path)).strip().lower()
    path = path.replace("\\", "/").lstrip("./")
    return path


def _files_match(cited: str, truth: str) -> bool:
    """Exact normalized match, or one path is a trailing path-suffix of the
    other (systems may cite a bare filename or a longer absolute path)."""
    a, b = _norm_file(cited), _norm_file(truth)
    if not a or not b:
        return False
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def _citation_matches(citation: dict, source: dict) -> bool:
    if not _files_match(citation.get("file", ""), source.get("file", "")):
        return False
    return citation.get("page") == source.get("page")


def score_citations(citations: list, source_groups: list) -> dict:
    """Score one answer's citations against the question's source groups.

    Returns groups_total/groups_satisfied, recall (fraction of groups
    satisfied — full credit requires satisfying every group), and precision
    (fraction of cited items that match some acceptable source; None when
    nothing was cited).
    """
    citations = [c for c in (citations or []) if isinstance(c, dict)]
    all_sources = [s for g in source_groups for s in g.get("any_of", [])]

    satisfied = sum(
        1 for group in source_groups
        if any(_citation_matches(c, s) for s in group.get("any_of", []) for c in citations)
    )
    matched_citations = sum(
        1 for c in citations if any(_citation_matches(c, s) for s in all_sources)
    )
    total = len(source_groups)
    return {
        "groups_total": total,
        "groups_satisfied": satisfied,
        "recall": satisfied / total if total else None,
        "precision": matched_citations / len(citations) if citations else None,
        "cited_count": len(citations),
    }

"""The wire shape both UI views speak, and the two adapters onto it.

The Live view (a fresh ``rag.types.Answer``) and the QA-History view (a line
from one of the repo's answer/judgment JSONLs) render with the SAME components,
so they must produce the same model. The JSONLs disagree with each other about
field names — ``category`` vs ``domain``, ``cost_estimate`` vs ``cost_usd`` —
and every one of those disagreements is absorbed here, once.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote as urlquote

from pydantic import BaseModel, Field

from rag.types import Answer


class SupportCitation(BaseModel):
    """One source card. ``thumbnail_url`` is set only for paged (PDF) sources;
    TXT corpus pages have no page image, they get a text preview instead."""

    id: str
    file_name: str
    page_number: int | None = None
    content_preview: str | None = None
    thumbnail_url: str | None = None


class JudgeGrades(BaseModel):
    """The committee's verdict on one answer (eval_results/**/judgments.jsonl).
    ``reasoning`` is a per-judge map."""

    correctness: float | None = None
    completeness: float | None = None
    conversational_quality: float | None = None
    verdict: str | None = None
    hallucination: bool | None = None
    reasoning: dict[str, str] | None = None


class SupportPair(BaseModel):
    """A question and what the system said about it — live or replayed."""

    id: str
    question: str | None = Field(None, description="None when no reference set has this id")
    answer: str | None = Field(None, description="None for a judgments-only dataset")
    citations: list[SupportCitation] = Field(default_factory=list)
    domain: str | None = None
    confidence: float | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    trace: list[dict[str, Any]] | None = None
    classification: dict[str, Any] | None = None
    difficulty: str | None = None
    reference_answer: str | None = None
    judgment: JudgeGrades | None = None


def _first(*values: Any) -> Any:
    """First value that is actually present. Not ``or``: a 0.0 cost and a 0.0
    confidence are real measurements, not missing ones."""
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _resolve_preview(file_name: str, page: int | None, quote: str | None) -> str | None:
    """Citation preview text. Precedence — defined here and nowhere else:
    the record's own quote, then the resolved corpus page, then nothing.

    The page number matters: a preview of the wrong page is worse than none.
    """
    if quote:
        return quote
    from webapi import corpus_view  # local: corpus/ is gitignored and optional

    return corpus_view.preview_text(file_name, page)


def _citations(raw: list[dict[str, Any]] | None, pair_id: str) -> list[SupportCitation]:
    cards: list[SupportCitation] = []
    for i, citation in enumerate(raw or []):
        if not isinstance(citation, dict):
            continue
        file_name = str(citation.get("file") or "")
        page = citation.get("page")
        thumbnail = (
            f"/api/citation/thumbnail?file={urlquote(file_name, safe='')}&page={page}"
            if page is not None
            else None
        )
        cards.append(
            SupportCitation(
                id=f"{pair_id}:{i}",
                file_name=file_name,
                page_number=page,
                content_preview=_resolve_preview(file_name, page, citation.get("quote")),
                thumbnail_url=thumbnail,
            )
        )
    return cards


def answer_to_pair(answer: Answer, *, pair_id: str, question: str | None) -> SupportPair:
    """Live path: the engine's own :class:`~rag.types.Answer`."""
    return SupportPair(
        id=pair_id,
        question=question,
        answer=answer.text,
        citations=_citations(
            [c.model_dump() for c in answer.citations], pair_id
        ),
        domain=answer.category,
        confidence=answer.confidence,
        latency_ms=answer.latency_ms,
        cost_usd=answer.cost_estimate,
        trace=answer.trace,
        classification=(
            answer.classification.model_dump() if answer.classification is not None else None
        ),
    )


def record_to_pair(
    record: dict[str, Any],
    *,
    pair_id: str,
    question_record: dict[str, Any] | None = None,
    judgment: dict[str, Any] | None = None,
) -> SupportPair:
    """Offline path: one line of an answer JSONL (``record``), optionally
    joined to its reference question and its judgment record.

    ``record`` may be empty — a judged run whose answers file is unresolvable
    still renders, with ``answer=None``.
    """
    question_record = question_record or {}
    judgment = judgment or {}
    grades = judgment.get("judgment") or None

    return SupportPair(
        id=pair_id,
        question=question_record.get("question"),
        answer=record.get("answer"),
        citations=_citations(
            record.get("citations") or judgment.get("citations"), pair_id
        ),
        # `category` is the CLI/agent name, `domain` the contract's; the
        # question's own domain is the last resort.
        domain=_first(
            record.get("category"), record.get("domain"),
            judgment.get("domain"), question_record.get("domain"),
        ),
        confidence=record.get("confidence"),
        latency_ms=_first(record.get("latency_ms"), judgment.get("latency_ms")),
        cost_usd=_first(record.get("cost_usd"), record.get("cost_estimate")),
        trace=record.get("trace"),
        classification=record.get("classification"),
        difficulty=_first(judgment.get("difficulty"), question_record.get("difficulty")),
        reference_answer=question_record.get("ground_truth_answer"),
        judgment=JudgeGrades(**{
            key: grades.get(key) for key in JudgeGrades.model_fields
        }) if grades else None,
    )

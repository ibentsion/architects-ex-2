"""Shared pydantic data contracts (rag_plan.md §8).

``Citation`` mirrors ``contract.py``'s ``Citation`` field-for-field so the
Stage-3 FastAPI wiring is a trivial adapter. ``Answer`` maps 1:1 onto
``contract.py``'s ``AskResponse`` (``text``→``answer``, ``category``→``domain``,
``cost_estimate``→``cost_usd``).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """One indexed unit of corpus text (rag_plan.md §8).

    Invariants (unit-tested in test_chunking.py):
      * PDF chunks have ``page >= 1`` (Docling ``prov.page_no`` is 1-based).
      * TXT chunks have ``page is None`` (matches ground-truth ``page: null``).
      * ``category`` is one of the 12 corpus directory names.
      * ``file`` starts with ``"{category}/"``.
      * ``file`` is the category-relative POSIX path, NFC-normalized, no
        leading ``./`` — must byte-match ``reference_questions.json``'s
        ``file`` field (e.g. ``apartment/files/הודעה-על-תקופת-התיישנות.pdf``).
      * ``text`` is ORIGINAL text (never normalized); tables as Markdown.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        ..., description='"{file}#p{page}#c{n}" — stable id joining dense/sparse indexes'
    )
    file: str = Field(
        ..., description="Category-relative POSIX path, NFC — exactly as in reference_questions.json"
    )
    page: int | None = Field(None, description="1-based for PDFs; None for TXT")
    category: str = Field(..., description='Corpus dir name, e.g. "apartment"')
    text: str = Field(..., description="Original (never normalized) text; tables as Markdown")
    source_url: str | None = Field(None, description="From corpus/manifest.json (data only, never fetched)")
    chunker: str = Field(..., description="Provenance: which chunking strategy produced this chunk")


class RetrievedChunk(BaseModel):
    """A chunk plus the per-stage retrieval scores it accumulated.

    Scores are optional because a chunk may reach fusion from only one
    backend, and reranking happens only to fusion survivors.
    """

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    dense_score: float | None = Field(None, description="Vector-store similarity score")
    sparse_score: float | None = Field(None, description="BM25 score")
    rrf_score: float | None = Field(None, description="Reciprocal-rank-fusion score")
    rerank_score: float | None = Field(None, description="CrossEncoder sigmoid relevance (also the gate signal)")


class Citation(BaseModel):
    """Mirrors contract.py's Citation — do not diverge."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(..., description="Source document path or URL")
    page: int | None = Field(None, description="1-based page number for PDFs")
    quote: str | None = Field(None, description="The supporting passage (optional but persuasive)")


class Answer(BaseModel):
    """Query-pipeline output (rag_plan.md §6 stage 8).

    Field-compatible with contract.py's AskResponse:
    text→answer, category→domain, cost_estimate→cost_usd.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="The answer, in the language of the question")
    citations: list[Citation] = Field(default_factory=list)
    category: str | None = Field(None, description="Category filter used / routed domain")
    confidence: float | None = Field(None, ge=0, le=1, description="max rerank sigmoid; 0.0 on gate-fail fallback")
    latency_ms: float | None = None
    cost_estimate: float | None = Field(None, description="Estimated $ cost of answering (LLM + embedding calls)")
    citation_fallback: bool = Field(
        False,
        description="True when the LLM's citations failed validation and the retrieved top-3 {file,page} were attached instead",
    )
    retrieved: list[RetrievedChunk] = Field(
        default_factory=list, description="Chunks that survived the relevance gate (retrieval debugging / eval)"
    )
    retrieval_stats: dict[str, dict[str, int]] | None = Field(
        None,
        description="Per-stage {n_chunks, n_documents} (dense/sparse/fused/gated) — "
        "measures how much each stage filters/re-ranks the candidate pool",
    )
    max_tokens_hit: bool = Field(
        False, description="True if any LLM call in this answer hit max_tokens (possible truncation)"
    )
    n_retries: int = Field(0, description="Number of corrective-nudge LLM retries used for this answer")
    retrieval_ms: float | None = Field(None, description="Wall time spent in Retriever.retrieve() (dense+sparse+rerank+gate)")
    generation_ms: float | None = Field(None, description="Wall time spent in Generator.generate() (0.0 on gate-fail, no LLM call)")

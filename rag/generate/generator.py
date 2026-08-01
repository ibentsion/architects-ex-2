"""Context assembly + tf_client LLM call + one retry on citation failure
(rag_plan.md §6 stages 5-7).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tf_client import chat as tf_chat

from rag.generate.citations import parse_sources_block, validate_citations
from rag.generate.prompts import CORRECTIVE_NUDGE, FALLBACK_TEXT, PROMPT_REGISTRY
from rag.types import Citation, RetrievedChunk

logger = logging.getLogger(__name__)

#: ~6k token budget for the assembled context (rag_plan.md §6 stage 5).
CONTEXT_TOKEN_BUDGET = 6000

#: Number of retrieved {file,page} attached when citations still fail after
#: the retry (rag_plan.md §6 stage 7).
FALLBACK_CITATION_COUNT = 3


def _approx_tokens(text: str) -> int:
    """Cheap ~4-chars/token approximation. Used only for the context trim
    budget and the cost estimate — tf_client.chat's ``return_usage=True``
    gives exact counts for the actual LLM call; this is for text that never
    reaches the model (candidate chunks trimmed from the tail)."""
    return max(1, len(text) // 4)


def _chunk_header(chunk: Any) -> str:
    page = chunk.page if chunk.page is not None else "-"
    return f"[מקור: {chunk.file} | עמוד: {page} | תחום: {chunk.category}]"


def assemble_context(
    retrieved: list[RetrievedChunk], token_budget: int = CONTEXT_TOKEN_BUDGET
) -> str:
    """Render surviving chunks with machine-readable headers (rag_plan.md §6
    stage 5). ``retrieved`` is assumed already in rerank-score order (the
    gate returns it sorted desc). Token budget guard (~6k tokens) trims the
    tail — always keeps at least the first chunk even if it alone exceeds
    the budget (nothing to answer from otherwise)."""
    blocks: list[str] = []
    used = 0
    for r in retrieved:
        block = f"{_chunk_header(r.chunk)}\n{r.chunk.text}"
        block_tokens = _approx_tokens(block)
        if blocks and used + block_tokens > token_budget:
            break
        blocks.append(block)
        used += block_tokens
    return "\n\n".join(blocks)


@dataclass
class GenerationResult:
    text: str
    citations: list[Citation]
    citation_fallback: bool
    cost_estimate: float
    tokens: dict[str, int]
    max_tokens_hit: bool = False
    n_retries: int = 0


def _zero_tokens() -> dict[str, int]:
    return {"prompt": 0, "completion": 0}


def build_generator(config: Any) -> "Generator":
    """Construct a Generator from a validated RagConfig's generation block
    (shared by the query CLI's QueryEngine and the agent engine)."""
    generation = config.generation
    return Generator(
        model=generation.model,
        prompt=generation.prompt,
        max_tokens=generation.max_tokens,
        temperature=generation.temperature,
        retry_on_citation_failure=generation.retry_on_citation_failure,
        extra_params=generation.extra_params,
    )


class Generator:
    """``generate(question, retrieved)`` — context assembly, tf_client.chat
    call, citation validation with one corrective retry
    (``retry_on_citation_failure``). Callers must have already applied the
    relevance gate; an empty ``retrieved`` list returns the fallback text
    with ZERO LLM calls (rag_plan.md §6 stage 4) — defense in depth even
    though the query engine is expected to short-circuit before this point.
    """

    def __init__(
        self,
        *,
        model: str,
        prompt: str = "grounded_cite",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        retry_on_citation_failure: bool = True,
        extra_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if prompt not in PROMPT_REGISTRY:
            raise ValueError(
                f"Unknown prompt variant '{prompt}'. Available: {sorted(PROMPT_REGISTRY)}"
            )
        if kwargs:
            raise TypeError(f"Unknown generation params: {sorted(kwargs)}")
        self.model = model
        self.prompt_name = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry_on_citation_failure = retry_on_citation_failure
        self.extra_params = extra_params or {}

    def _call(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int], float, str | None]:
        text, usage, cost = tf_chat(
            messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            quiet=False,
            return_usage=True,
            **self.extra_params,
        )
        finish_reason = usage.pop("finish_reason", None)
        if finish_reason == "length":
            logger.warning(
                "LLM call hit max_tokens (%d) — answer may be truncated (model=%s)",
                self.max_tokens,
                self.model,
                extra={
                    "rag_event": "generation_max_tokens_hit",
                    "rag_detail": {"max_tokens": self.max_tokens, "model": self.model},
                },
            )
        if text is None:
            # Reasoning models (gpt-oss, Nemotron, ...) can spend the entire
            # max_tokens budget on hidden chain-of-thought and return no
            # visible content at all -- always a "length" finish_reason.
            # Normalize to "" so downstream string ops (FALLBACK_TEXT in
            # text, parse_sources_block) never crash on None; the citation-
            # retry path treats it exactly like any other citation-less reply.
            logger.warning(
                "LLM returned no content (reasoning budget exhausted?) — model=%s",
                self.model,
            )
            text = ""
        return text, usage, cost, finish_reason

    def generate(
        self, question: str, retrieved: list[RetrievedChunk]
    ) -> GenerationResult:
        if not retrieved:
            return GenerationResult(
                text=FALLBACK_TEXT,
                citations=[],
                citation_fallback=False,
                cost_estimate=0.0,
                tokens=_zero_tokens(),
            )

        system_prompt = PROMPT_REGISTRY[self.prompt_name]()
        context = assemble_context(retrieved)
        user_message = f"קטעי מקור:\n\n{context}\n\nשאלה: {question}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        text, usage, cost, finish_reason = self._call(messages)
        total_tokens = dict(usage)
        total_cost = cost
        max_tokens_hit = finish_reason == "length"
        n_retries = 0

        if FALLBACK_TEXT in text:
            # The model itself declared it can't answer -- a legitimate
            # refusal, not a citation-validation failure. No retry, no
            # fabricated top-3 citations attached.
            return GenerationResult(
                text=text,
                citations=[],
                citation_fallback=False,
                cost_estimate=total_cost,
                tokens=total_tokens,
                max_tokens_hit=max_tokens_hit,
                n_retries=n_retries,
            )

        citations = validate_citations(parse_sources_block(text), retrieved)

        if not citations and self.retry_on_citation_failure:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": CORRECTIVE_NUDGE})
            logger.info("Citation validation failed; retrying with corrective nudge")
            text, usage, cost, finish_reason = self._call(messages)
            n_retries += 1
            max_tokens_hit = max_tokens_hit or finish_reason == "length"
            total_tokens = {k: total_tokens[k] + usage[k] for k in total_tokens}
            total_cost += cost
            if FALLBACK_TEXT in text:
                return GenerationResult(
                    text=text,
                    citations=[],
                    citation_fallback=False,
                    cost_estimate=total_cost,
                    tokens=total_tokens,
                    max_tokens_hit=max_tokens_hit,
                    n_retries=n_retries,
                )
            citations = validate_citations(parse_sources_block(text), retrieved)

        if citations:
            return GenerationResult(
                text=text,
                citations=citations,
                citation_fallback=False,
                cost_estimate=total_cost,
                tokens=total_tokens,
                max_tokens_hit=max_tokens_hit,
                n_retries=n_retries,
            )

        # Still failing (or missing/all invalid, retry disabled): keep the
        # answer text, attach the top-3 retrieved {file,page} as citations.
        logger.warning("Citation validation failed after retry; attaching top-3 retrieved as fallback citations")
        fallback_citations = [
            Citation(file=r.chunk.file, page=r.chunk.page)
            for r in retrieved[:FALLBACK_CITATION_COUNT]
        ]
        return GenerationResult(
            text=text,
            citations=fallback_citations,
            citation_fallback=True,
            cost_estimate=total_cost,
            tokens=total_tokens,
            max_tokens_hit=max_tokens_hit,
            n_retries=n_retries,
        )

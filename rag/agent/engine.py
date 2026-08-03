"""AgentEngine: hybrid fast-path agentic answering (quick task 260801-002).

Same answer contract as the query CLI's QueryEngine (``answer``,
``_answer`` -> (Answer, tokens), ``close``), selected with ``--engine agent``.

Flow per query:
  1. Classify (orchestrator model, one quick call): sub-questions, category
     tags, ``needs_calculation``/``dependent`` flags.
  2. Prefetch: retrieve every sub-question CONCURRENTLY (thread pool;
     category filter when a sub-question maps to exactly one category);
     pool + dedupe the gated chunks.
  3. Only if the query needs calculation or dependent reasoning: a native
     tool-calling loop on the orchestrator model with two tools —
     ``retrieve`` (follow-up evidence) and ``calculate`` (ALL arithmetic;
     the LLM writes expressions, rag/agent/calculator.py computes them).
     Tool calls within one turn also execute concurrently. Hops capped by
     ``harness.max_hops``; ANY loop failure degrades to the prefetched pool.
  4. Synthesize ONE final answer via the standard Generator (grounded_cite
     prompt + citation validation) over the pooled evidence; calculator
     results are appended to the question so the answer uses computed numbers
     verbatim. Difficulty routing picks the model: easy/medium single-topic
     questions go to ``harness.fast_synthesis_model``, everything harder to
     ``generation.model``.

Serial by necessity: classification (everything routes on it), agent hops
(hop N+1 consumes hop N results), calculation (after its inputs are
retrieved), synthesis (after all evidence). Concurrent: per-sub-question
retrievals, tool calls within a hop.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from tf_client import chat as tf_chat

from rag.agent.calculator import CalculationError, calculate
from rag.classify import CATEGORIES, build_classifier, expand_families, build_hint
from rag.generate.generator import Generator, build_generator
from rag.generate.prompts import FALLBACK_TEXT
from rag.retrieve.retriever import Retriever, load_retriever
from rag.types import Answer, Classification, RetrievedChunk

logger = logging.getLogger(__name__)

#: Per-chunk character cap when rendering evidence into orchestrator context
#: (the orchestrator only needs enough to spot numbers/gaps, not full pages).
ORCHESTRATOR_CHUNK_CHARS = 1500

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Fetch relevant Hebrew insurance policy passages from the corpus "
            "for a search query. Use when the gathered evidence is missing a fact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query in Hebrew"},
                    "category": {
                        "type": "string",
                        "enum": sorted(CATEGORIES),
                        "description": "optional category filter",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression (+ - * / // % ** and "
            "round/abs/min/max). You MUST use this for ALL arithmetic on numbers — "
            "never compute numbers yourself.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

#: System-prompt addendum for synthesis when calculator results exist — the
#: base grounded_cite rules forbid numbers that aren't in the sources, so the
#: computed values must be explicitly authorized.
CALCULATION_ADDENDUM = """כלל נוסף: בהודעת המשתמש מופיע בלוק "תוצאות חישוב" — חישובים אריתמטיים שבוצעו על-ידי מחשבון המערכת על נתונים מהשאלה ומקטעי המקור. כאשר חלק מהשאלה דורש חישוב על נתונים שהלקוח עצמו סיפק, תוצאת המחשבון היא התשובה לאותו חלק — השתמש בה כפי שהיא גם אם הסכום אינו מופיע בקטעי המקור, ואל תסרב לענות על חלק זה. אין לציין מקור עבור תוצאת חישוב בבלוק המקורות, ואין לחשב אותה מחדש בעצמך."""

ORCHESTRATOR_SYSTEM = """You orchestrate evidence gathering for a Hebrew insurance-support answer \
(Harel Insurance corpus). Evidence passages for the customer's question are provided; you have two tools:
- retrieve(query, category?): fetch more passages when a needed fact is missing.
- calculate(expression): evaluate arithmetic. You MUST use it for ALL arithmetic on numbers \
from the evidence or the question — never compute numbers yourself.

Another model writes the final customer answer from the gathered evidence and your calculation \
results; your ONLY job is to make sure every needed fact is retrieved and every needed number is \
computed. Work in as few turns as possible. When everything needed is available, reply with the \
single word DONE. Do not answer the customer yourself."""


#: A live consumer of trace records (webapi/agent_app.py streams them as SSE).
EventSink = Callable[[dict[str, Any]], None]


def _emit(trace: list[dict[str, Any]], sink: EventSink | None, record: dict[str, Any]) -> None:
    """Record one pipeline step: append it to ``trace``, then hand it to
    ``sink`` (when there is one) so a live consumer sees the step while the
    next one is still running.

    The sink receives the SAME dict instance that stays in ``answer.trace`` —
    consumers must treat it as read-only; mutating it corrupts the answer.

    A sink failure never fails the answer: a browser disconnecting mid-stream
    is a normal event, not a pipeline error.
    """
    trace.append(record)
    if sink is None:
        return
    try:
        sink(record)
    except Exception as exc:
        logger.warning("event_sink raised (%s: %s) — continuing", type(exc).__name__, exc)


def _render_evidence(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "(אין קטעים — לא נמצאו מקורות רלוונטיים)"
    blocks = []
    for r in retrieved:
        page = r.chunk.page if r.chunk.page is not None else "-"
        text = r.chunk.text[:ORCHESTRATOR_CHUNK_CHARS]
        blocks.append(f"[{r.chunk.file} | עמוד {page}]\n{text}")
    return "\n\n".join(blocks)


def _assistant_message_dict(message: Any) -> dict[str, Any]:
    """Serialize a litellm assistant message (possibly with tool_calls) back
    into a plain dict for the follow-up request."""
    result: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if getattr(message, "tool_calls", None):
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return result


def build_fast_generator(config: Any) -> Generator | None:
    """Second synthesis Generator on ``harness.fast_synthesis_model`` (None
    when the routing is disabled). Everything except model/max_tokens/
    extra_params is the generation block's — the answer contract (prompt
    variant, temperature, citation retry) must not depend on which model
    the query routed to."""
    harness = config.harness
    if not harness.fast_synthesis_model:
        return None
    generation = config.generation
    return Generator(
        model=harness.fast_synthesis_model,
        prompt=generation.prompt,
        max_tokens=harness.fast_synthesis_max_tokens,
        temperature=generation.temperature,
        retry_on_citation_failure=generation.retry_on_citation_failure,
        extra_params=harness.fast_synthesis_extra_params,
    )


class AgentEngine:
    """See module docstring. Construction mirrors QueryEngine: manifest
    verification + lazy component loads happen inside load_retriever."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.harness = config.harness
        self.classifier = build_classifier(config)
        self.retriever: Retriever = load_retriever(config)
        self.generator = build_generator(config)
        self.fast_generator = build_fast_generator(config)

    # ------------------------------------------------------------------ #
    # Public contract (same as QueryEngine)
    # ------------------------------------------------------------------ #

    def answer(
        self, question: str, category: str | None = None, *, event_sink: EventSink | None = None
    ) -> Answer:
        return self._answer(question, category, event_sink=event_sink)[0]

    def _answer(
        self,
        question: str,
        category: str | None = None,
        *,
        event_sink: EventSink | None = None,
    ) -> tuple[Answer, dict[str, int] | None]:
        # category is part of the engine interface but the agent predicts its
        # own filters; the CLI rejects --category with --engine agent.
        # event_sink is keyword-only and optional: every existing positional
        # caller (contract.py, rag/cli/query.py, evalharness) is unaffected.
        sink = event_sink
        trace: list[dict[str, Any]] = []
        total_tokens = {"prompt": 0, "completion": 0}

        # Retrieval evidence first: the classifier tags the query far better
        # when it can see which corpus slices the raw question actually hits
        # (260802-003 — correct filters 73% -> 81%). Fuse-only, so this costs
        # an embedding + BM25 and no cross-encoder pass.
        t_hint = time.monotonic()
        hint, hint_summary = build_hint(self.retriever, question)
        hint_ms = (time.monotonic() - t_hint) * 1000
        _emit(
            trace,
            sink,
            {
                "step": "hint",
                "ms": round(hint_ms),
                "top_category": hint_summary["top_category"],
                "top_share": hint_summary["top_share"],
                "n_hits": hint_summary["n_hits"],
            },
        )

        t0 = time.monotonic()
        classification = self.classifier.classify(question, hint=hint)
        classification_ms = (time.monotonic() - t0) * 1000
        _emit(
            trace,
            sink,
            {
                "step": "classify",
                "ms": round(classification_ms),
                "mode": classification.mode,
                "categories": classification.categories,
                "sub_questions": [sq.question for sq in classification.sub_questions],
                "needs_calculation": classification.needs_calculation,
                "dependent": classification.dependent,
                "difficulty": classification.estimated_difficulty,
            },
        )

        # Prefetch all sub-questions concurrently.
        t1 = time.monotonic()
        requests = [
            (sq.question, self._filter_categories(sq.categories))
            for sq in classification.sub_questions
        ]
        pool: dict[str, RetrievedChunk] = {}
        stats_sum: dict[str, dict[str, int]] = {}
        for (sub_q, sub_cat), (results, stats, retried) in zip(requests, self._retrieve_many(requests)):
            self._merge_pool(pool, results)
            self._merge_stats(stats_sum, stats)
            entry = {
                "step": "retrieve",
                "phase": "prefetch",
                "query": sub_q,
                "category": sub_cat,
                "n_gated": len(results),
            }
            if retried:
                entry["retried_unfiltered"] = True
            _emit(trace, sink, entry)
        retrieval_ms = (time.monotonic() - t1) * 1000

        # Agent loop only when the fast paths can't handle the query.
        calc_results: list[tuple[str, float]] = []
        loop_cost = 0.0
        if classification.needs_calculation or classification.dependent:
            loop_cost = self._run_loop(
                question, classification, pool, calc_results, trace, sink, total_tokens
            )

        retrieved = sorted(
            pool.values(), key=lambda r: (-(r.rerank_score or 0.0), r.chunk.chunk_id)
        )
        routed_category = (
            classification.categories[0] if len(classification.categories) == 1 else None
        )
        base_cost = classification.cost_estimate + loop_cost

        if not retrieved:
            answer = Answer(
                text=FALLBACK_TEXT,
                citations=[],
                category=routed_category,
                confidence=0.0,
                latency_ms=classification_ms + retrieval_ms,
                cost_estimate=base_cost,
                retrieved=[],
                retrieval_stats=stats_sum or None,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                classification=classification,
                classification_ms=classification_ms,
                trace=trace,
            )
            return answer, None

        synth_question = question
        addendum = None
        if calc_results:
            lines = "\n".join(f"- {expr} = {value:g}" for expr, value in calc_results)
            synth_question = f"{question}\n\nתוצאות חישוב:\n{lines}"
            addendum = CALCULATION_ADDENDUM

        generator, routed_fast = self._select_generator(classification)
        t2 = time.monotonic()
        result = generator.generate(synth_question, retrieved, system_addendum=addendum)
        generation_ms = (time.monotonic() - t2) * 1000
        _emit(
            trace,
            sink,
            {
                "step": "synthesize",
                "ms": round(generation_ms),
                "model": generator.model,
                "fast_synthesis": routed_fast,
            },
        )
        for key in total_tokens:
            total_tokens[key] += result.tokens.get(key, 0)

        answer = Answer(
            text=result.text,
            citations=result.citations,
            category=routed_category,
            confidence=max((r.rerank_score or 0.0) for r in retrieved),
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_estimate=result.cost_estimate + base_cost,
            citation_fallback=result.citation_fallback,
            retrieved=retrieved,
            retrieval_stats=stats_sum or None,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            max_tokens_hit=result.max_tokens_hit,
            n_retries=result.n_retries,
            classification=classification,
            classification_ms=classification_ms,
            trace=trace,
        )
        return answer, total_tokens

    def close(self) -> None:
        self.retriever.close()

    # ------------------------------------------------------------------ #
    # Synthesis routing
    # ------------------------------------------------------------------ #

    def _select_generator(self, classification: Classification) -> tuple[Any, bool]:
        """Pick the synthesis model. The strong ``generation.model`` earns its
        cost only on hard questions (committee eval: 6.44 vs 4.12 on hard,
        a tie on easy/medium), so everything that is hard, spans categories,
        needs arithmetic, or has dependent parts goes there; the rest goes to
        the fast model. Returns (generator, routed_to_fast)."""
        if self.fast_generator is None:
            return self.generator, False
        needs_strong = (
            classification.estimated_difficulty == "hard"
            or classification.mode == "multi"
            or classification.needs_calculation
            or classification.dependent
        )
        if needs_strong:
            return self.generator, False
        return self.fast_generator, True

    # ------------------------------------------------------------------ #
    # Concurrency helpers
    # ------------------------------------------------------------------ #

    def _filter_categories(self, categories: list[str]) -> str | list[str] | None:
        """The retrieval filter for one sub-question's tags, under
        ``harness.category_filter``.

        ``single`` is the historical behaviour and the reason a filter is a
        coin flip: one wrong tag filters the answer out of reach entirely
        (260802-004 measured +7/-6 groups against no filter, where a gold tag
        is +9/-0). ``family`` widens the filter to the categories that tag is
        confused with, which is the cheapest way to keep the right one in.
        """
        policy = self.harness.category_filter
        if policy == "none" or not categories:
            return None
        if policy == "single":
            return categories[0] if len(categories) == 1 else None
        if policy == "family":
            return expand_families(categories)
        return list(categories)

    def _retrieve_sub(
        self, question: str, category: str | list[str] | None
    ) -> tuple[list[RetrievedChunk], dict[str, dict[str, int]], bool]:
        """One sub-question retrieval with the unfiltered-retry policy: a
        category filter must never be the reason for an empty pool (the
        classifier's tags are ~76-88% accurate — a wrong single tag used to
        turn into a hard refusal). Returns (results, stats, retried)."""
        results, stats = self.retriever.retrieve_with_stats(question, category=category)
        if results or category is None:
            return results, stats, False
        logger.info(
            "Category-filtered retrieval gated to zero (category=%s) — retrying unfiltered",
            category,
        )
        results, stats = self.retriever.retrieve_with_stats(question, category=None)
        return results, stats, True

    def _retrieve_many(
        self, requests: list[tuple[str, str | list[str] | None]]
    ) -> list[tuple[list[RetrievedChunk], dict[str, dict[str, int]], bool]]:
        """Run retrievals concurrently (embedding is HTTP-bound; the sparse
        stage self-serializes inside Retriever). Order preserved."""
        if len(requests) == 1:
            question, category = requests[0]
            return [self._retrieve_sub(question, category)]
        workers = min(self.harness.max_workers, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._retrieve_sub, question, category)
                for question, category in requests
            ]
            return [future.result() for future in futures]

    @staticmethod
    def _merge_pool(pool: dict[str, RetrievedChunk], results: list[RetrievedChunk]) -> None:
        for r in results:
            prev = pool.get(r.chunk.chunk_id)
            if prev is None or (r.rerank_score or 0.0) > (prev.rerank_score or 0.0):
                pool[r.chunk.chunk_id] = r

    @staticmethod
    def _merge_stats(
        stats_sum: dict[str, dict[str, int]], stats: dict[str, dict[str, int]]
    ) -> None:
        """Element-wise sum across retrievals (documents may double-count)."""
        for stage, counts in stats.items():
            acc = stats_sum.setdefault(stage, {"n_chunks": 0, "n_documents": 0})
            for key, value in counts.items():
                acc[key] += value

    # ------------------------------------------------------------------ #
    # Tool-calling loop
    # ------------------------------------------------------------------ #

    def _run_loop(
        self,
        question: str,
        classification: Classification,
        pool: dict[str, RetrievedChunk],
        calc_results: list[tuple[str, float]],
        trace: list[dict[str, Any]],
        sink: EventSink | None,
        total_tokens: dict[str, int],
    ) -> float:
        """Native tool-calling loop on the orchestrator model. Mutates
        ``pool``/``calc_results``/``trace``/``total_tokens`` in place and
        returns the loop's LLM cost. Never raises — any failure logs and
        degrades to the prefetched pool."""
        cost = 0.0
        try:
            evidence = _render_evidence(
                sorted(pool.values(), key=lambda r: -(r.rerank_score or 0.0))
            )
            sub_qs = "\n".join(f"- {sq.question}" for sq in classification.sub_questions)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {
                    "role": "user",
                    "content": f"שאלת הלקוח:\n{question}\n\nתת-שאלות מזוהות:\n{sub_qs}\n\n"
                    f"ראיות שנאספו:\n{evidence}",
                },
            ]
            for hop in range(self.harness.max_hops):
                t0 = time.monotonic()
                message, usage, call_cost = tf_chat(
                    messages,
                    model=self.harness.orchestrator_model,
                    max_tokens=self.harness.orchestrator_max_tokens,
                    temperature=0.0,
                    quiet=False,
                    return_message=True,
                    tools=TOOLS,
                    tool_choice="auto",
                    **self.harness.orchestrator_extra_params,
                )
                cost += call_cost
                for key in total_tokens:
                    total_tokens[key] += usage.get(key, 0)
                tool_calls = getattr(message, "tool_calls", None) or []
                _emit(
                    trace,
                    sink,
                    {
                        "step": "orchestrator",
                        "hop": hop,
                        "ms": round((time.monotonic() - t0) * 1000),
                        "n_tool_calls": len(tool_calls),
                        "content": (message.content or "")[:200],
                    },
                )
                if not tool_calls:
                    break
                messages.append(_assistant_message_dict(message))
                messages.extend(
                    self._execute_tool_calls(tool_calls, pool, calc_results, trace, sink)
                )
            else:
                logger.warning("Agent loop hit max_hops=%d — proceeding to synthesis", self.harness.max_hops)
        except Exception as exc:
            logger.warning(
                "Agent loop failed (%s: %s) — degrading to prefetched evidence",
                type(exc).__name__,
                exc,
            )
            _emit(trace, sink, {"step": "orchestrator", "error": f"{type(exc).__name__}: {exc}"})
        return cost

    def _execute_tool_calls(
        self,
        tool_calls: list[Any],
        pool: dict[str, RetrievedChunk],
        calc_results: list[tuple[str, float]],
        trace: list[dict[str, Any]],
        sink: EventSink | None,
    ) -> list[dict[str, Any]]:
        """Execute a turn's tool calls (concurrently when >1) and return the
        tool-role messages, in call order."""
        def run_one(tc: Any) -> tuple[str, dict[str, Any], list[RetrievedChunk]]:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                return f"error: invalid JSON arguments ({exc.msg})", {"error": "bad-args"}, []
            if tc.function.name == "calculate":
                expression = str(args.get("expression", ""))
                try:
                    value = calculate(expression)
                except CalculationError as exc:
                    return f"error: {exc}", {"expression": expression, "error": str(exc)}, []
                return str(value), {"expression": expression, "value": value}, []
            if tc.function.name == "retrieve":
                query = str(args.get("query", "")).strip()
                category = args.get("category")
                if category not in CATEGORIES:
                    category = None
                if not query:
                    return "error: empty query", {"error": "empty-query"}, []
                # The tool's category is a single name (the enum the model sees),
                # widened by the same policy the prefetch uses.
                results, _stats, retried = self._retrieve_sub(
                    query, self._filter_categories([category] if category else []))
                detail = {"query": query, "category": category, "n_gated": len(results)}
                if retried:
                    detail["retried_unfiltered"] = True
                return _render_evidence(results), detail, results
            return f"error: unknown tool {tc.function.name}", {"error": "unknown-tool"}, []

        if len(tool_calls) == 1:
            outcomes = [run_one(tool_calls[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.harness.max_workers, len(tool_calls))
            ) as executor:
                outcomes = list(executor.map(run_one, tool_calls))

        tool_messages: list[dict[str, Any]] = []
        for tc, (content, detail, results) in zip(tool_calls, outcomes):
            if tc.function.name == "calculate" and "value" in detail:
                calc_results.append((detail["expression"], detail["value"]))
            self._merge_pool(pool, results)  # main thread — no pool races
            _emit(trace, sink, {"step": tc.function.name, "phase": "loop", **detail})
            tool_messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": content}
            )
        return tool_messages

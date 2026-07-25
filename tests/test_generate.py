"""Generation + citation tests (rag_plan.md §9 test_generate.py list).

LLM mocked throughout (monkeypatched ``rag.generate.generator.tf_chat``) --
prompt variants contain the citation mandate + fallback instruction;
SOURCES-block parser on well-formed / RTL-punctuated / missing-block output;
validator rejects {file,page} not in the retrieved set; fabricated citation
retries once; gate-fail (empty retrieved) never calls the LLM.
"""
from __future__ import annotations

import pytest

from rag.generate.citations import parse_sources_block, validate_citations
from rag.generate.generator import CONTEXT_TOKEN_BUDGET, Generator, assemble_context
from rag.generate.prompts import FALLBACK_TEXT, PROMPT_REGISTRY, SOURCES_HEADER
from rag.types import Chunk, Citation, RetrievedChunk

ANCHOR_FILE = "apartment/files/הודעה-על-תקופת-התיישנות.pdf"


def make_chunk(file: str = ANCHOR_FILE, page: int | None = 1, category: str = "apartment",
                text: str = "תקופת ההתיישנות היא שלוש שנים.") -> Chunk:
    return Chunk(
        chunk_id=f"{file}#p{page}#c0",
        file=file,
        page=page,
        category=category,
        text=text,
        source_url=None,
        chunker="per_page",
    )


def make_retrieved(**overrides) -> RetrievedChunk:
    chunk = overrides.pop("chunk", None) or make_chunk(**{
        k: overrides.pop(k) for k in ("file", "page", "category", "text") if k in overrides
    })
    return RetrievedChunk(chunk=chunk, rerank_score=overrides.pop("rerank_score", 0.9), **overrides)


# --------------------------------------------------------------------------- #
# Prompt variants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(PROMPT_REGISTRY))
def test_prompt_variants_mandate_citations_and_fallback(name):
    text = PROMPT_REGISTRY[name]()
    assert "עמוד" in text  # citation mandate references page
    assert SOURCES_HEADER in text
    assert FALLBACK_TEXT in text
    assert "file" in text and "page" in text  # sources-block field names


def test_strict_extractive_adds_quote_requirement():
    text = PROMPT_REGISTRY["strict_extractive"]()
    assert "quote" in text
    assert PROMPT_REGISTRY["grounded_cite"]() in text  # extends the base rules


def test_few_shot_cite_has_two_worked_examples():
    text = PROMPT_REGISTRY["few_shot_cite"]()
    assert text.count(SOURCES_HEADER) >= 2  # base rules mention it + 1 worked example
    assert FALLBACK_TEXT in text
    assert ANCHOR_FILE in text  # worked example 1 uses the real anchor doc


# --------------------------------------------------------------------------- #
# SOURCES-block parser
# --------------------------------------------------------------------------- #


def test_parse_well_formed_sources_block():
    text = f"""התשובה היא כך וכך.

{SOURCES_HEADER}
- file: {ANCHOR_FILE} | page: 1"""
    citations = parse_sources_block(text)
    assert citations == [Citation(file=ANCHOR_FILE, page=1)]


def test_parse_multiple_lines_and_null_page():
    text = f"""תשובה.

{SOURCES_HEADER}
- file: {ANCHOR_FILE} | page: 1
- file: travel/files/notes.txt | page: -"""
    citations = parse_sources_block(text)
    assert citations == [
        Citation(file=ANCHOR_FILE, page=1),
        Citation(file="travel/files/notes.txt", page=None),
    ]


def test_parse_quote_variant():
    text = f"""תשובה.

{SOURCES_HEADER}
- file: {ANCHOR_FILE} | page: 1 | quote: "שלוש שנים מיום קרות מקרה הביטוח" """
    citations = parse_sources_block(text)
    assert citations == [
        Citation(file=ANCHOR_FILE, page=1, quote="שלוש שנים מיום קרות מקרה הביטוח")
    ]


def test_parse_tolerant_of_bidi_controls_and_punct_lookalikes():
    # LRM/RLM sprinkled around punctuation; maqaf instead of "-"; fullwidth
    # pipe/colon instead of ASCII.
    text = (
        f"תשובה.\n\n{SOURCES_HEADER}\n"
        f"-‏ file‏： {ANCHOR_FILE} ‎｜‎ page： ־"
    )
    citations = parse_sources_block(text)
    assert citations == [Citation(file=ANCHOR_FILE, page=None)]


def test_parse_missing_block_returns_empty():
    assert parse_sources_block("תשובה בלי בלוק מקורות כלל.") == []


def test_parse_uses_last_sources_header_occurrence():
    # A retry conversation may echo the corrective nudge (which also mentions
    # מקורות:) before the real answer -- only the LAST block should count.
    text = (
        f"{SOURCES_HEADER}\n- file: bogus.pdf | page: 9\n\n"
        f"תשובה סופית.\n\n{SOURCES_HEADER}\n- file: {ANCHOR_FILE} | page: 1"
    )
    assert parse_sources_block(text) == [Citation(file=ANCHOR_FILE, page=1)]


# --------------------------------------------------------------------------- #
# Citation validation
# --------------------------------------------------------------------------- #


def test_validate_keeps_citation_matching_retrieved_set():
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    citations = [Citation(file=ANCHOR_FILE, page=1)]
    assert validate_citations(citations, retrieved) == citations


def test_validate_drops_fabricated_citation():
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    citations = [Citation(file="apartment/files/does-not-exist.pdf", page=5)]
    assert validate_citations(citations, retrieved) == []


def test_validate_drops_wrong_page_same_file():
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    citations = [Citation(file=ANCHOR_FILE, page=2)]
    assert validate_citations(citations, retrieved) == []


def test_validate_handles_none_page_txt_source():
    retrieved = [make_retrieved(file="travel/files/notes.txt", page=None)]
    citations = [Citation(file="travel/files/notes.txt", page=None)]
    assert validate_citations(citations, retrieved) == citations


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #


def test_assemble_context_header_format_and_order():
    r1 = make_retrieved(file=ANCHOR_FILE, page=1, category="apartment", text="טקסט א", rerank_score=0.9)
    r2 = make_retrieved(file="travel/files/notes.txt", page=None, category="travel", text="טקסט ב", rerank_score=0.5)
    context = assemble_context([r1, r2])
    assert context.index(f"[מקור: {ANCHOR_FILE} | עמוד: 1 | תחום: apartment]") < context.index(
        "[מקור: travel/files/notes.txt | עמוד: - | תחום: travel]"
    )
    assert "טקסט א" in context and "טקסט ב" in context


def test_assemble_context_trims_tail_to_token_budget():
    big_text = "מ" * (CONTEXT_TOKEN_BUDGET * 4 * 2)  # ~2x budget alone
    r1 = make_retrieved(file=ANCHOR_FILE, page=1, text=big_text, rerank_score=0.9)
    r2 = make_retrieved(file="travel/files/notes.txt", page=None, text="קצר", rerank_score=0.5)
    context = assemble_context([r1, r2])
    assert big_text in context  # first chunk always kept even alone over budget
    assert "travel/files/notes.txt" not in context  # tail trimmed


def test_assemble_context_keeps_first_chunk_alone_over_budget():
    huge = "א" * (CONTEXT_TOKEN_BUDGET * 4 * 3)
    context = assemble_context([make_retrieved(text=huge)])
    assert huge in context


# --------------------------------------------------------------------------- #
# Generator.generate -- LLM mocked
# --------------------------------------------------------------------------- #


def _sources_block(file: str, page) -> str:
    page_str = "-" if page is None else str(page)
    return f"{SOURCES_HEADER}\n- file: {file} | page: {page_str}"


def make_generator(**overrides) -> Generator:
    kwargs = dict(model="deepseek-ai/DeepSeek-V4-Pro", prompt="grounded_cite")
    kwargs.update(overrides)
    return Generator(**kwargs)


def test_gate_fail_never_calls_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rag.generate.generator.tf_chat",
        lambda *a, **kw: calls.append((a, kw)) or pytest.fail("LLM must not be called on gate-fail"),
    )
    generator = make_generator()
    result = generator.generate("מה מזג האוויר?", [])
    assert result.text == FALLBACK_TEXT
    assert result.citations == []
    assert result.citation_fallback is False
    assert result.cost_estimate == 0.0
    assert calls == []


def test_generate_happy_path_valid_citation(monkeypatch):
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    reply = f"התשובה: שלוש שנים.\n\n{_sources_block(ANCHOR_FILE, 1)}"
    monkeypatch.setattr(
        "rag.generate.generator.tf_chat",
        lambda messages, **kw: (reply, {"prompt": 50, "completion": 20}, 0.001),
    )
    generator = make_generator()
    result = generator.generate("שאלה", retrieved)
    assert result.citations == [Citation(file=ANCHOR_FILE, page=1)]
    assert result.citation_fallback is False
    assert result.tokens == {"prompt": 50, "completion": 20}
    assert result.cost_estimate == pytest.approx(0.001)


def test_generate_model_refusal_no_retry_no_fallback_citations(monkeypatch):
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    calls = []
    monkeypatch.setattr(
        "rag.generate.generator.tf_chat",
        lambda messages, **kw: calls.append(1) or (FALLBACK_TEXT, {"prompt": 10, "completion": 5}, 0.0001),
    )
    generator = make_generator()
    result = generator.generate("מה שער הדולר?", retrieved)
    assert result.text == FALLBACK_TEXT
    assert result.citations == []
    assert result.citation_fallback is False
    assert len(calls) == 1  # no retry on a legitimate refusal


def test_generate_fabricated_citation_retries_once_then_succeeds(monkeypatch):
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    replies = [
        f"תשובה שגויה.\n\n{_sources_block('apartment/files/fabricated.pdf', 99)}",
        f"תשובה מתוקנת.\n\n{_sources_block(ANCHOR_FILE, 1)}",
    ]
    calls = []

    def fake_chat(messages, **kw):
        calls.append(messages)
        return replies[len(calls) - 1], {"prompt": 10, "completion": 10}, 0.0005

    monkeypatch.setattr("rag.generate.generator.tf_chat", fake_chat)
    generator = make_generator()
    result = generator.generate("שאלה", retrieved)
    assert len(calls) == 2
    assert result.citations == [Citation(file=ANCHOR_FILE, page=1)]
    assert result.citation_fallback is False
    assert result.tokens == {"prompt": 20, "completion": 20}
    # the retry's corrective nudge was appended as a follow-up user turn
    assert calls[1][-1]["role"] == "user"
    assert "מקורות" in calls[1][-1]["content"]


def test_generate_all_invalid_after_retry_falls_back_to_top3(monkeypatch):
    r1 = make_retrieved(file=ANCHOR_FILE, page=1, rerank_score=0.9)
    r2 = make_retrieved(file="travel/files/a.pdf", page=2, rerank_score=0.8)
    r3 = make_retrieved(file="travel/files/b.pdf", page=3, rerank_score=0.7)
    r4 = make_retrieved(file="travel/files/c.pdf", page=4, rerank_score=0.6)
    retrieved = [r1, r2, r3, r4]
    bogus_reply = f"תשובה.\n\n{_sources_block('does/not/exist.pdf', 1)}"
    monkeypatch.setattr(
        "rag.generate.generator.tf_chat",
        lambda messages, **kw: (bogus_reply, {"prompt": 10, "completion": 10}, 0.0005),
    )
    generator = make_generator()
    result = generator.generate("שאלה", retrieved)
    assert result.citation_fallback is True
    assert result.citations == [
        Citation(file=ANCHOR_FILE, page=1),
        Citation(file="travel/files/a.pdf", page=2),
        Citation(file="travel/files/b.pdf", page=3),
    ]
    assert result.text == bogus_reply  # answer text kept even though citations were fabricated


def test_generate_retry_disabled_falls_back_immediately(monkeypatch):
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    calls = []
    bogus_reply = f"תשובה.\n\n{_sources_block('does/not/exist.pdf', 1)}"
    monkeypatch.setattr(
        "rag.generate.generator.tf_chat",
        lambda messages, **kw: calls.append(1) or (bogus_reply, {"prompt": 1, "completion": 1}, 0.0),
    )
    generator = make_generator(retry_on_citation_failure=False)
    result = generator.generate("שאלה", retrieved)
    assert len(calls) == 1
    assert result.citation_fallback is True
    assert result.citations == [Citation(file=ANCHOR_FILE, page=1)]


def test_generate_missing_sources_block_treated_as_citation_failure(monkeypatch):
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    replies = ["תשובה בלי בלוק מקורות בכלל.", f"תשובה מתוקנת.\n\n{_sources_block(ANCHOR_FILE, 1)}"]
    calls = []

    def fake_chat(messages, **kw):
        calls.append(messages)
        return replies[len(calls) - 1], {"prompt": 5, "completion": 5}, 0.0002

    monkeypatch.setattr("rag.generate.generator.tf_chat", fake_chat)
    generator = make_generator()
    result = generator.generate("שאלה", retrieved)
    assert len(calls) == 2
    assert result.citations == [Citation(file=ANCHOR_FILE, page=1)]
    assert result.citation_fallback is False


def test_generate_tracks_max_tokens_hit(monkeypatch):
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    reply = f"תשובה חלקית שנקטעה\n\n{_sources_block(ANCHOR_FILE, 1)}"
    monkeypatch.setattr(
        "rag.generate.generator.tf_chat",
        lambda messages, **kw: (reply, {"prompt": 50, "completion": 20, "finish_reason": "length"}, 0.001),
    )
    generator = make_generator()
    result = generator.generate("שאלה", retrieved)
    assert result.max_tokens_hit is True
    assert result.n_retries == 0
    assert result.tokens == {"prompt": 50, "completion": 20}  # finish_reason popped out, not summed


def test_generate_no_max_tokens_hit_on_normal_stop(monkeypatch):
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    reply = f"תשובה מלאה.\n\n{_sources_block(ANCHOR_FILE, 1)}"
    monkeypatch.setattr(
        "rag.generate.generator.tf_chat",
        lambda messages, **kw: (reply, {"prompt": 50, "completion": 20, "finish_reason": "stop"}, 0.001),
    )
    generator = make_generator()
    result = generator.generate("שאלה", retrieved)
    assert result.max_tokens_hit is False


def test_generate_n_retries_counts_the_corrective_retry(monkeypatch):
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    replies = [
        f"תשובה שגויה.\n\n{_sources_block('apartment/files/fabricated.pdf', 99)}",
        f"תשובה מתוקנת.\n\n{_sources_block(ANCHOR_FILE, 1)}",
    ]
    calls = []

    def fake_chat(messages, **kw):
        calls.append(messages)
        return replies[len(calls) - 1], {"prompt": 10, "completion": 10}, 0.0005

    monkeypatch.setattr("rag.generate.generator.tf_chat", fake_chat)
    generator = make_generator()
    result = generator.generate("שאלה", retrieved)
    assert result.n_retries == 1
    assert result.max_tokens_hit is False


def test_generate_none_content_normalized_never_crashes(monkeypatch):
    # Reasoning models (gpt-oss, Nemotron) can return content=None with
    # finish_reason="length" when the whole budget went to hidden reasoning.
    # This must not crash the batch (regression: TypeError on `FALLBACK_TEXT
    # in text` when text is None) -- it should behave like any other
    # citation-less reply: retry once, then fall back to top-3 citations.
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    calls = []

    def fake_chat(messages, **kw):
        calls.append(messages)
        return None, {"prompt": 50, "completion": 1024, "finish_reason": "length"}, 0.002

    monkeypatch.setattr("rag.generate.generator.tf_chat", fake_chat)
    generator = make_generator()
    result = generator.generate("שאלה", retrieved)
    assert len(calls) == 2  # citation-retry still attempted
    assert result.max_tokens_hit is True
    assert result.citation_fallback is True
    assert result.citations == [Citation(file=ANCHOR_FILE, page=1)]
    assert result.text == ""


def test_generate_max_tokens_hit_on_retry_call_is_still_tracked(monkeypatch):
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    replies = [
        (f"תשובה שגויה.\n\n{_sources_block('apartment/files/fabricated.pdf', 99)}",
         {"prompt": 10, "completion": 10, "finish_reason": "stop"}, 0.0005),
        (f"תשובה מתוקנת.\n\n{_sources_block(ANCHOR_FILE, 1)}",
         {"prompt": 10, "completion": 10, "finish_reason": "length"}, 0.0005),
    ]
    calls = []

    def fake_chat(messages, **kw):
        calls.append(messages)
        return replies[len(calls) - 1]

    monkeypatch.setattr("rag.generate.generator.tf_chat", fake_chat)
    generator = make_generator()
    result = generator.generate("שאלה", retrieved)
    assert result.n_retries == 1
    assert result.max_tokens_hit is True  # hit on the retry call, not the first


def test_generator_rejects_unknown_prompt_variant():
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        make_generator(prompt="does-not-exist")


def test_generator_rejects_unknown_params():
    with pytest.raises(TypeError, match="Unknown generation params"):
        make_generator(bogus_param=1)


def test_generator_forwards_extra_params_to_tf_chat(monkeypatch):
    retrieved = [make_retrieved(file=ANCHOR_FILE, page=1)]
    reply = f"תשובה.\n\n{_sources_block(ANCHOR_FILE, 1)}"
    seen_kwargs = {}

    def fake_chat(messages, **kw):
        seen_kwargs.update(kw)
        return reply, {"prompt": 10, "completion": 10}, 0.0001

    monkeypatch.setattr("rag.generate.generator.tf_chat", fake_chat)
    generator = make_generator(
        extra_params={"reasoning_effort": "low", "allowed_openai_params": ["reasoning_effort"]}
    )
    generator.generate("שאלה", retrieved)
    assert seen_kwargs["reasoning_effort"] == "low"
    assert seen_kwargs["allowed_openai_params"] == ["reasoning_effort"]


def test_generator_default_extra_params_is_empty():
    generator = make_generator()
    assert generator.extra_params == {}


# --------------------------------------------------------------------------- #
# One live smoke call via tf_client on the real index (slow + llm).
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.llm
def test_smoke_live_generation_on_real_index(repo_root, capsys):
    from rag.config import load_config
    from rag.index.manifest import load_manifest
    from rag.retrieve.retriever import load_retriever

    config = load_config(repo_root / "configs" / "default.yaml")
    try:
        load_manifest(config.index_dir)
    except Exception:
        pytest.skip("no ingested index at rag_index/default — run T6 subset ingest first")

    retriever = load_retriever(config)
    generator = Generator(
        model=config.generation.model,
        prompt=config.generation.prompt,
        max_tokens=config.generation.max_tokens,
        temperature=config.generation.temperature,
        retry_on_citation_failure=config.generation.retry_on_citation_failure,
    )
    try:
        question = (
            "קרה נזק בדירה שלי לפני חודש. האם נכון שיש לי שלוש שנים מיום המקרה "
            "להגיש תביעה לתגמולי ביטוח?"
        )
        retrieved = retriever.retrieve(question, category="apartment")
        assert retrieved, "relevance gate rejected every candidate for the dev question"
        result = generator.generate(question, retrieved)
        with capsys.disabled():
            print(f"\n[smoke] answer: {result.text}")
            print(f"[smoke] citations: {result.citations}")
            print(f"[smoke] citation_fallback={result.citation_fallback} cost=${result.cost_estimate:.4f} tokens={result.tokens}")
        assert result.text.strip()
        assert result.citations, "expected at least one citation on an answerable question"
    finally:
        retriever.close()

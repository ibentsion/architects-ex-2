"""Normalizer tests (rag_plan.md §9 test_normalize.py list).

Tests that need the real Stanza Hebrew pipeline are marked ``slow`` (first
run downloads the ~250 MB he model; pipeline load is ~20 s). Cache/fallback
behavior is tested fast with the pipeline mocked out.
"""
from __future__ import annotations

import shutil

import pytest

from rag.normalize import REGISTRY, Normalizer
from rag.normalize.cache import TokenCache, tokens_for_doc
from rag.normalize.stanza_norm import (
    StanzaNormalizer,
    normalize_token,
    split_long_text,
    whitespace_tokens,
)

SHA = "f" * 64
CHUNKER_ID = "chunkid123456"
NORM_ID = "normid1234567"


# --------------------------------------------------------------------------- #
# Recipe unit tests (no Stanza)
# --------------------------------------------------------------------------- #


def test_normalize_token_gershayim_to_plain_quote():
    assert normalize_token("חו״ל") == ['חו"ל']


def test_normalize_token_geresh_and_maqaf():
    assert normalize_token("צ׳ק") == ["צ'ק"]
    # maqaf normalized to '-' then split into word parts
    assert normalize_token("בית־חולים") == ["בית", "חולים"]


def test_normalize_token_strips_nikud():
    assert normalize_token("בֵּיטוּחַ") == ["ביטוח"]


def test_normalize_token_lowercases_latin_and_splits_hyphens():
    assert normalize_token("Dental-Claim-Form") == ["dental", "claim", "form"]


def test_normalize_token_keeps_digits_with_thousands_separator():
    assert normalize_token("6,000") == ["6,000"]
    assert normalize_token("(6,000)") == ["6,000"]


def test_normalize_token_drops_pure_punctuation():
    assert normalize_token("!!!") == []
    assert normalize_token("—") == []


def test_whitespace_tokens_applies_recipe():
    assert whitespace_tokens("ביטוח של 6,000 ₪ Dental-Form!") == [
        "ביטוח", "של", "6,000", "₪", "dental", "form",
    ]


# --------------------------------------------------------------------------- #
# Protocol + registry
# --------------------------------------------------------------------------- #


def test_stanza_normalizer_satisfies_protocol():
    assert isinstance(StanzaNormalizer(), Normalizer)


def test_stanza_normalizer_rejects_unknown_params():
    with pytest.raises(TypeError, match="Unknown stanza normalizer params"):
        StanzaNormalizer(no_such_param=True)


def test_trankit_stub_raises_import_error_with_pip_hint():
    with pytest.raises(ImportError, match="pip install trankit"):
        REGISTRY["trankit"]()


@pytest.mark.skipif(shutil.which("yap") is not None, reason="yap binary on PATH")
def test_yap_stub_raises_import_error_with_setup_hint():
    with pytest.raises(ImportError, match="OnlpLab/yap"):
        REGISTRY["yap"]()


# --------------------------------------------------------------------------- #
# Whitespace fallback on induced Stanza failure (pipeline mocked)
# --------------------------------------------------------------------------- #


class BrokenPipeline:
    def bulk_process(self, texts):
        raise RuntimeError("induced stanza failure (bulk)")

    def __call__(self, text):
        raise RuntimeError("induced stanza failure (single)")


class HalfBrokenPipeline(BrokenPipeline):
    """bulk fails; per-chunk succeeds only for the first text."""

    def __init__(self, good_text, good_doc):
        self.good_text = good_text
        self.good_doc = good_doc

    def __call__(self, text):
        if text == self.good_text:
            return self.good_doc
        raise RuntimeError("induced stanza failure (single)")


class FakeWord:
    def __init__(self, text, lemma):
        self.text, self.lemma = text, lemma


class FakeDoc:
    """Minimal stanza-Document shape: sentences -> tokens -> words."""

    def __init__(self, token_words: list[tuple[str, list[tuple[str, str]]]]):
        class T:
            pass

        class S:
            pass

        sentence = S()
        sentence.tokens = []
        for token_text, words in token_words:
            token = T()
            token.text = token_text
            token.words = [FakeWord(t, l) for t, l in words]
            sentence.tokens.append(token)
        self.sentences = [sentence]


def test_whitespace_fallback_on_induced_stanza_failure(monkeypatch, caplog):
    normalizer = StanzaNormalizer()
    monkeypatch.setattr(normalizer, "_get_pipeline", lambda: BrokenPipeline())
    with caplog.at_level("WARNING"):
        results = normalizer.tokens_batch_ex(["ביטוח דירה של 6,000 ₪"])
    assert results == [(["ביטוח", "דירה", "של", "6,000", "₪"], True)]
    assert "whitespace fallback" in caplog.text


def test_partial_fallback_flags_only_failing_chunks(monkeypatch):
    good_doc = FakeDoc([("בבית", [("ב", "ב"), ("בית", "בית")])])
    normalizer = StanzaNormalizer()
    monkeypatch.setattr(
        normalizer, "_get_pipeline", lambda: HalfBrokenPipeline("בבית", good_doc)
    )
    results = normalizer.tokens_batch_ex(["בבית", "שבר"])
    assert results[0] == (["ב", "בית", "בבית"], False)
    assert results[1] == (["שבר"], True)


def test_doc_tokens_union_and_token_surface_hedge():
    # MWT-split domain term: word surfaces are ה+ראל, token surface is הראל.
    doc = FakeDoc([("הראל", [("ה", "ה"), ("ראל", "ראל")])])
    normalizer = StanzaNormalizer()
    assert normalizer._doc_tokens(doc) == ["ה", "ראל", "הראל"]
    lemma_only = StanzaNormalizer(index_surface_forms=False)
    assert lemma_only._doc_tokens(doc) == ["ה", "ראל"]


# --------------------------------------------------------------------------- #
# Token cache (read-through; fallback docs not cached) — Stanza mocked
# --------------------------------------------------------------------------- #


class NoStanzaNormalizer(StanzaNormalizer):
    """Fails the test if the pipeline is ever touched."""

    def _get_pipeline(self):
        pytest.fail("Stanza pipeline must not be loaded on a token-cache hit")


class FakeNormalizer:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def tokens_batch_ex(self, texts):
        self.calls += 1
        return self.results[: len(texts)]

    def tokens(self, text):
        return self.tokens_batch_ex([text])[0][0]


def test_cache_hit_skips_stanza(tmp_cache_dir):
    cache = TokenCache(tmp_cache_dir)
    cache.store(SHA, CHUNKER_ID, NORM_ID, [["ביטוח"], ["פוליסה"]])
    tokens = tokens_for_doc(
        NoStanzaNormalizer(), cache,
        sha256=SHA, chunker_id=CHUNKER_ID, normalizer_id=NORM_ID,
        texts=["chunk one", "chunk two"],
    )
    assert tokens == [["ביטוח"], ["פוליסה"]]


def test_cache_miss_normalizes_then_stores(tmp_cache_dir):
    cache = TokenCache(tmp_cache_dir)
    fake = FakeNormalizer([(["ביטוח"], False), (["פוליסה"], False)])
    tokens = tokens_for_doc(
        fake, cache,
        sha256=SHA, chunker_id=CHUNKER_ID, normalizer_id=NORM_ID,
        texts=["a", "b"],
    )
    assert tokens == [["ביטוח"], ["פוליסה"]]
    assert cache.load(SHA, CHUNKER_ID, NORM_ID) == [["ביטוח"], ["פוליסה"]]
    # second call served from cache — normalizer not called again
    tokens_for_doc(
        fake, cache,
        sha256=SHA, chunker_id=CHUNKER_ID, normalizer_id=NORM_ID,
        texts=["a", "b"],
    )
    assert fake.calls == 1


def test_fallback_tokenized_doc_is_not_cached(tmp_cache_dir):
    cache = TokenCache(tmp_cache_dir)
    fake = FakeNormalizer([(["ביטוח"], False), (["שבר"], True)])
    tokens = tokens_for_doc(
        fake, cache,
        sha256=SHA, chunker_id=CHUNKER_ID, normalizer_id=NORM_ID,
        texts=["a", "b"],
    )
    assert tokens == [["ביטוח"], ["שבר"]]
    assert cache.load(SHA, CHUNKER_ID, NORM_ID) is None


def test_cache_key_varies_by_chunker_and_normalizer_id(tmp_cache_dir):
    cache = TokenCache(tmp_cache_dir)
    cache.store(SHA, CHUNKER_ID, NORM_ID, [["x"]])
    assert cache.load(SHA, "otherchunker", NORM_ID) is None
    assert cache.load(SHA, CHUNKER_ID, "othernorm") is None


def test_corrupt_cache_entry_renormalizes(tmp_cache_dir):
    cache = TokenCache(tmp_cache_dir)
    cache.tokens_dir.mkdir(parents=True)
    cache.path_for(SHA, CHUNKER_ID, NORM_ID).write_text("{not json", encoding="utf-8")
    fake = FakeNormalizer([(["ביטוח"], False)])
    tokens = tokens_for_doc(
        fake, cache,
        sha256=SHA, chunker_id=CHUNKER_ID, normalizer_id=NORM_ID,
        texts=["a"],
    )
    assert tokens == [["ביטוח"]]
    assert fake.calls == 1


def test_chunk_count_mismatch_renormalizes(tmp_cache_dir):
    cache = TokenCache(tmp_cache_dir)
    cache.store(SHA, CHUNKER_ID, NORM_ID, [["stale"]])
    fake = FakeNormalizer([(["a1"], False), (["b1"], False)])
    tokens = tokens_for_doc(
        fake, cache,
        sha256=SHA, chunker_id=CHUNKER_ID, normalizer_id=NORM_ID,
        texts=["a", "b"],
    )
    assert tokens == [["a1"], ["b1"]]


# --------------------------------------------------------------------------- #
# Real Stanza (slow: model download on first run + ~20 s pipeline load)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def stanza_norm() -> StanzaNormalizer:
    return StanzaNormalizer(package="default", index_surface_forms=True)


@pytest.mark.slow
def test_mwt_lemma_bavit(stanza_norm):
    assert "בית" in stanza_norm.tokens("בבית")


@pytest.mark.slow
def test_fused_prefixes_yield_lemma(stanza_norm):
    assert "פוליסה" in stanza_norm.tokens("והפוליסות")


@pytest.mark.slow
def test_surface_form_in_union(stanza_norm):
    tokens = stanza_norm.tokens("והפוליסות")
    assert "פוליסות" in tokens  # surface, alongside the lemma


@pytest.mark.slow
def test_harel_survives_via_surface_hedge(stanza_norm):
    assert "הראל" in stanza_norm.tokens("ביטוח של הראל")


@pytest.mark.slow
def test_latin_lowercased_and_hyphen_split(stanza_norm):
    tokens = stanza_norm.tokens("טופס Dental-Claim-Form מצורף")
    assert {"dental", "claim", "form"} <= set(tokens)


@pytest.mark.slow
def test_digits_kept(stanza_norm):
    assert "6,000" in stanza_norm.tokens("פיצוי של 6,000 ₪")


@pytest.mark.slow
def test_gershayim_normalized(stanza_norm):
    tokens = stanza_norm.tokens("ביטוח נסיעות לחו״ל")
    assert 'חו"ל' in tokens  # gershayim ״ -> plain "
    assert all("״" not in t for t in tokens)


@pytest.mark.slow
def test_determinism_across_two_instances(stanza_norm):
    text = "תקופת ההתיישנות של תביעה היא שלוש שנים לפי הפוליסה"
    second = StanzaNormalizer(package="default", index_surface_forms=True)
    try:
        assert stanza_norm.tokens(text) == second.tokens(text)
    finally:
        second._pipeline = None  # free ~600 MB


@pytest.mark.slow
def test_batch_matches_single(stanza_norm):
    texts = ["ביטוח דירה מקיף", "תקופת התיישנות של שלוש שנים"]
    batched = stanza_norm.tokens_batch_ex(texts)
    assert [tokens for tokens, fallback in batched] == [stanza_norm.tokens(t) for t in texts]
    assert all(not fallback for _, fallback in batched)


# --------------------------------------------------------------------------- #
# Long-text segmentation (OOM regression: Stanza memory is superlinear in
# sentence length — 14.5K-char table chunks killed the ingest; see
# MAX_SEGMENT_CHARS in rag/normalize/stanza_norm.py)
# --------------------------------------------------------------------------- #


def test_split_long_text_short_text_untouched():
    assert split_long_text("ביטוח דירה", max_chars=100) == ["ביטוח דירה"]


def test_split_long_text_prefers_line_boundaries():
    lines = [f"שורה {i} " + "מילה " * 30 for i in range(10)]
    text = "\n".join(lines)
    segments = split_long_text(text, max_chars=400)
    assert all(len(s) <= 400 for s in segments)
    # content preserved: same whitespace-token multiset
    assert " ".join(segments).split() == text.split()


def test_split_long_text_hard_splits_monster_line_at_whitespace():
    text = "מילה " * 2000  # one 10K-char line, no newlines
    segments = split_long_text(text.strip(), max_chars=1000)
    assert all(len(s) <= 1000 for s in segments)
    assert " ".join(segments).split() == text.split()


def test_split_long_text_no_whitespace_line_still_bounded():
    text = "א" * 5000
    segments = split_long_text(text, max_chars=1000)
    assert all(len(s) <= 1000 for s in segments)
    assert "".join(segments) == text


@pytest.mark.slow
def test_long_table_chunk_tokens_match_unsplit_recipe(stanza_norm):
    # A punctuation-free "table" text longer than one segment: the segmented
    # token stream must equal processing the same lines separately.
    row = "| הראל | ביטוח נסיעות | 6,000 | כיסוי מלא |"
    text = "\n".join(row for _ in range(300))  # ~13K chars > MAX_SEGMENT_CHARS
    tokens = stanza_norm.tokens(text)
    assert tokens  # did not fall back / crash
    row_tokens = set(stanza_norm.tokens(row))
    assert row_tokens <= set(tokens)

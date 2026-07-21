"""Retrieval tests (rag_plan.md §9 test_retrieve.py list).

RRF determinism + hand-computed example; #1-in-both wins; relevance gate
(CrossEncoder mocked); top_n truncation; and the ``slow`` retrieval smoke
check against the real apartment+travel index built by T6.
"""
from __future__ import annotations

import time

import pytest

from rag.retrieve import Reranker
from rag.retrieve.fusion import rrf
from rag.retrieve.rerank import BgeReranker, apply_gate
from rag.retrieve.retriever import Retriever, load_retriever
from rag.types import Chunk, RetrievedChunk

from tests.conftest import ANCHOR_PDF_REL


def make_chunk(chunk_id: str, category: str = "apartment", text: str = "טקסט") -> Chunk:
    file = chunk_id.split("#", 1)[0]
    return Chunk(
        chunk_id=chunk_id,
        file=file,
        page=1,
        category=category,
        text=text,
        source_url=None,
        chunker="per_page",
    )


def make_candidate(chunk_id: str, **scores) -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(chunk_id), **scores)


# --------------------------------------------------------------------------- #
# RRF fusion
# --------------------------------------------------------------------------- #

A = "apartment/files/a.pdf#p1#c0"
B = "apartment/files/b.pdf#p1#c0"
C = "apartment/files/c.pdf#p1#c0"


def test_rrf_hand_computed_three_doc_example():
    # dense: a > b > c; sparse: c > a (b absent). k=60, ranks 1-based:
    #   a = 1/61 + 1/62 = 0.03252…, c = 1/63 + 1/61 = 0.03226…, b = 1/62
    fused = rrf([[A, B, C], [C, A]], k=60)
    assert [cid for cid, _ in fused] == [A, C, B]
    scores = dict(fused)
    assert scores[A] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[C] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[B] == pytest.approx(1 / 62)


def test_rrf_deterministic():
    rankings = [[A, B, C], [C, A, B]]
    assert rrf(rankings, k=60) == rrf(rankings, k=60)


def test_rrf_tie_breaks_by_chunk_id_ascending():
    # a is #1 dense / #2 sparse; b is #2 dense / #1 sparse — identical scores.
    fused = rrf([[A, B], [B, A]], k=60)
    assert fused[0][1] == pytest.approx(fused[1][1])
    assert [cid for cid, _ in fused] == sorted([A, B])


def test_rrf_number_one_in_both_wins():
    # b is #1 in BOTH rankings — must beat a, which is #2 in both but would
    # win either single list's tail.
    fused = rrf([[B, A, C], [B, C, A]], k=60)
    assert fused[0][0] == B
    assert fused[0][1] == pytest.approx(2 / 61)


def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError, match="positive"):
        rrf([[A]], k=0)


# --------------------------------------------------------------------------- #
# Rerank + relevance gate (CrossEncoder mocked — no model download)
# --------------------------------------------------------------------------- #


class FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder."""

    def __init__(self, fixed_scores):
        self.fixed_scores = fixed_scores
        self.seen_pairs = []

    def predict(self, pairs, **kwargs):
        self.seen_pairs.extend(pairs)
        return self.fixed_scores[: len(pairs)]


def test_gate_all_below_threshold_returns_empty(monkeypatch):
    reranker = BgeReranker()
    fake = FakeCrossEncoder([0.10, 0.05, 0.20])
    monkeypatch.setattr(reranker, "_get_model", lambda: fake)
    candidates = [make_candidate(cid, rrf_score=0.03) for cid in (A, B, C)]
    scored = reranker.score("מה תקופת ההתיישנות?", candidates)
    assert [c.rerank_score for c in scored] == pytest.approx([0.10, 0.05, 0.20])
    assert fake.seen_pairs[0] == ("מה תקופת ההתיישנות?", candidates[0].chunk.text)
    assert apply_gate(scored, gate_threshold=0.35, top_n=6) == []


def test_gate_keeps_passing_candidates_sorted_desc(monkeypatch):
    reranker = BgeReranker()
    monkeypatch.setattr(reranker, "_get_model", lambda: FakeCrossEncoder([0.4, 0.9, 0.2]))
    scored = reranker.score("שאלה", [make_candidate(cid) for cid in (A, B, C)])
    survivors = apply_gate(scored, gate_threshold=0.35, top_n=6)
    assert [c.chunk.chunk_id for c in survivors] == [B, A]
    assert [c.rerank_score for c in survivors] == pytest.approx([0.9, 0.4])


def test_gate_top_n_truncation():
    candidates = [
        make_candidate(f"apartment/files/d.pdf#p1#c{i}", rerank_score=0.9 - i * 0.01)
        for i in range(10)
    ]
    survivors = apply_gate(candidates, gate_threshold=0.35, top_n=6)
    assert len(survivors) == 6
    assert [c.rerank_score for c in survivors] == pytest.approx(
        [0.9, 0.89, 0.88, 0.87, 0.86, 0.85]
    )


def test_gate_unscored_candidates_fail():
    assert apply_gate([make_candidate(A)], gate_threshold=0.35, top_n=6) == []


def test_reranker_empty_candidates_no_model_load():
    reranker = BgeReranker()  # _get_model would download 2.2 GB — must not run
    assert reranker.score("שאלה", []) == []
    assert isinstance(reranker, Reranker)


def test_reranker_rejects_unknown_params():
    with pytest.raises(TypeError, match="Unknown reranker params"):
        BgeReranker(bogus=1)


# --------------------------------------------------------------------------- #
# Retriever orchestration (all backends faked)
# --------------------------------------------------------------------------- #

APT1 = "apartment/files/x.pdf#p1#c0"
APT2 = "apartment/files/y.pdf#p2#c0"
TRV1 = "travel/files/z.pdf#p1#c0"


class FakeEmbedder:
    def embed_query(self, text):
        return [1.0, 0.0]


class FakeNormalizer:
    def tokens(self, text):
        return ["התיישנות"]


class FakeDense:
    def __init__(self, hits, store):
        self.hits = hits
        self.store = store
        self.fetched = []

    def search(self, vector, top_k, category=None):
        return self.hits[:top_k]

    def fetch(self, chunk_ids):
        self.fetched.extend(chunk_ids)
        return {cid: self.store[cid] for cid in chunk_ids if cid in self.store}


class FakeSparse:
    def __init__(self, hits):
        self.hits = hits
        self.requested_k = None

    def search(self, query_tokens, top_k):
        self.requested_k = top_k
        return self.hits[:top_k]


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def score(self, question, candidates):
        return [
            c.model_copy(update={"rerank_score": self.scores.get(c.chunk.chunk_id, 0.0)})
            for c in candidates
        ]


def make_retriever(dense_hits, sparse_hits, store, rerank_scores, **overrides):
    kwargs = dict(
        embedder=FakeEmbedder(),
        normalizer=FakeNormalizer(),
        dense=FakeDense(dense_hits, store),
        sparse=FakeSparse(sparse_hits),
        reranker=FakeReranker(rerank_scores),
        dense_top_k=20,
        sparse_top_k=20,
        rrf_k=60,
        gate_threshold=0.35,
        top_n=6,
    )
    kwargs.update(overrides)
    return Retriever(**kwargs)


def test_retriever_populates_all_scores_and_hydrates_sparse_only_hits():
    apt1, apt2 = make_chunk(APT1), make_chunk(APT2)
    retriever = make_retriever(
        dense_hits=[(apt1, 0.9)],
        sparse_hits=[(APT2, 7.5), (APT1, 3.0)],  # APT2 is sparse-only
        store={APT2: apt2},
        rerank_scores={APT1: 0.8, APT2: 0.6},
    )
    results = retriever.retrieve("מה תקופת ההתיישנות?")
    assert [r.chunk.chunk_id for r in results] == [APT1, APT2]
    top = results[0]
    assert top.dense_score == 0.9 and top.sparse_score == 3.0
    assert top.rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert top.rerank_score == 0.8
    hydrated = results[1]  # reached fusion only via BM25 — hydrated via fetch
    assert retriever.dense.fetched == [APT2]
    assert hydrated.dense_score is None and hydrated.sparse_score == 7.5
    assert hydrated.chunk == apt2


def test_retriever_gate_fail_returns_empty_list():
    apt1 = make_chunk(APT1)
    retriever = make_retriever(
        dense_hits=[(apt1, 0.9)],
        sparse_hits=[(APT1, 3.0)],
        store={},
        rerank_scores={APT1: 0.1},  # below gate_threshold=0.35
    )
    assert retriever.retrieve("מה מזג האוויר?") == []


def test_retriever_sparse_category_post_filter_fetches_3x():
    apt1 = make_chunk(APT1)
    trv1 = make_chunk(TRV1, category="travel")
    retriever = make_retriever(
        dense_hits=[(apt1, 0.9)],
        sparse_hits=[(TRV1, 9.0), (APT1, 5.0)],
        store={TRV1: trv1},
        rerank_scores={APT1: 0.9, TRV1: 0.9},
        sparse_top_k=5,
    )
    results = retriever.retrieve("שאלה", category="apartment")
    assert retriever.sparse.requested_k == 15  # 3x fetch with category filter
    assert [r.chunk.chunk_id for r in results] == [APT1]  # travel post-filtered


def test_retriever_no_hits_returns_empty():
    retriever = make_retriever([], [], {}, {})
    assert retriever.retrieve("שאלה") == []


# --------------------------------------------------------------------------- #
# Retrieval smoke check on the REAL apartment+travel index (T6 output).
# Live TF query embedding + local CrossEncoder — slow + llm.
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.llm
def test_smoke_anchor_chunk_in_top3_and_rerank_latency(repo_root, capsys):
    from rag.config import load_config
    from rag.index.manifest import load_manifest

    config = load_config(repo_root / "configs" / "default.yaml")
    try:
        load_manifest(config.index_dir)
    except Exception:
        pytest.skip("no ingested index at rag_index/default — run T6 subset ingest first")

    retriever = load_retriever(config)
    try:
        # Measure rerank latency separately: retrieve WITHOUT the gate first
        # by scoring the full fused candidate set (~20 pairs) directly.
        question = "תקופת התיישנות"
        dense_hits = retriever._dense_search(question, None)
        sparse_hits = retriever._sparse_search(question, None)
        from rag.retrieve.fusion import rrf as _rrf

        fused = _rrf(
            [[c.chunk_id for c, _ in dense_hits], [cid for cid, _ in sparse_hits]],
            k=retriever.rrf_k,
        )[:20]
        chunks_by_id = {c.chunk_id: c for c, _ in dense_hits}
        chunks_by_id.update(retriever.dense.fetch([cid for cid, _ in fused]))
        candidates = [
            RetrievedChunk(chunk=chunks_by_id[cid], rrf_score=score)
            for cid, score in fused
            if cid in chunks_by_id
        ]
        retriever.reranker._get_model()  # load outside the timed section
        t0 = time.monotonic()
        scored = retriever.reranker.score(question, candidates)
        rerank_seconds = time.monotonic() - t0

        results = apply_gate(scored, retriever.gate_threshold, retriever.top_n)
        assert results, "relevance gate rejected every candidate"
        top3 = [(r.chunk.file, r.chunk.page) for r in results[:3]]
        anchor_hits = [
            (rank, r)
            for rank, r in enumerate(results, start=1)
            if r.chunk.file == ANCHOR_PDF_REL and r.chunk.page == 1
        ]
        assert anchor_hits, f"anchor chunk not retrieved; top-3 was {top3}"
        rank, anchor = anchor_hits[0]
        assert rank <= 3, f"anchor chunk ranked #{rank} (>3); top-3 was {top3}"
        with capsys.disabled():
            print(
                f"\n[smoke] anchor rank #{rank}, rerank score "
                f"{anchor.rerank_score:.4f}; rerank latency for "
                f"{len(candidates)} pairs: {rerank_seconds:.2f}s "
                f"({rerank_seconds / len(candidates) * 1000:.0f} ms/pair)"
            )
    finally:
        retriever.close()

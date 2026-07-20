"""Embedder + index + manifest tests (rag_plan.md §9 test_index.py list).

API-dependent paths run with litellm mocked; ONE live 1-text embedding call
is marked ``llm`` (needs NEBIUS_API_KEY in .env).
"""
from __future__ import annotations

import numpy as np
import pytest

from rag.config import load_config
from rag.embed import Embedder
from rag.embed.cache import EmbeddingCache, embed_doc, embedder_cache_key
from rag.embed.st_embedder import SentenceTransformersEmbedder
from rag.embed.tf_embedder import TokenFactoryEmbedder
from rag.embed import tf_embedder as tf_module
from rag.index.dense import DENSE_REGISTRY, QdrantIndex, VectorIndex, point_id
from rag.index.manifest import (
    ManifestError,
    ManifestMismatchError,
    build_manifest,
    load_manifest,
    verify_manifest,
    write_manifest,
)
from rag.index.sparse import SPARSE_REGISTRY, Bm25sIndex, KeywordIndex
from rag.types import Chunk

SHA = "e" * 64
CHUNKER_ID = "chunkid123456"
EMB_KEY = embedder_cache_key("embid12345678", 8)


def make_chunk(n: int, category: str = "apartment", page: int | None = 1) -> Chunk:
    file = f"{category}/files/doc-{n // 100}.pdf"
    return Chunk(
        chunk_id=f"{file}#p{page}#c{n}",
        file=file,
        page=page,
        category=category,
        text=f"טקסט של פוליסת ביטוח מספר {n}",
        source_url=None,
        chunker="per_page",
    )


def unit_vectors(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Dense index (Qdrant local)
# --------------------------------------------------------------------------- #


@pytest.fixture
def dense_index(tmp_index_dir):
    index = QdrantIndex(path=tmp_index_dir / "qdrant")
    yield index
    index.close()


def test_dense_round_trip_payload_intact(dense_index):
    chunks = [make_chunk(i) for i in range(10)]
    vectors = unit_vectors(10)
    dense_index.add(chunks, vectors.tolist())
    results = dense_index.search(vectors[3].tolist(), top_k=1)
    assert len(results) == 1
    found, score = results[0]
    assert found == chunks[3]  # full Chunk metadata survives the payload round-trip
    assert score == pytest.approx(1.0, abs=1e-5)


def test_dense_category_filter_excludes_other_categories(dense_index):
    chunks = [make_chunk(i, category="apartment" if i < 5 else "travel") for i in range(10)]
    vectors = unit_vectors(10)
    dense_index.add(chunks, vectors.tolist())
    # nearest neighbor overall is the apartment chunk itself — filter must skip it
    results = dense_index.search(vectors[0].tolist(), top_k=10, category="travel")
    assert results
    assert {chunk.category for chunk, _ in results} == {"travel"}


def test_dense_persists_across_instances(tmp_index_dir):
    chunks = [make_chunk(i) for i in range(3)]
    vectors = unit_vectors(3)
    first = QdrantIndex(path=tmp_index_dir / "qdrant")
    first.add(chunks, vectors.tolist())
    first.close()
    second = QdrantIndex(path=tmp_index_dir / "qdrant")
    try:
        results = second.search(vectors[1].tolist(), top_k=1)
        assert results[0][0].chunk_id == chunks[1].chunk_id
    finally:
        second.close()


def test_dense_length_mismatch_rejected(dense_index):
    with pytest.raises(ValueError, match="chunks but"):
        dense_index.add([make_chunk(0)], [])


def test_qdrant_factories_require_location():
    with pytest.raises(ValueError, match="path"):
        DENSE_REGISTRY["qdrant_local"]()
    with pytest.raises(ValueError, match="url"):
        DENSE_REGISTRY["qdrant_server"]()


def test_point_id_is_stable_uuid():
    a = point_id("apartment/files/doc.pdf#p1#c0")
    assert a == point_id("apartment/files/doc.pdf#p1#c0")
    assert a != point_id("apartment/files/doc.pdf#p1#c1")


def test_dense_protocol_satisfied(dense_index):
    assert isinstance(dense_index, VectorIndex)


# --------------------------------------------------------------------------- #
# Sparse index (bm25s)
# --------------------------------------------------------------------------- #

TOKEN_LISTS = [
    ["ביטוח", "דירה", "מקיף"],
    ["פוליסה", "ביטוח", "נסיעות"],
    ["תקופה", "התיישנות", "שלוש", "שנה"],
    ["ספורט", "אתגרי", "נסיעות"],
]
CHUNK_IDS = [make_chunk(i).chunk_id for i in range(4)]


def build_sparse() -> Bm25sIndex:
    index = Bm25sIndex()
    index.add(CHUNK_IDS, TOKEN_LISTS)
    return index


def test_sparse_search_ranks_matching_doc_first():
    results = build_sparse().search(["התיישנות"], top_k=4)
    assert results[0][0] == CHUNK_IDS[2]
    assert results[0][1] > 0


def test_sparse_unknown_query_tokens_dropped_and_empty_query_ok():
    index = build_sparse()
    with_unknown = index.search(["התיישנות", "לא-קיים-באוצר"], top_k=4)
    without = index.search(["התיישנות"], top_k=4)
    assert [cid for cid, _ in with_unknown] == [cid for cid, _ in without]
    assert index.search(["לא-קיים-באוצר"], top_k=4) == []


def test_sparse_save_load_identical_scores(tmp_index_dir):
    index = build_sparse()
    before = index.search(["ביטוח", "נסיעות"], top_k=4)
    index.save(tmp_index_dir / "bm25")
    loaded = Bm25sIndex.load(tmp_index_dir / "bm25", mmap=True)
    after = loaded.search(["ביטוח", "נסיעות"], top_k=4)
    assert [cid for cid, _ in after] == [cid for cid, _ in before]
    assert [s for _, s in after] == pytest.approx([s for s, in ([s] for _, s in before)])


def test_sparse_chunk_id_join_matches_dense_namespace(dense_index):
    """Fusion joins on chunk_id: ids coming out of the sparse index must be
    the exact ids stored in the dense payload."""
    chunks = [make_chunk(i) for i in range(4)]
    vectors = unit_vectors(4)
    dense_index.add(chunks, vectors.tolist())
    sparse = Bm25sIndex()
    sparse.add([c.chunk_id for c in chunks], TOKEN_LISTS)
    sparse_ids = {cid for cid, _ in sparse.search(["ביטוח"], top_k=4)}
    dense_ids = {chunk.chunk_id for chunk, _ in dense_index.search(vectors[0].tolist(), top_k=4)}
    assert sparse_ids <= dense_ids


def test_sparse_search_before_add_raises():
    with pytest.raises(RuntimeError, match="empty"):
        Bm25sIndex().search(["ביטוח"], top_k=1)


def test_sparse_load_missing_chunk_ids_raises(tmp_index_dir):
    with pytest.raises(FileNotFoundError, match="re-ingest"):
        Bm25sIndex.load(tmp_index_dir / "nope")


def test_sparse_protocol_satisfied():
    assert isinstance(Bm25sIndex(), KeywordIndex)


def test_optional_sparse_backends_raise_import_error():
    with pytest.raises(ImportError, match="pip install elasticsearch"):
        SPARSE_REGISTRY["elasticsearch"]()
    with pytest.raises(ImportError, match="pip install opensearch-py"):
        SPARSE_REGISTRY["opensearch"]()


def test_optional_dense_backends_raise_import_error():
    with pytest.raises(ImportError, match="pip install chromadb"):
        DENSE_REGISTRY["chroma"]()
    with pytest.raises(ImportError, match="pip install pymilvus"):
        DENSE_REGISTRY["milvus"]()


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@pytest.fixture
def default_config(default_config_path):
    return load_config(default_config_path)


def test_manifest_write_then_verify(tmp_index_dir, default_config):
    manifest = build_manifest(
        default_config,
        chunk_counts={"apartment": 12, "travel": 7},
        files=[{"file": "apartment/files/a.pdf", "sha256": SHA, "status": "ok"}],
        canary={"ok": True},
    )
    write_manifest(tmp_index_dir, manifest)
    verified = verify_manifest(tmp_index_dir, default_config)
    assert verified["total_chunks"] == 19
    assert verified["embedder"]["provider"] == "tokenfactory"
    assert verified["embedder"]["model"] == "Qwen/Qwen3-Embedding-8B"
    assert verified["embedder"]["dimensions"] == 4096
    assert verified["files"][0]["status"] == "ok"
    assert verified["canary"] == {"ok": True}
    assert verified["created_at"]


def test_manifest_dimensions_mismatch_rejected(tmp_index_dir, default_config):
    write_manifest(tmp_index_dir, build_manifest(default_config))
    changed = default_config.model_copy(deep=True)
    changed.embedder.params["dimensions"] = 1024
    with pytest.raises(ManifestMismatchError) as excinfo:
        verify_manifest(tmp_index_dir, changed)
    message = str(excinfo.value)
    assert "dimensions" in message and "4096" in message and "1024" in message
    assert "Re-ingest" in message


def test_manifest_embedder_provider_and_model_mismatch_rejected(tmp_index_dir, default_config):
    write_manifest(tmp_index_dir, build_manifest(default_config))
    changed = default_config.model_copy(deep=True)
    changed.embedder.impl = "sentence_transformers"
    changed.embedder.params = {"model": "BAAI/bge-m3"}
    with pytest.raises(ManifestMismatchError, match="provider"):
        verify_manifest(tmp_index_dir, changed)


def test_manifest_normalizer_mismatch_rejected(tmp_index_dir, default_config):
    write_manifest(tmp_index_dir, build_manifest(default_config))
    changed = default_config.model_copy(deep=True)
    changed.normalizer.params["package"] = "iahltwiki"
    with pytest.raises(ManifestMismatchError, match="normalizer"):
        verify_manifest(tmp_index_dir, changed)


def test_manifest_missing_raises_with_ingest_hint(tmp_index_dir, default_config):
    with pytest.raises(ManifestError, match="rag.cli.ingest"):
        verify_manifest(tmp_index_dir, default_config)


def test_manifest_load_corrupt_raises(tmp_index_dir):
    (tmp_index_dir / "manifest.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ManifestError, match="re-ingest"):
        load_manifest(tmp_index_dir)


# --------------------------------------------------------------------------- #
# Token Factory embedder (litellm mocked)
# --------------------------------------------------------------------------- #


class FakeUsage:
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens


class FakeResponse:
    def __init__(self, n, dim):
        # reversed order on purpose: embedder must sort by index
        self.data = [
            {"index": i, "embedding": [float(i)] * dim} for i in reversed(range(n))
        ]
        self.usage = FakeUsage(prompt_tokens=n * 10)


class FakeRateLimitError(Exception):
    status_code = 429


class FakeBadRequestError(Exception):
    status_code = 400


def make_embedder(**overrides) -> TokenFactoryEmbedder:
    kwargs = dict(
        model="Qwen/Qwen3-Embedding-8B",
        dimensions=8,
        batch_size=2,
        query_instruct="Given a Hebrew insurance customer question, retrieve relevant policy passages",
        backoff_base=0.0,
    )
    kwargs.update(overrides)
    return TokenFactoryEmbedder(**kwargs)


def test_tf_embedder_batches_and_sends_docs_plain(monkeypatch):
    calls = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return FakeResponse(len(kwargs["input"]), 8)

    monkeypatch.setattr(tf_module.litellm, "embedding", fake_embedding)
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    embedder = make_embedder()
    texts = [f"מסמך {i}" for i in range(5)]
    vectors = embedder.embed_docs(texts)
    assert len(vectors) == 5 and all(len(v) == 8 for v in vectors)
    assert [len(c["input"]) for c in calls] == [2, 2, 1]  # batch_size=2 splits 5 -> 2+2+1
    assert [text for c in calls for text in c["input"]] == texts  # docs sent PLAIN
    assert all(c["model"] == "openai/Qwen/Qwen3-Embedding-8B" for c in calls)
    assert all(c["dimensions"] == 8 for c in calls)
    assert embedder.total_input_tokens == 50


def test_tf_embedder_sorts_response_by_index(monkeypatch):
    monkeypatch.setattr(tf_module.litellm, "embedding", lambda **kw: FakeResponse(len(kw["input"]), 4))
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    vectors = make_embedder(batch_size=3).embed_docs(["א", "ב", "ג"])
    assert vectors == [[0.0] * 4, [1.0] * 4, [2.0] * 4]


def test_tf_embedder_query_gets_instruct_framing(monkeypatch):
    calls = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return FakeResponse(1, 8)

    monkeypatch.setattr(tf_module.litellm, "embedding", fake_embedding)
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    make_embedder().embed_query("מה תקופת ההתיישנות?")
    assert calls[0]["input"] == [
        "Instruct: Given a Hebrew insurance customer question, retrieve relevant policy passages\n"
        "Query: מה תקופת ההתיישנות?"
    ]


def test_tf_embedder_retries_on_429_with_backoff(monkeypatch):
    attempts = []
    sleeps = []

    def fake_embedding(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise FakeRateLimitError("429 rate limited")
        return FakeResponse(1, 8)

    monkeypatch.setattr(tf_module.litellm, "embedding", fake_embedding)
    monkeypatch.setattr(tf_module.time, "sleep", sleeps.append)
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    vectors = make_embedder(backoff_base=1.0).embed_docs(["טקסט"])
    assert len(vectors) == 1
    assert len(attempts) == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_tf_embedder_gives_up_after_max_attempts(monkeypatch):
    def fake_embedding(**kwargs):
        raise FakeRateLimitError("429 rate limited")

    monkeypatch.setattr(tf_module.litellm, "embedding", fake_embedding)
    monkeypatch.setattr(tf_module.time, "sleep", lambda s: None)
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    with pytest.raises(FakeRateLimitError):
        make_embedder().embed_docs(["טקסט"])


def test_tf_embedder_does_not_retry_client_errors(monkeypatch):
    attempts = []

    def fake_embedding(**kwargs):
        attempts.append(1)
        raise FakeBadRequestError("400 bad request")

    monkeypatch.setattr(tf_module.litellm, "embedding", fake_embedding)
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    with pytest.raises(FakeBadRequestError):
        make_embedder().embed_docs(["טקסט"])
    assert len(attempts) == 1


def test_tf_embedder_missing_key_raises(monkeypatch):
    monkeypatch.setattr(tf_module.litellm, "embedding", lambda **kw: FakeResponse(1, 8))
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setattr(tf_module, "load_dotenv", lambda: None)
    with pytest.raises(RuntimeError, match="NEBIUS_API_KEY"):
        make_embedder().embed_docs(["טקסט"])


def test_tf_embedder_satisfies_protocol():
    assert isinstance(make_embedder(), Embedder)


# --------------------------------------------------------------------------- #
# sentence-transformers embedder (model mocked — no download)
# --------------------------------------------------------------------------- #


class FakeStModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        return np.ones((len(texts), 4), dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return 4


def test_st_embedder_e5_prefixes(monkeypatch):
    embedder = SentenceTransformersEmbedder(model="intfloat/multilingual-e5-large")
    fake = FakeStModel()
    monkeypatch.setattr(embedder, "_get_model", lambda: fake)
    embedder.embed_docs(["מסמך"])
    embedder.embed_query("שאלה")
    assert fake.seen == ["passage: מסמך", "query: שאלה"]
    assert embedder.dimensions == 4


def test_st_embedder_non_e5_no_prefixes(monkeypatch):
    embedder = SentenceTransformersEmbedder(model="BAAI/bge-m3")
    fake = FakeStModel()
    monkeypatch.setattr(embedder, "_get_model", lambda: fake)
    embedder.embed_docs(["מסמך"])
    embedder.embed_query("שאלה")
    assert fake.seen == ["מסמך", "שאלה"]
    assert isinstance(embedder, Embedder)


# --------------------------------------------------------------------------- #
# Embedding cache (read-through .npz)
# --------------------------------------------------------------------------- #


class NoApiEmbedder:
    def embed_docs(self, texts):
        pytest.fail("embedder (API) must not be called on an embedding-cache hit")

    def embed_query(self, text):
        pytest.fail("embed_query must not be called")


class CountingEmbedder:
    def __init__(self, dim=8):
        self.dim = dim
        self.calls = 0

    def embed_docs(self, texts):
        self.calls += 1
        return [[float(i)] * self.dim for i in range(len(texts))]

    def embed_query(self, text):
        return [0.0] * self.dim


def test_embedding_cache_hit_skips_api(tmp_cache_dir):
    cache = EmbeddingCache(tmp_cache_dir)
    stored = np.arange(16, dtype=np.float32).reshape(2, 8)
    cache.store(SHA, CHUNKER_ID, EMB_KEY, stored)
    vectors = embed_doc(
        NoApiEmbedder(), cache,
        sha256=SHA, chunker_id=CHUNKER_ID, embedder_key=EMB_KEY,
        texts=["a", "b"],
    )
    np.testing.assert_array_equal(vectors, stored)


def test_embedding_cache_miss_embeds_then_stores(tmp_cache_dir):
    cache = EmbeddingCache(tmp_cache_dir)
    embedder = CountingEmbedder()
    first = embed_doc(
        embedder, cache,
        sha256=SHA, chunker_id=CHUNKER_ID, embedder_key=EMB_KEY,
        texts=["a", "b"],
    )
    second = embed_doc(
        embedder, cache,
        sha256=SHA, chunker_id=CHUNKER_ID, embedder_key=EMB_KEY,
        texts=["a", "b"],
    )
    assert embedder.calls == 1  # second call served from cache
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32 and first.shape == (2, 8)


def test_embedding_cache_key_includes_dims(tmp_cache_dir):
    cache = EmbeddingCache(tmp_cache_dir)
    cache.store(SHA, CHUNKER_ID, embedder_cache_key("embid", 8), np.ones((1, 8), dtype=np.float32))
    assert cache.load(SHA, CHUNKER_ID, embedder_cache_key("embid", 4)) is None
    assert embedder_cache_key("abc123", 4096) == "abc123-4096"


def test_embedding_cache_row_count_mismatch_reembeds(tmp_cache_dir):
    cache = EmbeddingCache(tmp_cache_dir)
    cache.store(SHA, CHUNKER_ID, EMB_KEY, np.ones((1, 8), dtype=np.float32))
    embedder = CountingEmbedder()
    vectors = embed_doc(
        embedder, cache,
        sha256=SHA, chunker_id=CHUNKER_ID, embedder_key=EMB_KEY,
        texts=["a", "b"],
    )
    assert embedder.calls == 1
    assert vectors.shape == (2, 8)


# --------------------------------------------------------------------------- #
# Live API (one 1-text call; needs NEBIUS_API_KEY in .env)
# --------------------------------------------------------------------------- #


@pytest.mark.llm
def test_live_embedding_dims_match_config(default_config):
    params = default_config.embedder.params
    embedder = TokenFactoryEmbedder(**params)
    vectors = embedder.embed_docs(["מה תקופת ההתיישנות של תביעת ביטוח דירה?"])
    assert len(vectors) == 1
    assert len(vectors[0]) == params["dimensions"]
    assert embedder.total_input_tokens > 0

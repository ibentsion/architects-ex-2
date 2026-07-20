"""Config loading, validation, extends deep-merge, and registry resolution
(rag_plan.md §9: test_config.py)."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import (
    ConfigError,
    RagConfig,
    build,
    config_identity_hash,
    get_registry,
    impl_id,
    load_config,
)

#: Every impl name documented in rag_plan.md §2/§4, per phase.
DOCUMENTED_IMPLS: dict[str, list[str]] = {
    "parser": ["docling"],
    "chunker": ["per_page", "per_paragraph", "per_table"],
    "normalizer": ["stanza", "trankit", "yap"],
    "embedder": ["tokenfactory", "sentence_transformers"],
    "dense_index": ["qdrant_local", "qdrant_server", "chroma", "milvus"],
    "sparse_index": ["bm25s", "elasticsearch", "opensearch"],
    "reranker": ["bge", "jina"],
}

#: Optional-dependency impls whose FACTORY CALL must raise the documented
#: helpful ImportError (install hint) when the dependency is absent.
OPTIONAL_DEP_IMPLS: list[tuple[str, str, str]] = [
    # (phase, impl, substring expected in the error message)
    ("normalizer", "trankit", "pip install trankit"),
    ("normalizer", "yap", "OnlpLab/yap"),
    ("dense_index", "chroma", "pip install chromadb"),
    ("dense_index", "milvus", "pip install pymilvus"),
    ("sparse_index", "elasticsearch", "pip install elasticsearch"),
    ("sparse_index", "opensearch", "pip install opensearch-py"),
]


# --------------------------------------------------------------------------- #
# default.yaml loads and validates
# --------------------------------------------------------------------------- #


def test_default_config_loads_and_validates(default_config_path: Path):
    cfg = load_config(default_config_path)
    assert isinstance(cfg, RagConfig)
    assert cfg.corpus_dir == Path("corpus")
    assert cfg.index_dir == Path("rag_index/default")
    assert cfg.cache_dir == Path("cache")
    assert cfg.parser.impl == "docling"
    assert cfg.parser.params["rtl_canary"] is True
    assert cfg.chunker.impl == "per_page"
    assert cfg.chunker.params == {"max_tokens": 1800, "txt_max_tokens": 512}
    assert cfg.normalizer.impl == "stanza"
    assert cfg.normalizer.params["index_surface_forms"] is True
    assert cfg.embedder.impl == "tokenfactory"
    assert cfg.embedder.params["model"] == "Qwen/Qwen3-Embedding-8B"
    assert cfg.embedder.params["dimensions"] == 4096
    assert cfg.dense_index.impl == "qdrant_local"
    assert cfg.sparse_index.impl == "bm25s"
    assert cfg.retrieval.dense_top_k == 20
    assert cfg.retrieval.rrf_k == 60
    assert cfg.retrieval.rerank.impl == "bge"
    assert cfg.retrieval.rerank.top_n == 6
    assert cfg.retrieval.rerank.gate_threshold == pytest.approx(0.35)
    assert cfg.generation.model == "deepseek-ai/DeepSeek-V4-Pro"
    assert cfg.generation.prompt == "grounded_cite"
    assert cfg.generation.retry_on_citation_failure is True


# --------------------------------------------------------------------------- #
# unknown keys rejected (extra="forbid")
# --------------------------------------------------------------------------- #


def test_unknown_top_level_key_rejected(tmp_path: Path, default_config_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        f"extends: {default_config_path}\nfrobnicate: 1\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError, match="frobnicate"):
        load_config(bad)


def test_unknown_nested_key_rejected(tmp_path: Path, default_config_path: Path):
    bad = tmp_path / "bad_nested.yaml"
    bad.write_text(
        f"extends: {default_config_path}\nretrieval:\n  dense_topk: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="dense_topk"):
        load_config(bad)


# --------------------------------------------------------------------------- #
# extends: deep-merge
# --------------------------------------------------------------------------- #


def test_extends_deep_merge(swap_config_path: Path):
    cfg = load_config(swap_config_path)
    # Overridden values.
    assert cfg.index_dir == Path("rag_index/chroma-trankit")
    assert cfg.normalizer.impl == "trankit"
    assert cfg.dense_index.impl == "chroma"
    assert cfg.embedder.impl == "sentence_transformers"
    assert cfg.embedder.params == {"model": "BAAI/bge-m3", "batch_size": 16}
    assert cfg.generation.prompt == "strict_extractive"
    # impl changed -> the base impl's params must NOT leak through the merge.
    assert cfg.normalizer.params == {}
    assert "dimensions" not in cfg.embedder.params
    # Everything omitted inherits from default.yaml.
    assert cfg.corpus_dir == Path("corpus")
    assert cfg.cache_dir == Path("cache")
    assert cfg.parser.impl == "docling"
    assert cfg.chunker.params["max_tokens"] == 1800
    assert cfg.retrieval.rrf_k == 60
    assert cfg.retrieval.rerank.gate_threshold == pytest.approx(0.35)
    assert cfg.generation.model == "deepseek-ai/DeepSeek-V4-Pro"  # merged, not replaced
    assert cfg.generation.max_tokens == 1024


def test_extends_partial_block_merges(tmp_path: Path, default_config_path: Path):
    override = tmp_path / "partial.yaml"
    override.write_text(
        f"extends: {default_config_path}\n"
        "retrieval:\n  dense_top_k: 5\n",
        encoding="utf-8",
    )
    cfg = load_config(override)
    assert cfg.retrieval.dense_top_k == 5
    assert cfg.retrieval.sparse_top_k == 20  # untouched sibling key survives
    assert cfg.retrieval.rerank.top_n == 6  # untouched nested block survives


def test_extends_missing_base_helpful_error(tmp_path: Path):
    orphan = tmp_path / "orphan.yaml"
    orphan.write_text("extends: no/such/base.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no/such/base.yaml"):
        load_config(orphan)


# --------------------------------------------------------------------------- #
# registries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("phase", sorted(DOCUMENTED_IMPLS))
def test_registry_resolves_every_documented_impl(phase: str):
    registry = get_registry(phase)
    for name in DOCUMENTED_IMPLS[phase]:
        assert name in registry, f"{phase} registry missing documented impl '{name}'"
        assert callable(registry[name])


@pytest.mark.parametrize("phase,impl,hint", OPTIONAL_DEP_IMPLS)
def test_optional_dep_impl_raises_helpful_error(phase: str, impl: str, hint: str):
    """Optional-dep adapters must raise the documented ImportError with an
    install hint when their dependency is absent (rag_plan.md §2). If the
    optional dependency happens to be installed, resolution succeeding (or a
    NotImplementedError from the stub body) is also acceptable."""
    factory = get_registry(phase)[impl]
    try:
        factory()
    except ImportError as err:
        assert hint in str(err), f"ImportError for {phase}/{impl} lacks hint '{hint}': {err}"
    except NotImplementedError:
        pass  # dependency present; adapter body is later-wave work


def test_unpinned_impl_name_helpful_error(default_config_path: Path):
    cfg = load_config(default_config_path)
    cfg.embedder.impl = "does_not_exist"
    with pytest.raises(ConfigError) as excinfo:
        build("embedder", cfg)
    message = str(excinfo.value)
    assert "does_not_exist" in message
    assert "sentence_transformers" in message  # lists available impls
    assert "tokenfactory" in message


def test_unknown_phase_name_helpful_error(default_config_path: Path):
    with pytest.raises(ConfigError, match="Unknown phase"):
        get_registry("flux_capacitor")


def test_build_resolves_default_chunker(default_config_path: Path):
    cfg = load_config(default_config_path)
    chunker = build("chunker", cfg)
    assert chunker.params == {"max_tokens": 1800, "txt_max_tokens": 512}


# --------------------------------------------------------------------------- #
# identity hashes (§5 cache keys, index-manifest compatibility)
# --------------------------------------------------------------------------- #


def test_impl_id_short_and_stable(default_config_path: Path):
    cfg_a = load_config(default_config_path)
    cfg_b = load_config(default_config_path)
    assert impl_id(cfg_a.chunker) == impl_id(cfg_b.chunker)
    assert len(impl_id(cfg_a.chunker)) == 12
    cfg_b.chunker.params["max_tokens"] = 999
    assert impl_id(cfg_a.chunker) != impl_id(cfg_b.chunker)


def test_config_identity_hash_tracks_index_compat_subset(
    default_config_path: Path, swap_config_path: Path
):
    cfg = load_config(default_config_path)
    same = load_config(default_config_path)
    swapped = load_config(swap_config_path)
    assert config_identity_hash(cfg) == config_identity_hash(same)
    assert config_identity_hash(cfg) != config_identity_hash(swapped)  # embedder+normalizer differ
    # Generation settings are NOT part of index identity (no re-ingest needed).
    same.generation.temperature = 0.9
    same.generation.prompt = "few_shot_cite"
    assert config_identity_hash(cfg) == config_identity_hash(same)
    # Embedder dimensions ARE part of index identity (query must match ingest).
    same.embedder.params["dimensions"] = 1024
    assert config_identity_hash(cfg) != config_identity_hash(same)

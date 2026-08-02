"""Config models, YAML loader (with ``extends:`` deep-merge), and registry
resolution (rag_plan.md §4).

Every phase block is ``{impl: <registry key>, params: {…}}``. Unknown keys are
errors (``extra="forbid"`` everywhere — catches typos). ``build(phase_name,
config)`` resolves the ``impl`` name against the phase package's registry and
passes ``params`` through to the factory.

Identity helpers:
  * ``config_identity_hash(config)`` — sha256 of the canonical
    embedder/chunker/normalizer subset; stamped into the index manifest so the
    query CLI can refuse an incompatible index.
  * ``impl_id(block)`` — short stable id of a resolved ``{impl, params}``
    block; used in stage-cache keys (rag_plan.md §5).
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConfigError(Exception):
    """Configuration error with an actionable message."""


# --------------------------------------------------------------------------- #
# Pydantic config models (mirror configs/default.yaml — rag_plan.md §4)
# --------------------------------------------------------------------------- #


class PhaseConfig(BaseModel):
    """Generic swappable-phase block: {impl: <registry key>, params: {…}}."""

    model_config = ConfigDict(extra="forbid")

    impl: str
    params: dict[str, Any] = Field(default_factory=dict)


class RerankConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    impl: str = "bge"
    params: dict[str, Any] = Field(default_factory=dict)
    top_n: int = Field(6, description="Chunks passed to generation")
    gate_threshold: float = Field(
        0.35, description="Sigmoid relevance gate; below for ALL candidates → fallback answer"
    )


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = Field(60, description="Standard RRF constant")
    rerank: RerankConfig = Field(default_factory=RerankConfig)


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Any Token Factory model id (via tf_client)")
    prompt: str = Field("grounded_cite", description="Prompt variant name (registry in rag/generate/prompts.py)")
    max_tokens: int = 1024
    temperature: float = 0.2
    retry_on_citation_failure: bool = True
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Passed straight through to tf_client.chat -> litellm.completion "
        "(e.g. reasoning_effort + allowed_openai_params for gpt-oss's Harmony template, "
        "or extra_body: {chat_template_kwargs: {enable_thinking: false}} for Nemotron/Qwen3-"
        "style hybrid-reasoning models) — this is retrieval-grounded QA, not open-ended "
        "reasoning, so capping reasoning effort trades unneeded 'thinking' tokens for latency.",
    )


class HarnessConfig(BaseModel):
    """Agent-harness (rag/agent) model roles and limits. Orchestration
    (classifier, tool-calling loop, calculation expressions) runs on a fast
    model; final answer synthesis stays on ``generation.model``."""

    model_config = ConfigDict(extra="forbid")

    orchestrator_model: str = Field(..., description="Fast tool-calling model (classifier + agent loop)")
    orchestrator_max_tokens: int = 2048
    orchestrator_extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Passed through to tf_client.chat for orchestrator calls "
        "(e.g. reasoning_effort: low for gpt-oss's Harmony template)",
    )
    category_filter: Literal["single", "set", "family", "none"] = Field(
        "single",
        description="How a sub-question's category tags become a retrieval filter. "
        "single: filter only when the classifier gives exactly one tag (2+ or 0 = no "
        "filter). set: filter on whatever tags it gives. family: widen each tag to the "
        "categories it is confused with (rag.classify.CATEGORY_FAMILIES). none: never "
        "filter. An empty filtered pool always retries unfiltered, whatever the policy.",
    )
    fast_synthesis_model: str | None = Field(
        None,
        description="Synthesis model for easy/medium single-topic questions (the two "
        "models tie there, and the fast one is cheaper); hard/multi/calculation/"
        "dependent queries always synthesize on generation.model. None disables the "
        "routing entirely.",
    )
    fast_synthesis_max_tokens: int = 1024
    fast_synthesis_extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Passed through to tf_client.chat for fast-synthesis calls "
        "(same role as generation.extra_params)",
    )
    max_hops: int = Field(4, description="Agent tool-calling loop iteration cap")
    max_workers: int = Field(4, description="Thread pool size for concurrent sub-question retrievals")


class RagConfig(BaseModel):
    """Root config. All blocks required — configs/default.yaml is the single
    source of truth for defaults; overrides extend it via ``extends:``."""

    model_config = ConfigDict(extra="forbid")

    corpus_dir: Path
    index_dir: Path
    cache_dir: Path

    parser: PhaseConfig
    chunker: PhaseConfig
    normalizer: PhaseConfig
    embedder: PhaseConfig
    dense_index: PhaseConfig
    sparse_index: PhaseConfig

    retrieval: RetrievalConfig
    generation: GenerationConfig
    harness: HarnessConfig


# --------------------------------------------------------------------------- #
# YAML loading with `extends:` deep-merge
# --------------------------------------------------------------------------- #


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge ``override`` onto ``base``.

    Dicts merge key-wise; any non-dict override value replaces the base value.
    Special rule: when both sides are phase-like dicts and ``impl`` CHANGES,
    the base's ``params`` are discarded first — params written for one impl
    are meaningless (and often invalid) for another.
    """
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    if (
        "impl" in base
        and "impl" in override
        and override["impl"] != base["impl"]
    ):
        base = {k: v for k, v in base.items() if k != "params"}
    merged = dict(base)
    for key, value in override.items():
        merged[key] = deep_merge(base[key], value) if key in base else value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML mapping: {path}")
    return data


def _resolve_extends_path(value: str, extending_file: Path) -> Path:
    """``extends:`` paths are repo-root/cwd-relative (as in the §4 example:
    ``extends: configs/default.yaml``), relative to the extending file, or
    relative to the extending file's parent directories."""
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    tried = [candidate]
    bases = [extending_file.parent, *extending_file.resolve().parents[1:]]
    for base in bases:
        alt = base / value
        tried.append(alt)
        if alt.exists():
            return alt
    raise ConfigError(
        f"extends: '{value}' (referenced from {extending_file}) not found — "
        f"tried: {', '.join(str(p) for p in tried)}"
    )


def load_config_dict(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load YAML, resolving ``extends:`` chains with deep-merge (base first)."""
    path = Path(path)
    resolved = path.resolve()
    if resolved in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, resolved))
        raise ConfigError(f"Circular 'extends:' chain: {chain}")
    raw = _read_yaml(path)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    if not isinstance(extends, str):
        raise ConfigError(f"'extends:' must be a string path, got {type(extends).__name__} in {path}")
    base_path = _resolve_extends_path(extends, path)
    base = load_config_dict(base_path, (*_stack, resolved))
    return deep_merge(base, raw)


def load_config(path: str | Path) -> RagConfig:
    """Load + validate a YAML config file (unknown keys rejected)."""
    return RagConfig.model_validate(load_config_dict(path))


# --------------------------------------------------------------------------- #
# Identity hashes (index manifest compatibility + stage-cache keys, §5)
# --------------------------------------------------------------------------- #


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def impl_id(block: PhaseConfig | RerankConfig) -> str:
    """Short stable id of a resolved {impl, params} block (cache-key component)."""
    return hashlib.sha256(_canonical({"impl": block.impl, "params": block.params})).hexdigest()[:12]


def config_identity_hash(config: RagConfig) -> str:
    """sha256 over the embedder/chunker/normalizer subset — the parts whose
    change makes an existing index unusable (query must match ingest)."""
    subset = {
        phase: {"impl": block.impl, "params": block.params}
        for phase, block in (
            ("embedder", config.embedder),
            ("chunker", config.chunker),
            ("normalizer", config.normalizer),
        )
    }
    return hashlib.sha256(_canonical(subset)).hexdigest()


# --------------------------------------------------------------------------- #
# Registry resolution
# --------------------------------------------------------------------------- #

#: phase name -> (module holding the registry, registry attribute name).
#: Modules are imported lazily so `import rag.config` stays dependency-light.
_PHASE_REGISTRIES: dict[str, tuple[str, str]] = {
    "parser": ("rag.parsing", "REGISTRY"),
    "chunker": ("rag.chunking", "REGISTRY"),
    "normalizer": ("rag.normalize", "REGISTRY"),
    "embedder": ("rag.embed", "REGISTRY"),
    "dense_index": ("rag.index", "DENSE_REGISTRY"),
    "sparse_index": ("rag.index", "SPARSE_REGISTRY"),
    "reranker": ("rag.retrieve", "RERANK_REGISTRY"),
}


def get_registry(phase_name: str) -> dict[str, Callable[..., Any]]:
    """Return the ``{impl name: factory}`` registry for a phase."""
    try:
        module_name, attr = _PHASE_REGISTRIES[phase_name]
    except KeyError:
        raise ConfigError(
            f"Unknown phase '{phase_name}'. Known phases: {sorted(_PHASE_REGISTRIES)}"
        ) from None
    return getattr(importlib.import_module(module_name), attr)


def _phase_block(phase_name: str, config: RagConfig) -> PhaseConfig | RerankConfig:
    if phase_name == "reranker":
        return config.retrieval.rerank
    return getattr(config, phase_name)


def build(phase_name: str, config: RagConfig) -> Any:
    """Resolve ``config.<phase>.impl`` against the phase registry and call the
    factory with ``**params``. Raises ConfigError for unpinned impl names."""
    block = _phase_block(phase_name, config)
    registry = get_registry(phase_name)
    if block.impl not in registry:
        raise ConfigError(
            f"Unknown {phase_name} impl '{block.impl}'. "
            f"Available: {sorted(registry)}. "
            f"Fix the '{phase_name}.impl' value in your YAML config."
        )
    return registry[block.impl](**block.params)

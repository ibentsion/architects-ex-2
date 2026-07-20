"""Index manifest: write/verify ``<index_dir>/manifest.json``
(rag_plan.md §5 stage 7, §6 stage 0).

Records the config identity hash + resolved impl names, the
embedder/chunker/normalizer identity (embedder provider/model/dimensions —
the vector space), chunk counts per category, per-file sha256 + status
(ok/failed/cached), the RTL canary result, and timestamps. The query CLI
refuses to run against an index built with an incompatible
embedder/chunker/normalizer (clear error telling you to re-ingest).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.config import RagConfig, config_identity_hash, impl_id

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1

#: The phases whose change makes an existing index unusable.
IDENTITY_PHASES = ("embedder", "chunker", "normalizer")


class ManifestError(Exception):
    """Missing/corrupt manifest."""


class ManifestMismatchError(Exception):
    """Index was built with an incompatible embedder/chunker/normalizer."""


def _phase_identity(config: RagConfig, phase: str) -> dict[str, Any]:
    block = getattr(config, phase)
    identity: dict[str, Any] = {
        "impl": block.impl,
        "params": block.params,
        "id": impl_id(block),
    }
    if phase == "embedder":
        # Called out explicitly: provider/model/dimensions define the vector
        # space — query must match ingest (rag_plan.md §4).
        identity["provider"] = block.impl
        identity["model"] = block.params.get("model")
        identity["dimensions"] = block.params.get("dimensions")
    return identity


def build_manifest(
    config: RagConfig,
    *,
    chunk_counts: dict[str, int] | None = None,
    files: list[dict[str, Any]] | None = None,
    canary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dict. ``files`` entries:
    ``{"file": rel_path, "sha256": …, "status": "ok"|"failed"|"cached"}``."""
    counts = dict(chunk_counts or {})
    return {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_identity": config_identity_hash(config),
        "impls": {
            "parser": config.parser.impl,
            "chunker": config.chunker.impl,
            "normalizer": config.normalizer.impl,
            "embedder": config.embedder.impl,
            "dense_index": config.dense_index.impl,
            "sparse_index": config.sparse_index.impl,
        },
        "embedder": _phase_identity(config, "embedder"),
        "chunker": _phase_identity(config, "chunker"),
        "normalizer": _phase_identity(config, "normalizer"),
        "chunk_counts": counts,
        "total_chunks": sum(counts.values()),
        "files": list(files or []),
        "canary": canary,
    }


def write_manifest(index_dir: Path, manifest: dict[str, Any]) -> Path:
    """Atomic write (tmp + rename) of ``<index_dir>/manifest.json``."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / MANIFEST_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)
    return path


def load_manifest(index_dir: Path) -> dict[str, Any]:
    path = Path(index_dir) / MANIFEST_FILENAME
    if not path.is_file():
        raise ManifestError(
            f"No index manifest at {path} — this index_dir has not been ingested. "
            f"Run: python -m rag.cli.ingest --config <your config>"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Corrupt index manifest at {path}: {exc} — re-ingest.") from exc


def _mismatch_details(manifest: dict[str, Any], config: RagConfig) -> list[str]:
    details: list[str] = []
    for phase in IDENTITY_PHASES:
        recorded = manifest.get(phase, {})
        current = _phase_identity(config, phase)
        if recorded.get("id") == current["id"]:
            continue
        if phase == "embedder":
            for field in ("provider", "model", "dimensions"):
                if recorded.get(field) != current.get(field):
                    details.append(
                        f"embedder {field}: index has {recorded.get(field)!r}, "
                        f"config wants {current.get(field)!r}"
                    )
            if not any(d.startswith("embedder") for d in details):
                details.append(
                    f"embedder params differ (index id {recorded.get('id')}, config id {current['id']})"
                )
        else:
            details.append(
                f"{phase}: index has impl={recorded.get('impl')!r} params={recorded.get('params')!r}, "
                f"config wants impl={current['impl']!r} params={current['params']!r}"
            )
    return details


def verify_manifest(index_dir: Path, config: RagConfig) -> dict[str, Any]:
    """Load the manifest and verify the config's embedder/chunker/normalizer
    identity matches the one the index was built with. Returns the manifest;
    raises ``ManifestMismatchError`` with a re-ingest message on mismatch."""
    manifest = load_manifest(index_dir)
    if manifest.get("config_identity") == config_identity_hash(config):
        return manifest
    details = _mismatch_details(manifest, config) or [
        "config identity hash differs (manifest predates this format?)"
    ]
    raise ManifestMismatchError(
        f"Index at {index_dir} is incompatible with the active config:\n  - "
        + "\n  - ".join(details)
        + "\nRe-ingest with this config (python -m rag.cli.ingest --config …) "
        "or point index_dir at the matching index."
    )

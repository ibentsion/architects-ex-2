"""The one place a request parameter is allowed to become a filesystem path.

Every dataset id, citation file and thumbnail request the bridge serves is
attacker-controlled (threat T-260803-01). No other webapi module may build a
path from request data — it must call :func:`resolve_in_repo`.
"""
from __future__ import annotations

from pathlib import Path

#: Repo root. Tests monkeypatch this, so it is always read through the module
#: (``paths.REPO_ROOT``) rather than imported by value.
REPO_ROOT = Path(__file__).resolve().parents[1]


class PathEscape(ValueError):
    """A requested path resolved outside the repository root."""


def resolve_in_repo(rel: str) -> Path:
    """Resolve a repo-relative path, or raise :class:`PathEscape`.

    Absolute inputs are rejected outright; everything else is resolved (which
    also follows symlinks) and checked against the real repo root, so ``..``
    segments and symlinks pointing outside are both caught.
    """
    candidate = Path(rel)
    if candidate.is_absolute():
        raise PathEscape(f"absolute paths are not accepted: {rel!r}")
    root = REPO_ROOT.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise PathEscape(f"path escapes the repository root: {rel!r}")
    return resolved

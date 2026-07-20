"""trankit Hebrew normalizer adapter (optional dependency).

Selecting ``normalizer: {impl: trankit}`` without the package installed raises
the ImportError below with install instructions (rag_plan.md §2, §3).
"""
from __future__ import annotations

from typing import Any

try:
    import trankit  # noqa: F401
except ImportError as _err:
    raise ImportError(
        "normalizer impl 'trankit' requires the optional dependency 'trankit', "
        "which is not installed.\n"
        "Install it with: pip install trankit\n"
        "(or keep the default: normalizer: {impl: stanza})"
    ) from _err


class TrankitNormalizer:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def tokens(self, text: str) -> list[str]:
        raise NotImplementedError("TrankitNormalizer adapter is planned (rag_plan.md §2) but not implemented yet")

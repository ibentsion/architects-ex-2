"""YAP Hebrew normalizer adapter (optional external dependency).

YAP is a Go program served over HTTP, not a pip package. Selecting
``normalizer: {impl: yap}`` without a reachable YAP setup raises the
ImportError below with setup instructions (rag_plan.md §2, §3).
"""
from __future__ import annotations

import shutil
from typing import Any

if shutil.which("yap") is None:
    raise ImportError(
        "normalizer impl 'yap' requires the YAP morphological analyzer, which "
        "is not a pip package and was not found on PATH.\n"
        "Build it from https://github.com/OnlpLab/yap (Go required) and either "
        "put the 'yap' binary on PATH or run 'yap api' and point this adapter "
        "at the HTTP endpoint.\n"
        "(or keep the default: normalizer: {impl: stanza})"
    )


class YapNormalizer:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def tokens(self, text: str) -> list[str]:
        raise NotImplementedError("YapNormalizer adapter is planned (rag_plan.md §2) but not implemented yet")

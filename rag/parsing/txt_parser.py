"""TXT parser: UTF-8 read, ``page=None``, source_url from manifest.json.

Implemented in wave E2 (T2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class TxtParser:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def parse(self, path: Path) -> Any:
        raise NotImplementedError("TxtParser is implemented in wave E2 (rag_plan.md T2)")

"""Docling PDF parser adapter: DocumentConverter -> DoclingDocument ->
``export_to_dict()`` JSON (never Markdown — rag_plan.md §1.1).

Implemented in wave E2 (T2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class DoclingParser:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def parse(self, path: Path) -> Any:
        raise NotImplementedError("DoclingParser is implemented in wave E2 (rag_plan.md T2)")

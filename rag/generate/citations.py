"""Sources-block parsing + citation validation (rag_plan.md §6 stage 7).

Every cited {file, page} MUST exist in the retrieved chunk set — anything else
is a fabricated citation and is dropped. Implemented in wave E5 (T8)."""
from __future__ import annotations

from typing import Any


def parse_sources_block(text: str) -> list[Any]:
    raise NotImplementedError("Citation parsing is implemented in wave E5 (rag_plan.md T8)")


def validate_citations(citations: list[Any], retrieved: list[Any]) -> list[Any]:
    raise NotImplementedError("Citation validation is implemented in wave E5 (rag_plan.md T8)")

"""Context assembly + tf_client LLM call + one retry on citation failure
(rag_plan.md §6 stages 5-7). Implemented in wave E5 (T8)."""
from __future__ import annotations

from typing import Any


class Generator:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def generate(self, question: str, retrieved: list[Any]) -> Any:
        raise NotImplementedError("Generator is implemented in wave E5 (rag_plan.md T8)")

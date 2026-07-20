"""Stanza Hebrew normalizer: ``tokenize,mwt,pos,lemma``; lemma+surface union;
lowercase Latin; keep digits; normalize gershayim/geresh/maqaf; strip nikud +
punctuation (rag_plan.md §5 stage 4). Implemented in wave E3 (T4)."""
from __future__ import annotations

from typing import Any


class StanzaNormalizer:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def tokens(self, text: str) -> list[str]:
        raise NotImplementedError("StanzaNormalizer is implemented in wave E3 (rag_plan.md T4)")

"""per_page chunker (default): group DoclingDocument items by ``prov.page_no``,
one chunk per page; over-long pages split at paragraph boundaries, fragments
keep the same ``page``. TXT: paragraph-packed to ``txt_max_tokens``,
``page=None``. Implemented in wave E2 (T3)."""
from __future__ import annotations

from typing import Any


class PerPageChunker:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def chunk(self, doc: Any) -> list[Any]:
        raise NotImplementedError("PerPageChunker is implemented in wave E2 (rag_plan.md T3)")

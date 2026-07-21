"""per_page chunker (default) — rag_plan.md §5 stage 3.

One chunk per PDF page (citations correct by construction: the chunk's page
IS the citation page). Pages exceeding ``max_tokens`` are split at paragraph
boundaries — every fragment keeps the same ``page``. Tables stay whole within
their page (serialized as Markdown inline). TXT files are paragraph-packed to
``txt_max_tokens`` with ``page=None``.
"""
from __future__ import annotations

import logging

from rag.chunking.common import (
    DEFAULT_TOKENIZER_ID,
    TokenCounter,
    build_chunks,
    chunk_txt,
    iter_reading_order,
    load_docling,
    pack_paragraphs,
)
from rag.parsing import ParsedDoc
from rag.report import EVENT_NO_PAGE
from rag.types import Chunk

logger = logging.getLogger(__name__)


class PerPageChunker:
    name = "per_page"

    def __init__(
        self,
        max_tokens: int = 1800,
        txt_max_tokens: int = 512,
        tokenizer: str = DEFAULT_TOKENIZER_ID,
    ) -> None:
        self.max_tokens = max_tokens
        self.txt_max_tokens = txt_max_tokens
        self._counter = TokenCounter(tokenizer)

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        if doc.kind == "txt":
            return chunk_txt(doc, self.txt_max_tokens, self._counter, self.name)

        dl_doc = load_docling(doc)
        # Group item texts by page, preserving reading order within each page.
        by_page: dict[int, list[str]] = {}
        for _item, page, text in iter_reading_order(dl_doc):
            if page is None:
                logger.warning(
                    "Skipping PDF item without page provenance (%s): %r",
                    doc.source.rel_path,
                    text[:60],
                    extra={
                        "rag_event": EVENT_NO_PAGE,
                        "rag_file": doc.source.rel_path,
                        "rag_category": doc.source.category,
                        "rag_chunker": self.name,
                        "rag_detail": {"snippet": text[:120]},
                    },
                )
                continue
            if text.strip():
                by_page.setdefault(page, []).append(text)

        pieces: list[tuple[str, int | None]] = []
        for page in sorted(by_page):
            paragraphs = by_page[page]
            page_text = "\n\n".join(paragraphs)
            if self._counter.count(page_text) <= self.max_tokens:
                pieces.append((page_text, page))
            else:
                # Split at paragraph boundaries; all fragments keep this page.
                ctx = {
                    "file": doc.source.rel_path,
                    "category": doc.source.category,
                    "page": page,
                    "chunker": self.name,
                }
                for fragment in pack_paragraphs(
                    paragraphs, self.max_tokens, self._counter, context=ctx
                ):
                    pieces.append((fragment, page))
        return build_chunks(doc, pieces, self.name)

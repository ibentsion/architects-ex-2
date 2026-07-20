"""per_table chunker — rag_plan.md §5 stage 3.

Every ``TableItem`` becomes one ATOMIC chunk: the table serialized as
Markdown with its section-heading context prepended (orphaned table cells
are useless to both retrieval and the reader). Non-table prose falls back to
per_paragraph (HybridChunker), with table-bearing hybrid chunks dropped so
table content is never indexed twice. TXT files are paragraph-packed with
``page=None``.
"""
from __future__ import annotations

import logging

from rag.chunking.common import (
    DEFAULT_TOKENIZER_ID,
    build_chunks,
    chunk_txt,
    iter_reading_order,
    load_docling,
)
from rag.chunking.per_paragraph import PerParagraphChunker, chunk_page
from rag.parsing import ParsedDoc
from rag.types import Chunk

logger = logging.getLogger(__name__)


class PerTableChunker:
    name = "per_table"

    def __init__(
        self,
        prose_max_tokens: int = 512,
        merge_peers: bool = True,
        txt_max_tokens: int = 512,
        tokenizer: str = DEFAULT_TOKENIZER_ID,
    ) -> None:
        self._prose = PerParagraphChunker(
            max_tokens=prose_max_tokens,
            merge_peers=merge_peers,
            txt_max_tokens=txt_max_tokens,
            tokenizer=tokenizer,
        )
        self._counter = self._prose._counter

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        if doc.kind == "txt":
            return chunk_txt(doc, self._prose.txt_max_tokens, self._counter, self.name)

        from docling_core.types.doc import DocItemLabel
        from docling_core.types.doc.document import SectionHeaderItem, TableItem, TitleItem

        dl_doc = load_docling(doc)

        # Walk in reading order, tracking the most recent heading; each table
        # becomes one atomic Markdown chunk with that heading prepended.
        table_pieces: list[tuple[str, int | None]] = []
        current_heading: str | None = None
        for item, page, text in iter_reading_order(dl_doc):
            if isinstance(item, (SectionHeaderItem, TitleItem)):
                current_heading = text.strip() or current_heading
            elif isinstance(item, TableItem):
                if not text.strip():
                    continue
                table_text = f"{current_heading}\n\n{text}" if current_heading else text
                table_pieces.append((table_text, page))

        # Prose fallback: per_paragraph chunks, minus any that contain table
        # items (those are covered — atomically — above).
        prose_pieces: list[tuple[str, int | None]] = []
        hybrid = self._prose._get_hybrid()
        for hybrid_chunk in hybrid.chunk(dl_doc=dl_doc):
            if any(item.label == DocItemLabel.TABLE for item in hybrid_chunk.meta.doc_items):
                continue
            page = chunk_page(hybrid_chunk)
            if page is None:
                logger.warning(
                    "Dropping prose chunk without page provenance (%s): %r",
                    doc.source.rel_path,
                    hybrid_chunk.text[:60],
                )
                continue
            prose_pieces.append((hybrid_chunk.text, page))

        return build_chunks(doc, prose_pieces + table_pieces, self.name)

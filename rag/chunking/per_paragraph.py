"""per_paragraph chunker (alternative to the default per_table -- no table
atomicity: Docling's HybridChunker truncates/summarizes large tables rather
than keeping the full grid, verified 26.3% raw-text retention on a
table-heavy file vs 99.1% for per_table) — rag_plan.md §5 stage 3.

Docling ``HybridChunker`` (hierarchical, structure-aware, tokenizer-fitted
split/merge) over the reconstructed DoclingDocument. The tokenizer is the
embedder model's HF tokenizer (small download — the 8B weights are never
touched); ``page`` comes from the chunk's first ``doc_items[].prov[].page_no``.
TXT files are sentence-window chunked with ``page=None`` (no docling
structure, and no blank-line paragraph breaks either — see
``rag.chunking.common.chunk_txt``).
"""
from __future__ import annotations

import logging
from typing import Any

from rag.chunking.common import (
    DEFAULT_TOKENIZER_ID,
    TokenCounter,
    build_chunks,
    chunk_txt,
    load_docling,
)
from rag.parsing import ParsedDoc
from rag.report import EVENT_NO_PAGE
from rag.types import Chunk

logger = logging.getLogger(__name__)


def chunk_page(hybrid_chunk: Any) -> int | None:
    """First available ``prov.page_no`` across the chunk's doc items."""
    for item in hybrid_chunk.meta.doc_items:
        for prov in getattr(item, "prov", []) or []:
            return prov.page_no
    return None


class PerParagraphChunker:
    name = "per_paragraph"

    def __init__(
        self,
        max_tokens: int = 512,
        merge_peers: bool = True,
        txt_max_tokens: int = 512,
        txt_sentence_count: int = 7,
        txt_sentence_overlap: int = 2,
        tokenizer: str = DEFAULT_TOKENIZER_ID,
    ) -> None:
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.txt_max_tokens = txt_max_tokens
        self.txt_sentence_count = txt_sentence_count
        self.txt_sentence_overlap = txt_sentence_overlap
        self._counter = TokenCounter(tokenizer)
        self._hybrid: Any = None  # lazy: tokenizer load is deferred to first PDF

    def _get_hybrid(self) -> Any:
        if self._hybrid is None:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )

            self._hybrid = HybridChunker(
                tokenizer=HuggingFaceTokenizer(
                    tokenizer=self._counter.require_hf(), max_tokens=self.max_tokens
                ),
                merge_peers=self.merge_peers,
            )
        return self._hybrid

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        if doc.kind == "txt":
            return chunk_txt(
                doc,
                self.txt_max_tokens,
                self._counter,
                self.name,
                sentence_count=self.txt_sentence_count,
                sentence_overlap=self.txt_sentence_overlap,
            )

        dl_doc = load_docling(doc)
        hybrid = self._get_hybrid()
        pieces: list[tuple[str, int | None]] = []
        for hybrid_chunk in hybrid.chunk(dl_doc=dl_doc):
            page = chunk_page(hybrid_chunk)
            if page is None:
                logger.warning(
                    "Dropping HybridChunker chunk without page provenance (%s): %r",
                    doc.source.rel_path,
                    hybrid_chunk.text[:60],
                    extra={
                        "rag_event": EVENT_NO_PAGE,
                        "rag_file": doc.source.rel_path,
                        "rag_category": doc.source.category,
                        "rag_chunker": self.name,
                        "rag_detail": {"snippet": hybrid_chunk.text[:120]},
                    },
                )
                continue
            pieces.append((hybrid_chunk.text, page))
        return build_chunks(doc, pieces, self.name)

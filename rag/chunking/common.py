"""Shared chunking helpers (rag_plan.md §5 stage 3, §8 invariants).

* :class:`TokenCounter` — token counting with the embedder tokenizer, with a
  documented fallback ladder for offline environments.
* Paragraph splitting/packing for TXT files (``page=None`` always).
* :func:`build_chunks` — the single place where §8 invariants are enforced
  and chunk ids are assigned. Every chunker funnels through it.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Iterable

from rag.parsing import KNOWN_CATEGORIES, ParsedDoc
from rag.types import Chunk

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER_ID = "Qwen/Qwen3-Embedding-8B"
_FALLBACK_TOKENIZER_ID = "bert-base-multilingual-cased"


class ChunkInvariantError(ValueError):
    """A produced chunk violates the §8 metadata contract — programming error,
    never silently dropped (citations are graded on this metadata)."""


class TokenCounter:
    """Counts tokens with the embedder's tokenizer (rag_plan.md §4: chunk
    sizes are 'token-counted with the embedder tokenizer').

    Fallback ladder (each step logged):
      1. HF AutoTokenizer for the configured model id (the Qwen3-Embedding
         tokenizer is a small download; the 8B weights are never touched).
      2. Generic multilingual tokenizer (``bert-base-multilingual-cased``).
      3. Regex word-count heuristic (~1.3 tokens/word) — counting only;
         chunkers that NEED a real tokenizer (HybridChunker) raise instead.
    """

    def __init__(self, model_id: str = DEFAULT_TOKENIZER_ID) -> None:
        self.model_id = model_id
        self._hf_tokenizer: Any = None
        self._loaded = False
        self.fallback: str | None = None  # None | "generic" | "heuristic"

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        from transformers import AutoTokenizer

        for candidate, kind in ((self.model_id, None), (_FALLBACK_TOKENIZER_ID, "generic")):
            try:
                self._hf_tokenizer = AutoTokenizer.from_pretrained(candidate)
                self.fallback = kind
                if kind:
                    logger.warning(
                        "Tokenizer %s unavailable — falling back to %s (token counts are approximate)",
                        self.model_id,
                        candidate,
                    )
                return
            except Exception as exc:  # noqa: BLE001 — offline/DNS/auth all land here
                logger.warning("Could not load tokenizer %s: %s", candidate, exc)
        self.fallback = "heuristic"
        logger.warning(
            "No HF tokenizer available — using word-count heuristic (offline mode)."
        )

    @property
    def hf_tokenizer(self) -> Any | None:
        self._load()
        return self._hf_tokenizer

    def count(self, text: str) -> int:
        self._load()
        if self._hf_tokenizer is not None:
            return len(self._hf_tokenizer.encode(text, add_special_tokens=False))
        # ~1.3 tokens/word is a safe over-estimate for Hebrew subword vocabularies
        return int(len(re.findall(r"\S+", text)) * 1.3) + 1

    def require_hf(self) -> Any:
        """The HF tokenizer object, or an actionable error (HybridChunker
        cannot run on the heuristic)."""
        self._load()
        if self._hf_tokenizer is None:
            raise RuntimeError(
                f"per_paragraph/per_table need a real HF tokenizer but neither "
                f"'{self.model_id}' nor '{_FALLBACK_TOKENIZER_ID}' could be loaded "
                f"(offline?). Pre-download one into the HF cache and retry."
            )
        return self._hf_tokenizer


def load_docling(doc: ParsedDoc) -> Any:
    """Reconstruct the DoclingDocument model from the cached dict (§1.1: the
    dict round-trip preserves ``prov.page_no``; Markdown would not)."""
    from docling_core.types.doc import DoclingDocument

    assert doc.docling is not None, f"not a parsed PDF: {doc.source.rel_path}"
    return DoclingDocument.model_validate(doc.docling)


def iter_reading_order(dl_doc: Any) -> Iterable[tuple[Any, int | None, str]]:
    """Yield ``(item, page_no, text)`` for every text/table item in reading
    order. Tables serialize to Markdown (tables stay whole within their page).
    Section headers/titles are TextItem subclasses and are included."""
    from docling_core.types.doc.document import TableItem, TextItem

    for item, _level in dl_doc.iterate_items():
        if isinstance(item, TableItem):
            text = item.export_to_markdown(dl_doc)
        elif isinstance(item, TextItem):
            text = item.text
        else:
            continue
        page = item.prov[0].page_no if item.prov else None
        yield item, page, text


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; strips whitespace; drops empty paragraphs."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def pack_paragraphs(
    paragraphs: Iterable[str], max_tokens: int, counter: TokenCounter
) -> list[str]:
    """Greedy paragraph packing: consecutive paragraphs are joined while the
    pack stays ≤ ``max_tokens``. A single over-long paragraph becomes its own
    fragment (paragraph boundaries are the split unit — plan §5 stage 3)."""
    packs: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        para_tokens = counter.count(para)
        if current and current_tokens + para_tokens > max_tokens:
            packs.append("\n\n".join(current))
            current, current_tokens = [], 0
        if para_tokens > max_tokens and not current:
            logger.warning(
                "Paragraph longer than max_tokens (%d > %d) kept whole", para_tokens, max_tokens
            )
            packs.append(para)
            continue
        current.append(para)
        current_tokens += para_tokens
    if current:
        packs.append("\n\n".join(current))
    return packs


def chunk_txt(doc: ParsedDoc, max_tokens: int, counter: TokenCounter, chunker_name: str) -> list[Chunk]:
    """TXT files: paragraph-packed to ``max_tokens``; ``page=None`` always."""
    assert doc.text is not None
    packs = pack_paragraphs(split_paragraphs(doc.text), max_tokens, counter)
    return build_chunks(doc, [(text, None) for text in packs], chunker_name)


def chunk_id_for(file: str, page: int | None, n: int) -> str:
    return f"{file}#p{page if page is not None else 'null'}#c{n}"


def build_chunks(
    doc: ParsedDoc, pieces: list[tuple[str, int | None]], chunker_name: str
) -> list[Chunk]:
    """Turn ``(text, page)`` pieces into Chunks: drop empties (warned), assign
    per-file ids, enforce every §8 invariant."""
    source = doc.source
    if source.category not in KNOWN_CATEGORIES:
        raise ChunkInvariantError(
            f"Unknown category '{source.category}' for {source.rel_path} — "
            f"known: {sorted(KNOWN_CATEGORIES)}"
        )
    if not source.rel_path.startswith(f"{source.category}/"):
        raise ChunkInvariantError(
            f"file '{source.rel_path}' does not start with 'category/' ({source.category}/)"
        )
    if source.rel_path != unicodedata.normalize("NFC", source.rel_path):
        raise ChunkInvariantError(f"file path is not NFC-normalized: {source.rel_path!r}")

    chunks: list[Chunk] = []
    n = 0
    for text, page in pieces:
        if not text or not text.strip():
            logger.warning("Dropping empty chunk (%s, page=%s)", source.rel_path, page)
            continue
        if source.kind == "pdf" and (page is None or page < 1):
            raise ChunkInvariantError(
                f"PDF chunk must have 1-based page, got {page!r} ({source.rel_path})"
            )
        if source.kind == "txt" and page is not None:
            raise ChunkInvariantError(
                f"TXT chunk must have page=None, got {page!r} ({source.rel_path})"
            )
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(source.rel_path, page, n),
                file=source.rel_path,
                page=page,
                category=source.category,
                text=text,
                source_url=source.source_url,
                chunker=chunker_name,
            )
        )
        n += 1
    return chunks

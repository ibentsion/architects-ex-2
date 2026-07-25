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
from rag.report import EVENT_EMPTY_CHUNK, EVENT_LONG_PARAGRAPH
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


#: Sentence end = ".", "!", or "?" followed by whitespace. Good enough for
#: this corpus's scraped .txt pages (Hebrew/English prose, no abbreviation
#: list needed in practice) -- never used on PDFs, which keep Docling's
#: structure-aware paragraph/page split instead.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation; strips whitespace, drops empties."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def pack_sentences(
    sentences: list[str],
    sentence_count: int,
    overlap: int,
    max_tokens: int,
    counter: TokenCounter,
    *,
    context: dict[str, Any] | None = None,
) -> list[str]:
    """Sliding window over sentences: ``sentence_count`` sentences per chunk,
    the last ``overlap`` of which are repeated at the start of the next chunk
    (stride = ``sentence_count - overlap``) -- the fix for TXT sources that
    have no blank-line paragraph breaks at all (this corpus's scraped pages
    are single unbroken lines; ``pack_paragraphs`` degenerates to one
    giant chunk on them).

    Each window is additionally capped at ``max_tokens`` by dropping trailing
    sentences (never split mid-sentence); a single sentence alone over
    ``max_tokens`` is kept whole and warned about, same as
    ``pack_paragraphs``. If the cap shrinks a window, the advance to the next
    window shrinks with it (never the fixed stride) so no sentence is ever
    skipped without landing in some chunk -- content-loss safety takes
    priority over exact overlap width in that degenerate case.
    """
    if sentence_count <= overlap:
        raise ValueError(f"sentence_count ({sentence_count}) must be > overlap ({overlap})")
    if not sentences:
        return []
    ctx = context or {}
    packs: list[str] = []
    n = len(sentences)
    start = 0
    while start < n:
        window = sentences[start : start + sentence_count]
        while len(window) > 1 and counter.count(" ".join(window)) > max_tokens:
            window = window[:-1]
        text = " ".join(window)
        if len(window) == 1 and counter.count(text) > max_tokens:
            logger.warning(
                "Sentence longer than max_tokens (%d > %d) kept whole [%s]",
                counter.count(text),
                max_tokens,
                ctx.get("file", "?"),
                extra={
                    "rag_event": EVENT_LONG_PARAGRAPH,
                    "rag_file": ctx.get("file"),
                    "rag_category": ctx.get("category"),
                    "rag_page": ctx.get("page"),
                    "rag_chunker": ctx.get("chunker"),
                    "rag_detail": {"tokens": counter.count(text), "max_tokens": max_tokens},
                },
            )
        packs.append(text)
        # Break once THIS window's actual coverage (not the intended
        # sentence_count) reaches the end -- a max_tokens-shrunk window
        # covers fewer sentences than intended, so checking the intended
        # size here would break early and silently drop the tail.
        if start + len(window) >= n:
            break
        # Advance by (this window's actual length - overlap): equals the
        # configured stride in the common case (full-size window); if
        # max_tokens forced a smaller window, the advance shrinks with it so
        # no sentence is ever skipped without landing in some chunk.
        start += max(1, len(window) - overlap)
    return packs


def pack_paragraphs(
    paragraphs: Iterable[str],
    max_tokens: int,
    counter: TokenCounter,
    *,
    context: dict[str, Any] | None = None,
) -> list[str]:
    """Greedy paragraph packing: consecutive paragraphs are joined while the
    pack stays ≤ ``max_tokens``. A single over-long paragraph becomes its own
    fragment (paragraph boundaries are the split unit — plan §5 stage 3).

    ``context`` (file/category/page/chunker), when given, is attached to the
    long-paragraph warning for the ingestion report (rag/report.py) — the
    paragraph text is never split further or dropped, just kept as one
    oversized chunk, so this is a size-tuning signal, not a content-loss one.
    """
    ctx = context or {}
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
                "Paragraph longer than max_tokens (%d > %d) kept whole [%s]",
                para_tokens,
                max_tokens,
                ctx.get("file", "?"),
                extra={
                    "rag_event": EVENT_LONG_PARAGRAPH,
                    "rag_file": ctx.get("file"),
                    "rag_category": ctx.get("category"),
                    "rag_page": ctx.get("page"),
                    "rag_chunker": ctx.get("chunker"),
                    "rag_detail": {"tokens": para_tokens, "max_tokens": max_tokens},
                },
            )
            packs.append(para)
            continue
        current.append(para)
        current_tokens += para_tokens
    if current:
        packs.append("\n\n".join(current))
    return packs


def chunk_txt(
    doc: ParsedDoc,
    max_tokens: int,
    counter: TokenCounter,
    chunker_name: str,
    *,
    sentence_count: int = 7,
    sentence_overlap: int = 2,
) -> list[Chunk]:
    """TXT files: sentence-window chunked (``sentence_count`` sentences per
    chunk, ``sentence_overlap`` shared with the next -- this corpus's .txt
    pages are single unbroken lines with no paragraph breaks to pack on),
    capped at ``max_tokens``; ``page=None`` always."""
    assert doc.text is not None
    ctx = {
        "file": doc.source.rel_path,
        "category": doc.source.category,
        "page": None,
        "chunker": chunker_name,
    }
    packs = pack_sentences(
        split_sentences(doc.text), sentence_count, sentence_overlap, max_tokens, counter, context=ctx
    )
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
            logger.warning(
                "Dropping empty chunk (%s, page=%s)",
                source.rel_path,
                page,
                extra={
                    "rag_event": EVENT_EMPTY_CHUNK,
                    "rag_file": source.rel_path,
                    "rag_category": source.category,
                    "rag_page": page,
                    "rag_chunker": chunker_name,
                },
            )
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

"""Parsing phase: corpus discovery + file -> ParsedDoc (rag_plan.md §5 stages 1-2).

Load-bearing rule (§1.1): downstream chunkers consume the DoclingDocument
dict — NEVER a Markdown export (Markdown destroys ``prov.page_no`` and with
it citation pages).
"""
from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

#: The 12 corpus category directories (rag_plan.md §8 invariant:
#: ``Chunk.category`` must be one of these).
KNOWN_CATEGORIES: frozenset[str] = frozenset(
    {
        "apartment",
        "business",
        "car",
        "dental",
        "diseases-disabilities",
        "health",
        "life",
        "long-term-care",
        "loss-of-working-ability",
        "mortgage",
        "personal-accident",
        "travel",
    }
)


class ParseError(Exception):
    """A single file failed to parse (logged + recorded in the index manifest's
    ``failed_files``; ingest continues — canary failure is the only hard stop).

    Raised by every parser impl, so the ingest CLI has one thing to catch."""


class SourceFile(BaseModel):
    """One discovered corpus file (rag_plan.md §5 stage 1)."""

    model_config = ConfigDict(extra="forbid")

    abs_path: Path
    rel_path: str = Field(
        ...,
        description=(
            "Category-relative POSIX path, NFC-normalized — byte-matches "
            "reference_questions.json's 'file' field. For markdown sources this "
            "names the *PDF* the markdown was rendered from: markdown is another "
            "parse of the same document, so citations stay comparable across "
            "parsers. Only abs_path/sha256 refer to the .md itself."
        ),
    )
    category: str
    kind: Literal["pdf", "txt", "md"]
    sha256: str
    source_url: str | None = Field(None, description="From corpus/manifest.json (data only, never fetched)")


class ParsedDoc(BaseModel):
    """Parse output for one source file (rag_plan.md §5 stage 2).

    PDFs carry the ``DoclingDocument.export_to_dict()`` dict in ``docling``;
    TXT docs carry the whole file text in ``text`` (their chunks get
    ``page=None``). Exactly one of the two is set, matching ``source.kind``.
    """

    model_config = ConfigDict(extra="forbid")

    source: SourceFile
    docling: dict[str, Any] | None = Field(None, description="DoclingDocument dict (PDF only)")
    text: str | None = Field(None, description="Raw UTF-8 text (TXT only)")

    @property
    def kind(self) -> str:
        return self.source.kind


@runtime_checkable
class Parser(Protocol):
    """Parse one corpus file into a ParsedDoc."""

    def parse(self, source: SourceFile) -> ParsedDoc: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_url_manifest(corpus_dir: Path) -> dict[str, str]:
    """``corpus/manifest.json``: {NFC rel_path: source_url}. Data only."""
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {unicodedata.normalize("NFC", key): value for key, value in raw.items()}


#: Which rendering of the corpus documents to walk. ``pdf`` is the Docling
#: path (``files/*.pdf``); ``markdown`` is the Marker/DataLab path
#: (``markdown-files/*.md``). Scraped TXT pages are shared by both.
DocSource = Literal["pdf", "markdown"]


def discover(corpus_dir: Path, doc_source: DocSource = "pdf") -> list[SourceFile]:
    """Stage 1 — walk ``corpus_dir`` and return SourceFile records.

    Layout: ``<corpus_dir>/<category>/files/*.pdf``,
    ``<corpus_dir>/<category>/markdown-files/*.md`` and
    ``<corpus_dir>/<category>/pages/*.txt``. Other assets are skipped.
    ``rel_path`` is the category-relative POSIX path, NFC-normalized.

    ``doc_source="markdown"`` walks the markdown rendering instead of the PDFs.
    Those records still carry the **PDF's** ``rel_path`` (see
    :class:`SourceFile`) so citations mean the same thing in both arms; a
    markdown file with no matching PDF is skipped with a warning rather than
    inventing a citation target.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir} — run get_corpus.py to fetch the frozen snapshot."
        )
    urls = load_url_manifest(corpus_dir)
    docs = (
        ("files", ".pdf", "pdf")
        if doc_source == "pdf"
        else ("markdown-files", ".md", "md")
    )
    sources: list[SourceFile] = []
    for category_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        category = category_dir.name
        if category not in KNOWN_CATEGORIES:
            logger.warning("Skipping unknown category dir: %s", category_dir)
            continue
        for subdir, suffix, kind in (docs, ("pages", ".txt", "txt")):
            base = category_dir / subdir
            if not base.is_dir():
                continue
            for path in sorted(base.iterdir()):
                if not path.is_file():
                    continue
                if path.suffix.lower() != suffix:
                    logger.debug("Skipping non-%s asset: %s", suffix, path)
                    continue
                if kind == "md":
                    pdf = category_dir / "files" / f"{path.stem}.pdf"
                    if not pdf.is_file():
                        logger.warning(
                            "Markdown with no source PDF, skipping: %s", path
                        )
                        continue
                    rel_path = unicodedata.normalize("NFC", f"{category}/files/{pdf.name}")
                else:
                    rel_path = unicodedata.normalize("NFC", f"{category}/{subdir}/{path.name}")
                sources.append(
                    SourceFile(
                        abs_path=path,
                        rel_path=rel_path,
                        category=category,
                        kind=kind,  # type: ignore[arg-type]
                        sha256=sha256_file(path),
                        source_url=urls.get(rel_path),
                    )
                )
    return sources


def _docling_factory(**params: Any) -> Any:
    from rag.parsing.docling_parser import DoclingParser

    return DoclingParser(**params)


def _markdown_factory(**params: Any) -> Any:
    from rag.parsing.markdown_parser import MarkdownParser

    return MarkdownParser(**params)


#: ``markdown`` reads the Marker/DataLab rendering of the same PDFs
#: (``markdown-files/*.md``) instead of running Docling. marker-pdf as a
#: per-file fallback inside the docling parser's remediation ladder (§5 stage
#: 2) is a different thing and is not a registry entry.
REGISTRY: dict[str, Callable[..., Any]] = {
    "docling": _docling_factory,
    "markdown": _markdown_factory,
}


def doc_source_for(parser_impl: str) -> DocSource:
    """Which corpus rendering a parser impl reads."""
    return "markdown" if parser_impl == "markdown" else "pdf"

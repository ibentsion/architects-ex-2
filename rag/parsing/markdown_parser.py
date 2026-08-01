"""Marker/DataLab markdown adapter (alternative to the Docling parser).

`ilan3580/apex-ex2-harel-corpus-markdown` carries a Marker rendering of the
same 350 PDFs at ``<category>/markdown-files/<stem>.md``. Marker's
``--paginate_output`` writes ``{N}------…`` separators, N 0-indexed — verified
on all 350 files, where the recovered page count matches ``pdfinfo`` exactly.
Page provenance therefore survives, and citations stay ``{file, page}``.

Two things make this a drop-in parser rather than a second pipeline:

* **The document identity stays the PDF.** ``SourceFile.rel_path`` names
  ``<category>/files/<stem>.pdf`` even though the bytes parsed are the ``.md``
  — markdown is a different *parse* of the same policy document, so citations
  are directly comparable to the Docling arm. Only ``sha256`` is of the
  markdown, which is what keys the caches.
* **Output is a synthetic ``DoclingDocument`` dict** (``texts[]`` with
  ``prov.page_no``, no ``tables[]``). Every chunker already consumes that
  shape, so none of them need to learn what markdown is. Markdown tables ride
  along inside their page's text, which is why the fair comparison uses the
  page-atomic ``per_page`` chunker (``per_table`` keys off Docling
  ``TableItem``s that have no markdown equivalent).
"""
from __future__ import annotations

import re
from typing import Any

from rag.parsing import ParseError, ParsedDoc, SourceFile

#: Marker page separator: ``{0}------------------------…``
PAGE_MARKER = re.compile(r"^\{(\d+)\}-{5,}\s*$", re.MULTILINE)

#: Marker artifacts that carry no document text. Image references and the
#: LLM-written ``<p>Image: …</p>`` captions are dropped outright — captions are
#: model output, not corpus content, and indexing them would let the parser
#: invent retrievable text. Structural HTML is unwrapped, keeping its contents.
_IMAGE_CAPTION = re.compile(r"<p>\s*Image:.*?</p>", re.DOTALL | re.IGNORECASE)
_IMG_TAG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_CHECKBOX = re.compile(r"<input\b[^>]*>", re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?(?:div|b|i|u|em|strong|td|tr|th|table|li|ul|ol|p|span|sup|sub)\b[^>]*>", re.IGNORECASE)

#: Marker escapes ``$`` as ``\$`` in markdown; the corpus is full of dollar
#: premiums and the backslash would otherwise reach the index and the answer.
_ESCAPES = re.compile(r"\\([$*_#\[\]()~`>+=|{}.!-])")


def clean_markdown(text: str) -> str:
    """Strip Marker artifacts, keeping document text (tables included)."""
    text = _IMAGE_CAPTION.sub(" ", text)
    text = _IMG_TAG.sub(" ", text)
    text = _MD_IMAGE.sub(" ", text)
    text = _CHECKBOX.sub(" ", text)
    text = _BR.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _ESCAPES.sub(r"\1", text)
    # Collapse the blank-line runs the substitutions leave behind, but keep
    # paragraph breaks — chunkers split on them.
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_pages(text: str) -> dict[int, str]:
    """``{1-based page number: markdown}`` from Marker's page separators.

    A separator *precedes* the page it labels, so the text before the first
    marker belongs to no page and is dropped (it is empty in this corpus).
    """
    pages: dict[int, str] = {}
    matches = list(PAGE_MARKER.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        pages[int(match.group(1)) + 1] = body
    return pages


#: ``ProvenanceItem`` requires a bbox and the chunkers reconstruct a real
#: ``DoclingDocument`` from this dict, so a placeholder is mandatory. Markdown
#: carries no geometry; nothing downstream reads these coordinates (the BiDi
#: repair, the only bbox consumer, runs on the Docling arm alone).
_NO_BBOX = {"l": 0.0, "t": 0.0, "r": 0.0, "b": 0.0, "coord_origin": "BOTTOMLEFT"}


def to_docling_dict(text: str) -> dict[str, Any]:
    """Synthetic ``DoclingDocument.export_to_dict()`` shape.

    One text item per markdown block per page, in reading order. Blocks are
    kept whole so a markdown table stays a single item (the per_page chunker
    packs on block boundaries).
    """
    texts: list[dict[str, Any]] = []
    page_nos = sorted(split_pages(text))
    for page_no, raw in sorted(split_pages(text).items()):
        cleaned = clean_markdown(raw)
        if not cleaned:
            continue
        for block in re.split(r"\n\s*\n", cleaned):
            block = block.strip()
            if block:
                texts.append(
                    {
                        "self_ref": f"#/texts/{len(texts)}",
                        # iterate_items() resolves parent/child refs both ways;
                        # an item without a parent crashes the traversal.
                        "parent": {"$ref": "#/body"},
                        "children": [],
                        "label": "text",
                        "content_layer": "body",
                        "prov": [{"page_no": page_no, "bbox": dict(_NO_BBOX), "charspan": [0, len(block)]}],
                        "orig": block,
                        "text": block,
                    }
                )
    return {
        "schema_name": "DoclingDocument",
        "version": "1.7.0",
        "name": "markdown",
        "body": {"self_ref": "#/body", "children": [{"$ref": t["self_ref"]} for t in texts],
                 "content_layer": "body", "name": "_root_", "label": "unspecified"},
        "furniture": {"self_ref": "#/furniture", "children": [], "content_layer": "furniture",
                      "name": "_root_", "label": "unspecified"},
        "groups": [],
        "texts": texts,
        "pictures": [],
        "tables": [],
        "key_value_items": [],
        "form_items": [],
        # The document's real page set. The citation judge treats this as
        # ground truth for "does this page exist" (a page that parsed to
        # nothing is a parse gap, not a fabricated citation), so leaving it
        # empty would make every markdown-arm citation out-of-range. Size is
        # zeroed: markdown carries no geometry and nothing reads it.
        "pages": {
            str(n): {"page_no": n, "size": {"width": 0.0, "height": 0.0}} for n in page_nos
        },
    }


class MarkdownParser:
    """``.md`` -> ParsedDoc carrying a synthetic DoclingDocument dict."""

    #: Where ``rag.parsing.discover`` looks for this parser's documents.
    source_kind = "markdown"

    def parse(self, source: SourceFile) -> ParsedDoc:
        if source.kind != "md":
            raise ValueError(
                f"MarkdownParser only parses markdown, got {source.kind}: {source.rel_path}"
            )
        try:
            raw = source.abs_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ParseError(f"Could not read {source.rel_path}: {exc}") from exc
        if not PAGE_MARKER.search(raw):
            raise ParseError(
                f"No Marker page separators in {source.rel_path} — pages, and so "
                "citations, cannot be recovered"
            )
        return ParsedDoc(source=source, docling=to_docling_dict(raw))

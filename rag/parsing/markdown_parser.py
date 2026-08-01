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
* **Output is a synthetic ``DoclingDocument`` dict**: prose in ``texts[]``
  with ``prov.page_no``, and markdown pipe tables parsed into real
  ``tables[]`` entries. Every chunker already consumes that shape, so none of
  them need to learn what markdown is — including ``per_table``, which keys
  off ``TableItem``s and would otherwise have no markdown equivalent. Lifting
  tables out structurally also keeps their pipe syntax (371 pipe characters in
  one measured page-chunk) out of the prose that gets embedded.
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
#: Captured so the alt text can be matched against the caption line Marker
#: writes straight after the image ("![Accessibility icon](x.jpg)" followed by
#: a bare "Accessibility icon"), which would otherwise reach the index twice.
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
#: "[text](url)" — the link text is document content, the URL is not. Left
#: alone, a mailto link puts the same address in the chunk twice.
_MD_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\([^)]*\)")
_CHECKBOX = re.compile(r"<input\b[^>]*>", re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?(?:div|b|i|u|em|strong|td|tr|th|table|li|ul|ol|p|span|sup|sub)\b[^>]*>", re.IGNORECASE)

#: Marker escapes ``$`` as ``\$`` in markdown; the corpus is full of dollar
#: premiums and the backslash would otherwise reach the index and the answer.
_ESCAPES = re.compile(r"\\([$*_#\[\]()~`>+=|{}.!-])")


def clean_markdown(text: str) -> str:
    """Strip Marker artifacts, keeping document text.

    Markdown tables are left intact here — :func:`to_docling_dict` lifts them
    out into ``tables[]`` first, so by the time this runs on a prose block
    there is no table syntax left to preserve.
    """
    alts = {m.group(1).strip() for m in _MD_IMAGE.finditer(text) if m.group(1).strip()}
    text = _IMAGE_CAPTION.sub(" ", text)
    text = _IMG_TAG.sub(" ", text)
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _CHECKBOX.sub(" ", text)
    text = _BR.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _ESCAPES.sub(r"\1", text)
    if alts:
        # Marker repeats an image's alt text as a bare line under it; that is a
        # description of a logo or icon, not document content.
        text = "\n".join(
            line for line in text.splitlines() if line.strip() not in alts
        )
    # Collapse the blank-line runs the substitutions leave behind, but keep
    # paragraph breaks — chunkers split on them.
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


#: A markdown table row: a line whose first non-space character is a pipe.
_TABLE_ROW = re.compile(r"^\s*\|.*$")
#: The header/body separator: |---|:--:|---|
_TABLE_RULE = re.compile(r"^\s*\|[\s|:-]+\|?\s*$")


def _split_row(line: str) -> list[str]:
    cells = line.strip().split("|")
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return [c.strip() for c in cells]


def parse_table(lines: list[str]) -> dict[str, Any] | None:
    """Markdown pipe table -> DoclingDocument ``TableData``.

    Returns ``None`` for a table with no real content (Marker emits fully empty
    grids for ruled form boxes), so those never become chunks.
    """
    rows = [_split_row(l) for l in lines if not _TABLE_RULE.match(l)]
    rows = [[clean_markdown(c) for c in r] for r in rows if r]
    if not rows or not any(c for r in rows for c in r):
        return None
    # Marker renders ruled form layouts as grids that are mostly empty. Keeping
    # the empty rows/columns costs nothing in content and a great deal in
    # chunk size, because export_to_markdown pads every column to its widest
    # cell — one measured page went from 5.9k to 10.8k characters that way.
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows if any(r)]
    keep = [c for c in range(width) if any(r[c] for r in rows)]
    rows = [[r[c] for c in keep] for r in rows]
    if not rows or not keep:
        return None
    num_cols = len(keep)
    cells: list[dict[str, Any]] = []
    grid: list[list[dict[str, Any]]] = []
    for r, row in enumerate(rows):
        grid_row: list[dict[str, Any]] = []
        for c in range(num_cols):
            cell = {
                "bbox": dict(_NO_BBOX),
                "row_span": 1,
                "col_span": 1,
                "start_row_offset_idx": r,
                "end_row_offset_idx": r + 1,
                "start_col_offset_idx": c,
                "end_col_offset_idx": c + 1,
                "text": row[c] if c < len(row) else "",
                "column_header": r == 0,
                "row_header": False,
                "row_section": False,
            }
            grid_row.append(cell)
            if cell["text"]:
                cells.append(dict(cell))
        grid.append(grid_row)
    return {
        "table_cells": cells,
        "num_rows": len(rows),
        "num_cols": num_cols,
        "orientation": "rot_0",  # markdown tables have no rotation
        "grid": grid,
    }


def split_blocks(page: str) -> list[tuple[str, Any]]:
    """Page markdown -> ``[("text", str) | ("table", TableData)]`` in order."""
    blocks: list[tuple[str, Any]] = []
    buf: list[str] = []
    table: list[str] = []

    def flush_text() -> None:
        if buf:
            body = clean_markdown("\n".join(buf))
            for para in re.split(r"\n\s*\n", body):
                if para.strip():
                    blocks.append(("text", para.strip()))
            buf.clear()

    def flush_table() -> None:
        if table:
            data = parse_table(table)
            if data is not None:
                blocks.append(("table", data))
            table.clear()

    for line in page.splitlines():
        if _TABLE_ROW.match(line):
            flush_text()
            table.append(line)
        else:
            flush_table()
            buf.append(line)
    flush_text()
    flush_table()
    return blocks


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

    Prose becomes ``texts[]``; markdown pipe tables become real ``tables[]``
    entries. Emitting tables structurally rather than leaving their pipe syntax
    inside a text block is what lets the **per_table** chunker run on this arm —
    without it the markdown index can only be compared against ``per_page``,
    not against the production config. It also keeps raw table syntax (371 pipe
    characters in one measured page-chunk) out of the prose that gets embedded.
    """
    texts: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    body_children: list[dict[str, str]] = []
    page_nos = sorted(split_pages(text))
    for page_no, raw in sorted(split_pages(text).items()):
        for kind, block in split_blocks(raw):
            prov = [{"page_no": page_no, "bbox": dict(_NO_BBOX), "charspan": [0, 0]}]
            if kind == "table":
                ref = f"#/tables/{len(tables)}"
                tables.append(
                    {
                        "self_ref": ref,
                        "parent": {"$ref": "#/body"},
                        "children": [],
                        "label": "table",
                        "content_layer": "body",
                        "prov": prov,
                        "captions": [],
                        "references": [],
                        "footnotes": [],
                        "data": block,
                    }
                )
            else:
                ref = f"#/texts/{len(texts)}"
                prov[0]["charspan"] = [0, len(block)]
                texts.append(
                    {
                        "self_ref": ref,
                        # iterate_items() resolves parent/child refs both ways;
                        # an item without a parent crashes the traversal.
                        "parent": {"$ref": "#/body"},
                        "children": [],
                        "label": "text",
                        "content_layer": "body",
                        "prov": prov,
                        "orig": block,
                        "text": block,
                    }
                )
            body_children.append({"$ref": ref})
    return {
        "schema_name": "DoclingDocument",
        "version": "1.7.0",
        "name": "markdown",
        # Reading order: body children are emitted in document order, so
        # per_page/per_table see prose and tables interleaved as they appear.
        "body": {"self_ref": "#/body", "children": body_children,
                 "content_layer": "body", "name": "_root_", "label": "unspecified"},
        "furniture": {"self_ref": "#/furniture", "children": [], "content_layer": "furniture",
                      "name": "_root_", "label": "unspecified"},
        "groups": [],
        "texts": texts,
        "pictures": [],
        "tables": tables,
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

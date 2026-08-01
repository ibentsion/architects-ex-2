"""BiDi word-order repair for Docling output (rag_plan.md §5 stage 2, remediation ladder).

Docling emits Hebrew table cells in **visual** order: the letters inside each
word are correct (which is why the RTL canary passes — it checks letter order),
but the words of every line run left-to-right, i.e. backwards. Measured on this
corpus: 0.3% of body text items are affected, against **99.4% of multi-word
Hebrew table cells** — and tables are where coverage limits, deductibles and
premiums live.

    docling:  | יומית עלות | גיל קבוצת | הרחבה |
    correct:  | עלות יומית | קבוצת גיל | הרחבה |

Docling still owns the table *grid* and the page provenance; all this pass
changes is the word order inside an item's own text, using **pypdfium2**
(already a locked dependency) as the reading-order oracle — pdfium returns
Hebrew in logical order.

Two strategies, in order:

1. **bbox** — re-extract the item's own rectangle with pdfium. This is exact
   even for multi-line cells, which is where reordering heuristics break down.
   Accepted only when the re-extracted text has the *same characters* as
   Docling's, so the pass can reorder but never invent or drop content.
2. **segment** — when the bbox disagrees (a rectangle that clips a neighbour,
   a missing bbox), fall back to text: split the run into maximal pieces that
   occur within a single oracle line, reversed or forward, and emit them
   un-reversed. Applied only if *every* word is explained that way; one
   unexplained word rejects the run. That strictness matters — a greedy pass
   that repairs what it can produces text that is neither the original nor
   correct (e.g. locally swapping ``6 ,5`` inside a run it cannot explain).

Anything neither strategy can adjudicate is left exactly as Docling produced
it, which also makes the pass **idempotent**: repaired text matches forward, so
a second run is a no-op.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

#: Longest word run considered for a single line match. Real PDF lines are far
#: shorter; the cap bounds the segmentation search.
LINE_CAP = 48

#: A segment must contain at least this many words. Single words carry no order
#: information, so allowing them would let any run be "explained".
MIN_RUN = 2

#: Docling and pdfium disagree on punctuation, decoration (``**שטח`` vs
#: ``שטח``) and even word boundaries — Docling splits final letters off
#: ("ם נוספי" for "נוספים"). Matching therefore happens on the *characters* of
#: a run with everything non-alphanumeric removed, which is blind to all three;
#: the original words are what gets emitted.
_NOISE = re.compile(r"[^\w֐-׿]+|_+")

#: Shortest character key that may be matched — two Hebrew words are normally
#: well above this, and it keeps very short runs from matching by accident.
MIN_KEY_CHARS = 4

#: Character n-gram width used to shortlist candidate lines.
_GRAM = 4

#: Points of slack around a cell rectangle: Docling's bbox is derived from the
#: cell's glyphs, so an exact rectangle can clip the outermost ones.
_BBOX_PAD = 1.0

_HEBREW = re.compile(r"[֐-׿]")


class RepairStats(BaseModel):
    """What one document's repair pass changed."""

    model_config = ConfigDict(extra="forbid")

    cells_total: int = 0
    cells_repaired: int = 0
    texts_total: int = 0
    texts_repaired: int = 0
    by_bbox: int = 0
    by_segment: int = 0
    pages_without_oracle: list[int] = Field(
        default_factory=list, description="Pages pdfium returned no text for — left untouched"
    )

    @property
    def repaired(self) -> int:
        return self.cells_repaired + self.texts_repaired

    def merge(self, other: "RepairStats") -> None:
        self.cells_total += other.cells_total
        self.cells_repaired += other.cells_repaired
        self.texts_total += other.texts_total
        self.texts_repaired += other.texts_repaired
        self.by_bbox += other.by_bbox
        self.by_segment += other.by_segment
        self.pages_without_oracle = sorted(
            set(self.pages_without_oracle) | set(other.pages_without_oracle)
        )


def _key(words: Iterable[str]) -> str:
    return _NOISE.sub("", " ".join(words)).casefold()


def has_hebrew(text: str) -> bool:
    return bool(_HEBREW.search(text))


class PageOracle:
    """One page's reading order as pdfium sees it."""

    __slots__ = ("lines", "height", "_by_gram", "_textpage")

    def __init__(self, text: str, height: float, textpage: Any = None) -> None:
        self.height = height
        self._textpage = textpage
        #: one character key per visual line — a match may never span a line
        #: break, because only *within* a line is the word order reversed
        self.lines: list[str] = []
        self._by_gram: dict[str, set[int]] = {}
        for raw in text.splitlines():
            key = _key(raw.split())
            if not key:
                continue
            idx = len(self.lines)
            self.lines.append(key)
            for pos in range(len(key) - _GRAM + 1):
                self._by_gram.setdefault(key[pos : pos + _GRAM], set()).add(idx)

    def __bool__(self) -> bool:
        return bool(self.lines)

    def contains(self, words: list[str]) -> bool:
        """Do these words, as characters, occur inside a single line?"""
        key = _key(words)
        if len(key) < MIN_KEY_CHARS:
            return False
        candidates = self._by_gram.get(key[:_GRAM])
        if not candidates:
            return False
        return any(key in self.lines[idx] for idx in candidates)

    def text_in(self, bbox: dict[str, Any]) -> str | None:
        """Text inside a Docling bbox, in pdfium's reading order."""
        if self._textpage is None or not bbox:
            return None
        try:
            left, right = float(bbox["l"]), float(bbox["r"])
            top, bottom = float(bbox["t"]), float(bbox["b"])
        except (KeyError, TypeError, ValueError):
            return None
        if bbox.get("coord_origin") == "TOPLEFT":
            top, bottom = self.height - top, self.height - bottom
        if bottom > top:
            top, bottom = bottom, top
        try:
            raw = self._textpage.get_text_bounded(
                left=left - _BBOX_PAD,
                bottom=bottom - _BBOX_PAD,
                right=right + _BBOX_PAD,
                top=top + _BBOX_PAD,
            )
        except Exception:  # noqa: BLE001 — degenerate rectangles raise; fall back
            return None
        return " ".join(raw.split()) or None


def repair_run(text: str, oracle: PageOracle) -> str:
    """Re-order one text run against the page oracle's lines.

    Returns ``text`` unchanged unless every word is explained by an oracle line.
    """
    words = text.split()
    if len(words) < MIN_RUN or not has_hebrew(text):
        return text
    if oracle.contains(words):
        return text  # already reads correctly

    out: list[str] = []
    i = 0
    n = len(words)
    reversed_any = False
    while i < n:
        matched = 0
        for length in range(min(LINE_CAP, n - i), MIN_RUN - 1, -1):
            run = words[i : i + length]
            if oracle.contains(run):  # forward-readable stretch — keep as is
                out.extend(run)
                matched = length
                break
            if oracle.contains(run[::-1]):
                out.extend(reversed(run))
                matched = length
                reversed_any = True
                break
        if not matched:
            return text  # unexplained word — reject the whole run
        i += matched

    if not reversed_any:
        return text
    repaired = " ".join(out)
    return repaired if repaired != " ".join(words) else text


def repair_item_text(text: str, bbox: dict[str, Any] | None, oracle: PageOracle) -> tuple[str, str]:
    """``(text, strategy)`` where strategy is ``bbox`` / ``segment`` / ``""``."""
    if not text.strip() or not has_hebrew(text):
        return text, ""
    from_bbox = oracle.text_in(bbox or {})
    if from_bbox is not None and from_bbox != text:
        here, there = _key([text]), _key([from_bbox])
        # Same characters (so this can only reorder, never invent or drop
        # content) but a different sequence (so it is a real reordering, not
        # pdfium and Docling merely disagreeing about spaces or decoration —
        # in which case Docling's text is left alone, keeping markers like **).
        if here != there and sorted(here) == sorted(there):
            return from_bbox, "bbox"
    repaired = repair_run(text, oracle)
    return (repaired, "segment") if repaired != text else (text, "")


def _iter_table_cells(table: dict[str, Any]) -> Iterable[tuple[dict[str, Any], bool]]:
    """``(cell, counted)`` for every cell dict of a table.

    ``table_cells`` and ``grid`` hold *separate copies* of the same cells in the
    exported JSON and both must be repaired (``export_to_markdown`` reads
    ``grid``, the canary reads ``table_cells``), but only one copy is counted.
    """
    data = table.get("data") or {}
    for cell in data.get("table_cells") or []:
        yield cell, True
    for row in data.get("grid") or []:
        for cell in row:
            yield cell, False


def _page_of(item: dict[str, Any]) -> int | None:
    prov = item.get("prov") or []
    return prov[0].get("page_no") if prov else None


def _repairable(doc_dict: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any], dict | None, str, bool]]:
    """``(page, item, bbox, kind, counted)`` for everything the pass may touch."""
    for item in doc_dict.get("texts", []):
        page = _page_of(item)
        prov = item.get("prov") or []
        if page is not None:
            yield page, item, (prov[0].get("bbox") if prov else None), "text", True
    for table in doc_dict.get("tables", []):
        page = _page_of(table)
        if page is None:
            continue
        for cell, counted in _iter_table_cells(table):
            yield page, cell, cell.get("bbox"), "cell", counted


def repair_docling(doc_dict: dict[str, Any], pdf_path: Path) -> RepairStats:
    """Repair a ``DoclingDocument.export_to_dict()`` in place; return what changed.

    Text items keep their ``orig`` field untouched (Docling's raw extraction);
    only ``text``, which is what the chunkers consume, is re-ordered.
    """
    import pypdfium2 as pdfium

    stats = RepairStats()
    by_page: dict[int, list[tuple[dict[str, Any], dict | None, str, bool]]] = {}
    for page, item, bbox, kind, counted in _repairable(doc_dict):
        by_page.setdefault(page, []).append((item, bbox, kind, counted))
        if counted:
            if kind == "cell":
                stats.cells_total += 1
            else:
                stats.texts_total += 1
    if not by_page:
        return stats

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 — a broken PDF must not fail ingestion
        logger.warning("BiDi repair skipped for %s — pdfium could not open it: %s", pdf_path, exc)
        return stats

    missing: list[int] = []
    try:
        n_pages = len(doc)
        for page_no in sorted(by_page):
            if not 1 <= page_no <= n_pages:
                missing.append(page_no)
                continue
            page = doc[page_no - 1]
            textpage = page.get_textpage()
            try:
                oracle = PageOracle(textpage.get_text_range(), page.get_height(), textpage)
                if not oracle:
                    missing.append(page_no)
                    continue
                for item, bbox, kind, counted in by_page[page_no]:
                    text = item.get("text") or ""
                    repaired, strategy = repair_item_text(text, bbox, oracle)
                    if not strategy:
                        continue
                    item["text"] = repaired
                    if not counted:
                        continue
                    if kind == "cell":
                        stats.cells_repaired += 1
                    else:
                        stats.texts_repaired += 1
                    if strategy == "bbox":
                        stats.by_bbox += 1
                    else:
                        stats.by_segment += 1
            finally:
                textpage.close()
                page.close()
    finally:
        doc.close()

    stats.pages_without_oracle = sorted(set(missing))
    return stats

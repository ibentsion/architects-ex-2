"""BiDi word-order repair tests (rag/parsing/rtl_repair.py).

Docling emits Hebrew table cells in visual (backwards) word order; these tests
pin the repair's two guarantees — it fixes reversed runs, and it never invents,
drops or half-rewrites content.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.parsing import SourceFile
from rag.parsing.docling_parser import DoclingParser
from rag.parsing.rtl_repair import (
    PageOracle,
    RepairStats,
    _key,
    repair_docling,
    repair_run,
)


# --------------------------------------------------------------------------- #
# PageOracle
# --------------------------------------------------------------------------- #


def test_oracle_matches_within_a_line_only() -> None:
    oracle = PageOracle("עלות יומית לנוסע\nקבוצת גיל", height=800.0)
    assert oracle.contains(["עלות", "יומית"])
    assert oracle.contains(["קבוצת", "גיל"])
    # "לנוסע קבוצת" spans a line break — order across lines is not reversed,
    # so matching there would licence bogus repairs.
    assert not oracle.contains(["לנוסע", "קבוצת"])


def test_oracle_ignores_punctuation_and_word_splits() -> None:
    oracle = PageOracle("פרק ט - ביטוח תאונות אישיות", height=800.0)
    assert oracle.contains(["פרק", "ט", "ביטוח"])
    # Docling splits final letters off ("ם נוספי" for "נוספים"); matching is
    # character-level, so the split is invisible to it.
    split = PageOracle("כיסויים נוספים", height=800.0)
    assert split.contains(["כיסויי", "ם", "נוספים"])


def test_empty_oracle_is_falsy() -> None:
    assert not PageOracle("   \n\n", height=800.0)


# --------------------------------------------------------------------------- #
# repair_run — the text-only fallback strategy
# --------------------------------------------------------------------------- #


def test_repairs_a_reversed_line() -> None:
    oracle = PageOracle("חיפוש איתור וחילוץ", height=800.0)
    assert repair_run("וחילוץ איתור חיפוש", oracle) == "חיפוש איתור וחילוץ"


def test_leaves_correct_text_alone() -> None:
    oracle = PageOracle("חיפוש איתור וחילוץ", height=800.0)
    assert repair_run("חיפוש איתור וחילוץ", oracle) == "חיפוש איתור וחילוץ"


def test_is_idempotent() -> None:
    oracle = PageOracle("חיפוש איתור וחילוץ", height=800.0)
    once = repair_run("וחילוץ איתור חיפוש", oracle)
    assert repair_run(once, oracle) == once


def test_multi_line_cell_keeps_line_order_and_reverses_each_line() -> None:
    oracle = PageOracle("ניתן לבטח רק את הפריטים\nכסא גלגלים כלי נגינה", height=800.0)
    # Docling concatenates reversed line 1 then reversed line 2.
    docling = "הפריטים את רק לבטח ניתן נגינה כלי גלגלים כסא"
    assert repair_run(docling, oracle) == "ניתן לבטח רק את הפריטים כסא גלגלים כלי נגינה"


def test_rejects_runs_it_cannot_fully_explain() -> None:
    """A partial repair is worse than none — it yields text that is neither the
    original nor correct (the ``6 ,5`` swap that an earlier greedy pass made)."""
    oracle = PageOracle("הרחבות פרקים 5, 6, 14-25", height=800.0)
    docling = "הרחבות )פרקים 5, 6, אחר-לגמרי("
    assert repair_run(docling, oracle) == docling


def test_ignores_runs_without_hebrew() -> None:
    oracle = PageOracle("total daily cost", height=800.0)
    assert repair_run("cost daily total", oracle) == "cost daily total"


def test_single_word_is_never_touched() -> None:
    oracle = PageOracle("הרחבה אחרת", height=800.0)
    assert repair_run("הרחבה", oracle) == "הרחבה"


# --------------------------------------------------------------------------- #
# repair_docling — whole-document pass over a real PDF
# --------------------------------------------------------------------------- #


def _reversed_words(text: str) -> str:
    return " ".join(reversed(text.split()))


@pytest.fixture(scope="module")
def anchor_page_line(anchor_pdf: Path) -> str:
    """A real multi-word Hebrew line from page 1 of the ground-truth anchor."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(anchor_pdf))
    try:
        textpage = doc[0].get_textpage()
        raw = textpage.get_text_range()
        textpage.close()
    finally:
        doc.close()
    lines = [" ".join(l.split()) for l in raw.splitlines()]
    candidates = [l for l in lines if len(l.split()) >= 4 and any("א" <= c <= "ת" for c in l)]
    assert candidates, "anchor PDF page 1 has no usable Hebrew line"
    return candidates[0]


def test_repairs_reversed_table_cell_in_real_pdf(anchor_pdf: Path, anchor_page_line: str) -> None:
    doc_dict = {
        "texts": [],
        "tables": [
            {
                "prov": [{"page_no": 1}],
                "data": {
                    "table_cells": [{"text": _reversed_words(anchor_page_line)}],
                    "grid": [[{"text": _reversed_words(anchor_page_line)}]],
                },
            }
        ],
    }
    stats = repair_docling(doc_dict, anchor_pdf)
    assert stats.cells_repaired == 1
    assert stats.cells_total == 1  # grid copy is repaired but not double-counted
    cells = doc_dict["tables"][0]["data"]
    assert cells["table_cells"][0]["text"] == anchor_page_line
    # grid holds a separate copy and feeds export_to_markdown — it must be fixed too
    assert cells["grid"][0][0]["text"] == anchor_page_line


def test_repair_is_idempotent_on_real_pdf(anchor_pdf: Path, anchor_page_line: str) -> None:
    doc_dict = {
        "texts": [{"text": _reversed_words(anchor_page_line), "prov": [{"page_no": 1}]}],
        "tables": [],
    }
    first = repair_docling(doc_dict, anchor_pdf)
    assert first.texts_repaired == 1
    after_once = doc_dict["texts"][0]["text"]
    second = repair_docling(doc_dict, anchor_pdf)
    assert second.texts_repaired == 0
    assert doc_dict["texts"][0]["text"] == after_once


def test_never_invents_or_drops_characters(anchor_pdf: Path, anchor_page_line: str) -> None:
    reversed_text = _reversed_words(anchor_page_line)
    doc_dict = {"texts": [{"text": reversed_text, "prov": [{"page_no": 1}]}], "tables": []}
    repair_docling(doc_dict, anchor_pdf)
    assert sorted(_key([doc_dict["texts"][0]["text"]])) == sorted(_key([reversed_text]))


def test_orig_field_is_left_untouched(anchor_pdf: Path, anchor_page_line: str) -> None:
    reversed_text = _reversed_words(anchor_page_line)
    doc_dict = {
        "texts": [{"text": reversed_text, "orig": reversed_text, "prov": [{"page_no": 1}]}],
        "tables": [],
    }
    repair_docling(doc_dict, anchor_pdf)
    assert doc_dict["texts"][0]["orig"] == reversed_text
    assert doc_dict["texts"][0]["text"] != reversed_text


def test_page_beyond_the_pdf_is_left_alone(anchor_pdf: Path) -> None:
    doc_dict = {"texts": [{"text": "וחילוץ איתור חיפוש", "prov": [{"page_no": 999}]}], "tables": []}
    stats = repair_docling(doc_dict, anchor_pdf)
    assert stats.repaired == 0
    assert stats.pages_without_oracle == [999]
    assert doc_dict["texts"][0]["text"] == "וחילוץ איתור חיפוש"


def test_unreadable_pdf_does_not_raise(tmp_path: Path) -> None:
    """A broken file must not fail ingestion — the parse is kept as-is."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    doc_dict = {"texts": [{"text": "וחילוץ איתור חיפוש", "prov": [{"page_no": 1}]}], "tables": []}
    stats = repair_docling(doc_dict, broken)
    assert stats.repaired == 0
    assert doc_dict["texts"][0]["text"] == "וחילוץ איתור חיפוש"


def test_items_without_page_provenance_are_skipped(anchor_pdf: Path) -> None:
    doc_dict = {"texts": [{"text": "וחילוץ איתור חיפוש", "prov": []}], "tables": []}
    assert repair_docling(doc_dict, anchor_pdf).repaired == 0


# --------------------------------------------------------------------------- #
# Parser wiring
# --------------------------------------------------------------------------- #


def test_docling_parser_applies_the_repair(anchor_pdf: Path, anchor_page_line: str, monkeypatch) -> None:
    """The repair is part of parsing, not an optional post-step."""
    reversed_text = _reversed_words(anchor_page_line)

    class FakeDocument:
        def export_to_dict(self) -> dict:
            return {
                "texts": [{"text": reversed_text, "prov": [{"page_no": 1}]}],
                "tables": [],
            }

    class FakeConverter:
        def convert(self, path):  # noqa: ANN001
            return type("Result", (), {"document": FakeDocument()})()

    parser = DoclingParser()
    monkeypatch.setattr(parser, "_get_converter", lambda: FakeConverter())
    source = SourceFile(
        abs_path=anchor_pdf,
        rel_path="apartment/files/anchor.pdf",
        category="apartment",
        kind="pdf",
        sha256="b" * 64,
    )
    parsed = parser.parse(source)
    assert parsed.docling is not None
    assert parsed.docling["texts"][0]["text"] == anchor_page_line


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def test_stats_merge_unions_pages_without_oracle() -> None:
    a = RepairStats(cells_repaired=1, by_bbox=1, pages_without_oracle=[3, 5])
    b = RepairStats(texts_repaired=2, by_segment=2, pages_without_oracle=[5, 7])
    a.merge(b)
    assert a.repaired == 3
    assert a.by_bbox == 1 and a.by_segment == 2
    assert a.pages_without_oracle == [3, 5, 7]

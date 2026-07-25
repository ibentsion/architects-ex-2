"""Chunking-phase tests (rag_plan.md §9 test_chunking.py list).

Fast tests build synthetic DoclingDocuments programmatically via docling_core
(schema-proof: the dict round-trip is the real ``export_to_dict()``). Tests
that need a real Docling parse of the mini corpus are marked ``slow`` (served
from the shared parse cache after the first run).
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from rag.chunking.common import (
    ChunkInvariantError,
    TokenCounter,
    build_chunks,
    chunk_id_for,
    pack_sentences,
    split_paragraphs,
    split_sentences,
)
from rag.chunking.per_page import PerPageChunker
from rag.chunking.per_paragraph import PerParagraphChunker
from rag.chunking.per_table import PerTableChunker
from rag.parsing import ParsedDoc, SourceFile
from tests.conftest import ANCHOR_PDF_REL

# --------------------------------------------------------------------------- #
# Synthetic document builders
# --------------------------------------------------------------------------- #


def make_source(rel_path: str = "apartment/files/doc.pdf", kind: str = "pdf") -> SourceFile:
    return SourceFile(
        abs_path=Path("/nonexistent") / rel_path,
        rel_path=rel_path,
        category=rel_path.split("/", 1)[0],
        kind=kind,  # type: ignore[arg-type]
        sha256="c" * 64,
        source_url="https://example.invalid/doc",
    )


def _new_doc(n_pages: int):
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.base import Size

    doc = DoclingDocument(name="synthetic")
    for page_no in range(1, n_pages + 1):
        doc.add_page(page_no=page_no, size=Size(width=595, height=842))
    return doc


def _prov(page_no: int, text: str):
    from docling_core.types.doc import BoundingBox, ProvenanceItem

    return ProvenanceItem(
        page_no=page_no, bbox=BoundingBox(l=0, t=0, r=500, b=50), charspan=(0, len(text))
    )


PAGE1_TEXTS = ["הודעה על תקופת התיישנות", "תקופת ההתיישנות היא שלוש שנים ממועד קרות מקרה הביטוח"]
PAGE2_TEXTS = ["פרטים נוספים זמינים באתר הראל", "ניתן לפנות למוקד שירות הלקוחות בכל שאלה"]


def two_page_parsed() -> ParsedDoc:
    from docling_core.types.doc import DocItemLabel

    doc = _new_doc(2)
    for text in PAGE1_TEXTS:
        doc.add_text(label=DocItemLabel.PARAGRAPH, text=text, prov=_prov(1, text))
    for text in PAGE2_TEXTS:
        doc.add_text(label=DocItemLabel.PARAGRAPH, text=text, prov=_prov(2, text))
    return ParsedDoc(source=make_source(), docling=doc.export_to_dict())


def table_parsed() -> ParsedDoc:
    from docling_core.types.doc import DocItemLabel, TableCell, TableData

    doc = _new_doc(1)
    intro = "מסמך זה מפרט את הכיסויים בפוליסת ביטוח הדירה"
    doc.add_text(label=DocItemLabel.PARAGRAPH, text=intro, prov=_prov(1, intro))
    heading = "טבלת סכומי כיסוי"
    doc.add_heading(text=heading, prov=_prov(1, heading))

    def cell(text: str, row: int, col: int, header: bool = False) -> TableCell:
        return TableCell(
            text=text,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
            column_header=header,
        )

    data = TableData(
        num_rows=2,
        num_cols=2,
        table_cells=[
            cell("כיסוי", 0, 0, header=True),
            cell("סכום מרבי", 0, 1, header=True),
            cell("נזקי מים", 1, 0),
            cell("6,000 ₪", 1, 1),
        ],
    )
    doc.add_table(data=data, prov=_prov(1, "table"))
    return ParsedDoc(source=make_source(), docling=doc.export_to_dict())


def txt_parsed(sentence_count: int = 20) -> ParsedDoc:
    """Single unbroken line, no blank-line paragraph breaks -- matches the
    real corpus's scraped .txt pages (see rag/chunking/common.py split_sentences)."""
    text = " ".join(
        f"משפט מספר {i} על ביטול פוליסת ביטוח דירה ותנאי ההחזר הכספי."
        for i in range(sentence_count)
    )
    return ParsedDoc(source=make_source("apartment/pages/cancellation.txt", kind="txt"), text=text)


@pytest.fixture(scope="module")
def counter() -> TokenCounter:
    return TokenCounter()


# --------------------------------------------------------------------------- #
# split_sentences / pack_sentences (TXT chunking primitives)
# --------------------------------------------------------------------------- #


def test_split_sentences_splits_on_terminal_punctuation():
    text = "משפט ראשון. משפט שני! משפט שלישי?"
    assert split_sentences(text) == ["משפט ראשון.", "משפט שני!", "משפט שלישי?"]


def test_split_sentences_drops_empty_and_strips_whitespace():
    text = "  משפט אחד.   \n\n  משפט שני.  "
    assert split_sentences(text) == ["משפט אחד.", "משפט שני."]


def test_split_sentences_single_run_on_line_stays_one_sentence():
    # This corpus's real .txt pages are single unbroken lines with no
    # terminal punctuation between some clauses -- no split point means
    # no split, which is exactly what exposed the old paragraph-packing bug.
    assert split_sentences("אין כאן סימני פיסוק בסוף") == ["אין כאן סימני פיסוק בסוף"]


def test_pack_sentences_rejects_overlap_ge_sentence_count(counter):
    with pytest.raises(ValueError, match="must be > overlap"):
        pack_sentences(["א", "ב"], sentence_count=2, overlap=2, max_tokens=512, counter=counter)


def test_pack_sentences_empty_input_returns_empty(counter):
    assert pack_sentences([], sentence_count=7, overlap=2, max_tokens=512, counter=counter) == []


def test_pack_sentences_fewer_than_window_returns_one_pack(counter):
    sentences = ["משפט אחד.", "משפט שני.", "משפט שלישי."]
    packs = pack_sentences(sentences, sentence_count=7, overlap=2, max_tokens=512, counter=counter)
    assert packs == [" ".join(sentences)]


def test_pack_sentences_stride_and_overlap():
    sentences = [f"משפט{i}" for i in range(20)]  # no internal spaces -> 1 "word" each

    class _WordCounter:
        def count(self, text):
            return len(text.split())

    packs = pack_sentences(sentences, sentence_count=7, overlap=2, max_tokens=10_000, counter=_WordCounter())
    windows = [p.split(" ") for p in packs]
    starts = [sentences.index(w[0]) for w in windows]
    assert starts == [0, 5, 10, 15]  # stride = sentence_count - overlap = 5
    assert [len(w) for w in windows] == [7, 7, 7, 5]


def test_pack_sentences_max_tokens_shrinks_window_and_advance_without_skipping():
    sentences = [f"משפט{i}" for i in range(10)]  # 1 "word"/sentence

    class _WordCounter:
        def count(self, text):
            return len(text.split())

    # cap=1 word forces every window down to a single sentence
    packs = pack_sentences(sentences, sentence_count=7, overlap=2, max_tokens=1, counter=_WordCounter())
    assert packs == sentences  # one sentence per chunk, none skipped, none duplicated


# --------------------------------------------------------------------------- #
# per_page
# --------------------------------------------------------------------------- #


def test_per_page_one_chunk_per_page_order_preserved():
    chunks = PerPageChunker().chunk(two_page_parsed())
    assert [c.page for c in chunks] == [1, 2]
    assert all(c.chunker == "per_page" for c in chunks)
    assert PAGE1_TEXTS[0] in chunks[0].text and PAGE1_TEXTS[1] in chunks[0].text
    assert PAGE2_TEXTS[0] in chunks[1].text


def test_per_page_overlong_page_splits_at_paragraph_boundaries_same_page():
    chunks = PerPageChunker(max_tokens=12).chunk(two_page_parsed())
    page1 = [c for c in chunks if c.page == 1]
    assert len(page1) == 2  # split, one fragment per paragraph
    assert [c.text for c in page1] == PAGE1_TEXTS
    assert all(c.page == 1 for c in page1)  # fragments keep the same page


def test_per_page_txt_sentence_windowed_with_overlap():
    # 20 sentences, default window=7/overlap=2 (stride 5) -> 4 windows:
    # [0:7], [5:12], [10:17], [15:20] (tail, shorter).
    doc = txt_parsed(sentence_count=20)
    chunks = PerPageChunker(txt_max_tokens=512).chunk(doc)
    assert all(c.page is None for c in chunks)
    assert len(chunks) == 4
    windows = [split_sentences(c.text) for c in chunks]
    assert [len(w) for w in windows] == [7, 7, 7, 5]
    # consecutive windows share exactly the configured 2-sentence overlap
    assert windows[0][-2:] == windows[1][:2]
    assert windows[1][-2:] == windows[2][:2]
    assert windows[2][-2:] == windows[3][:2]
    # nothing lost: every sentence appears in at least one chunk
    all_sentences = split_sentences(doc.text)
    assert set(s for w in windows for s in w) == set(all_sentences)


def test_txt_sentence_window_shrinks_under_tight_max_tokens_without_losing_sentences(counter):
    # max_tokens too small for even 2 sentences together -> every window
    # shrinks to 1 sentence; advance must shrink too, or sentences get skipped.
    doc = txt_parsed(sentence_count=20)
    chunks = PerPageChunker(txt_max_tokens=40).chunk(doc)
    assert all(c.page is None for c in chunks)
    assert all(counter.count(c.text) <= 40 for c in chunks)
    all_sentences = split_sentences(doc.text)
    assert [c.text for c in chunks] == all_sentences  # one sentence per chunk, none skipped


def test_chunk_metadata_and_id_format():
    chunks = PerPageChunker().chunk(two_page_parsed())
    first = chunks[0]
    assert first.chunk_id == "apartment/files/doc.pdf#p1#c0"
    assert first.file == "apartment/files/doc.pdf"
    assert first.category == "apartment"
    assert first.source_url == "https://example.invalid/doc"
    txt_chunks = PerPageChunker().chunk(txt_parsed())
    assert txt_chunks[0].chunk_id.startswith("apartment/pages/cancellation.txt#pnull#c")


# --------------------------------------------------------------------------- #
# §8 invariants (build_chunks is the single enforcement point)
# --------------------------------------------------------------------------- #


def test_invariant_unknown_category_rejected():
    doc = ParsedDoc(source=make_source("nonsense/files/x.pdf"), docling={"texts": [], "tables": []})
    with pytest.raises(ChunkInvariantError, match="Unknown category"):
        build_chunks(doc, [("טקסט", 1)], "per_page")


def test_invariant_pdf_chunk_requires_1based_page():
    doc = ParsedDoc(source=make_source(), docling={"texts": [], "tables": []})
    with pytest.raises(ChunkInvariantError, match="1-based page"):
        build_chunks(doc, [("טקסט", None)], "per_page")
    with pytest.raises(ChunkInvariantError, match="1-based page"):
        build_chunks(doc, [("טקסט", 0)], "per_page")


def test_invariant_txt_chunk_requires_page_none():
    doc = txt_parsed()
    with pytest.raises(ChunkInvariantError, match="page=None"):
        build_chunks(doc, [("טקסט", 1)], "per_page")


def test_invariant_file_must_start_with_category():
    source = make_source().model_copy(update={"rel_path": "files/doc.pdf"})
    doc = ParsedDoc(source=source, docling={"texts": [], "tables": []})
    with pytest.raises(ChunkInvariantError, match="category"):
        build_chunks(doc, [("טקסט", 1)], "per_page")


def test_empty_chunks_dropped_with_ids_stable():
    doc = ParsedDoc(source=make_source(), docling={"texts": [], "tables": []})
    chunks = build_chunks(doc, [("", 1), ("  \n ", 1), ("תוכן אמיתי", 2)], "per_page")
    assert len(chunks) == 1
    assert chunks[0].chunk_id == chunk_id_for("apartment/files/doc.pdf", 2, 0)


# --------------------------------------------------------------------------- #
# per_paragraph (HybridChunker)
# --------------------------------------------------------------------------- #


def test_per_paragraph_chunks_within_max_tokens(counter):
    chunker = PerParagraphChunker(max_tokens=64)
    chunks = chunker.chunk(two_page_parsed())
    assert chunks
    assert all(c.chunker == "per_paragraph" for c in chunks)
    assert all(c.page in (1, 2) for c in chunks)
    for c in chunks:
        assert counter.count(c.text) <= 64


def test_per_paragraph_txt_sentence_windowed():
    chunks = PerParagraphChunker(txt_max_tokens=512).chunk(txt_parsed(sentence_count=20))
    assert chunks
    assert all(c.page is None for c in chunks)
    assert len(chunks) == 4  # default sentence_count=7, overlap=2 -> same windowing as per_page


# --------------------------------------------------------------------------- #
# per_table
# --------------------------------------------------------------------------- #


def test_per_table_table_chunk_is_atomic_markdown_with_heading():
    chunks = PerTableChunker().chunk(table_parsed())
    table_chunks = [c for c in chunks if "|" in c.text]
    assert len(table_chunks) == 1, "exactly one atomic chunk per TableItem"
    table_chunk = table_chunks[0]
    assert table_chunk.page == 1
    # Markdown pipe rows with all cells present
    assert "נזקי מים" in table_chunk.text and "6,000 ₪" in table_chunk.text
    # Section-heading context prepended
    assert table_chunk.text.startswith("טבלת סכומי כיסוי")


def test_per_table_prose_does_not_duplicate_table_content():
    chunks = PerTableChunker().chunk(table_parsed())
    prose_chunks = [c for c in chunks if "|" not in c.text]
    assert prose_chunks, "non-table prose must fall back to per_paragraph"
    for c in prose_chunks:
        assert "נזקי מים" not in c.text and "6,000 ₪" not in c.text
    assert any("מפרט את הכיסויים" in c.text for c in prose_chunks)


def test_per_table_txt_page_none():
    chunks = PerTableChunker().chunk(txt_parsed())
    assert chunks and all(c.page is None for c in chunks)
    assert all(c.chunker == "per_table" for c in chunks)


# --------------------------------------------------------------------------- #
# Registry integration
# --------------------------------------------------------------------------- #


def test_registry_builds_all_three_strategies():
    from rag.chunking import REGISTRY

    assert isinstance(REGISTRY["per_page"](), PerPageChunker)
    assert isinstance(REGISTRY["per_paragraph"](max_tokens=512), PerParagraphChunker)
    assert isinstance(REGISTRY["per_table"](prose_max_tokens=512), PerTableChunker)


# --------------------------------------------------------------------------- #
# Real mini-corpus parses (slow; shared parse cache makes reruns cheap)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_real_per_page_anchor_pdf(parsed_anchor):
    chunks = PerPageChunker().chunk(parsed_anchor)
    n_pages = len((parsed_anchor.docling or {}).get("pages", {}))
    assert [c.page for c in chunks] == list(range(1, n_pages + 1))  # one per page, in order
    # Hebrew filename survives NFC-byte-identical to the ground-truth path
    assert chunks[0].file == ANCHOR_PDF_REL
    assert chunks[0].file == unicodedata.normalize("NFC", chunks[0].file)
    assert chunks[0].category == "apartment"
    assert "התיישנות" in chunks[0].text and "שלוש שנים" in chunks[0].text


@pytest.mark.slow
def test_real_per_paragraph_anchor_pdf(parsed_anchor, counter):
    chunks = PerParagraphChunker(max_tokens=512).chunk(parsed_anchor)
    assert chunks
    for c in chunks:
        assert c.page is not None and c.page >= 1
        assert counter.count(c.text) <= 512
    assert any("התיישנות" in c.text for c in chunks)


@pytest.mark.slow
def test_real_per_table_travel_pdf(parsed_travel_pdf, parsed_anchor):
    for parsed in (parsed_travel_pdf, parsed_anchor):
        chunks = PerTableChunker().chunk(parsed)
        assert chunks
        for c in chunks:
            assert c.page is not None and c.page >= 1
            assert c.chunker == "per_table"
            assert c.file.startswith(f"{c.category}/")

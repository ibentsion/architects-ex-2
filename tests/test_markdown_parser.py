"""Marker/DataLab markdown parser tests (rag/parsing/markdown_parser.py).

The markdown arm exists to be *compared* against Docling, so the properties
that matter are the ones that keep the comparison honest: pages must line up
with the PDF's, citations must name the PDF, and LLM-written image captions
must never reach the index.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.parsing import ParseError, SourceFile, discover, doc_source_for
from rag.parsing.markdown_parser import (
    MarkdownParser,
    clean_markdown,
    split_pages,
    to_docling_dict,
)

PAGED = """{0}------------------------------------------------

# עמוד ראשון

טקסט של העמוד הראשון.

{1}------------------------------------------------

## עמוד שני

| עלות יומית | קבוצת גיל |
|------------|-----------|
| \\$0.20     | כל הגילאים |
"""


def make_md_source(path: Path, rel_path: str = "travel/files/doc.pdf") -> SourceFile:
    return SourceFile(
        abs_path=path,
        rel_path=rel_path,
        category=rel_path.split("/", 1)[0],
        kind="md",
        sha256="c" * 64,
    )


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #


def test_pages_are_1_based() -> None:
    """Marker's markers are 0-indexed; Docling's prov.page_no is 1-based, and
    citations are graded against the latter."""
    pages = split_pages(PAGED)
    assert sorted(pages) == [1, 2]
    assert "עמוד ראשון" in pages[1]
    assert "עמוד שני" in pages[2]


def test_marker_precedes_its_page() -> None:
    pages = split_pages(PAGED)
    assert "עמוד שני" not in pages[1]


def test_text_before_the_first_marker_belongs_to_no_page() -> None:
    pages = split_pages("preamble\n" + PAGED)
    assert "preamble" not in "".join(pages.values())


# --------------------------------------------------------------------------- #
# Artifact stripping
# --------------------------------------------------------------------------- #


def test_llm_image_captions_are_dropped() -> None:
    """Marker's <p>Image: …</p> is model output, not corpus text — indexing it
    would let the parser invent retrievable content."""
    out = clean_markdown("לפני <p>Image: Icon of binoculars</p> אחרי")
    assert "Icon of binoculars" not in out
    assert "לפני" in out and "אחרי" in out


def test_image_and_checkbox_markup_is_dropped() -> None:
    out = clean_markdown('א <img alt="x" src="y.jpg"/> ב ![alt](z.jpg) ג <input type="checkbox"/> ד')
    for junk in ("<img", "y.jpg", "![alt]", "<input"):
        assert junk not in out
    assert all(w in out for w in ("א", "ב", "ג", "ד"))


def test_structural_html_is_unwrapped_not_deleted() -> None:
    out = clean_markdown("<div><b>סכום הביטוח</b></div>")
    assert "סכום הביטוח" in out
    assert "<b>" not in out and "<div>" not in out


def test_dollar_escapes_are_removed() -> None:
    """The corpus is full of dollar premiums; a stray backslash would reach the
    index and the answer text."""
    assert clean_markdown(r"עלות יומית \$0.20") == "עלות יומית $0.20"


def test_table_pipes_survive_cleaning() -> None:
    out = clean_markdown("| עלות יומית | קבוצת גיל |\n|---|---|\n| $0.20 | כל הגילאים |")
    assert out.count("|") >= 8


# --------------------------------------------------------------------------- #
# Synthetic DoclingDocument
# --------------------------------------------------------------------------- #


def test_every_text_item_carries_page_provenance() -> None:
    doc = to_docling_dict(PAGED)
    assert doc["texts"]
    for item in doc["texts"]:
        assert item["prov"][0]["page_no"] >= 1


def test_a_table_stays_one_block() -> None:
    doc = to_docling_dict(PAGED)
    tables = [t for t in doc["texts"] if t["text"].startswith("|")]
    assert len(tables) == 1
    assert "כל הגילאים" in tables[0]["text"]


def test_pages_map_lists_every_page() -> None:
    """The citation judge treats ``pages`` as the document's real page set — an
    empty map makes every markdown-arm citation 'page out of range'."""
    doc = to_docling_dict(PAGED)
    assert sorted(int(p) for p in doc["pages"]) == [1, 2]


def test_shape_loads_as_a_real_docling_document() -> None:
    """The chunkers reconstruct a DoclingDocument from this dict — if it does
    not validate, every chunker breaks on the markdown arm."""
    from docling_core.types.doc import DoclingDocument

    dl = DoclingDocument.model_validate(to_docling_dict(PAGED))
    pages = {item.prov[0].page_no for item, _ in dl.iterate_items() if item.prov}
    assert pages == {1, 2}


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def test_parses_a_markdown_file(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text(PAGED, encoding="utf-8")
    parsed = MarkdownParser().parse(make_md_source(md))
    assert parsed.docling is not None
    assert parsed.source.rel_path.endswith(".pdf")


def test_markdown_without_page_markers_is_a_parse_error(tmp_path: Path) -> None:
    """Silently indexing it would produce chunks that can never be cited."""
    md = tmp_path / "doc.md"
    md.write_text("# כותרת\n\nטקסט ללא סימני עמוד", encoding="utf-8")
    with pytest.raises(ParseError, match="page separators"):
        MarkdownParser().parse(make_md_source(md))


def test_parser_rejects_non_markdown_sources(tmp_path: Path) -> None:
    src = make_md_source(tmp_path / "doc.md").model_copy(update={"kind": "pdf"})
    with pytest.raises(ValueError, match="only parses markdown"):
        MarkdownParser().parse(src)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _mini_corpus(root: Path, *, with_pdf: bool = True) -> Path:
    (root / "travel" / "files").mkdir(parents=True)
    (root / "travel" / "markdown-files").mkdir(parents=True)
    (root / "travel" / "pages").mkdir(parents=True)
    if with_pdf:
        (root / "travel" / "files" / "doc.pdf").write_bytes(b"%PDF-1.4 stub")
    (root / "travel" / "markdown-files" / "doc.md").write_text(PAGED, encoding="utf-8")
    (root / "travel" / "pages" / "page.txt").write_text("טקסט", encoding="utf-8")
    return root


def test_markdown_discovery_cites_the_pdf(tmp_path: Path) -> None:
    """rel_path is the citation target; markdown is a parse of the PDF, so both
    arms must cite the same thing or the eval is not comparable."""
    sources = discover(_mini_corpus(tmp_path), "markdown")
    md = [s for s in sources if s.kind == "md"]
    assert len(md) == 1
    assert md[0].rel_path == "travel/files/doc.pdf"
    assert md[0].abs_path.suffix == ".md"


def test_markdown_without_a_source_pdf_is_skipped(tmp_path: Path) -> None:
    sources = discover(_mini_corpus(tmp_path, with_pdf=False), "markdown")
    assert [s for s in sources if s.kind == "md"] == []


def test_txt_pages_are_shared_by_both_arms(tmp_path: Path) -> None:
    root = _mini_corpus(tmp_path)
    for doc_source in ("pdf", "markdown"):
        txts = [s for s in discover(root, doc_source) if s.kind == "txt"]
        assert len(txts) == 1


def test_pdf_discovery_ignores_markdown(tmp_path: Path) -> None:
    sources = discover(_mini_corpus(tmp_path), "pdf")
    assert {s.kind for s in sources} == {"pdf", "txt"}


def test_doc_source_is_derived_from_the_parser_impl() -> None:
    assert doc_source_for("markdown") == "markdown"
    assert doc_source_for("docling") == "pdf"


# --------------------------------------------------------------------------- #
# Chunker integration
# --------------------------------------------------------------------------- #


def test_page_store_serves_markdown_when_asked(tmp_path: Path) -> None:
    """Judging markdown-arm citations against Docling's text would score a page
    the retriever never saw, so PageStore has to follow the arm."""
    from evalharness.pages import PageStore

    root = _mini_corpus(tmp_path)
    text, reason = PageStore(root, tmp_path / "cache", "markdown").resolve(
        "travel/files/doc.pdf", 2
    )
    assert reason is None
    assert "כל הגילאים" in text


def test_page_store_rejects_a_page_the_markdown_does_not_have(tmp_path: Path) -> None:
    from evalharness.pages import PageStore

    root = _mini_corpus(tmp_path)
    text, reason = PageStore(root, tmp_path / "cache", "markdown").resolve(
        "travel/files/doc.pdf", 99
    )
    assert text is None and reason == "page_out_of_range"


def test_per_page_chunker_keeps_markdown_pages(tmp_path: Path) -> None:
    from rag.chunking.per_page import PerPageChunker

    md = tmp_path / "doc.md"
    md.write_text(PAGED, encoding="utf-8")
    parsed = MarkdownParser().parse(make_md_source(md))
    chunks = PerPageChunker().chunk(parsed)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.file.endswith(".pdf") for c in chunks)

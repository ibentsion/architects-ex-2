"""Corpus page inventory + the sampler that decides what each question is
written from.

The inventory is every substantive page of a category, read through the eval
harness's `PageStore` — so a page offered to a question writer is a page the
citation judge will later be able to resolve and read back.

Sampling is deterministic (seeded per category) and coverage-driven: it hands
out pages from as many distinct files as possible, never reuses a page inside a
category, and never offers a page the held-out v1 set quizzes on.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Below this, a page is a cover sheet, a header, or a stub — nothing to ask about.
MIN_PAGE_CHARS = 500
#: Hard/medium questions need a quantity to be about (a limit, a percentage, a
#: sum, a number of days). A lone digit is usually a list marker, so require a
#: currency/percent marker or a multi-digit number.
_QUANTITY = re.compile(r"[₪%]|ש\"ח|אחוז|\d{2,}")
_WORD = re.compile(r"[֐-׿\w]{3,}")
#: Hebrew words, for checking a page's word order against the PDF oracle.
_HEBREW_WORD = re.compile(r"[֐-׿]{2,}")
#: A page must have at least this fraction of its Hebrew lines in the same word
#: order as pypdfium2 reports, or it is not fit to write a question from.
MIN_ORDER_AGREEMENT = 0.7


@dataclass(frozen=True)
class Page:
    """One candidate page a question can be written from."""

    file: str
    page: int | None
    text: str

    @property
    def key(self) -> tuple[str, int | None]:
        return (self.file, self.page)

    @property
    def has_numbers(self) -> bool:
        return bool(_QUANTITY.search(self.text))


def order_agreement(text: str, oracle: str) -> float:
    """Fraction of a page's multi-word Hebrew lines whose word order matches
    the PDF's own text layer.

    Docling emits some Hebrew in visual rather than logical order.
    `rag.parsing.rtl_repair` fixes the great majority of it, but what survives
    is invisible to the acceptance gates: a question written from a scrambled
    page gets a scrambled ground truth, and the derivability judge — reading
    that same scrambled page — agrees with it. The only defence is to not
    write questions from those pages at all.
    """
    oracle_words = " ".join(_HEBREW_WORD.findall(oracle))
    lines = [_HEBREW_WORD.findall(line) for line in text.split("\n")]
    lines = [words for words in lines if len(words) >= 4]
    if not lines or not oracle_words:
        return 1.0  # nothing checkable — not evidence of a problem
    agreed = sum(1 for words in lines if " ".join(words) in oracle_words)
    return agreed / len(lines)


def _oracle_pages(pdf_path) -> dict[int, str]:
    """The PDF's own text layer, per 1-based page, via pypdfium2 (which
    returns Hebrew in logical order)."""
    import pypdfium2 as pdfium

    out: dict[int, str] = {}
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in range(len(pdf)):
            out[index + 1] = pdf[index].get_textpage().get_text_bounded()
    except Exception:  # a PDF pdfium cannot read gives no opinion
        return {}
    finally:
        pdf.close()
    return out


def build_inventory(category: str, store, min_chars: int = MIN_PAGE_CHARS,
                    verify_order: bool = True) -> list[Page]:
    """Every substantive page of one category, in stable corpus order.

    With `verify_order`, pages whose Hebrew word order disagrees with the PDF's
    own text layer are dropped — see `order_agreement`.
    """
    sources = store._load_sources()
    pages: list[Page] = []
    for rel_path, source in sorted(sources.items()):
        if source.category != category:
            continue
        if source.kind == "txt":
            text, _ = store.resolve(rel_path, None)
            if text and len(text) >= min_chars:
                pages.append(Page(rel_path, None, text))
            continue
        if rel_path not in store._pages:
            store._extract(rel_path)
        candidates = [(page_no, text) for page_no, text in sorted(store._pages[rel_path].items())
                      if text and len(text) >= min_chars]
        if not candidates:
            continue
        oracle = _oracle_pages(source.abs_path) if verify_order else {}
        for page_no, text in candidates:
            if verify_order and page_no in oracle:
                if order_agreement(text, oracle[page_no]) < MIN_ORDER_AGREEMENT:
                    logger.debug("Skipping scrambled page: %s p%s", rel_path, page_no)
                    continue
            pages.append(Page(rel_path, page_no, text))
    return pages


def _overlap(a: Page, b: Page) -> int:
    """Shared distinctive words — a cheap topical link between two pages."""
    return len(set(_WORD.findall(a.text)) & set(_WORD.findall(b.text)))


@dataclass
class Sampler:
    """Hands out pages for one category, spreading them across files.

    `used` accumulates across the whole category (not just one cell), so the
    dataset never asks two questions about the same page, and `file_uses`
    biases each draw toward files that have not been used yet.
    """

    pages: list[Page]
    seed: int = 0
    excluded: set = field(default_factory=set)
    used: set = field(default_factory=set)
    file_uses: dict = field(default_factory=dict)
    _rng: random.Random = field(init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def available(self, prefer_numbers: bool = False, cell_used: set | None = None) -> list[Page]:
        """Unused, non-excluded pages. A thin category (dental has 16 usable
        pages for 11 questions) can exhaust the pool; `cell_used` then relaxes
        the constraint to what the *spec* actually forbids — reuse within one
        category×difficulty cell."""
        pool = [p for p in self.pages
                if p.key not in self.used and p.key not in self.excluded]
        if not pool and cell_used is not None:
            pool = [p for p in self.pages
                    if p.key not in cell_used and p.key not in self.excluded]
        if prefer_numbers:
            with_numbers = [p for p in pool if p.has_numbers]
            if with_numbers:
                return with_numbers
        return pool

    def draw(self, prefer_numbers: bool = False, cell_used: set | None = None) -> Page | None:
        """One page, biased toward the least-used files. Marks it used."""
        pool = self.available(prefer_numbers, cell_used)
        if not pool:
            return None
        fewest = min(self.file_uses.get(p.file, 0) for p in pool)
        candidates = [p for p in pool if self.file_uses.get(p.file, 0) == fewest]
        page = self._rng.choice(candidates)
        self.take(page)
        return page

    def draw_pair(self, top_n: int = 5) -> tuple[Page, Page] | None:
        """Two topically related pages from two different files.

        A multi-source question only works if the two pages are about
        overlapping subject matter — a random pair usually has no question that
        needs both. Pick a seed page, then its most word-overlapping page in
        another file (sampled among the top few so the choice is not always the
        single densest page).
        """
        seed_page = self.draw()
        if seed_page is None:
            return None
        others = [p for p in self.available() if p.file != seed_page.file]
        if not others:
            return None
        ranked = sorted(others, key=lambda p: _overlap(seed_page, p), reverse=True)
        partner = self._rng.choice(ranked[:top_n])
        self.take(partner)
        return seed_page, partner

    def take(self, page: Page) -> None:
        self.used.add(page.key)
        self.file_uses[page.file] = self.file_uses.get(page.file, 0) + 1

    def release(self, *pages: Page) -> None:
        """Put rejected pages back — a page that produced a bad question is
        not necessarily a bad page, but try everything else first."""
        for page in pages:
            self.used.discard(page.key)
            self.file_uses[page.file] = max(0, self.file_uses.get(page.file, 1) - 1)


def bm25_search(query: str, pages: list[Page], k: int = 20) -> list[Page]:
    """Top-k pages of a category for a query — the recall net for proving an
    `unanswerable` question really has no answer in its category.

    Plain token BM25 (no Stanza lemmatization, unlike the production index), so
    Hebrew morphology costs some recall. It is a net, not the verdict: the LLM
    gate reads the top-k and rules on whether any of them answers the question.
    """
    import bm25s

    corpus = [_WORD.findall(p.text) for p in pages]
    if not corpus:
        return []
    retriever = bm25s.BM25()
    retriever.index(corpus, show_progress=False)
    tokens = _WORD.findall(query)
    if not tokens:
        return []
    hits, _ = retriever.retrieve([tokens], k=min(k, len(pages)), show_progress=False)
    return [pages[i] for i in hits[0]]

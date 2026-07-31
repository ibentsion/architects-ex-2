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

import random
import re
from dataclasses import dataclass, field

#: Below this, a page is a cover sheet, a header, or a stub — nothing to ask about.
MIN_PAGE_CHARS = 500
#: Hard/medium questions need a quantity to be about (a limit, a percentage, a
#: sum, a number of days). A lone digit is usually a list marker, so require a
#: currency/percent marker or a multi-digit number.
_QUANTITY = re.compile(r"[₪%]|ש\"ח|אחוז|\d{2,}")
_WORD = re.compile(r"[֐-׿\w]{3,}")


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


def build_inventory(category: str, store, min_chars: int = MIN_PAGE_CHARS) -> list[Page]:
    """Every substantive page of one category, in stable corpus order."""
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
        for page_no, text in sorted(store._pages[rel_path].items()):
            if text and len(text) >= min_chars:
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

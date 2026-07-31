"""Resolve a citation's {file, page} to the actual corpus page text.

Direct lookup, never a search: the corpus walk gives `rel_path -> sha256`, the
sha256 keys the Docling parse cache, and the cached DoclingDocument yields the
page's text in reading order. The judge therefore sees exactly the text the
retriever indexed (tables as Markdown, Docling reading order) — so "the cited
page does not establish this" is a statement about the corpus, not about a
second, differently-parsed view of it.

Requires `corpus/` and `cache/parsed/` to be present (both are shipped to the
cloud node by `cloud/upload_artifacts.sh`).
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

#: Reasons a citation could not be resolved to a page (all count as invalid;
#: reported separately so parse gaps are distinguishable from bad citations).
UNKNOWN_FILE = "unknown_file"
AMBIGUOUS_FILE = "ambiguous_file"
PAGE_OUT_OF_RANGE = "page_out_of_range"
MISSING_PAGE = "missing_page"
EMPTY_PAGE = "empty_page"
PAGE_ON_TXT = "page_on_txt"


def norm_file(path: str) -> str:
    """Normalize a file reference for comparison: NFC unicode, lowercase,
    forward slashes, no leading './'."""
    path = unicodedata.normalize("NFC", str(path)).strip().lower()
    return path.replace("\\", "/").lstrip("./")


def files_match(cited: str, truth: str) -> bool:
    """Exact normalized match, or one path is a trailing path-suffix of the
    other (systems may cite a bare filename or a longer absolute path)."""
    a, b = norm_file(cited), norm_file(truth)
    if not a or not b:
        return False
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


class PageStore:
    """`{file, page}` -> page text, backed by the corpus + Docling parse cache.

    Everything is lazy: the corpus walk happens on first lookup, a document's
    pages are extracted on first citation of that document, and both are then
    memoized for the run.
    """

    def __init__(self, corpus_dir: Path | str = "corpus",
                 cache_dir: Path | str = "cache") -> None:
        self.corpus_dir = Path(corpus_dir)
        self.cache_dir = Path(cache_dir)
        self._sources: dict[str, object] | None = None  # rel_path -> SourceFile
        self._pages: dict[str, dict[int | None, str]] = {}  # rel_path -> {page: text}
        self._page_ids: dict[str, set] = {}  # rel_path -> page numbers in the PDF
        self._file_alias: dict[str, tuple] = {}  # cited path -> (rel_path, reason)

    # -- corpus / cache plumbing -------------------------------------------

    def _load_sources(self) -> dict:
        if self._sources is None:
            from rag.parsing import discover

            self._sources = {s.rel_path: s for s in discover(self.corpus_dir)}
        return self._sources

    def _match_file(self, cited: str) -> tuple[str | None, str | None]:
        """Map a cited path to a corpus rel_path (exact, then suffix match).

        Returns `(rel_path, invalid_reason)`. A bare filename that several
        categories share (e.g. הודעה-על-תקופת-התיישנות.pdf) is ambiguous, not
        resolvable — the citation does not identify a page.
        """
        if cited in self._file_alias:
            return self._file_alias[cited]
        sources = self._load_sources()
        result: tuple[str | None, str | None]
        if cited in sources:
            result = (cited, None)
        else:
            normalized = {norm_file(rel): rel for rel in sources}
            exact = normalized.get(norm_file(cited))
            if exact is not None:
                result = (exact, None)
            else:
                hits = [rel for rel in sources if files_match(cited, rel)]
                if len(hits) == 1:
                    result = (hits[0], None)
                else:
                    result = (None, AMBIGUOUS_FILE if hits else UNKNOWN_FILE)
        self._file_alias[cited] = result
        return result

    def _extract(self, rel_path: str) -> None:
        """Populate the page text and page-number sets for one corpus file."""
        from rag.parsing import ParsedDoc
        from rag.parsing.cache import ParseCache

        source = self._load_sources()[rel_path]
        if source.kind == "txt":
            self._pages[rel_path] = {None: source.abs_path.read_text(encoding="utf-8")}
            self._page_ids[rel_path] = set()
            return

        docling = ParseCache(self.cache_dir).load(source.sha256)
        if docling is None:
            raise FileNotFoundError(
                f"No parse-cache entry for {rel_path} (sha256 {source.sha256[:12]}). "
                f"Citation judging needs the same cache/ the index was built from."
            )

        from rag.chunking.common import iter_reading_order, load_docling

        by_page: dict[int, list[str]] = {}
        for _item, page, text in iter_reading_order(load_docling(ParsedDoc(source=source, docling=docling))):
            if page is not None and text.strip():
                by_page.setdefault(page, []).append(text)
        self._pages[rel_path] = {p: "\n\n".join(t) for p, t in by_page.items()}
        # The `pages` map is the document's real page set: a page that exists
        # but parsed to nothing is a parse gap, not a fabricated citation.
        self._page_ids[rel_path] = {int(p) for p in (docling.get("pages") or {})}

    # -- public API ---------------------------------------------------------

    def resolve(self, file: str, page) -> tuple[str | None, str | None]:
        """Return `(page_text, invalid_reason)` — exactly one is non-None."""
        rel_path, reason = self._match_file(file or "")
        if rel_path is None:
            return None, reason
        if rel_path not in self._pages:
            self._extract(rel_path)

        is_txt = self._load_sources()[rel_path].kind == "txt"
        if is_txt:
            if page is not None:
                return None, PAGE_ON_TXT
            return self._pages[rel_path][None], None
        if page is None:
            return None, MISSING_PAGE
        try:
            page = int(page)
        except (TypeError, ValueError):
            return None, PAGE_OUT_OF_RANGE
        if page not in self._page_ids[rel_path]:
            return None, PAGE_OUT_OF_RANGE
        text = self._pages[rel_path].get(page, "")
        if not text.strip():
            return None, EMPTY_PAGE
        return text, None

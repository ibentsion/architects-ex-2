"""Citation previews from the local corpus: PDF page images and TXT page text.

``corpus/`` is gitignored, so on most machines this whole module has nothing to
work with. That is a supported state: every entry point either returns ``None``
or raises :class:`CorpusUnavailable`, which the routes answer with 404. Nothing
here ever 500s because a file is missing.

Rendering uses pypdfium2, which is already in the tree as a Docling dependency
— no new upstream for the sake of a thumbnail. Page TEXT comes from
``evalharness.pages.PageStore``: corpus walk -> sha256 -> Docling parse cache,
the same resolution the citation judge uses. Nothing is parsed on a request.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path

from webapi import paths
from webapi.paths import PathEscape, resolve_in_repo

logger = logging.getLogger(__name__)

#: Roughly a readable page on a sidebar-sized card without shipping a megabyte.
MAX_THUMBNAIL_WIDTH = 900
#: pypdfium2 scale 2 is ~144 dpi.
RENDER_SCALE = 2
JPEG_QUALITY = 80
#: How much TXT page text a citation card shows before the user opens it.
PREVIEW_CHARS = 600


class CorpusUnavailable(RuntimeError):
    """The requested corpus page cannot be served — no corpus/ on this machine,
    no such file, or no such page in the document. All of them are a 404: the
    UI shows a placeholder and moves on."""


def _corpus_path(file: str) -> Path:
    """Resolve a citation's category-relative path inside ``corpus/``.

    Raises :class:`~webapi.paths.PathEscape` (400) for traversal and
    :class:`CorpusUnavailable` (404) for anything simply not there.
    """
    path = resolve_in_repo(f"corpus/{file}")
    if not path.is_file():
        raise CorpusUnavailable(f"not in the local corpus/: {file}")
    return path


@lru_cache(maxsize=1)
def _manifest() -> dict[str, str]:
    """``corpus/manifest.json`` — a flat map of the same category-relative
    paths citations use, to the URL the document came from. Data only; never
    fetched."""
    try:
        path = resolve_in_repo("corpus/manifest.json")
    except PathEscape:  # pragma: no cover - constant path
        return {}
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("corpus/manifest.json unreadable (%s) — no source URLs", exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def source_url(file: str) -> str | None:
    return _manifest().get(file)


def page_text(file: str) -> tuple[str, str | None]:
    """(text, source_url) for a TXT corpus page."""
    path = _corpus_path(file)
    return path.read_text(encoding="utf-8"), source_url(file)


# --------------------------------------------------------------------------- #
# Paged (PDF) page text, via the evalharness page store
# --------------------------------------------------------------------------- #

# One store per repo root, built on first use and kept for the process: the
# first lookup walks the corpus (~570 files, ~3 s), everything after it is
# memoized. The bridge is long-lived, so re-walking per request is not an
# option. PageStore's memoization is plain dict mutation, so the lock guards
# lookups too — uvicorn runs sync endpoints on a thread pool.
_store_lock = threading.Lock()
_stores: dict[str, object] = {}

# PDFium is not thread-safe. uvicorn runs sync endpoints on a thread pool and a
# citation list fires one thumbnail request per card, so concurrent renders are
# the normal case, not an edge one — unsynchronized they fail with
# "PdfiumError: Data format error" on documents that load fine on their own.
# Only the render is serialized; cache hits never take this lock.
_render_lock = threading.Lock()


def _page_store():
    """The evalharness PageStore for the current repo root.

    Reuses the citation-judging machinery rather than a second, differently
    parsed view of the corpus: corpus walk -> sha256 -> Docling parse cache,
    direct lookup and never a search. That is the same text the retriever
    indexed, which is exactly what a citation preview should show.
    """
    root = str(paths.REPO_ROOT)
    store = _stores.get(root)
    if store is None:
        from evalharness.pages import PageStore

        store = PageStore(paths.REPO_ROOT / "corpus", paths.REPO_ROOT / "cache")
        _stores[root] = store
    return store


def page_preview(file: str, page: int) -> str | None:
    """Text of one PDF page, or None if it cannot be resolved.

    Never raises. The store raises for a missing corpus/ and for a document
    with no parse-cache entry, and returns a reason code for an unknown file
    or an out-of-range page — for a preview all of those are simply "nothing
    to show", and a citation list must render regardless.
    """
    try:
        with _store_lock:
            text, reason = _page_store().resolve(file, page)
    except Exception as exc:  # missing corpus/, missing parse, parse error
        logger.debug("no page preview for %s p%s (%s: %s)", file, page, type(exc).__name__, exc)
        return None
    if text is None:
        logger.debug("no page preview for %s p%s (%s)", file, page, reason)
    return text


def reset_page_store() -> None:
    """Drop the memoized stores (tests move REPO_ROOT under the process)."""
    with _store_lock:
        _stores.clear()


def preview_text(file: str, page: int | None = None) -> str | None:
    """Best-effort preview for a citation with no quote of its own.

    Paged citations resolve through the parse cache; page-less TXT citations
    read the corpus page directly. Never raises — an absent corpus/ (gitignored)
    is the normal case, not an error path.
    """
    if not file:
        return None
    try:
        resolve_in_repo(f"corpus/{file}")  # the chokepoint applies to previews too
    except PathEscape:
        return None

    if page is not None:
        text = page_preview(file, page)
    elif file.endswith(".txt"):
        try:
            text, _url = page_text(file)
        except (CorpusUnavailable, OSError, UnicodeDecodeError):
            return None
    else:
        return None  # a paged document cited without a page identifies nothing

    if text is None:
        return None
    return text.strip()[:PREVIEW_CHARS] or None


def _thumbnail_cache_path(file: str, page: int) -> Path:
    # Hash the path: corpus filenames are long Hebrew strings, and a flat
    # cache dir of those is a filesystem-encoding problem waiting to happen.
    digest = hashlib.sha256(file.encode("utf-8")).hexdigest()[:16]
    return paths.REPO_ROOT / "cache" / "webui_thumbs" / f"{digest}-p{page}.jpg"


def page_thumbnail(file: str, page: int) -> Path:
    """Render (once) and return the cached JPEG of a PDF page.

    ``page`` is 1-based, exactly as in a citation.
    """
    # Validate the path first, always: a traversal attempt is a 400 whatever
    # else is wrong with the request.
    path = resolve_in_repo(f"corpus/{file}")
    if not file.lower().endswith(".pdf"):
        raise CorpusUnavailable(f"not a paged document: {file}")

    cached = _thumbnail_cache_path(file, page)
    if cached.is_file():
        return cached
    if not path.is_file():
        raise CorpusUnavailable(f"not in the local corpus/: {file}")

    import pypdfium2  # heavyweight; the bridge starts without touching it

    with _render_lock:
        if cached.is_file():
            return cached  # another thread rendered it while we waited

        try:
            document = pypdfium2.PdfDocument(path)
        except pypdfium2.PdfiumError as exc:
            # A corrupt or unsupported PDF is a placeholder, not a 500.
            raise CorpusUnavailable(f"cannot open {file}: {exc}") from exc
        try:
            if not 1 <= page <= len(document):
                raise CorpusUnavailable(
                    f"page {page} is outside {file} (1..{len(document)})"
                )
            image = document[page - 1].render(scale=RENDER_SCALE).to_pil()
        except pypdfium2.PdfiumError as exc:
            raise CorpusUnavailable(f"cannot render {file} p{page}: {exc}") from exc
        finally:
            document.close()

        if image.width > MAX_THUMBNAIL_WIDTH:
            height = round(image.height * MAX_THUMBNAIL_WIDTH / image.width)
            image = image.resize((MAX_THUMBNAIL_WIDTH, height))

        # Write-then-rename: a half-written JPEG would be cached forever and
        # served as a broken image on every later request.
        cached.parent.mkdir(parents=True, exist_ok=True)
        staging = cached.with_suffix(".jpg.tmp")
        image.convert("RGB").save(staging, "JPEG", quality=JPEG_QUALITY)
        staging.replace(cached)
    return cached

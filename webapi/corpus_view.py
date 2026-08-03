"""Citation previews from the local corpus: PDF page images and TXT page text.

``corpus/`` is gitignored, so on most machines this whole module has nothing to
work with. That is a supported state: every entry point either returns ``None``
or raises :class:`CorpusUnavailable`, which the routes answer with 404. Nothing
here ever 500s because a file is missing.

Rendering uses pypdfium2, which is already in the tree as a Docling dependency
— no new upstream for the sake of a thumbnail.
"""
from __future__ import annotations

import hashlib
import json
import logging
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


def preview_text(file: str) -> str | None:
    """Best-effort preview for a citation with no quote of its own.

    Only TXT sources: a PDF page's text would mean re-running the ingest
    parser on a UI request. Never raises — a missing corpus is the normal case.
    """
    if not file.endswith(".txt"):
        return None
    try:
        text, _url = page_text(file)
    except (CorpusUnavailable, PathEscape, OSError, UnicodeDecodeError):
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

    document = pypdfium2.PdfDocument(path)
    try:
        if not 1 <= page <= len(document):
            raise CorpusUnavailable(
                f"page {page} is outside {file} (1..{len(document)})"
            )
        image = document[page - 1].render(scale=RENDER_SCALE).to_pil()
    finally:
        document.close()

    if image.width > MAX_THUMBNAIL_WIDTH:
        height = round(image.height * MAX_THUMBNAIL_WIDTH / image.width)
        image = image.resize((MAX_THUMBNAIL_WIDTH, height))

    cached.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(cached, "JPEG", quality=JPEG_QUALITY)
    return cached

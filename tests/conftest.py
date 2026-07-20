"""Shared pytest fixtures (rag_plan.md §9).

Mini corpus: 2 categories (apartment, travel), 2 small real PDFs (incl. the
ground-truth anchor הודעה-על-תקופת-התיישנות.pdf) + 2 TXTs + mini manifest.json,
preserving the corpus category/files|pages/ layout and NFC Hebrew filenames.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Stale ~/.cache/huggingface/token on this machine 401s every implicit-token
# Hub request; anonymous access works (see rag/__init__.py).
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: Ground-truth anchor (rag_plan.md §5 stage 2): p.1 must contain התיישנות.
ANCHOR_PDF_REL = "apartment/files/הודעה-על-תקופת-התיישנות.pdf"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def mini_corpus_dir() -> Path:
    path = FIXTURES_DIR / "mini_corpus"
    assert path.is_dir(), f"mini corpus fixture missing: {path}"
    return path


@pytest.fixture(scope="session")
def anchor_pdf(mini_corpus_dir: Path) -> Path:
    path = mini_corpus_dir / ANCHOR_PDF_REL
    assert path.is_file(), f"ground-truth anchor PDF missing: {path}"
    return path


@pytest.fixture(scope="session")
def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture(scope="session")
def swap_config_path() -> Path:
    return REPO_ROOT / "configs" / "swap-example.yaml"


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    path = tmp_path / "cache"
    path.mkdir()
    return path


# --------------------------------------------------------------------------- #
# Real-parse fixtures (slow on first run, then served from the SHARED repo
# parse cache — rag_plan.md: "mini_corpus real parse is fine, cache makes it
# a one-time cost"). cache/ is gitignored.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def mini_sources(mini_corpus_dir: Path):
    from rag.parsing import discover

    return discover(mini_corpus_dir)


def _source_for(sources, rel_path: str):
    matches = [s for s in sources if s.rel_path == rel_path]
    assert matches, f"fixture source not discovered: {rel_path}"
    return matches[0]


@pytest.fixture(scope="session")
def real_cached_parser(repo_root: Path):
    from rag.parsing.cache import CachedParser
    from rag.parsing.docling_parser import DoclingParser

    return CachedParser(DoclingParser(), repo_root / "cache")


@pytest.fixture(scope="session")
def parsed_anchor(real_cached_parser, mini_sources):
    """Real Docling parse of the ground-truth anchor PDF (cache-backed)."""
    return real_cached_parser.parse(_source_for(mini_sources, ANCHOR_PDF_REL))


@pytest.fixture(scope="session")
def parsed_travel_pdf(real_cached_parser, mini_sources):
    return real_cached_parser.parse(
        _source_for(mini_sources, "travel/files/הודעה-על-הגדרת-ספורט-אתגרי.pdf")
    )


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    path = tmp_path / "rag_index"
    path.mkdir()
    return path

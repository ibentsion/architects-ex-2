"""Shared pytest fixtures (rag_plan.md §9).

Mini corpus: 2 categories (apartment, travel), 2 small real PDFs (incl. the
ground-truth anchor הודעה-על-תקופת-התיישנות.pdf) + 2 TXTs + mini manifest.json,
preserving the corpus category/files|pages/ layout and NFC Hebrew filenames.
"""
from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    path = tmp_path / "rag_index"
    path.mkdir()
    return path

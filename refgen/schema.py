"""The v2 dataset spec, as code.

Two layers:

* **Item schema** (`RefQuestion`) — what one question must look like, including
  the per-kind structural invariants (an `unanswerable` item cites nothing; a
  `multi_source` item cites two groups from two different files).
* **Dataset invariants** (`check_dataset`) — what the collection must look like:
  every category present, the agreed per-cell counts, distinct pages within a
  cell, no leakage from the held-out v1 set.

Everything here is deterministic — no LLM, no corpus access beyond an optional
`PageStore` for resolving cited pages. `refgen.audit` runs it over a file;
`refgen.generate` runs it while building one.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.parsing import KNOWN_CATEGORIES

DIFFICULTIES = ("easy", "medium", "hard")
KINDS = ("standard", "multi_source", "unanswerable")

#: Per category: 3 questions at each difficulty, plus one multi-source item
#: (always hard — v1 treats multi-document as what makes a question hard) and
#: one unanswerable item (difficulty rotates across categories).
STANDARD_PER_CELL = 3
MULTI_SOURCE_PER_CATEGORY = 1
UNANSWERABLE_PER_CATEGORY = 1
ITEMS_PER_CATEGORY = (STANDARD_PER_CELL * len(DIFFICULTIES)
                      + MULTI_SOURCE_PER_CATEGORY + UNANSWERABLE_PER_CATEGORY)

#: Token-overlap above which two questions are "the same question asked twice".
DUPLICATE_JACCARD = 0.6

_HEBREW = re.compile(r"[֐-׿]")
_WORD = re.compile(r"[֐-׿\w]+")


class Source(BaseModel):
    """One {file, page} the answer can be read from."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(..., min_length=1, description="Category-relative corpus path")
    page: int | None = Field(None, ge=1, description="1-based for PDFs; null for TXT")


class SourceGroup(BaseModel):
    """Interchangeable sources for one fact — citing any member satisfies it."""

    model_config = ConfigDict(extra="forbid")

    any_of: list[Source] = Field(..., min_length=1)


class Provenance(BaseModel):
    """Who wrote and who checked this item. A verifier is never the generator."""

    model_config = ConfigDict(extra="forbid")

    generator_model: str
    verifier_models: list[str] = Field(default_factory=list)
    attempts: int = Field(1, ge=1, description="Generation attempts before acceptance")
    gates: dict[str, str] = Field(default_factory=dict,
                                  description="Gate name -> the verdict that passed it")

    @model_validator(mode="after")
    def _verifier_is_not_the_generator(self):
        if self.generator_model in self.verifier_models:
            raise ValueError(f"{self.generator_model} verified its own item")
        return self


class RefQuestion(BaseModel):
    """One dataset item. Field order matches v1 so the files read alike."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^v2-\d{3}-[a-z-]+-(easy|medium|hard)$")
    domain: str
    difficulty: Literal["easy", "medium", "hard"]
    kind: Literal["standard", "multi_source", "unanswerable"] = "standard"
    answerable: bool = True
    question: str = Field(..., min_length=20)
    ground_truth_answer: str = Field(..., min_length=10)
    ground_truth_sources: list[SourceGroup] = Field(default_factory=list)
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _kind_invariants(self):
        if self.domain not in KNOWN_CATEGORIES:
            raise ValueError(f"unknown domain: {self.domain}")
        if not _HEBREW.search(self.question):
            raise ValueError("question must be Hebrew")
        if not _HEBREW.search(self.ground_truth_answer):
            raise ValueError("ground-truth answer must be Hebrew")
        if not self.id.endswith(f"-{self.domain}-{self.difficulty}"):
            raise ValueError(f"id {self.id!r} disagrees with domain/difficulty")

        groups = self.ground_truth_sources
        if self.kind == "unanswerable":
            if self.answerable or groups:
                raise ValueError("an unanswerable item must be answerable=false with no sources")
        else:
            if not self.answerable:
                raise ValueError(f"a {self.kind} item must be answerable=true")
            if any(s.file.split("/")[0] != self.domain for g in groups for s in g.any_of):
                raise ValueError("every source must live under the item's own category")

        if self.kind == "standard" and len(groups) != 1:
            raise ValueError(f"a standard item needs exactly 1 source group, got {len(groups)}")
        if self.kind == "multi_source":
            if len(groups) != 2:
                raise ValueError(f"a multi-source item needs exactly 2 source groups, got {len(groups)}")
            if self.difficulty != "hard":
                raise ValueError("multi-source items are hard by definition")
            files = {s.file for g in groups for s in g.any_of}
            if len(files) < 2:
                raise ValueError("a multi-source item's two groups must be different files")
        return self

    def pages(self) -> list[tuple[str, int | None]]:
        """Every (file, page) this item cites."""
        return [(s.file, s.page) for g in self.ground_truth_sources for s in g.any_of]


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(unicodedata.normalize("NFC", text.lower())))


def jaccard(a: str, b: str) -> float:
    """Token overlap between two questions — the near-duplicate signal."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_duplicate(question: str, existing: list[str],
                 threshold: float = DUPLICATE_JACCARD) -> str | None:
    """Return the first near-duplicate of `question`, or None."""
    return next((other for other in existing
                 if jaccard(question, other) >= threshold), None)


def load_v1_exclusions(path) -> tuple[set[tuple[str, int | None]], list[str]]:
    """Pages and questions of the held-out v1 set, which v2 must not reuse.

    v1 is the validation set: a v2 item built from a page v1 quizzes on — or
    paraphrasing a v1 question — would leak the test set into the dev set.
    """
    import json
    from pathlib import Path

    items = json.loads(Path(path).read_text(encoding="utf-8"))
    pages = {(s["file"], s.get("page"))
             for item in items for g in item["ground_truth_sources"] for s in g["any_of"]}
    return pages, [item["question"] for item in items]


def check_item_sources(item: RefQuestion, store) -> list[str]:
    """Every cited page must resolve to real corpus text (uses the eval
    harness's PageStore, so "resolvable" means exactly what it means at
    scoring time)."""
    problems = []
    for file, page in item.pages():
        text, reason = store.resolve(file, page)
        if text is None:
            problems.append(f"{item.id}: cited {file} p{page} is unresolvable ({reason})")
    return problems


def check_dataset(items: list[RefQuestion], v1_pages=(), v1_questions=(),
                  store=None, strict_counts: bool = True) -> list[str]:
    """Every deterministic dataset invariant. Returns human-readable problems;
    empty means the dataset conforms to the spec."""
    problems: list[str] = []

    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        problems.append(f"duplicate ids: {sorted({i for i in ids if ids.count(i) > 1})}")

    by_category: dict[str, list[RefQuestion]] = {}
    for item in items:
        by_category.setdefault(item.domain, []).append(item)

    if strict_counts:
        missing = sorted(KNOWN_CATEGORIES - set(by_category))
        if missing:
            problems.append(f"categories with no questions: {', '.join(missing)}")

    for category, group in sorted(by_category.items()):
        standard = [i for i in group if i.kind == "standard"]
        if strict_counts:
            for difficulty in DIFFICULTIES:
                n = sum(1 for i in standard if i.difficulty == difficulty)
                if n != STANDARD_PER_CELL:
                    problems.append(f"{category}/{difficulty}: {n} standard items, "
                                    f"expected {STANDARD_PER_CELL}")
            for kind, expected in (("multi_source", MULTI_SOURCE_PER_CATEGORY),
                                   ("unanswerable", UNANSWERABLE_PER_CATEGORY)):
                n = sum(1 for i in group if i.kind == kind)
                if n != expected:
                    problems.append(f"{category}: {n} {kind} items, expected {expected}")

        # Coverage: no two items in a cell may be built from the same page.
        for difficulty in DIFFICULTIES:
            seen: dict[tuple, str] = {}
            for item in (i for i in group if i.difficulty == difficulty):
                for page in item.pages():
                    if page in seen:
                        problems.append(f"{category}/{difficulty}: {item.id} reuses "
                                        f"{page[0]} p{page[1]} already used by {seen[page]}")
                    seen[page] = item.id

    v1_pages = set(v1_pages)
    for item in items:
        leaked = [p for p in item.pages() if p in v1_pages]
        if leaked:
            problems.append(f"{item.id} cites held-out v1 page(s): "
                            + ", ".join(f"{f} p{p}" for f, p in leaked))

    questions = list(v1_questions)
    for item in items:
        duplicate = is_duplicate(item.question, questions)
        if duplicate is not None:
            problems.append(f"{item.id} near-duplicates an existing question: {duplicate[:60]}...")
        questions.append(item.question)

    if store is not None:
        for item in items:
            problems += check_item_sources(item, store)
    return problems


def coverage(items: list[RefQuestion]) -> dict:
    """Per-category corpus coverage of the emitted dataset."""
    out = {}
    for item in items:
        bucket = out.setdefault(item.domain, {"items": 0, "files": set(), "pages": set(),
                                              "file_uses": {}})
        bucket["items"] += 1
        for file, page in item.pages():
            bucket["files"].add(file)
            bucket["pages"].add((file, page))
            bucket["file_uses"][file] = bucket["file_uses"].get(file, 0) + 1
    return {
        category: {
            "items": b["items"],
            "distinct_files": len(b["files"]),
            "distinct_pages": len(b["pages"]),
            "overused_files": sorted(f for f, n in b["file_uses"].items() if n > 2),
        }
        for category, b in sorted(out.items())
    }

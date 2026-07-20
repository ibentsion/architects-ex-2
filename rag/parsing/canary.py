"""RTL sanity gate (rag_plan.md §5 stage 2) — blocking, first-run.

Asserts canary tokens (ביטוח, הראל, פוליסה) appear and their reversals
(e.g. חוטיב) do NOT, plus the ground-truth anchor check
(apartment/files/הודעה-על-תקופת-התיישנות.pdf p.1 contains התיישנות and
שלוש שנים). Hard stop on failure. Implemented in wave E2 (T2).
"""
from __future__ import annotations

from typing import Any


def run_canary(parsed_docs: Any, sample: int = 10) -> None:
    raise NotImplementedError("RTL canary gate is implemented in wave E2 (rag_plan.md T2)")

"""Prompt variants — ALL citation-mandating, all instruct answering in the
question's language, mandating the fallback sentence and a parseable
``מקורות:`` sources block (rag_plan.md §6 stage 6).

Implemented in wave E5 (T8).
"""
from __future__ import annotations

from typing import Any, Callable


def _grounded_cite(**kwargs: Any) -> Any:
    raise NotImplementedError("Prompt 'grounded_cite' is implemented in wave E5 (rag_plan.md T8)")


def _strict_extractive(**kwargs: Any) -> Any:
    raise NotImplementedError("Prompt 'strict_extractive' is implemented in wave E5 (rag_plan.md T8)")


def _few_shot_cite(**kwargs: Any) -> Any:
    raise NotImplementedError("Prompt 'few_shot_cite' is implemented in wave E5 (rag_plan.md T8)")


PROMPT_REGISTRY: dict[str, Callable[..., Any]] = {
    "grounded_cite": _grounded_cite,        # default: answer only from sources, cite file+page per claim
    "strict_extractive": _strict_extractive,  # + short verbatim quote per claim (feeds Citation.quote)
    "few_shot_cite": _few_shot_cite,        # grounded_cite + 2 worked Hebrew examples
}

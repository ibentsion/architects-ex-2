"""LLM query classification: category tags + sub-question decomposition.

Real-world queries never name a corpus category, and one query may pack
several questions ("does my apartment policy cover flooding, and how do I
file a car claim?"). ``QueryClassifier.classify(question)`` makes ONE
tf_client call that decomposes the query into self-contained sub-questions,
each tagged with categories from the closed 12-category corpus list; the
Classification's ``mode`` (single/multi) and top-level ``categories`` are
derived from the validated sub-questions, never trusted from the LLM.

Failure policy: any LLM/parse failure degrades to a single-mode,
no-category-filter classification of the original question (warning logged)
— classification must never block answering.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from tf_client import chat as tf_chat

from rag.types import Classification, SubQuestion

logger = logging.getLogger(__name__)

#: Closed category list = the 12 corpus directory names (Chunk.category
#: invariant, rag/types.py), with Hebrew descriptions for the prompt.
CATEGORIES: dict[str, str] = {
    "apartment": "ביטוח דירה ותכולה",
    "business": "ביטוח עסק",
    "car": "ביטוח רכב",
    "dental": "ביטוח שיניים",
    "diseases-disabilities": "ביטוח מחלות קשות ונכויות",
    "health": "ביטוח בריאות",
    "life": "ביטוח חיים",
    "long-term-care": "ביטוח סיעודי",
    "loss-of-working-ability": "ביטוח אובדן כושר עבודה",
    "mortgage": "ביטוח משכנתא",
    "personal-accident": "ביטוח תאונות אישיות",
    "travel": "ביטוח נסיעות לחו\"ל",
}


def _system_prompt() -> str:
    category_lines = "\n".join(f"- {cid}: {desc}" for cid, desc in CATEGORIES.items())
    return f"""אתה מסווג פניות עבור סוכן תמיכה של הראל ביטוח. בהינתן שאלת לקוח:
1. פרק את השאלה לתת-שאלות עצמאיות שניתן לענות על כל אחת בנפרד. שאלה בנושא יחיד נשארת תת-שאלה אחת, כלשונה.
2. שייך לכל תת-שאלה את תחומי הביטוח הרלוונטיים מהרשימה הסגורה הבאה (השתמש ב-id באנגלית בלבד):

{category_lines}

3. קבע שני דגלים:
   - needs_calculation: האם נדרש חישוב אריתמטי על מספרים (אחוזים, סכומים, הפרשים וכו').
   - dependent: האם תת-שאלה תלויה בתשובה של תת-שאלה אחרת (לא ניתן לענות עליהן במקביל).

החזר JSON בלבד, בדיוק בפורמט הבא, ללא טקסט נוסף:
{{"sub_questions": [{{"question": "...", "categories": ["<id>"]}}], "needs_calculation": false, "dependent": false}}

כללים:
- השתמש רק ב-ids מהרשימה. אם אף תחום לא מתאים לתת-שאלה, החזר עבורה רשימת categories ריקה.
- שייך תחום אחד לכל תת-שאלה, אלא אם היא באמת נוגעת לכמה תחומים.
- אל תמציא תת-שאלות שהלקוח לא שאל."""


def _extract_json(text: str) -> dict[str, Any]:
    """Tolerate code fences / prose around the JSON object: parse the
    outermost ``{...}`` span."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in classifier reply")
    return json.loads(text[start : end + 1])


def _fallback(question: str, cost: float) -> Classification:
    return Classification(
        mode="single",
        categories=[],
        sub_questions=[SubQuestion(question=question, categories=[])],
        cost_estimate=cost,
    )


def build_classifier(config: Any, model: str | None = None) -> "QueryClassifier":
    """Classifier runs on the fast orchestrator model (harness config block);
    its extra_params (reasoning knobs) apply only when the model is unchanged."""
    chosen = model or config.harness.orchestrator_model
    extra = config.harness.orchestrator_extra_params if chosen == config.harness.orchestrator_model else {}
    return QueryClassifier(chosen, extra_params=extra)


class QueryClassifier:
    """One-shot tf_client classifier; see module docstring for the contract."""

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 768,
        temperature: float = 0.0,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_params = extra_params or {}

    def classify(self, question: str) -> Classification:
        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": question},
        ]
        cost = 0.0
        try:
            text, _usage, cost = tf_chat(
                messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                quiet=False,
                return_usage=True,
                **self.extra_params,
            )
            data = _extract_json(text or "")
            sub_questions = self._parse_sub_questions(data)
        except Exception as exc:
            logger.warning(
                "Query classification failed (%s: %s) — falling back to single/no-filter",
                type(exc).__name__,
                exc,
            )
            return _fallback(question, cost)
        if not sub_questions:
            logger.warning("Classifier returned no usable sub-questions — falling back")
            return _fallback(question, cost)

        # Derived, not trusted: ordered union of sub-question categories.
        categories = list(dict.fromkeys(c for sq in sub_questions for c in sq.categories))
        return Classification(
            mode="multi" if len(categories) > 1 else "single",
            categories=categories,
            sub_questions=sub_questions,
            needs_calculation=bool(data.get("needs_calculation", False)),
            dependent=bool(data.get("dependent", False)),
            cost_estimate=cost,
        )

    def _parse_sub_questions(self, data: dict[str, Any]) -> list[SubQuestion]:
        sub_questions: list[SubQuestion] = []
        for item in data.get("sub_questions", []):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            raw_categories = item.get("categories") or []
            if not isinstance(raw_categories, list):
                raw_categories = []
            categories, unknown = [], []
            for cat in raw_categories:
                (categories if cat in CATEGORIES else unknown).append(cat)
            if unknown:
                logger.warning("Classifier produced unknown categories (dropped): %s", unknown)
            sub_questions.append(
                SubQuestion(question=question, categories=list(dict.fromkeys(categories)))
            )
        return sub_questions

"""LLM query classification: category tags + sub-question decomposition.

Real-world queries never name a corpus category, and one query may pack
several questions ("does my apartment policy cover flooding, and how do I
file a car claim?"). ``QueryClassifier.classify(question)`` makes ONE
tf_client call that decomposes the query into self-contained sub-questions,
each tagged with categories from the closed 12-category corpus list; the
Classification's ``mode`` (single/multi) and top-level ``categories`` are
derived from the validated sub-questions, never trusted from the LLM.

Two hooks exist for the classification A/B harness (evalharness/classify_eval)
and are inert by default: ``system_prompt=`` overrides the built-in prompt, and
``classify(question, hint=...)`` appends a delimited retrieval-evidence block to
the user turn.

Failure policy: any LLM/parse failure degrades to a single-mode,
no-category-filter classification of the original question (warning logged)
— classification must never block answering. That fallback is safe but not
free (a lost filter is a worse retrieval pool), so a malformed reply is first
run through :func:`_repair_json` for the two mechanical defects gpt-oss-120b
actually emits; only an unrepairable reply falls back.
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
4. הערך את רמת הקושי של השאלה כולה (difficulty):
   - easy: עובדה בודדת ומפורשת שסביר שכתובה במסמך אחד.
   - medium: דורש קריאת תנאים או פרטים מדויקים.
   - hard: דורש שילוב מספר מקורות, חריגים, טבלאות, או הסקה זהירה.

החזר JSON בלבד, בדיוק בפורמט הבא, ללא טקסט נוסף:
{{"sub_questions": [{{"question": "...", "categories": ["<id>"]}}], "needs_calculation": false, "dependent": false, "difficulty": "easy|medium|hard"}}

כללים:
- השתמש רק ב-ids מהרשימה. אם אף תחום לא מתאים לתת-שאלה, החזר עבורה רשימת categories ריקה.
- שייך תחום אחד לכל תת-שאלה, אלא אם היא באמת נוגעת לכמה תחומים.
- אל תמציא תת-שאלות שהלקוח לא שאל."""


def _user_message(question: str, hint: str | None) -> str:
    """The user turn: the question alone, or the question followed by a
    delimited retrieval-evidence block. The block is self-describing because
    the system prompt is not required to mention it (evalharness's hint arms
    pair it with the unmodified production prompt)."""
    if not hint:
        return question
    return (
        f"{question}\n\n"
        "--- ראיות מהאינדקס (תוצאות חיפוש ראשוניות, לידיעה בלבד — השאלה עצמה קובעת) ---\n"
        f"{hint}\n"
        "--- סוף ראיות ---"
    )


#: A closing quote inside a JSON string is only really a closing quote when
#: JSON structure follows it. See :func:`_string_ends_at`.
_STRUCTURE_AFTER_STRING = frozenset(":}]")

#: Every character a JSON value may begin with.
_VALUE_START = frozenset('"{[-0123456789tfn')

_OPENER_OF = {"]": "[", "}": "{"}


def _string_ends_at(text: str, pos: int) -> bool:
    """Does the ``"`` just before ``pos`` terminate its string?

    In well-formed JSON a string is always followed by ``:``, ``,``, ``}``,
    ``]`` or the end of input, so anything else means the quote was written
    *inside* the value. A comma is the ambiguous one — ``"travel", "difficulty"``
    is structure, ``בחו"ל, ולכן`` is prose — so it only counts when a fresh
    value starts after it."""
    rest = text[pos:].lstrip()
    if not rest:
        return True
    if rest[0] in _STRUCTURE_AFTER_STRING:
        return True
    return rest[0] == "," and rest[1:].lstrip()[:1] in _VALUE_START


def _escape_stray_quotes(text: str) -> str:
    """Escape ASCII double-quotes the model left unescaped inside a string.

    Hebrew gershayim are typed as ``"`` (``בחו"ל``), and the model copies the
    customer's spelling straight into the JSON."""
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        char = text[i]
        if in_string and char == "\\":
            out.append(text[i : i + 2])
            i += 2
            continue
        if char == '"':
            if not in_string:
                in_string = True
            elif _string_ends_at(text, i + 1):
                in_string = False
            else:
                out.append("\\")
        out.append(char)
        i += 1
    return "".join(out)


def _drop_unmatched_closers(text: str) -> str:
    """Drop closing brackets that close nothing.

    gpt-oss-120b duplicates the ``]`` that ends ``sub_questions``, emitting
    ``...}]], "needs_calculation": false``. Nested arrays are untouched because
    their closers do match the open stack. Must run after
    :func:`_escape_stray_quotes` so the string boundaries are trustworthy."""
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        char = text[i]
        if in_string:
            if char == "\\":
                out.append(text[i : i + 2])
                i += 2
                continue
            if char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in _OPENER_OF:
            if not stack or stack[-1] != _OPENER_OF[char]:
                i += 1
                continue
            stack.pop()
        out.append(char)
        i += 1
    return "".join(out)


def _repair_json(text: str) -> str:
    """Best-effort repair of the two defects gpt-oss-120b actually emits.

    Both are mechanical and lossless; neither invents content. Truncated
    replies (finish_reason ``length``) are deliberately not repaired — closing
    a cut-off object would fabricate a classification."""
    return _drop_unmatched_closers(_escape_stray_quotes(text))


def _extract_json(text: str) -> dict[str, Any]:
    """Tolerate code fences / prose around the JSON object: parse the
    outermost ``{...}`` span, retrying once through :func:`_repair_json`.

    The retry exists because 4-12% of gpt-oss-120b replies carry one of two
    mechanical defects, and every one of them used to degrade a perfectly good
    classification to the no-filter fallback."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in classifier reply")
    span = text[start : end + 1]
    try:
        return json.loads(span)
    except json.JSONDecodeError:
        repaired = _repair_json(span)
        if repaired == span:
            raise
        data = json.loads(repaired)
        logger.info("Repaired malformed JSON in the classifier reply")
        return data


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
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_params = extra_params or {}
        #: Prompt override exists so evalharness can A/B prompt variants
        #: against the production one; None keeps today's behaviour exactly.
        self.system_prompt = _system_prompt() if system_prompt is None else system_prompt

    def classify(self, question: str, hint: str | None = None) -> Classification:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": _user_message(question, hint)},
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
            estimated_difficulty=(
                data.get("difficulty") if data.get("difficulty") in ("easy", "medium", "hard") else "medium"
            ),
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

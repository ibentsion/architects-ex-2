"""Prompts for writing and checking v2 questions.

Generation is one prompt per kind, specialized by difficulty. v1 items are used
as *form* anchors only — few-shot examples are always drawn from a different
category than the one being written, so the model copies the voice and the
answer style without seeing content from the category it must invent for. v1
itself is held out as the validation set.

Verification splits into gates with narrow questions, because a single "is this
a good question?" call grades everything and refuses nothing. Derivability is
not here at all: it reuses the citation judge from `evalharness.prompts`, which
already asks whether given pages establish a given answer.
"""
from __future__ import annotations

import json

#: What makes a question easy/medium/hard in v1 — the structural definition
#: every generator and the difficulty gate share.
DIFFICULTY_RULES = {
    "easy": (
        "EASY: one fact, stated outright on the page. The customer asks about a "
        "single coverage/condition and the ground-truth answer restates what the "
        "page says. No arithmetic, no combining of clauses."
    ),
    "medium": (
        "MEDIUM: a specific limit, condition, exception or procedure that sits "
        "inside a longer passage — the reader must find the right clause among "
        "several. Alternatively, the customer states a common misconception and "
        "the ground truth corrects it. Still a single source, no arithmetic."
    ),
    "hard": (
        "HARD: the answer requires real work on top of retrieval — either "
        "arithmetic over numbers on the page (a percentage of a sum, a "
        "deductible, a pro-rata refund), or combining two separate clauses into "
        "one conclusion. The ground-truth answer must show the computation or "
        "the combination explicitly."
    ),
}

_FORM_RULES = """The question must read like a real customer of an Israeli insurance company wrote it:
- First person, colloquial Hebrew, describing a concrete situation ("קרה נזק בדירה שלי לפני חודש...", "אני נוסע לחו\"ל בשבוע הבא...").
- Self-contained: never mention the document, the policy booklet, a page, a section number, "according to the text", or "based on the attached". The customer has not read the policy.
- One clear thing is being asked. It may be a yes/no question or a "how much / how long / how do I" question.
- Do NOT quote the page verbatim in the question — a customer would not know the wording.

The ground-truth answer must be:
- Short and decisive (one to three sentences), in Hebrew, usually opening with כן or לא when the question is yes/no.
- Carrying the exact number, limit, condition or exception the page states — not a vague paraphrase.
- Supported entirely by the page(s) shown. Never add knowledge from outside them."""

_OUTPUT = """Return ONLY a JSON object (no markdown fences, no prose):
{
  "question": "<the customer's question, in Hebrew>",
  "ground_truth_answer": "<the authoritative answer, in Hebrew>",
  "rationale": "<1-2 sentences in English: which part of the page it comes from and why it is %(difficulty)s>"
}
If the page does not support a good %(difficulty)s question, return {"skip": "<why, in English>"} instead. Skipping is always better than inventing."""


def _examples_block(examples: list[dict]) -> str:
    """v1 items rendered as form anchors — question and answer only, never
    their sources (v2 must not build on the held-out set's pages)."""
    return "\n\n".join(
        f"Example {i} (category: {ex['domain']}, difficulty: {ex['difficulty']}):\n"
        f"Question: {ex['question']}\nGround-truth answer: {ex['ground_truth_answer']}"
        for i, ex in enumerate(examples, start=1))


def _page_block(pages: list) -> str:
    return "\n\n".join(
        f"--- PAGE {i} | file: {p.file} | page: {'n/a' if p.page is None else p.page} ---\n{p.text}"
        for i, p in enumerate(pages, start=1))


SYSTEM_STANDARD = """You write evaluation questions for a Hebrew insurance customer-support RAG system. You are given ONE page from the insurance corpus and must write one question a customer would ask, whose answer is established by that page.

""" + _FORM_RULES + """

%(difficulty_rule)s

""" + _OUTPUT


SYSTEM_MULTI = """You write evaluation questions for a Hebrew insurance customer-support RAG system. You are given TWO pages from two different documents of the same insurance category. Write one question whose answer needs BOTH pages.

""" + _FORM_RULES + """

The two-page requirement is tested mechanically, so it must genuinely hold:
- Neither page alone may be enough to answer the question. A judge will be shown each page separately; if either one establishes your answer on its own, the question is rejected.
- Both pages together must fully establish it. Do not ask about anything the pages do not state.
- Natural shapes: comparing what two products/policies say about the same subject, or a situation where one page gives the rule and the other gives the number/exception it depends on.
- The ground-truth answer must state both halves explicitly.

%(difficulty_rule)s

""" + _OUTPUT


SYSTEM_UNANSWERABLE = """You write evaluation questions for a Hebrew insurance customer-support RAG system. Your job here is the opposite of usual: write a question the corpus CANNOT answer, to test whether the system correctly says "I don't have enough information" instead of inventing an answer.

You are shown a sample of pages from one insurance category so you can see what the corpus does and does not cover.

""" + _FORM_RULES.replace("Supported entirely by the page(s) shown. Never add knowledge from outside them.",
                          "NOT an answer. See the ground-truth rule below.") + """

The question must be:
- Squarely inside this insurance category — a question this insurer's customers really would ask. Not off-topic, not about another category, not a joke.
- Genuinely unanswerable from the corpus: nothing in these documents states the answer. A retrieval + judge pass over the category will check this, and a question that turns out to be answered somewhere is rejected.
- %(unanswerable_rule)s

The "ground_truth_answer" field here does NOT answer the question. It states, in Hebrew, that the available documents do not contain this information and names what the customer should be told or asked to do instead (e.g. contact the agent / the insurer). This is the behaviour a correct system must exhibit.

""" + _OUTPUT

#: An unanswerable question's difficulty is how tempting the gap is.
UNANSWERABLE_RULES = {
    "easy": ("EASY: plainly outside what these documents cover — a customer-service "
             "matter the corpus never touches (personal account details, current "
             "prices, the status of an individual claim). A system with any "
             "discipline should notice at once."),
    "medium": ("MEDIUM: the sort of detail these documents COULD have contained but "
               "do not — a specific limit, waiting period or procedure for something "
               "the corpus mentions only in passing."),
    "hard": ("HARD: right next to something the corpus DOES state. The documents "
             "cover the neighbouring case (another policy, another peril, another "
             "population) so retrieval will return confident-looking pages that do "
             "not actually answer this question. This is the trap that catches "
             "systems which answer from whatever they retrieved."),
}


def build_generation_messages(kind: str, difficulty: str, pages: list,
                              examples: list[dict], category: str) -> list:
    """Messages for writing one item of `kind` at `difficulty`."""
    fill = {"difficulty": difficulty,
            "difficulty_rule": DIFFICULTY_RULES[difficulty],
            "unanswerable_rule": UNANSWERABLE_RULES.get(difficulty, "")}
    system = {"standard": SYSTEM_STANDARD, "multi_source": SYSTEM_MULTI,
              "unanswerable": SYSTEM_UNANSWERABLE}[kind] % fill

    header = (f"Insurance category: {category}\n\n"
              "Here is how questions in this dataset are written (different "
              "categories — copy the voice and the answer style, not the "
              "subject):\n\n" + _examples_block(examples))
    if kind == "unanswerable":
        body = ("\n\nA sample of the pages this category's corpus contains:\n\n"
                + _page_block(pages)
                + f"\n\nWrite one {difficulty} unanswerable question for this category.")
    else:
        label = "page" if len(pages) == 1 else "two pages"
        body = (f"\n\nThe {label} to write from:\n\n" + _page_block(pages)
                + f"\n\nWrite one {difficulty} question.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": header + body}]


SYSTEM_FORM_GATE = """You check whether a candidate evaluation question is written in the right form. You are NOT checking whether it is factually right — another judge does that. Judge only the writing.

Reject ("fail") if any of these is true:
- The question refers to the source material: "according to the document/policy/section/page", "in the attached text", "as written", or quotes it verbatim.
- The question is not in a customer's first-person voice, or reads like an exam question about a document.
- The question is not self-contained — it cannot be understood without seeing the page it was written from (e.g. "what is the limit mentioned there?").
- The question or the answer is not in Hebrew.
- The ground-truth answer is vague where the question demands a specific number/limit/condition, or rambles well beyond three sentences.

Return ONLY a JSON object: {"verdict": "pass" | "fail", "reason": "<1 sentence in English; if fail, the single most important reason>"}"""


SYSTEM_DIFFICULTY_GATE = """You classify how hard an evaluation question is for a retrieval-augmented insurance assistant, using these definitions ONLY:

%(easy)s

%(medium)s

%(hard)s

You are shown the question, its ground-truth answer, and how many source pages it was built from. Classify it, then say whether it matches the requested label.

Return ONLY a JSON object: {"difficulty": "easy" | "medium" | "hard", "reason": "<1 sentence in English>"}""" % {
    "easy": DIFFICULTY_RULES["easy"], "medium": DIFFICULTY_RULES["medium"],
    "hard": DIFFICULTY_RULES["hard"]}


SYSTEM_TOPICALITY_GATE = """You check whether a question belongs to an insurance category — whether a real customer holding this kind of insurance would plausibly ask it of the insurer's support line.

You are shown the category name and a sample of what its documents cover. A question can be about a matter the documents do not answer (that is expected here); what you are judging is only whether it is the right SUBJECT for this category, and a real customer question rather than a trick or a joke.

Return ONLY a JSON object: {"verdict": "pass" | "fail", "reason": "<1 sentence in English>"}"""


def build_form_messages(item: dict) -> list:
    user = (f"Question:\n{item['question']}\n\n"
            f"Ground-truth answer:\n{item['ground_truth_answer']}")
    return [{"role": "system", "content": SYSTEM_FORM_GATE},
            {"role": "user", "content": user}]


def build_difficulty_messages(item: dict, n_sources: int) -> list:
    user = (f"Question:\n{item['question']}\n\n"
            f"Ground-truth answer:\n{item['ground_truth_answer']}\n\n"
            f"Built from {n_sources} source page(s).")
    return [{"role": "system", "content": SYSTEM_DIFFICULTY_GATE},
            {"role": "user", "content": user}]


def build_topicality_messages(question: str, category: str, pages: list) -> list:
    inventory = "\n".join(f"- {p.file}: {p.text[:200].strip()}..." for p in pages[:12])
    user = (f"Insurance category: {category}\n\n"
            f"What this category's documents look like:\n{inventory}\n\n"
            f"Question:\n{question}")
    return [{"role": "system", "content": SYSTEM_TOPICALITY_GATE},
            {"role": "user", "content": user}]


def retry_message(reason: str) -> dict:
    """Fed back to the generator after a rejection, so the next attempt is not
    a blind re-roll."""
    return {"role": "user", "content":
            f"That question was rejected: {reason}\n\n"
            "Write a different one that fixes this. Same JSON schema, same page(s). "
            "If you cannot, return {\"skip\": \"<why>\"}."}


def parse_generation(raw: dict) -> dict:
    """Validate a generator reply: either a skip or a well-formed item."""
    if "skip" in raw:
        raise Skip(str(raw["skip"])[:200])
    question = str(raw.get("question", "")).strip()
    answer = str(raw.get("ground_truth_answer", "")).strip()
    if len(question) < 20:
        raise ValueError(f"question too short: {question!r}")
    if len(answer) < 10:
        raise ValueError(f"ground_truth_answer too short: {answer!r}")
    return {"question": question, "ground_truth_answer": answer,
            "rationale": str(raw.get("rationale", "")).strip()}


class Skip(Exception):
    """The generator declined this page — resample rather than retry."""


def parse_verdict(raw: dict) -> dict:
    """Validate a pass/fail gate reply."""
    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "fail"):
        raise ValueError(f"bad verdict: {verdict!r}")
    return {"verdict": verdict, "reason": str(raw.get("reason", "")).strip()}


def parse_difficulty(raw: dict) -> dict:
    difficulty = str(raw.get("difficulty", "")).strip().lower()
    if difficulty not in DIFFICULTY_RULES:
        raise ValueError(f"bad difficulty: {difficulty!r}")
    return {"difficulty": difficulty, "reason": str(raw.get("reason", "")).strip()}


def dumps(item: dict) -> str:
    """Compact JSON for logging a generated item."""
    return json.dumps(item, ensure_ascii=False)

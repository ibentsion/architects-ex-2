"""Judge prompts.

Two independent judgments, each with its own prompt and its own output schema:

* **Answer quality** — three variants sharing one schema, so parsing and
  aggregation are identical; they differ in how the judge is told to reason.
      rubric — anchored 0-10 scales per category (default; most consistent numbers)
      strict — terse fact-agreement check; fastest, cheapest
      claims — decompose the ground truth into atomic facts and count
               supported/contradicted ones before scoring (best diagnostics)
* **Citation accuracy** — does the cited evidence actually establish the
  answer? The judge reads the real corpus pages the system cited (resolved by
  `evalharness.pages`) and rules fully / partially / not_at_all.
"""

# Fields every judge must return. `claims` is optional extra from the claims variant.
OUTPUT_SCHEMA = """Return ONLY a JSON object (no markdown fences, no prose) with exactly these fields:
{
  "verdict": "correct" | "partially_correct" | "incorrect" | "refusal",
  "hallucination": true | false,
  "correctness": <integer 0-10>,
  "completeness": <integer 0-10>,
  "conversational_quality": <integer 0-10>,
  "reasoning": "<1-3 sentences in English explaining the verdict and scores>"
}"""

_SHARED_RULES = """Rules:
- The question, ground-truth answer, and system answer are in Hebrew. Your reasoning and output must be in English.
- Judge ONLY against the ground-truth answer. Do not use your own knowledge of insurance to overrule it — the ground truth is authoritative even if you believe otherwise.
- "refusal" = the system declined to answer or said it lacks information ("איני יודע", "אין לי מספיק מידע" etc.). A refusal is NEVER a hallucination.
- "hallucination" = true only when the answer confidently asserts something that CONTRADICTS the ground truth on the asked fact. Extra unverifiable-but-plausible detail that does not contradict the ground truth is not a hallucination.
- Ignore citations entirely — they are scored separately by a dedicated citation judge."""

SYSTEM_RUBRIC = """You are a meticulous evaluator of customer-support answers for an Israeli insurance company. You compare a system's answer against an authoritative ground-truth answer and grade it on fixed scales.

""" + _SHARED_RULES + """

Scoring anchors:
- correctness (does the answer agree with the ground truth on the asked fact?):
  10 = fully agrees on every asked fact, including exact numbers/limits/conditions;
  7-9 = agrees on the main fact, minor imprecision on secondary details;
  4-6 = partially correct — right on some asked facts, wrong or missing on others;
  1-3 = mostly wrong but touches the right topic;
  0 = contradicts the ground truth on the main fact, or is a refusal, or is off-topic.
- completeness (does it cover everything in the ground truth?):
  10 = every component of the ground truth (numbers, conditions, exceptions, required documents) is present;
  5 = about half the components; 0 = none (including refusals).
- conversational_quality (would this be a good first-line support reply, judged independently of factual correctness?):
  10 = clear, well-structured, polite, natural Hebrew, appropriate length;
  5 = understandable but rambling, poorly structured, or with awkward language;
  0 = incoherent or inappropriate. Do NOT penalize this score for factual errors.

""" + OUTPUT_SCHEMA

SYSTEM_STRICT = """You are a strict fact-checker. Compare the system answer to the authoritative ground-truth answer and decide whether they agree on the asked fact.

""" + _SHARED_RULES + """

Focus on the single question: does the system answer state the same fact(s) as the ground truth? Set correctness accordingly (10 = same fact including numbers, 0 = contradicts or refusal). Set completeness and conversational_quality with quick judgment — they are secondary here.

""" + OUTPUT_SCHEMA

SYSTEM_CLAIMS = """You are a claim-level evaluator of customer-support answers for an Israeli insurance company.

""" + _SHARED_RULES + """

Method — do this before scoring:
1. Decompose the ground-truth answer into its atomic factual claims (a number, a limit, a condition, a required document each count as one claim).
2. For each claim, classify the system answer as: SUPPORTED (states it correctly), CONTRADICTED (states it incorrectly), or MISSING (does not address it).
3. correctness = 10 * supported / (supported + contradicted), or 0 if nothing is supported; completeness = 10 * supported / total claims. Round to integers.
4. hallucination = true if any claim is CONTRADICTED confidently.
5. conversational_quality: judge clarity/tone/structure independently of facts.

In addition to the standard fields, include: "claims": {"total": <int>, "supported": <int>, "contradicted": <int>, "missing": <int>}

""" + OUTPUT_SCHEMA

SYSTEM_PROMPTS = {
    "rubric": SYSTEM_RUBRIC,
    "strict": SYSTEM_STRICT,
    "claims": SYSTEM_CLAIMS,
}


CITATION_OUTPUT_SCHEMA = """Return ONLY a JSON object (no markdown fences, no prose) with exactly these fields:
{
  "citation_support": "fully" | "partially" | "not_at_all",
  "per_citation": [{"idx": <integer, the CITATION index shown>, "label": "establishes" | "partial" | "unrelated"}],
  "reasoning": "<1-3 sentences in English, quoting or paraphrasing the page text you relied on>"
}
"per_citation" must contain exactly one entry per citation shown, in order."""

SYSTEM_CITATION = """You are verifying the evidence behind a customer-support answer for an Israeli insurance company. You are shown a question, its authoritative ground-truth answer, and the actual corpus pages the system cited. Decide whether those pages establish the ground-truth answer.

Rules:
- The question, ground-truth answer, and page text are in Hebrew. Your reasoning and output must be in English.
- Judge ONLY from the page text shown. Do not use your own knowledge of insurance, and never assume that some other, unshown page says the right thing.
- The same fact appears in several corpus documents. ANY page that genuinely establishes the fact earns credit, no matter which document or category it comes from. There is no expected list of "correct" sources — do not reason about which document the answer "should" have come from.
- The pages are automated extractions of Hebrew right-to-left PDFs: word order, punctuation and number placement are often mangled, and tables are flattened to Markdown. Judge the substance, not the formatting. A number or condition that is present but awkwardly placed still counts.
- Pages marked [TRUNCATED] were cut for length; judge what is shown.

Verdict for "citation_support" — the cited pages, taken TOGETHER:
- "fully" = a reader of these pages could derive the ground-truth answer, including its key specifics (the asked number, limit, condition or exception).
- "partially" = the pages support part of the ground-truth answer — e.g. the right topic and some conditions but not the decisive number, or one of two facts the ground truth combines.
- "not_at_all" = none of the pages support the ground-truth answer; they are off-topic, or address the topic without stating anything the ground truth asserts.

Label for each individual citation:
- "establishes" = this page alone states the ground-truth answer (or its decisive fact).
- "partial" = this page contributes something relevant but does not establish the answer by itself.
- "unrelated" = this page contributes nothing to the ground-truth answer.

""" + CITATION_OUTPUT_SCHEMA

#: Page-text budget per judge call. Corpus pages run ~1.6k chars (p95 ~2.9k),
#: so these caps almost never fire — they exist so an answer citing many long
#: pages cannot blow the judge's context.
MAX_PAGE_CHARS = 6000
MAX_TOTAL_CHARS = 40000


def _render_pages(resolved: list) -> str:
    """Render resolved citations as numbered, delimited page blocks."""
    blocks, budget = [], MAX_TOTAL_CHARS
    for idx, item in enumerate(resolved):
        text = item["text"]
        truncated = len(text) > MAX_PAGE_CHARS or len(text) > budget
        text = text[: min(MAX_PAGE_CHARS, max(budget, 0))]
        budget -= len(text)
        page = "n/a" if item["page"] is None else item["page"]
        blocks.append(f"--- CITATION {idx} | file: {item['file']} | page: {page} ---\n"
                      + text + ("\n[TRUNCATED]" if truncated else ""))
    return "\n\n".join(blocks)


def build_citation_messages(question: dict, resolved: list) -> list:
    """Build the chat messages for judging one answer's resolved citations.

    `resolved` is the list of citations that resolved to real corpus pages —
    each {file, page, text}. Unresolvable citations are never shown to the
    judge; they are penalized arithmetically in `evalharness.citations`.
    """
    user = f"""Question (Hebrew):
{question["question"]}

Ground-truth answer (authoritative):
{question["ground_truth_answer"]}

Cited pages, resolved to the real corpus text ({len(resolved)} citation(s)):

{_render_pages(resolved)}"""
    return [
        {"role": "system", "content": SYSTEM_CITATION},
        {"role": "user", "content": user},
    ]


def build_messages(question: dict, answer: dict, variant: str) -> list:
    """Build the chat messages for judging one answer against one dev question."""
    user = f"""Question (Hebrew):
{question["question"]}

Ground-truth answer (authoritative):
{question["ground_truth_answer"]}

System answer to evaluate:
{answer.get("answer", "")}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[variant]},
        {"role": "user", "content": user},
    ]

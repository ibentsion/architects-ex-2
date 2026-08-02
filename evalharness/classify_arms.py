"""The 12 arms of the classifier sweep — one dataclass per experimental cell.

An arm is a complete recipe for producing a prediction: which prompt variant,
which model, which decoding knobs, whether it sees the retrieval hint, and
which strategy runs it. Everything the runner needs is here, so adding an arm
never means editing the runner.

``strategy`` selects the execution path in ``classify_eval``:

  llm            one classify call (optionally repeated for self-consistency)
  hint_vote      no LLM at all — the filter is the index's own top category
  verify_2stage  a baseline classify call, then a cheap retraction call that
                 sees the retrieval evidence and may drop tags (never add)

``model=None`` and ``extra_params=None`` mean "whatever the config says",
which reproduces ``rag.classify.build_classifier`` exactly — that is what makes
``baseline`` the production path rather than an imitation of it. Arms that
change the model pass ``extra_params={}`` because the orchestrator's reasoning
knobs are model-specific (same rule as ``build_classifier``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Share of the top-10 hits that must agree before hint-vote commits to a
#: filter. Below it the arm emits no tag — a deliberate abstention, since the
#: whole point of the arm is to measure the index's *confident* signal.
HINT_VOTE_THRESHOLD = 0.6


@dataclass(frozen=True)
class Arm:
    id: str
    what: str
    strategy: str = "llm"
    prompt: str = "baseline"
    model: str | None = None
    extra_params: dict | None = None
    temperature: float = 0.0
    #: Self-consistency: number of samples, and how many must carry a category
    #: for it to survive. samples=1 keeps the single-shot path.
    samples: int = 1
    min_votes: int = 1
    use_hint: bool = False
    hint_vote_threshold: float = HINT_VOTE_THRESHOLD

    @property
    def needs_hints(self) -> bool:
        return self.use_hint or self.strategy in ("hint_vote", "verify_2stage")


#: gpt-oss-120b reasoning knobs, spelled out rather than derived, because
#: effort-medium must differ from the config default in exactly one field.
_REASONING = lambda effort: {  # noqa: E731
    "reasoning_effort": effort,
    "allowed_openai_params": ["reasoning_effort"],
}

ARMS: list[Arm] = [
    Arm("baseline", "today's production prompt, gpt-oss-120b, reasoning low, temp 0"),
    Arm("abstain", "+ a wrong tag is worse than none — abstain when undetermined",
        prompt="abstain"),
    Arm("examples", "+ 2 invented example questions per category",
        prompt="examples"),
    Arm("rich-desc", "category descriptions with belongs / does-NOT-belong + overlap "
                     "disambiguation",
        prompt="rich-desc"),
    Arm("decision-rules", "+ ordered decision rules with a stated precedence for overlaps",
        prompt="decision-rules"),
    Arm("hint-sparse", "baseline prompt + retrieval evidence block",
        use_hint=True),
    Arm("hint-vote", "no LLM — filter = top hint category when its share is high enough",
        strategy="hint_vote"),
    Arm("model-qwen", "baseline prompt on Qwen3-235B-A22B-Instruct",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507", extra_params={}),
    Arm("model-deepseek", "baseline prompt on DeepSeek-V4-Pro",
        model="deepseek-ai/DeepSeek-V4-Pro", extra_params={}),
    Arm("effort-medium", "gpt-oss-120b at reasoning_effort medium",
        extra_params=_REASONING("medium")),
    Arm("selfcons-3", "baseline prompt, temp 0.7, 3 samples, keep categories with >=2 votes",
        temperature=0.7, samples=3, min_votes=2),
    Arm("verify-2stage", "baseline tags, then a cheap call that sees the evidence and "
                         "may retract a tag",
        strategy="verify_2stage"),
]

ARMS_BY_ID = {arm.id: arm for arm in ARMS}
BASELINE_ARM = "baseline"


def get_arm(arm_id: str) -> Arm:
    if arm_id not in ARMS_BY_ID:
        raise KeyError(f"unknown arm {arm_id!r}; have: {', '.join(ARMS_BY_ID)}")
    return ARMS_BY_ID[arm_id]

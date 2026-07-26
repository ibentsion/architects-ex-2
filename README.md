# Exercise 2 — Harel Insurance Support Agent

Read **`exercise2_customer_support_agent.md`** for the full exercise.

## What's here

| Path | What |
|---|---|
| `exercise2_customer_support_agent.md` | The exercise |
| `reference_questions.json` | Dev Q&A set: questions labeled easy/medium/hard, ground-truth answers and file+page citations |
| `contract.py` | The FastAPI `/ask` schema your system must expose (grading calls it) |
| `baseline_runner.py` | Stage 1: run the questions through a bare model, answers JSONL out |
| `submit_runner.py` | Batch-asks your `/ask` endpoint → answers JSONL (used for final submission) |
| `tf_client.py` | Minimal Token Factory client with per-call cost estimate (shared key — play fair) |

There is deliberately no evaluation script here: **building your own harness is
a Stage 1 deliverable** — the exercise page documents exactly what it must
measure (relevance, hallucination rate, citation accuracy, latency).

## Evaluation harness (`evalharness/`)

Our Stage 1 harness. Scores an answers JSONL against the dev set with an
LLM-as-judge (numeric 0–10 grades for correctness / completeness /
conversational quality, plus verdict and hallucination flags), deterministic
citation accuracy (`any_of` group matching), and latency stats. Writes
`judgments.jsonl`, `metrics.json`, and a markdown `report.md` with per-domain /
per-difficulty / per-source-count breakdowns and prioritized improvement
suggestions.

```bash
uv run -m evalharness.run \
    --questions reference_questions.json \
    --answers baseline_answers.jsonl \
    --out eval_results/baseline

# options: --prompt {rubric,strict,claims}   judge prompt variant (default rubric)
#          --judges MODEL [MODEL ...]        several models = judging committee
#          --limit N --workers N
```

The judge is pinned to `deepseek-ai/DeepSeek-V4-Pro` at temperature 0 for
run-to-run comparability. Questions/answers are Hebrew; judge output and
reports are English.

## Quickstart

```bash
uv sync
export NEBIUS_API_KEY=...               # the shared course Token Factory key
export OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
export OPENAI_API_KEY=$NEBIUS_API_KEY

# Stage 1: baseline answers, then score them with YOUR harness
uv run baseline_runner.py --model deepseek-ai/DeepSeek-V4-Pro
```

The document corpus: `uv run get_corpus.py` downloads the frozen snapshot
from the public HF dataset `orik/apex-ex2-harel-corpus` into a local `corpus/`
dir (gitignored). Ground-truth answers are anchored to that snapshot, not the
live site.

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
| `webapi/` | FastAPI layer for the web UI: an SSE app that streams the agent's pipeline trace, plus a local bridge serving the QA-history and citation endpoints |
| `webui/` | React web UI — live support chat with the pipeline trace as it happens, and a QA-history browser over the eval runs (see `webui/README.md`) |

`webui/` and `webapi/` are Stage 3's **optional bonus** ("voice interface,
simple UI") and are explicitly not part of the graded contract — grading calls
`contract.py`, which they do not touch.

There is deliberately no evaluation script here: **building your own harness is
a Stage 1 deliverable** — the exercise page documents exactly what it must
measure (relevance, hallucination rate, citation accuracy, latency).

## Evaluation harness (`evalharness/`)

Our Stage 1 harness. Scores an answers JSONL against the dev set with an
LLM-as-judge (numeric 0–10 grades for correctness / completeness /
conversational quality, plus verdict and hallucination flags), LLM-judged
citation accuracy, and latency stats. Writes `judgments.jsonl`, `metrics.json`,
and a markdown `report.md` with per-domain / per-difficulty / per-source-count
breakdowns and prioritized improvement suggestions.

**Citation accuracy** — does the cited evidence actually establish the answer?
Every cited `{file, page}` is resolved to the *actual* corpus page (corpus walk
→ sha256 → Docling parse cache; no search), and a judge rules whether the cited
pages establish the ground-truth answer: fully / partially / not at all. The
same fact appears in several corpus documents, so any page that truly
establishes it earns credit — there is no fixed list of "correct" sources.
`ground_truth_sources` records where each reference answer was authored from
and is kept only as an unscored retrieval diagnostic (`gt_source_hit_rate`).
A citation pointing at a nonexistent file or page earns nothing and dilutes the
rest:

    citation_accuracy = credit (1.0 / 0.5 / 0.0) × (citations resolving to a real page ÷ citations made)

```bash
python -m evalharness.run \
    --questions reference_questions.json \
    --answers baseline_answers.jsonl \
    --out eval_results/baseline

# options: --prompt {rubric,strict,claims}   judge prompt variant (default rubric)
#          --judges MODEL [MODEL ...]        several models = judging committee
#          --corpus DIR --cache-dir DIR      where cited pages are resolved from
#          --limit N --workers N
```

Citation judging needs `corpus/` and the `cache/parsed/` Docling parses the
index was built from (`get_corpus.py`, then any ingest run). It costs one extra
judge call per answer that cited at least one resolvable page.

The judge is pinned to `deepseek-ai/DeepSeek-V4-Pro` at temperature 0 for
run-to-run comparability. Questions/answers are Hebrew; judge output and
reports are English.

## Quickstart

```bash
pip install -r requirements.txt
export NEBIUS_API_KEY=...               # the shared course Token Factory key
export OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
export OPENAI_API_KEY=$NEBIUS_API_KEY

# Stage 1: baseline answers, then score them with YOUR harness
python baseline_runner.py --model deepseek-ai/DeepSeek-V4-Pro
```

The document corpus: `python get_corpus.py` downloads the frozen snapshot
from the public HF dataset `orik/apex-ex2-harel-corpus` into a local `corpus/`
dir (gitignored). Ground-truth answers are anchored to that snapshot, not the
live site.

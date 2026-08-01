#!/usr/bin/env bash
# Markdown-vs-Docling parser comparison (2026-08-01).
#
# Builds the Marker/DataLab markdown index, then evaluates it against the
# Docling per_page index with everything else held constant: same questions,
# same judges, same chunker (per_page), same embedder (bge-m3), same generator
# (gpt-oss-120b low reasoning). The only variable is the parse.
#
# Each arm's citations are judged against ITS OWN pages (--doc-source);
# scoring markdown citations against Docling's text would grade a page the
# retriever never saw.
#
# Run on the node from the repo root:
#   cloud/submit_job.sh run 'bash cloud/markdown_arm_eval.sh'
set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
JUDGES=(openai/gpt-oss-120b Qwen/Qwen3-235B-A22B-Instruct-2507 deepseek-ai/DeepSeek-V4-Pro)
RUN="parser-markdown-vs-docling-$TS"

# Build the markdown index (the Docling per_page index already exists).
python -m rag.cli.ingest --config configs/markdown-per_page-bgem3.yaml

ANS_MD="rag_answers_${RUN}_markdown.jsonl"
ANS_DL="rag_answers_${RUN}_docling.jsonl"
python -m rag.cli.query --config configs/markdown-per_page-bgem3.yaml \
    --questions reference_questions.json --out "$ANS_MD"
python -m rag.cli.query --config configs/final-per_page-bgem3-gptoss-low.yaml \
    --questions reference_questions.json --out "$ANS_DL"

fail=0
python -m evalharness.run --questions reference_questions.json --answers "$ANS_DL" \
    --out "eval_results/$RUN/docling-per_page" --doc-source pdf \
    --judges "${JUDGES[@]}" || fail=1
python -m evalharness.run --questions reference_questions.json --answers "$ANS_MD" \
    --out "eval_results/$RUN/markdown-per_page" --doc-source markdown \
    --judges "${JUDGES[@]}" || fail=1

# Ship results home even if a judge failed on some questions.
hf upload "$ARTIFACTS_REPO" "eval_results/$RUN" "results/$RUN" --repo-type dataset
hf upload "$ARTIFACTS_REPO" "$ANS_MD" "results/$RUN/$ANS_MD" --repo-type dataset
hf upload "$ARTIFACTS_REPO" "$ANS_DL" "results/$RUN/$ANS_DL" --repo-type dataset
echo "RESULTS_PREFIX=results/$RUN"
exit $fail

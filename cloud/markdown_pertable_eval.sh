#!/usr/bin/env bash
# Markdown vs Docling, apples-to-apples (2026-08-01, second pass).
#
# The first pass had to handicap markdown onto per_page, because per_table
# keys off Docling TableItems. The markdown parser now lifts pipe tables into
# real tables[], so both arms run the PRODUCTION config: per_table + bge-m3 +
# stanza (chunker/embedder/normalizer identity hashes verified equal). The
# only difference is where the documents come from.
#
# Also folds in the markdown-pipeline fixes found by diagnosing the first
# pass's retrieval losses: markdown links unwrapped (a mailto link put the
# same address in a chunk twice), image alt-text captions dropped, and table
# syntax kept out of embedded prose.
#
# Run on the node from the repo root:
#   cloud/submit_job.sh run 'bash cloud/markdown_pertable_eval.sh'
set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
JUDGES=(openai/gpt-oss-120b Qwen/Qwen3-235B-A22B-Instruct-2507 deepseek-ai/DeepSeek-V4-Pro)
RUN="parser-pertable-markdown-vs-docling-$TS"

python -m rag.cli.ingest --config configs/markdown-per_table-bgem3.yaml

ANS_MD="rag_answers_${RUN}_markdown.jsonl"
ANS_DL="rag_answers_${RUN}_docling.jsonl"
python -m rag.cli.query --config configs/markdown-per_table-bgem3.yaml \
    --questions reference_questions.json --out "$ANS_MD"
python -m rag.cli.query --config configs/final-per_table-bgem3-gptoss-low.yaml \
    --questions reference_questions.json --out "$ANS_DL"

fail=0
python -m evalharness.run --questions reference_questions.json --answers "$ANS_DL" \
    --out "eval_results/$RUN/docling-per_table" --doc-source pdf \
    --judges "${JUDGES[@]}" || fail=1
python -m evalharness.run --questions reference_questions.json --answers "$ANS_MD" \
    --out "eval_results/$RUN/markdown-per_table" --doc-source markdown \
    --judges "${JUDGES[@]}" || fail=1

hf upload "$ARTIFACTS_REPO" "eval_results/$RUN" "results/$RUN" --repo-type dataset
hf upload "$ARTIFACTS_REPO" "$ANS_MD" "results/$RUN/$ANS_MD" --repo-type dataset
hf upload "$ARTIFACTS_REPO" "$ANS_DL" "results/$RUN/$ANS_DL" --repo-type dataset
echo "RESULTS_PREFIX=results/$RUN"
exit $fail

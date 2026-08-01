#!/usr/bin/env bash
# Post-BiDi-repair eval (2026-08-01): re-run of the per_table + bge-m3 /
# gpt-oss-120b-low arm from cloud/committee_eval.sh, against indexes rebuilt
# from the repaired parse cache (rag/parsing/rtl_repair.py).
#
# Same questions, same judges, same generation config as
# eval_results/committee-pertable-bgem3-gptosslow-20260727T195127Z, so the
# delta isolates one variable: Hebrew table cells that used to be word-order
# reversed (99.4% of multi-word cells -> 5.3%).
#
# Baseline to compare against (2026-07-27, reversed index):
#   correctness 5.65  completeness 5.44  hallucination 0.17  refusal 0.27
#   verdicts 24 correct / 5 partial / 6 incorrect / 13 refusal
#   citation recall 0.33  precision 0.26
#
# Run on the node from the repo root:
#   cloud/submit_job.sh run 'bash cloud/postrepair_eval.sh'
set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
JUDGES=(openai/gpt-oss-120b Qwen/Qwen3-235B-A22B-Instruct-2507 zai-org/GLM-5.1)
RUN="postrepair-pertable-bgem3-gptosslow-$TS"

ANS="rag_answers_$RUN.jsonl"
python -m rag.cli.query --config configs/final-per_table-bgem3-gptoss-low.yaml \
    --questions reference_questions.json --out "$ANS"

fail=0
python -m evalharness.run --questions reference_questions.json --answers "$ANS" \
    --out "eval_results/$RUN" --judges "${JUDGES[@]}" || fail=1

# Ship results home even if a judge failed on some questions.
hf upload "$ARTIFACTS_REPO" "eval_results/$RUN" "results/$RUN" --repo-type dataset
hf upload "$ARTIFACTS_REPO" "$ANS" "results/$RUN/$ANS" --repo-type dataset
echo "RESULTS_PREFIX=results/$RUN"
exit $fail

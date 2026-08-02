#!/usr/bin/env bash
# Depth-to-cover + filter-mode study (2026-08-02, quick tasks 260802-004/005).
#
# Retrieval only — no generation, no judges, and no classification calls: every
# predicted-filter mode replays the tags 260802-003's best classifier arm
# already produced.
#
# Six filter modes on the validation half. The holdout half is deliberately NOT
# run: it exists to confirm whatever policy this study picks, and a number you
# have already seen is not a confirmation.
#
# Run on the node from the repo root:
#   cloud/submit_job.sh run 'bash cloud/retrieval_depth.sh'
set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
RUN="retrieval-depth-$TS"
PREDICTIONS="eval_results/classify-merge-20260802T050208Z/merged-B/predictions.jsonl"

# The split is deterministic in its seed, so regenerating it on the node
# reproduces the committed files rather than depending on them being shipped.
python -m evalharness.split

python -m evalharness.retrieval_audit --out "eval_results/$RUN" --workers 4 \
    --arm pdf-per_table=configs/final-per_table-bgem3-gptoss-low.yaml \
    --dataset validation=ref_q_validation_set_v1.jsonl \
    --deep-k 100 \
    --filter-mode none gold gold-family predicted predicted-set predicted-family \
    --predictions "$PREDICTIONS"

hf upload "$ARTIFACTS_REPO" "eval_results/$RUN" "results/$RUN" --repo-type dataset
echo "RESULTS_PREFIX=results/$RUN"

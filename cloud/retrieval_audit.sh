#!/usr/bin/env bash
# Retrieval hit-rate audit: where the ground-truth page is lost, per arm
# (2026-08-02, quick task 260802-002 T3).
#
# {pdf, markdown} x {per_table, per_page} x {v1, v2}, retrieval only — no
# generation and no judges, so the whole matrix costs GPU time and nothing
# else. Each ground-truth source group is filed under the furthest stage it
# reached (missing_from_index / not_retrieved / not_gated / gated), which
# separates corpus and parse gaps from ranking failures.
#
# The markdown arms need their indexes ingested; arms whose index is absent
# are skipped by the auditor rather than failing the run.
#
# Run on the node from the repo root:
#   cloud/submit_job.sh run 'bash cloud/retrieval_audit.sh'
set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
RUN="retrieval-audit-$TS"

# Ingest only what is missing: an existing manifest means the index is there.
for cfg in configs/final-per_table-bgem3-gptoss-low.yaml \
           configs/final-per_page-bgem3-gptoss-low.yaml \
           configs/markdown-per_table-bgem3.yaml \
           configs/markdown-per_page-bgem3.yaml; do
    index_dir=$(python -c "import sys;from rag.config import load_config;print(load_config(sys.argv[1]).index_dir)" "$cfg")
    if [ -f "$index_dir/manifest.json" ]; then
        echo "=== index present for $cfg ($index_dir)"
    else
        echo "=== ingest $cfg -> $index_dir"
        python -m rag.cli.ingest --config "$cfg"
    fi
done

python -m evalharness.retrieval_audit --out "eval_results/$RUN" --workers 4 \
    --dataset v1=reference_questions.json \
    --dataset v2=reference_questions_v2.json

hf upload "$ARTIFACTS_REPO" "eval_results/$RUN" "results/$RUN" --repo-type dataset
echo "RESULTS_PREFIX=results/$RUN"

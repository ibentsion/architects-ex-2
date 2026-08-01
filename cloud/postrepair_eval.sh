#!/usr/bin/env bash
# Post-BiDi-repair eval (2026-08-01): does repairing Docling's reversed Hebrew
# table cells (rag/parsing/rtl_repair.py) move the scores?
#
# Two arms, SAME questions and SAME judges, differing only in the index the
# answers came from:
#   before/  2026-07-27 answers, generated against the reversed index (reused)
#   after/   fresh answers from the indexes rebuilt off the repaired cache
#
# Both arms are judged here rather than comparing against the published
# 2026-07-27 numbers, because the committee changed: GLM-5.1 is dropped (too
# slow) and DeepSeek-V4-Pro takes its place — it was DOWN on the Token Factory
# at committee time and is back up. Re-judging the old answers with the new
# committee keeps the delta attributable to retrieval alone.
#
# gpt-oss-120b judges an arm it also generated; as before, the median across
# three judges is what mitigates the self-judging bias.
#
# Run on the node from the repo root:
#   cloud/submit_job.sh run 'bash cloud/postrepair_eval.sh'
set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
JUDGES=(openai/gpt-oss-120b Qwen/Qwen3-235B-A22B-Instruct-2507 deepseek-ai/DeepSeek-V4-Pro)
RUN="postrepair-pertable-bgem3-gptosslow-$TS"

# Arm 1 — answers off the reversed index (2026-07-27), re-judged unchanged.
ANS_BEFORE=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$ARTIFACTS_REPO', 'results/committee-20260727T195127Z/rag_answers_cloud_pertable-bgem3-gptosslow_20260727T195127Z.jsonl', repo_type='dataset'))")

# Arm 2 — fresh answers off the repaired index.
ANS_AFTER="rag_answers_$RUN.jsonl"
python -m rag.cli.query --config configs/final-per_table-bgem3-gptoss-low.yaml \
    --questions reference_questions.json --out "$ANS_AFTER"

fail=0
python -m evalharness.run --questions reference_questions.json --answers "$ANS_BEFORE" \
    --out "eval_results/$RUN/before-reversed-index" --judges "${JUDGES[@]}" || fail=1
python -m evalharness.run --questions reference_questions.json --answers "$ANS_AFTER" \
    --out "eval_results/$RUN/after-repaired-index" --judges "${JUDGES[@]}" || fail=1

# Ship results home even if a judge failed on some questions.
hf upload "$ARTIFACTS_REPO" "eval_results/$RUN" "results/$RUN" --repo-type dataset
hf upload "$ARTIFACTS_REPO" "$ANS_AFTER" "results/$RUN/$ANS_AFTER" --repo-type dataset
echo "RESULTS_PREFIX=results/$RUN"
exit $fail

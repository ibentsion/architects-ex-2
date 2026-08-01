#!/usr/bin/env bash
# SUPERSEDED — kept as the record of the 2026-07-27 run. Do not re-run as-is:
# GLM-5.1 is too slow to keep in a judging loop, and the index this scored has
# since been rebuilt (Hebrew table cells were word-order reversed). The current
# comparison lives in cloud/postrepair_eval.sh.
#
# Judge-committee eval (2026-07-27): per_table + bge-m3 index, two generation
# arms, both scored by a 3-model committee (median scores / majority verdict):
#   openai/gpt-oss-120b            western member (also a contestant — the
#                                  median mitigates self-judging bias)
#   Qwen/Qwen3-235B-A22B-Instruct-2507
#   zai-org/GLM-5.1
# DeepSeek-V4-Pro was DOWN on the Token Factory at setup time (listed in
# /v1/models but requests hang), so the DeepSeek arm reuses its 2026-07-25
# answers file (generated before the outage; predates the 22:12 bge-m3
# re-ingest — retrieval not byte-identical to the fresh gpt-oss arm, and its
# latencies are from the CPU dev machine).
#
# Run on the node from the repo root:
#   cloud/submit_job.sh run 'bash cloud/committee_eval.sh'
set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
JUDGES=(openai/gpt-oss-120b Qwen/Qwen3-235B-A22B-Instruct-2507 zai-org/GLM-5.1)

# Arm 1: DeepSeek-V4-Pro answers, reused (see header).
ANS_DS=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$ARTIFACTS_REPO', 'answers/rag_answers_bgem3_20260725T175049Z.jsonl', repo_type='dataset'))")

# Arm 2: gpt-oss-120b low reasoning, generated fresh on this node.
ANS_GO="rag_answers_cloud_pertable-bgem3-gptosslow_$TS.jsonl"
python -m rag.cli.query --config configs/final-per_table-bgem3-gptoss-low.yaml \
    --questions reference_questions.json --out "$ANS_GO"

fail=0
python -m evalharness.run --questions reference_questions.json --answers "$ANS_DS" \
    --out "eval_results/committee-pertable-bgem3-deepseek-$TS" --judges "${JUDGES[@]}" || fail=1
python -m evalharness.run --questions reference_questions.json --answers "$ANS_GO" \
    --out "eval_results/committee-pertable-bgem3-gptosslow-$TS" --judges "${JUDGES[@]}" || fail=1

# Ship results home even if a judge failed on some questions.
hf upload "$ARTIFACTS_REPO" "eval_results/committee-pertable-bgem3-deepseek-$TS" \
    "results/committee-$TS/deepseek" --repo-type dataset
hf upload "$ARTIFACTS_REPO" "eval_results/committee-pertable-bgem3-gptosslow-$TS" \
    "results/committee-$TS/gptosslow" --repo-type dataset
hf upload "$ARTIFACTS_REPO" "$ANS_GO" "results/committee-$TS/$ANS_GO" --repo-type dataset
echo "RESULTS_PREFIX=results/committee-$TS"
exit $fail

#!/usr/bin/env bash
# Re-score the ship config end-to-end (2026-08-02, quick task 260802-005).
#
# configs/ship.yaml + --engine agent is the production candidate. The last
# scored numbers (agent-eval-20260801T193537Z: v1 6.46, v2 4.80) predate the
# unfiltered retry, the family category filter, and the classifier's decision
# rules + retrieval hint. Same index, same judge committee, same reference
# sets, so the only difference against that run is those three changes.
#
# v1 first and uploaded before v2 starts: v1 is a quarter of the work and the
# result is wanted as soon as it exists.
#
# Run on the node from the repo root:
#   cloud/submit_job.sh run 'bash cloud/ship_eval.sh' --timeout 3h
set -uo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
JUDGES=(openai/gpt-oss-120b Qwen/Qwen3-235B-A22B-Instruct-2507 deepseek-ai/DeepSeek-V4-Pro)
RUN="ship-eval-$TS"
CFG=configs/ship.yaml

mkdir -p "eval_results/$RUN/answers" "eval_results/$RUN/logs"
fail=0

score() { # name questions_file
    local name=$1 questions=$2
    local out="eval_results/$RUN/answers/$name.jsonl"
    echo "=== generate $name ($CFG, agent)"
    if python -m rag.cli.query --config "$CFG" --engine agent \
            --questions "$questions" --out "$out" \
            2> "eval_results/$RUN/logs/$name.stderr.log"; then
        echo "=== OK generate $name ($(wc -l < "$out") answers)"
    else
        fail=1
        echo "=== FAIL generate $name (see logs/$name.stderr.log)"
        return
    fi
    echo "=== judge $name"
    python -m evalharness.run --questions "$questions" --answers "$out" \
        --out "eval_results/$RUN/$name" --judges "${JUDGES[@]}" || fail=1
    hf upload "$ARTIFACTS_REPO" "eval_results/$RUN" "results/$RUN" --repo-type dataset
    echo "=== UPLOADED after $name -> results/$RUN"
}

score v1 reference_questions.json
score v2 reference_questions_v2.json

echo "RESULTS_PREFIX=results/$RUN"
exit $fail

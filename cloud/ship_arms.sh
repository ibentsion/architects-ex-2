#!/usr/bin/env bash
# What the submitted system scores, and what the two open questions cost
# (2026-08-03). v1 (48 questions, ground truth) — the blind set cannot be
# scored, it has no reference answers.
#
# Arms (agent engine throughout, same index unless stated):
#   routed    configs/ship.yaml as submitted: gpt-oss-120b-low synthesizes
#             easy/medium single-topic questions, DeepSeek the rest
#   markdown  configs/ship-markdown.yaml: same, from the Marker markdown parse
#
# The DeepSeek-only arm is NOT re-run: ship-eval-20260802T204751Z/v1 is exactly
# that config (routing off) on the same questions with the same judges —
# correctness 6.27, citation accuracy 0.71.
#
#   cloud/submit_job.sh run 'bash cloud/ship_arms.sh' --timeout 2h
set -uo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
JUDGES=(openai/gpt-oss-120b Qwen/Qwen3-235B-A22B-Instruct-2507 deepseek-ai/DeepSeek-V4-Pro)
RUN="ship-arms-$TS"
V1=reference_questions.json

mkdir -p "eval_results/$RUN/answers" "eval_results/$RUN/logs"
fail=0

score() { # name config doc_source
    local name=$1 config=$2 doc_source=$3
    local out="eval_results/$RUN/answers/$name.jsonl"
    echo "=== generate $name ($config)"
    if python -m rag.cli.query --config "$config" --engine agent \
            --questions "$V1" --out "$out" \
            2> "eval_results/$RUN/logs/$name.stderr.log"; then
        echo "=== OK generate $name ($(wc -l < "$out") answers)"
    else
        fail=1
        echo "=== FAIL generate $name (see logs/$name.stderr.log)"
        return
    fi
    # The citation judge must read the same parse the retriever indexed, or it
    # scores a page the system never saw.
    python -m evalharness.run --questions "$V1" --answers "$out" \
        --out "eval_results/$RUN/$name" --doc-source "$doc_source" \
        --judges "${JUDGES[@]}" || fail=1
    hf upload "$ARTIFACTS_REPO" "eval_results/$RUN" "results/$RUN" --repo-type dataset
    echo "=== UPLOADED after $name -> results/$RUN"
}

score routed   configs/ship.yaml          pdf
score markdown configs/ship-markdown.yaml markdown

echo "RESULTS_PREFIX=results/$RUN"
exit $fail

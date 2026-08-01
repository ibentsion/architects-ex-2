#!/usr/bin/env bash
# Agent-harness eval (2026-08-01, quick task 260801-004): score the new
# --engine agent harness on BOTH reference datasets, with baselines to
# attribute failures.
#
# Arms (per_table + bge-m3 index throughout; committee = postrepair panel):
#   v1-agent-deepseek   agent, DeepSeek-V4-Pro synthesis (production candidate)
#   v1-agent-gptoss     agent, gpt-oss-120b-low synthesis (apples-to-apples
#                       with the postrepair after-repaired-index baseline)
#   v2-agent-deepseek   agent, DeepSeek synthesis
#   v2-agent-gptoss     agent, gpt-oss-low synthesis
#   v2-rag-gptoss       classic rag engine (pipeline baseline for v2)
#   v2-norag-deepseek   bare model, no retrieval (Stage-1 baseline convention)
#
# Per-arm failure isolation: a failed arm sets fail=1 but never kills the job;
# whatever succeeded is judged and uploaded. Answers, judge outputs, and every
# arm's stderr log land under eval_results/$RUN and ship to the artifacts HF
# dataset.
#
# Run on the node from the repo root:
#   cloud/submit_job.sh run 'bash cloud/agent_eval.sh' --timeout 6h
set -uo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
JUDGES=(openai/gpt-oss-120b Qwen/Qwen3-235B-A22B-Instruct-2507 deepseek-ai/DeepSeek-V4-Pro)
RUN="agent-eval-$TS"
CFG_GO=configs/final-per_table-bgem3-gptoss-low.yaml
CFG_DS=configs/embedder-bge-m3.yaml   # generation defaults to DeepSeek-V4-Pro
V1=reference_questions.json
V2=reference_questions_v2.json

mkdir -p "eval_results/$RUN/answers" "eval_results/$RUN/logs"
fail=0

generate() { # name engine config questions
    local name=$1 engine=$2 config=$3 questions=$4
    local out="eval_results/$RUN/answers/$name.jsonl"
    echo "=== generate $name ($engine, $config)"
    if python -m rag.cli.query --config "$config" --engine "$engine" \
            --questions "$questions" --out "$out" \
            2> "eval_results/$RUN/logs/$name.stderr.log"; then
        echo "=== OK $name ($(wc -l < "$out") answers)"
    else
        fail=1
        echo "=== FAIL $name (see logs/$name.stderr.log)"
    fi
}

# qdrant-local is single-process — arms run serially by necessity.
generate v1-agent-deepseek agent "$CFG_DS" "$V1"
generate v1-agent-gptoss   agent "$CFG_GO" "$V1"
generate v2-agent-deepseek agent "$CFG_DS" "$V2"
generate v2-agent-gptoss   agent "$CFG_GO" "$V2"
generate v2-rag-gptoss     rag   "$CFG_GO" "$V2"

echo "=== generate v2-norag-deepseek (bare model, no retrieval)"
if python baseline_runner.py --questions "$V2" --model deepseek-ai/DeepSeek-V4-Pro \
        --out "eval_results/$RUN/answers/v2-norag-deepseek.jsonl" \
        2> "eval_results/$RUN/logs/v2-norag-deepseek.stderr.log"; then
    echo "=== OK v2-norag-deepseek"
else
    fail=1
    echo "=== FAIL v2-norag-deepseek"
fi

judge() { # name questions
    local name=$1 questions=$2
    local answers="eval_results/$RUN/answers/$name.jsonl"
    [ -s "$answers" ] || { echo "=== SKIP judging $name (no answers)"; return; }
    echo "=== judge $name"
    python -m evalharness.run --questions "$questions" --answers "$answers" \
        --out "eval_results/$RUN/$name" --judges "${JUDGES[@]}" \
        2> "eval_results/$RUN/logs/judge-$name.stderr.log" \
        || { fail=1; echo "=== FAIL judging $name"; }
}

judge v1-agent-deepseek "$V1"
judge v1-agent-gptoss   "$V1"
judge v2-agent-deepseek "$V2"
judge v2-agent-gptoss   "$V2"
judge v2-rag-gptoss     "$V2"
judge v2-norag-deepseek "$V2"

# Ship everything home even on partial failure.
hf upload "$ARTIFACTS_REPO" "eval_results/$RUN" "results/$RUN" --repo-type dataset || fail=1
echo "RESULTS_PREFIX=results/$RUN"
exit $fail

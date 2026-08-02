#!/usr/bin/env bash
# The graded run: serve the contract endpoint and answer blind_questions.json.
#
# Latency is measured client-side by submit_runner.py and is part of the grade,
# so this runs on the GPU node (the cross-encoder is 9 ms/pair there against
# ~1.7 s/pair on a laptop CPU) with nothing else on the box.
#
#   cloud/submit_job.sh run 'bash cloud/blind_run.sh' --timeout 3h
#
# The engine loads lazily on the first request, so one throwaway question is
# asked before the runner starts: a served system is warm, and question 1
# should not carry 30 s of model loading. submit_runner resumes from an
# existing --out file, so a re-run continues where a killed one stopped.
set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
TEAM="${TEAM:-ibentsion}"
QUESTIONS="${QUESTIONS:-blind_questions.json}"
OUT="submission_${TEAM}_${TS}.jsonl"
export RAG_CONFIG="${RAG_CONFIG:-configs/ship.yaml}"

echo "=== serving contract:app with RAG_CONFIG=$RAG_CONFIG"
python -m uvicorn contract:app --port 8000 > "uvicorn_${TS}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
    curl -sf http://localhost:8000/health >/dev/null && break
    sleep 5
done
curl -sf http://localhost:8000/health || { echo "endpoint never came up"; tail -40 "uvicorn_${TS}.log"; exit 1; }

echo "=== warming the engine (model loads, index open)"
curl -sf -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
    -d '{"question": "מה תקופת ההתיישנות להגשת תביעה?"}' -o /dev/null \
    --max-time 300 || echo "warm-up call failed — continuing, the runner records failures"

echo "=== answering $(python -c "import json,sys;d=json.load(open('$QUESTIONS'));print(len(d['questions'] if isinstance(d,dict) else d))") questions"
python submit_runner.py --questions "$QUESTIONS" --endpoint http://localhost:8000 \
    --out "$OUT" --timeout 120

python - "$OUT" <<'PY'
import json, statistics, sys
records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
latencies = sorted(r["latency_ms"] for r in records)
errors = [r for r in records if r.get("endpoint_error")]
empty = [r for r in records if not r["answer"].strip()]
cited = [r for r in records if r["citations"]]
print(f"\n{len(records)} answers | p50 {statistics.median(latencies):.0f} ms | "
      f"p95 {latencies[int(len(latencies) * 0.95)]:.0f} ms | max {latencies[-1]:.0f} ms")
print(f"endpoint failures: {len(errors)} | empty answers: {len(empty)} | "
      f"with citations: {len(cited)} ({len(cited) / len(records):.0%})")
print(f"est. cost: ${sum(r.get('cost_usd') or 0 for r in records):.2f}")
PY

hf upload "$ARTIFACTS_REPO" "$OUT" "submissions/$OUT" --repo-type dataset
echo "SUBMISSION=$OUT"
echo "RESULTS_PREFIX=submissions/$OUT"

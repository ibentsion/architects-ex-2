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
OUT="submission_${TEAM}.jsonl"   # stable name: submit_runner resumes it
                                 # across jobs, so a killed run continues
export RAG_CONFIG="${RAG_CONFIG:-configs/ship.yaml}"

# Import first: a broken import inside uvicorn only shows up as a health
# check that never passes, which costs 5 minutes to learn nothing.
# The graded questions are deliberately not in git — they live on the private
# artifacts dataset, the same place the corpus and indexes come from.
if [ ! -f "$QUESTIONS" ]; then
    echo "=== fetching $QUESTIONS from $ARTIFACTS_REPO"
    python - "$ARTIFACTS_REPO" "$QUESTIONS" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download

repo, name = sys.argv[1], sys.argv[2]
shutil.copyfile(hf_hub_download(repo, name, repo_type="dataset"), name)
print(f"fetched {name}")
PY
fi

echo "=== import check"
python -c "import contract; print('contract imports OK -> ' + contract._rag_config_path())"

echo "=== serving contract:app with RAG_CONFIG=$RAG_CONFIG"
python -m uvicorn contract:app --port 8000 > "uvicorn_${TS}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep 5
kill -0 "$SERVER_PID" 2>/dev/null || { echo "=== server died on startup:"; cat "uvicorn_${TS}.log"; exit 1; }

# The node image ships no curl (see cloud/setup_node.sh) — everything that
# talks to the endpoint goes through python's stdlib. `|| health=1` keeps
# set -e from killing the script before the log can be printed.
health=0
python - <<'PY' || health=1
import sys, time, urllib.request

for attempt in range(60):
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as response:
            if response.status == 200:
                print(f"=== endpoint up after ~{attempt * 5}s")
                sys.exit(0)
    except Exception as exc:
        if attempt % 6 == 0:
            print(f"  waiting for endpoint ({type(exc).__name__})", flush=True)
    time.sleep(5)
sys.exit(1)
PY
if [ "$health" -ne 0 ]; then
    echo "=== endpoint never came up — uvicorn log:"
    tail -60 "uvicorn_${TS}.log"
    exit 1
fi

echo "=== warming the engine (model loads, index open)"
python - <<'PY'
import json, urllib.request

request = urllib.request.Request(
    "http://localhost:8000/ask",
    data=json.dumps({"question": "מה תקופת ההתיישנות להגשת תביעה?"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
try:
    urllib.request.urlopen(request, timeout=300).read()
    print("=== warm")
except Exception as exc:  # the runner records per-question failures anyway
    print(f"warm-up call failed ({exc}) — continuing")
PY

echo "=== answering $(python -c "import json,sys;d=json.load(open('$QUESTIONS'));print(len(d['questions'] if isinstance(d,dict) else d))") questions"
python submit_runner.py --questions "$QUESTIONS" --endpoint http://localhost:8000 \
    --out "$OUT" --timeout 240   # DeepSeek is slow tonight; an answer that takes
                             # 200 s still scores, an abandoned one scores zero

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

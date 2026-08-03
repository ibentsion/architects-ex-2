#!/usr/bin/env bash
# Serve the streaming agent app for the web-UI demo (Stage 3 optional bonus).
#
#   cloud/submit_job.sh run 'bash cloud/serve_ui.sh' --timeout 3h
#
# Then, on the laptop:
#   ssh -N -L 8000:localhost:8000 <node>            # tunnel
#   ./venv/bin/python -m uvicorn webapi.bridge_app:app --port 8080
#   npm --prefix webui run dev
#
# This is NOT the graded endpoint — cloud/blind_run.sh serves contract:app for
# that. Same engine and same config; this one streams the pipeline's trace.
set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
PORT="${PORT:-8000}"
export RAG_CONFIG="${RAG_CONFIG:-configs/ship.yaml}"

# Import first: a broken import inside uvicorn only shows up as a health check
# that never passes, which costs 5 minutes to learn nothing.
echo "=== import check"
python -c "import webapi.agent_app as a; print('webapi.agent_app imports OK -> ' + a._rag_config_path())"

echo "=== serving webapi.agent_app:app on :$PORT with RAG_CONFIG=$RAG_CONFIG"
python -m uvicorn webapi.agent_app:app --host 0.0.0.0 --port "$PORT" \
    > "uvicorn_ui_${TS}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep 5
kill -0 "$SERVER_PID" 2>/dev/null || { echo "=== server died on startup:"; cat "uvicorn_ui_${TS}.log"; exit 1; }

# The node image ships no curl (see cloud/setup_node.sh) — everything that
# talks to the endpoint goes through python's stdlib. `|| health=1` keeps
# set -e from killing the script before the log can be printed.
health=0
python - "$PORT" <<'PY' || health=1
import sys, time, urllib.request

port = sys.argv[1]
for attempt in range(60):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as response:
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
    tail -60 "uvicorn_ui_${TS}.log"
    exit 1
fi

# The engine loads lazily on the first request. Warm it here so the first
# question someone types in the browser does not pay 30 s of model loading.
echo "=== warming the engine (model loads, index open)"
python - "$PORT" <<'PY'
import json, sys, urllib.request

request = urllib.request.Request(
    f"http://localhost:{sys.argv[1]}/query",
    data=json.dumps({"question": "מה תקופת ההתיישנות להגשת תביעה?"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(request, timeout=600) as response:
        events = [line for line in response.read().decode("utf-8").splitlines()
                  if line.startswith("event: ")]
    print(f"=== warm ({len(events)} frames: {' '.join(e[7:] for e in events)})")
except Exception as exc:
    print(f"warm-up call failed ({exc}) — continuing")
PY

NODE_HOST=$(hostname)
cat <<EOF

=== ready. On the laptop:
    ssh -N -L ${PORT}:localhost:${PORT} ${NODE_HOST}
    ./venv/bin/python -m uvicorn webapi.bridge_app:app --port 8080
    npm --prefix webui run dev

    logs: uvicorn_ui_${TS}.log
EOF

wait "$SERVER_PID"

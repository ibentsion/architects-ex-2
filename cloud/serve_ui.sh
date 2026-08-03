#!/usr/bin/env bash
# Serve the streaming agent app for the web-UI demo (Stage 3 optional bonus).
# Started on the node by cloud/serve_endpoint.sh — never run it by hand.
#
# Two topologies:
#
#   WITH_BRIDGE unset  — agent only, on $PORT. The bridge and frontend run on
#                        the laptop against it (webui/README.md).
#   WITH_BRIDGE=1      — agent on localhost:$PORT plus the bridge on
#                        $BRIDGE_PORT serving the built frontend, so the whole
#                        app is one public URL a browser can just open. The
#                        bridge is gated by UI_PASSWORD; refuses to start without
#                        one, because in this mode it is internet-facing and it
#                        reads repo files by design.
#
# This is NOT the graded endpoint — cloud/blind_run.sh serves contract:app for
# that. Same engine and same config; this one streams the pipeline's trace.
set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
PORT="${PORT:-8000}"
BRIDGE_PORT="${BRIDGE_PORT:-8080}"
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
export RAG_CONFIG="${RAG_CONFIG:-configs/ship.yaml}"

if [ -n "${WITH_BRIDGE:-}" ] && [ -z "${UI_PASSWORD:-}" ]; then
    echo "WITH_BRIDGE=1 needs UI_PASSWORD — refusing to serve the repo publicly with no gate" >&2
    exit 2
fi

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

if [ -z "${WITH_BRIDGE:-}" ]; then
    cat <<EOF

=== ready. The agent is on :${PORT}; point a laptop bridge at it with
    AGENT_BASE_URL (see webui/README.md and cloud/serve_endpoint.sh).

    logs: uvicorn_ui_${TS}.log
EOF
    wait "$SERVER_PID"
    exit 0
fi

# --- everything-on-the-node mode ------------------------------------------- #
# The frontend is a static bundle built on a laptop and shipped through the same
# HF artifacts dataset as corpus/cache/rag_index (cloud/upload_artifacts.sh
# webui-dist). Building it here would mean a Node toolchain on the GPU image for
# no reason.
if [ ! -f webui/dist/index.html ]; then
    echo "=== fetching webui-dist.tar.gz from $ARTIFACTS_REPO"
    tarball=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$ARTIFACTS_REPO', 'webui-dist.tar.gz', repo_type='dataset'))")
    tar xzf "$tarball"
fi
[ -f webui/dist/index.html ] || { echo "=== no webui/dist/index.html after fetch"; exit 1; }

export AGENT_BASE_URL="http://localhost:${PORT}"
export WEBUI_DIST="webui/dist"
# AGENT_TOKEN deliberately unset: the agent is on loopback inside this container
# and is not published. UI_PASSWORD on the bridge is the gate.
unset AGENT_TOKEN || true

echo "=== serving the bridge + UI on :$BRIDGE_PORT (gated by UI_PASSWORD)"
python -m uvicorn webapi.bridge_app:app --host 0.0.0.0 --port "$BRIDGE_PORT" \
    > "uvicorn_bridge_${TS}.log" 2>&1 &
BRIDGE_PID=$!
trap 'kill "$SERVER_PID" "$BRIDGE_PID" 2>/dev/null || true' EXIT
sleep 5
kill -0 "$BRIDGE_PID" 2>/dev/null || { echo "=== bridge died on startup:"; cat "uvicorn_bridge_${TS}.log"; exit 1; }

python - "$BRIDGE_PORT" <<'PY' || { echo "=== bridge health never passed"; exit 1; }
import sys, time, urllib.request
for attempt in range(30):
    try:
        with urllib.request.urlopen(f"http://localhost:{sys.argv[1]}/healthz", timeout=5) as r:
            if r.status == 200:
                print(f"=== bridge up after ~{attempt * 2}s")
                sys.exit(0)
    except Exception:
        pass
    time.sleep(2)
sys.exit(1)
PY

cat <<EOF

=== ready. The whole app is on :${BRIDGE_PORT} — open the endpoint's public URL
    in a browser and log in with UI_PASSWORD.

    logs: uvicorn_ui_${TS}.log (agent), uvicorn_bridge_${TS}.log (bridge)
EOF

wait "$SERVER_PID" "$BRIDGE_PID"

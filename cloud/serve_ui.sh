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

# PUBLIC_TUNNEL=1 fronts the bridge with a Cloudflare quick tunnel, purely to
# get *https*: browsers refuse microphone access outside a secure context, and
# the endpoint's own public route is plain http on a raw IP (the platform's
# managed https tunnel only forwards for --auth token endpoints, which a
# browser cannot satisfy on navigation). Pinned + checksummed rather than
# "curl | latest": this binary sits in front of an app that spends the shared
# Token Factory key.
CLOUDFLARED_VERSION="2026.7.3"
CLOUDFLARED_SHA256="9d71c677db00134c1bd4144b7783486b654ad281b1ea62b4972098d19f770f17"
# Normally exported by cloud/env.sh, which the endpoint sources first; defaulted
# here so the script is runnable on its own.
EX2_ROOT="${EX2_ROOT:-/mnt/data/ex2/ibentsion}"
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

# The bridge logs to STDOUT, not a file: it is the process users actually hit,
# and a file inside the container is unreachable without SSH (endpoints get no
# authorized keys). A 500 here has to show up in `nebius ai endpoint logs`.
echo "=== serving the bridge + UI on :$BRIDGE_PORT (gated by UI_PASSWORD)"
# --proxy-headers so X-Forwarded-Proto from the tunnel is honoured; uvicorn
# trusts it only from 127.0.0.1 by default, which is exactly where cloudflared
# connects from. Without it the session cookie never gets its Secure flag on
# an https visit, and direct hits on the raw public IP must NOT be able to
# spoof the scheme.
python -m uvicorn webapi.bridge_app:app --host 0.0.0.0 --port "$BRIDGE_PORT" --proxy-headers 2>&1 &
BRIDGE_PID=$!
trap 'kill "$SERVER_PID" "$BRIDGE_PID" ${TUNNEL_PID:-} 2>/dev/null || true' EXIT
sleep 5
kill -0 "$BRIDGE_PID" 2>/dev/null || { echo "=== bridge died on startup (see above)"; exit 1; }

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

if [ -n "${PUBLIC_TUNNEL:-}" ]; then
    if [ ! -x "$EX2_ROOT/cloudflared" ]; then
        echo "=== fetching cloudflared $CLOUDFLARED_VERSION"
        python - "$CLOUDFLARED_VERSION" "$EX2_ROOT/cloudflared.part" <<'PY'
import sys, urllib.request
version, dest = sys.argv[1], sys.argv[2]
url = (f"https://github.com/cloudflare/cloudflared/releases/download/{version}"
       "/cloudflared-linux-amd64")
with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
    f.write(r.read())
PY
        ACTUAL=$(sha256sum "$EX2_ROOT/cloudflared.part" | cut -d' ' -f1)
        if [ "$ACTUAL" != "$CLOUDFLARED_SHA256" ]; then
            rm -f "$EX2_ROOT/cloudflared.part"
            echo "=== cloudflared checksum mismatch (got $ACTUAL) — refusing to run it" >&2
            exit 1
        fi
        chmod +x "$EX2_ROOT/cloudflared.part"
        mv "$EX2_ROOT/cloudflared.part" "$EX2_ROOT/cloudflared"
    fi

    echo "=== starting cloudflare quick tunnel (for https, so the mic works)"
    "$EX2_ROOT/cloudflared" tunnel --no-autoupdate \
        --url "http://localhost:${BRIDGE_PORT}" > "tunnel_${TS}.log" 2>&1 &
    TUNNEL_PID=$!

    # The URL is assigned at runtime and only ever appears in cloudflared's own
    # output, so it has to be scraped back out and echoed to the container log.
    TUNNEL_URL=""
    for _ in $(seq 1 30); do
        sleep 2
        TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "tunnel_${TS}.log" | head -1 || true)
        [ -n "$TUNNEL_URL" ] && break
    done
    if [ -n "$TUNNEL_URL" ]; then
        echo "=== PUBLIC HTTPS URL: $TUNNEL_URL"
    else
        echo "=== tunnel did not report a URL; last lines:"; tail -20 "tunnel_${TS}.log"
    fi
fi

cat <<EOF

=== ready. The whole app is on :${BRIDGE_PORT} — open the public URL above
    (or the endpoint's raw IP) and log in with UI_PASSWORD.

    logs: uvicorn_ui_${TS}.log (agent, in-container); the bridge logs here.
EOF

wait "$SERVER_PID" "$BRIDGE_PID"

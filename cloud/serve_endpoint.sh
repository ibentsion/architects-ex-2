#!/usr/bin/env bash
# Serve the web UI's agent on a GPU node, reachable from a laptop browser.
#
#   cloud/serve_endpoint.sh create        # agent only; bridge + UI run locally
#   cloud/serve_endpoint.sh create --full # EVERYTHING on the node: one public
#                                         # URL anyone can open in a browser
#   cloud/serve_endpoint.sh status        # URL + state of the running endpoint
#   cloud/serve_endpoint.sh logs          # container logs
#   cloud/serve_endpoint.sh stop          # STOP IT -- it holds a GPU until you do
#
# --full needs the built UI on the artifacts dataset first:
#   npm --prefix webui run build && cloud/upload_artifacts.sh webui-dist
#
# Why an endpoint and not a job (cloud/submit_job.sh):
# a `nebius ai job` gets a private VPC IP only. No public IP, no SSH authorized
# keys, and `nebius ai job ssh` has no -L, so nothing served inside a job can be
# reached from outside the VPC -- `--container-port` on a job yields a private
# endpoint and nothing else. A `nebius ai endpoint` publishes each HTTP port on
# a managed https:// URL with no public IP required. That is the only route in.
#
# Auth: created with `--auth token` and a freshly generated token, because the
# URL is public and every question spends the SHARED course Token Factory key.
# The token is written to .agent_token (gitignored) and read back by `status`.
set -euo pipefail

IMAGE="cr.eu-north1.nebius.cloud/e00v1er5fasm8gmdwy/apex-ex-1"
PLATFORM="${PLATFORM:-gpu-l40s-d}"
PRESET="${PRESET:-1gpu-16vcpu-96gb}"
VOLUME="computefilesystem-e00hnnpfn5rr5aavma:/mnt/data"
REPO_URL="https://github.com/ibentsion/architects-ex-2.git"
EX2_ROOT="/mnt/data/ex2/ibentsion"
TF_BASE_URL="https://api.tokenfactory.nebius.com/v1"
NAME="${NAME:-ex2-ui-agent}"
BRANCH="${BRANCH:-main}"
TOKEN_FILE=".agent_token"

MODE="${1:-}"
[ -n "$MODE" ] || { echo "usage: cloud/serve_endpoint.sh create [--full]|status|logs|stop"; exit 2; }
shift || true

FULL=""
[ "${1:-}" = "--full" ] && { FULL=1; shift; }
PASSWORD_FILE=".ui_password"

_endpoint_id() {
    nebius ai endpoint get-by-name --name "$NAME" --format json 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["metadata"]["id"])' 2>/dev/null
}

_public_url() {
    nebius ai endpoint get "$1" --format json 2>/dev/null | python3 -c '
import json, sys
status = json.load(sys.stdin).get("status", {})
for url in status.get("public_endpoints", []):
    if str(url).startswith("https://"):
        print(url)
        break
'
}

case "$MODE" in
    create)
        # Secrets, same sources as cloud/submit_job.sh.
        if [ -z "${NEBIUS_API_KEY:-}" ] && [ -f .env ]; then
            NEBIUS_API_KEY=$(grep -oP '^NEBIUS_API_KEY=\K.*' .env || true)
        fi
        [ -n "${NEBIUS_API_KEY:-}" ] || { echo "NEBIUS_API_KEY not set (env or .env)"; exit 2; }
        if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
            HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
        fi
        [ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set (env or ~/.cache/huggingface/token)"; exit 2; }

        umask 077
        SERVE_CMD="bash cloud/serve_ui.sh"
        AUTH_FLAGS=()
        EXTRA_ENV=()

        if [ -n "$FULL" ]; then
            # Everything on the node behind one URL. The platform's --auth token
            # cannot be used here: it wants an Authorization header, and a
            # browser navigating to a URL sends none. So the endpoint is open at
            # the platform layer and the BRIDGE enforces access with a shared
            # password -- which is why serve_ui.sh refuses to start without one.
            UI_PASSWORD="${UI_PASSWORD:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')}"
            printf '%s\n' "$UI_PASSWORD" > "$PASSWORD_FILE"
            SERVE_CMD="WITH_BRIDGE=1 bash cloud/serve_ui.sh"
            PORT_SPEC="8080/http"
            AUTH_FLAGS=(--auth none)
            EXTRA_ENV=(--env UI_PASSWORD="$UI_PASSWORD")
        else
            AGENT_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
            printf '%s\n' "$AGENT_TOKEN" > "$TOKEN_FILE"
            PORT_SPEC="8000/http"
            AUTH_FLAGS=(--auth token --token "$AGENT_TOKEN")
        fi

        # Same bootstrap as submit_job.sh: clone-or-update on the volume, run the
        # idempotent setup, then serve. Everything persists across endpoints.
        REMOTE_CMD="mkdir -p $EX2_ROOT && \
if [ -d $EX2_ROOT/repo/.git ]; then git -C $EX2_ROOT/repo fetch origin $BRANCH && git -C $EX2_ROOT/repo reset --hard origin/$BRANCH; \
else git clone -b $BRANCH $REPO_URL $EX2_ROOT/repo; fi && \
cd $EX2_ROOT/repo && bash cloud/setup_node.sh && source cloud/env.sh && $SERVE_CMD"

        echo "creating endpoint $NAME (branch $BRANCH${FULL:+, full app})"
        nebius ai endpoint create \
            --name "$NAME" \
            --image "$IMAGE" \
            --container-command bash \
            --args "-c \"$REMOTE_CMD\"" \
            --platform "$PLATFORM" \
            --preset "$PRESET" \
            --container-port "$PORT_SPEC" \
            --public \
            "${AUTH_FLAGS[@]}" \
            --volume "$VOLUME" \
            --env NEBIUS_API_KEY="$NEBIUS_API_KEY" \
            --env HF_TOKEN="$HF_TOKEN" \
            --env OPENAI_BASE_URL="$TF_BASE_URL" \
            --env OPENAI_API_KEY="$NEBIUS_API_KEY" \
            "${EXTRA_ENV[@]}" \
            > /dev/null

        ID=$(_endpoint_id)
        echo "endpoint: $ID"
        echo
        if [ -n "$FULL" ]; then
            echo "password written to $PASSWORD_FILE (gitignored):  $UI_PASSWORD"
            echo "share the URL from 'cloud/serve_endpoint.sh status' plus that password."
            echo "startup clones, sets up, fetches the UI bundle and warms the engine (~4 min)."
        else
            echo "token written to $TOKEN_FILE (gitignored)"
            echo "the engine warms on startup; wait for /health to answer 200, then:"
            echo "    export AGENT_BASE_URL=\$(cloud/serve_endpoint.sh status | grep -o 'https://[^ ]*')"
            echo "    export AGENT_TOKEN=\$(cat $TOKEN_FILE)"
            echo "    ./venv/bin/python -m uvicorn webapi.bridge_app:app --port 8080"
            echo "    npm --prefix webui run dev"
        fi
        echo
        echo "IT HOLDS A GPU UNTIL STOPPED:  cloud/serve_endpoint.sh stop"
        ;;

    status)
        ID=$(_endpoint_id)
        [ -n "$ID" ] || { echo "no endpoint named $NAME"; exit 1; }
        STATE=$(nebius ai endpoint get "$ID" --format json 2>/dev/null \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"]["state"])')
        echo "id:    $ID"
        echo "state: $STATE"
        echo "url:   $(_public_url "$ID")"
        # `|| true`: a bare test as the last command would make `status` exit 1
        # whenever the token file is absent, which is not a failure.
        [ -f "$TOKEN_FILE" ] && echo "token: $TOKEN_FILE" || true
        ;;

    logs)
        ID=$(_endpoint_id)
        [ -n "$ID" ] || { echo "no endpoint named $NAME"; exit 1; }
        shift
        nebius ai endpoint logs "$ID" "$@"
        ;;

    stop)
        ID=$(_endpoint_id)
        [ -n "$ID" ] || { echo "no endpoint named $NAME — nothing to stop"; exit 0; }
        nebius ai endpoint stop "$ID"
        echo "stopped $ID"
        ;;

    *) echo "unknown mode: $MODE"; exit 2 ;;
esac

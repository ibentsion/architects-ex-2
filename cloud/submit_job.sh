#!/usr/bin/env bash
# Submit an ex-2 job to the Nebius GPU node (run from this machine, not the node).
#
#   cloud/submit_job.sh probe                 # venv build + GPU/torch/reranker check
#                                             # (no HF token or API key needed)
#   cloud/submit_job.sh setup                 # build venv + fetch artifacts/models
#   cloud/submit_job.sh smoke                 # setup (fast no-op) + cloud/smoke_test.py
#   cloud/submit_job.sh run '<shell command>' # setup + arbitrary command in repo root
#   options: --timeout 45m   --name my-job    --branch main
#
# Every job: clone-or-update the repo at /mnt/data/ex2/repo, run
# cloud/setup_node.sh (idempotent), then the mode's command — so a job always
# lands on a ready node no matter what ran before it. The run-mode command is
# passed inside double quotes end-to-end: use single quotes and avoid double
# quotes inside it.
#
# Outputs (answers JSONL, eval_results/, smoke_answers.jsonl) land in the repo
# checkout on the volume and persist across jobs.
set -euo pipefail

IMAGE="cr.eu-north1.nebius.cloud/e00v1er5fasm8gmdwy/apex-ex-1"
PLATFORM="gpu-l40s-d"
PRESET="1gpu-16vcpu-96gb"
VOLUME="computefilesystem-e00hnnpfn5rr5aavma:/mnt/data"
REPO_URL="https://github.com/ibentsion/architects-ex-2.git"
EX2_ROOT="/mnt/data/ex2"
TF_BASE_URL="https://api.tokenfactory.nebius.com/v1"

MODE="${1:-}"
shift || true
[ -n "$MODE" ] || { echo "usage: cloud/submit_job.sh setup|smoke|run [cmd] [--timeout T] [--name N] [--branch B]"; exit 2; }

USER_CMD=""
if [ "$MODE" = "run" ]; then
    USER_CMD="${1:-}"
    [ -n "$USER_CMD" ] || { echo "run mode needs a command argument"; exit 2; }
    shift
fi

TIMEOUT=""
NAME=""
BRANCH="main"
while [ $# -gt 0 ]; do
    case "$1" in
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --name)    NAME="$2"; shift 2 ;;
        --branch)  BRANCH="$2"; shift 2 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done

ENV_FLAGS=()
if [ "$MODE" != "probe" ]; then
    # Secrets: NEBIUS_API_KEY from env or repo .env; HF_TOKEN from env or the hf CLI cache.
    if [ -z "${NEBIUS_API_KEY:-}" ] && [ -f .env ]; then
        NEBIUS_API_KEY=$(grep -oP '^NEBIUS_API_KEY=\K.*' .env || true)
    fi
    [ -n "${NEBIUS_API_KEY:-}" ] || { echo "NEBIUS_API_KEY not set (env or .env)"; exit 2; }
    if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
        HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    fi
    [ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set (env or ~/.cache/huggingface/token)"; exit 2; }
    ENV_FLAGS=(
        --env NEBIUS_API_KEY="$NEBIUS_API_KEY"
        --env HF_TOKEN="$HF_TOKEN"
        --env OPENAI_BASE_URL="$TF_BASE_URL"
        --env OPENAI_API_KEY="$NEBIUS_API_KEY"
    )
fi

SETUP_ARGS=""
[ "$MODE" = "probe" ] && SETUP_ARGS=" --venv-only"
BOOTSTRAP="mkdir -p $EX2_ROOT && \
if [ -d $EX2_ROOT/repo/.git ]; then git -C $EX2_ROOT/repo fetch origin $BRANCH && git -C $EX2_ROOT/repo reset --hard origin/$BRANCH; \
else git clone -b $BRANCH $REPO_URL $EX2_ROOT/repo; fi && \
cd $EX2_ROOT/repo && bash cloud/setup_node.sh$SETUP_ARGS"

case "$MODE" in
    probe)
        REMOTE_CMD="$BOOTSTRAP && source cloud/env.sh && nvidia-smi && python cloud/smoke_test.py --gpu-only"
        TIMEOUT="${TIMEOUT:-30m}"
        ;;
    setup)
        REMOTE_CMD="$BOOTSTRAP"
        TIMEOUT="${TIMEOUT:-45m}"
        ;;
    smoke)
        REMOTE_CMD="$BOOTSTRAP && source cloud/env.sh && nvidia-smi && python cloud/smoke_test.py"
        TIMEOUT="${TIMEOUT:-30m}"
        ;;
    run)
        REMOTE_CMD="$BOOTSTRAP && source cloud/env.sh && $USER_CMD"
        TIMEOUT="${TIMEOUT:-2h}"
        ;;
    *) echo "unknown mode: $MODE"; exit 2 ;;
esac

NAME="${NAME:-ex2-$MODE-$(date +%Y%m%d-%H%M%S)}"

echo "submitting job $NAME (timeout $TIMEOUT)"
nebius ai job create \
    --name "$NAME" \
    --image "$IMAGE" \
    --container-command bash \
    --args "-c \"$REMOTE_CMD\"" \
    --platform "$PLATFORM" \
    --preset "$PRESET" \
    --timeout "$TIMEOUT" \
    "${ENV_FLAGS[@]}" \
    --volume "$VOLUME"

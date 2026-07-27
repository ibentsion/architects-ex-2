#!/usr/bin/env bash
# Idempotent GPU-node setup. Run from the repo root (jobs run it after the
# bootstrap clone/pull in cloud/submit_job.sh). Fast no-op when everything is
# already in place, so every job can run it unconditionally.
#
#   1. venv at $EX2_ROOT/venv  (reinstalled only when requirements.lock changes)
#   2. artifacts (corpus/ cache/ rag_index/) from the private HF dataset
#      $ARTIFACTS_REPO, extracted into the repo root  (needs HF_TOKEN)
#   3. local models pre-downloaded into $HF_HOME / $STANZA_RESOURCES_DIR
#
# --venv-only stops after step 1 (used by the no-HF-token probe job).
set -euo pipefail

EX2_ROOT="${EX2_ROOT:-/mnt/data/ex2/ibentsion}"
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
mkdir -p "$EX2_ROOT"

# --- 1. venv ---------------------------------------------------------------
# requirements.lock was resolved on Python 3.12 (numpy 2.5.1 needs >=3.12) but
# the job image ships 3.11, so the interpreter comes from uv's managed CPython,
# stored on the volume alongside uv itself and its cache.
export UV_PYTHON_INSTALL_DIR="$EX2_ROOT/uv-python"
export UV_CACHE_DIR="$EX2_ROOT/uv-cache"
export PATH="$EX2_ROOT/bin:$PATH"
if ! command -v uv >/dev/null; then
    echo "[setup] installing uv"
    # the pytorch image has no curl/wget — pip is the reliable path
    python3 -m pip install --quiet --root-user-action=ignore uv
fi
if ! "$EX2_ROOT/venv/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
    echo "[setup] (re)creating venv (python 3.12) at $EX2_ROOT/venv"
    rm -rf "$EX2_ROOT/venv"
    uv venv --python 3.12 "$EX2_ROOT/venv"
fi
source cloud/env.sh

STAMP="$EX2_ROOT/venv/.requirements.lock.sha256"
if [ ! -f "$STAMP" ] || ! sha256sum -c "$STAMP" --status; then
    echo "[setup] installing requirements.lock (this takes a few minutes on first run)"
    uv pip install --quiet --python "$EX2_ROOT/venv/bin/python" -r requirements.lock
    sha256sum requirements.lock > "$STAMP"
else
    echo "[setup] venv up to date (requirements.lock unchanged)"
fi

if [ "${1:-}" = "--venv-only" ]; then
    echo "[setup] --venv-only: skipping artifacts and model downloads"
    exit 0
fi

# --- 2. artifacts ----------------------------------------------------------
# cache/ holds the expensive-to-recreate stage caches (Docling parses, token
# lists, paid Token Factory embeddings); rag_index/ the built indexes; corpus/
# the frozen document snapshot. All gitignored, all extracted into the repo
# root so relative paths in configs/*.yaml work unchanged.
for name in corpus cache rag_index; do
    if [ -d "$name" ]; then
        echo "[setup] $name/ present — skipping download"
        continue
    fi
    echo "[setup] fetching $name.tar.gz from $ARTIFACTS_REPO"
    # hf_hub_download instead of the hf CLI: the CLI decorates its stdout
    # (colors + multi-line "Downloaded" banner), not a clean path
    tarball=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$ARTIFACTS_REPO', '$name.tar.gz', repo_type='dataset'))")
    tar xzf "$tarball"
done

# --- 3. models -------------------------------------------------------------
for model in BAAI/bge-m3 BAAI/bge-reranker-v2-m3; do
    echo "[setup] ensuring $model in $HF_HOME"
    hf download "$model" > /dev/null
done
if [ ! -d "$STANZA_RESOURCES_DIR/he" ]; then
    echo "[setup] downloading stanza he models"
    python -c "import stanza; stanza.download('he', verbose=False)"
fi

echo "[setup] done — venv, artifacts and models ready under $EX2_ROOT"

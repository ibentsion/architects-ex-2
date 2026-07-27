#!/usr/bin/env bash
# Idempotent GPU-node setup. Run from the repo root (jobs run it after the
# bootstrap clone/pull in cloud/submit_job.sh). Fast no-op when everything is
# already in place, so every job can run it unconditionally.
#
#   1. venv at $EX2_ROOT/venv  (reinstalled only when requirements.lock changes)
#   2. artifacts (corpus/ cache/ rag_index/) from the private HF dataset
#      $ARTIFACTS_REPO, extracted into the repo root  (needs HF_TOKEN)
#   3. local models pre-downloaded into $HF_HOME / $STANZA_RESOURCES_DIR
set -euo pipefail

EX2_ROOT="${EX2_ROOT:-/mnt/data/ex2}"
ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
mkdir -p "$EX2_ROOT"

# --- 1. venv ---------------------------------------------------------------
if [ ! -x "$EX2_ROOT/venv/bin/python" ]; then
    echo "[setup] creating venv at $EX2_ROOT/venv"
    python3 -m venv "$EX2_ROOT/venv"
fi
source cloud/env.sh

STAMP="$EX2_ROOT/venv/.requirements.lock.sha256"
if [ ! -f "$STAMP" ] || ! sha256sum -c "$STAMP" --status; then
    echo "[setup] installing requirements.lock (this takes a few minutes on first run)"
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.lock
    sha256sum requirements.lock > "$STAMP"
else
    echo "[setup] venv up to date (requirements.lock unchanged)"
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
    tarball=$(hf download "$ARTIFACTS_REPO" "$name.tar.gz" --repo-type dataset)
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

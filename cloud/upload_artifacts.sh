#!/usr/bin/env bash
# Package corpus/ cache/ rag_index/ and upload them to the private HF dataset
# used by cloud/setup_node.sh. Run locally from the repo root whenever the
# caches/indexes changed and the cloud copy should be refreshed:
#
#   cloud/upload_artifacts.sh [corpus] [cache] [rag_index]   # default: all three
#
# Uses the venv's hf CLI and your cached HF token. Re-uploading a subset is
# fine — each dir is its own tarball. On the node, delete the dir before the
# next job to force a re-download (setup skips dirs that already exist).
set -euo pipefail

ARTIFACTS_REPO="${ARTIFACTS_REPO:-ibentsion/apex-ex2-artifacts}"
HF=venv/bin/hf
if [ $# -gt 0 ]; then DIRS=("$@"); else DIRS=(corpus cache rag_index); fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

"$HF" repo create "$ARTIFACTS_REPO" --repo-type dataset --private --exist-ok >/dev/null

for name in "${DIRS[@]}"; do
    [ -d "$name" ] || { echo "skip $name (not a directory here)"; continue; }
    echo "packing $name/ ..."
    tar cf - "$name" | gzip -1 > "$WORK/$name.tar.gz"
    du -h "$WORK/$name.tar.gz"
    echo "uploading $name.tar.gz to $ARTIFACTS_REPO ..."
    "$HF" upload "$ARTIFACTS_REPO" "$WORK/$name.tar.gz" "$name.tar.gz" --repo-type dataset
done
echo "done"

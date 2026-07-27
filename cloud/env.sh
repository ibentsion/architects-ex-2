# Shared environment for the Nebius GPU node. Source from the repo root:
#   source cloud/env.sh
# Everything persistent lives on the mounted compute filesystem so models,
# venv, caches and indexes survive across jobs.
export EX2_ROOT="${EX2_ROOT:-/mnt/data/ex2}"
export HF_HOME="$EX2_ROOT/hf"
export STANZA_RESOURCES_DIR="$EX2_ROOT/stanza"
# Tokenizer-parallelism warning noise from transformers inside forked workers.
export TOKENIZERS_PARALLELISM=false
if [ -f "$EX2_ROOT/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$EX2_ROOT/venv/bin/activate"
fi

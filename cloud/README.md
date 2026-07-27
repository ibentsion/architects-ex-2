# Cloud dev/eval environment (Nebius, 1× L40S)

Everything persistent lives on the compute filesystem
(`computefilesystem-e00hnnpfn5rr5aavma`, mounted at `/mnt/data`; shared across users — everything of ours stays under `/mnt/data/ex2/ibentsion`):

```
/mnt/data/ex2/ibentsion/
  venv/     python env, built once from requirements.lock
  hf/       HF_HOME — bge-m3, bge-reranker-v2-m3, anything lazily downloaded
  stanza/   STANZA_RESOURCES_DIR — Hebrew stanza models
  repo/     git checkout of this repo; corpus/ cache/ rag_index/ extracted
            inside it (gitignored), so all relative paths in configs work
            exactly as on the dev machine; answers JSONL + eval_results/
            land here and persist across jobs
```

The one-time artifacts (`corpus/` 106M, `cache/` 447M with the paid Token
Factory embeddings, `rag_index/` 721M) are copied from the dev machine via a
private HF dataset (`ibentsion/apex-ex2-artifacts`), not rebuilt.

On the GPU node the reranker (bge-reranker-v2-m3, the CPU bottleneck at
~1.7 s/pair) and stanza auto-select CUDA; the local sentence-transformers
embedder auto-detects it too (`rag/embed/st_embedder.py`). No config changes
needed between CPU and GPU machines.

## Workflow (all from the dev machine, repo root)

```bash
cloud/upload_artifacts.sh            # once, or after cache/index changes
cloud/submit_job.sh probe            # optional: venv + GPU/torch/reranker check, no tokens needed
cloud/submit_job.sh setup            # once: venv + artifacts + models (~15 min)
cloud/submit_job.sh smoke            # GPU check, reranker bench, unit tests, 3-question E2E
cloud/submit_job.sh run 'python -m rag.cli.query --config configs/default.yaml \
    --questions reference_questions.json --out rag_answers_gpu.jsonl'
cloud/submit_job.sh run 'python -m rag.cli.ingest --config configs/foo.yaml' --timeout 4h

nebius ai job list                   # status
nebius ai job logs --id <job-id>     # output
```

Every job re-runs the idempotent bootstrap (clone/update repo → `cloud/
setup_node.sh`), so jobs are order-independent; setup steps that are already
done are skipped in seconds. Push to GitHub before submitting — jobs run
`origin/main` (or `--branch`).

Secrets: `NEBIUS_API_KEY` is read from `.env`, `HF_TOKEN` from
`~/.cache/huggingface/token`; both are injected into the job env (plus
`OPENAI_BASE_URL`/`OPENAI_API_KEY` for the evalharness judge).

Notes
- Ingest with a *new* config reuses `cache/` (parse/token/embedding caches),
  so most re-ingests cost little; a genuinely new embedder pays embedding
  compute (Token Factory API, or local GPU for sentence-transformers models).
- Docling layout models are not pre-downloaded (parses are cached); a fresh
  parse of new PDFs will lazily fetch them into `$HF_HOME` on first use.
- To force a fresh artifact copy on the node, delete the dir in
  `/mnt/data/ex2/ibentsion/repo` (e.g. `rm -rf rag_index`) and re-run any job —
  setup re-downloads missing dirs only.

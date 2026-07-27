"""Cloud-node smoke test: GPU availability, GPU reranker speed, quick unit
suite, and an end-to-end 3-question RAG run against the copied index.

Run on the node from the repo root, after cloud/setup_node.sh, with the venv
active (``source cloud/env.sh``)::

    python cloud/smoke_test.py

Exits non-zero on the first failed step. Needs NEBIUS_API_KEY (generation +
smoke config's TF calls) in the environment.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SMOKE_CONFIG = "configs/final-per_page-bgem3-gptoss-low.yaml"
SMOKE_N_QUESTIONS = 3
RERANK_PAIRS = 20  # matches the default candidate pool (dense 20 + sparse 20 -> RRF top-20)


def step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def check_gpu() -> None:
    step("1/4 GPU visibility")
    import torch

    print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        sys.exit(
            "FAIL: torch.cuda.is_available() is False — driver/wheel mismatch? "
            "Check nvidia-smi output above; the requirements.lock torch build "
            "may need a driver newer than this node has."
        )
    name = torch.cuda.get_device_name(0)
    x = torch.randn(512, 512, device="cuda")
    (x @ x).sum().item()  # forces a real kernel launch
    print(f"OK: CUDA available on {name}")


def bench_reranker() -> None:
    step(f"2/4 reranker on GPU ({RERANK_PAIRS} pairs)")
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    device = str(model.model.device)
    print(f"CrossEncoder device: {device}")
    if "cuda" not in device:
        sys.exit("FAIL: reranker did not land on CUDA — it would stay the latency bottleneck.")

    pairs = [
        (
            "האם ביטוח הדירה מכסה נזקי צנרת ומה גובה ההשתתפות העצמית?",
            "פוליסת ביטוח הדירה של הראל מכסה נזקי מים ונוזלים אחרים שמקורם בצנרת הדירה. " * 12,
        )
    ] * RERANK_PAIRS
    model.predict(pairs, activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)  # warmup
    t0 = time.monotonic()
    model.predict(pairs, activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
    dt = time.monotonic() - t0
    print(f"OK: {RERANK_PAIRS} pairs in {dt:.2f}s ({dt / RERANK_PAIRS * 1000:.0f} ms/pair; "
          f"CPU baseline was ~1700 ms/pair)")


def run_unit_tests() -> None:
    step('3/4 pytest -m "not slow"')
    proc = subprocess.run([sys.executable, "-m", "pytest", "-m", "not slow", "-q"])
    if proc.returncode != 0:
        sys.exit("FAIL: quick unit suite failed on this node.")


def run_e2e() -> None:
    step(f"4/4 end-to-end: {SMOKE_N_QUESTIONS} reference questions via {SMOKE_CONFIG}")
    questions = json.loads(Path("reference_questions.json").read_text(encoding="utf-8"))
    subset = questions[:SMOKE_N_QUESTIONS]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)
        subset_path = f.name

    out_path = Path("smoke_answers.jsonl")
    proc = subprocess.run(
        [
            sys.executable, "-m", "rag.cli.query",
            "--config", SMOKE_CONFIG,
            "--questions", subset_path,
            "--out", str(out_path),
        ]
    )
    if proc.returncode != 0:
        sys.exit("FAIL: batch query run failed.")

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != SMOKE_N_QUESTIONS:
        sys.exit(f"FAIL: expected {SMOKE_N_QUESTIONS} answers, got {len(records)}")
    for r in records:
        if not r["answer"].strip():
            sys.exit(f"FAIL: empty answer for {r['id']}")
        print(
            f"OK: {r['id']}  total {r['latency_ms']:.0f} ms "
            f"(retrieval {r['retrieval_ms']:.0f} ms, generation {r['generation_ms']:.0f} ms)"
        )
    print(f"answers written to {out_path}")


if __name__ == "__main__":
    check_gpu()
    bench_reranker()
    run_unit_tests()
    run_e2e()
    print("\nSMOKE TEST PASSED")

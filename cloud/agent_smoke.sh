#!/usr/bin/env bash
# One-query smoke of the agent engine on the node before committing to the
# full 6-arm eval job (cloud/agent_eval.sh): exercises classify (gpt-oss
# orchestrator), concurrent prefetch, the tool loop with calculate, and
# DeepSeek synthesis against the real bge-m3 index.
set -euo pipefail

OUT=$(mktemp)
python -m rag.cli.query --config configs/embedder-bge-m3.yaml --engine agent --json \
    "אם הנזק לתכולת הדירה הוערך ב-10,000 שקל וההשתתפות העצמית היא 1,500 שקל, איזה סכום אקבל בפועל? ומה תקופת ההתיישנות להגשת תביעה?" \
    > "$OUT"

python - "$OUT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
steps = [t["step"] for t in d.get("trace") or []]
print("ANSWER:", d["text"][:300].replace("\n", " "))
print("STEPS:", steps)
print("CITATIONS:", len(d["citations"]), "LATENCY_S:", round((d["latency_ms"] or 0) / 1000, 1))
assert d["text"].strip(), "empty answer"
assert "classify" in steps and "synthesize" in steps, f"unexpected trace: {steps}"
print("AGENT_SMOKE_OK")
EOF

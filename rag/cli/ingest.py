"""Ingestion CLI: ``python -m rag.cli.ingest --config configs/default.yaml``
(rag_plan.md §7). Flags: --config, --categories, --force-reparse,
--skip-canary, --dry-run. Exit codes: 0 ok, 2 canary failure, 3 config error.
Implemented in wave E4 (T6)."""
from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Ingestion CLI is implemented in wave E4 (rag_plan.md T6)")


if __name__ == "__main__":
    raise SystemExit(main())

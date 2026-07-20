"""RAG pipeline for the Harel insurance corpus (rag_plan.md).

Two pipelines — ingestion (``rag.cli.ingest``) and query (``rag.cli.query``) —
share a YAML config (``rag/config.py``), a chunk-metadata contract
(``rag/types.py``), and swappable per-phase module interfaces.

Each phase package's ``__init__`` holds a ``Protocol`` class and a
``REGISTRY: dict[str, factory]``; ``rag.config.build(phase_name, config)``
resolves ``impl`` names to factories. Adding a backend = one file + one
registry entry, no call-site changes.
"""
import os

# This machine has a stale OAuth token in ~/.cache/huggingface/token whose
# signature no longer verifies — every implicit-token Hub request (docling
# model downloads, tokenizers, rerankers) fails with 401 while anonymous
# access works fine. Disable the implicit token unless the user explicitly
# overrides. setdefault → a deliberate HF_HUB_DISABLE_IMPLICIT_TOKEN=0 or a
# refreshed login via HF_TOKEN still wins.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

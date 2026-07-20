"""RAG pipeline for the Harel insurance corpus (rag_plan.md).

Two pipelines — ingestion (``rag.cli.ingest``) and query (``rag.cli.query``) —
share a YAML config (``rag/config.py``), a chunk-metadata contract
(``rag/types.py``), and swappable per-phase module interfaces.

Each phase package's ``__init__`` holds a ``Protocol`` class and a
``REGISTRY: dict[str, factory]``; ``rag.config.build(phase_name, config)``
resolves ``impl`` names to factories. Adding a backend = one file + one
registry entry, no call-site changes.
"""

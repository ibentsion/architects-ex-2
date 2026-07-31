# Project State

Exercise 2 — Harel Insurance customer support agent (APEX). Standalone quick-task
tracking (no ROADMAP.md — work proceeds as ad-hoc quick tasks).

Last activity: 2026-07-31 - Completed quick task 260731-001: citation accuracy is now LLM-judged against the actual cited corpus page (resolved via the Docling parse cache), not exact-matched against ground_truth_sources

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260715-001 | Baseline evaluation harness and report | 2026-07-15 | 315450d | [260715-001-eval-harness](./quick/260715-001-eval-harness/) |
| 260720-001 | Research + implementation plan for Hebrew RAG ingestion/query pipelines (rag_plan.md) | 2026-07-20 | 3094391 | [260720-001-rag-pipeline-plan](./quick/260720-001-rag-pipeline-plan/) |
| 260720-002 | Implemented rag_plan.md T1-T10: parsing/chunking/normalize/embed/index/retrieve/generate + both CLIs + tests; live E2E verified; full-corpus ingest partial (2/12 categories) | 2026-07-21 | 9430f03 | [260720-002-rag-implementation](./quick/260720-002-rag-implementation/) |
| 260731-001 | Citation accuracy re-based on LLM judging of the actual cited page (PageStore + citation judge + committee); ground_truth_sources demoted to an unscored diagnostic; 27 new tests | 2026-07-31 | 42c55e9 | [260731-001-citation-judge](./quick/260731-001-citation-judge/) |

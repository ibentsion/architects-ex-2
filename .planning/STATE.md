# Project State

Exercise 2 — Harel Insurance customer support agent (APEX). Standalone quick-task
tracking (no ROADMAP.md — work proceeds as ad-hoc quick tasks).

Last activity: 2026-08-02 - Completed quick task 260801-004: six-arm committee eval of the agent harness on v1+v2 (agent ties rag at equal model, DeepSeek synthesis lifts it; refusal inflation from category filters is the top fix) — see 260801-004-ANALYSIS.md
Previous: 2026-08-01 - Completed quick task 260801-003: agentic harness (--engine agent) — hybrid fast-path with concurrent sub-question retrieval, native tool-calling loop (retrieve + AST calculator), gpt-oss orchestration + DeepSeek synthesis; --route removed
Previous: 2026-08-01 - Completed quick task 260801-002: Marker/DataLab markdown parser + parser comparison (markdown is the better parse, Docling the better pipeline; staying on Docling+BiDi repair)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260715-001 | Baseline evaluation harness and report | 2026-07-15 | 315450d | [260715-001-eval-harness](./quick/260715-001-eval-harness/) |
| 260720-001 | Research + implementation plan for Hebrew RAG ingestion/query pipelines (rag_plan.md) | 2026-07-20 | 3094391 | [260720-001-rag-pipeline-plan](./quick/260720-001-rag-pipeline-plan/) |
| 260720-002 | Implemented rag_plan.md T1-T10: parsing/chunking/normalize/embed/index/retrieve/generate + both CLIs + tests; live E2E verified; full-corpus ingest partial (2/12 categories) | 2026-07-21 | 9430f03 | [260720-002-rag-implementation](./quick/260720-002-rag-implementation/) |
| 260731-001 | Citation accuracy re-based on LLM judging of the actual cited page (PageStore + citation judge + committee); ground_truth_sources demoted to an unscored diagnostic; 27 new tests | 2026-07-31 | 42c55e9 | [260731-001-citation-judge](./quick/260731-001-citation-judge/) |
| 260801-001 | Retrieval stages as parameterized query-CLI tools (dense/sparse/fuse/rerank/retrieve, no generation) + LLM query classification into categories/sub-questions (classify subcommand + --route pooled answering); 26 new tests, live-verified | 2026-08-01 | a3f6bb2 | [260801-001-retrieval-tools-classify](./quick/260801-001-retrieval-tools-classify/) |
| 260731-002 | Reference dataset v2: `refgen` package (spec-as-code, sampler, gated generation, audit CLI) + harness support for unanswerable questions; produced 108 verified questions over all 12 categories, 3 generator models, v1 held out | 2026-08-01 | (this push) | [260731-002-refset-v2](./quick/260731-002-refset-v2/) |
| 260801-002 | Marker/DataLab markdown as an alternative parser (`markdown` impl, synthetic DoclingDocument from `{N}---` page markers, evalharness `--doc-source`) + full parser comparison; markdown wins on parse quality (clause ordering 40.5% vs 32.6%), Docling wins end-to-end (correctness 5.69 vs 5.02) — recommendation: stay on Docling+repair, keep markdown for per-file routing | 2026-08-01 | (this push) | [260801-002-markdown-corpus-comparison](./quick/260801-002-markdown-corpus-comparison/) |
| 260801-003 | Agentic harness: --engine agent on the query CLI — classify + concurrent sub-question retrieval + native tool-calling loop (retrieve/calculate, AST calculator, capped hops, degrade-on-failure) + DeepSeek synthesis; harness config block (per-role models); --route replaced; partial-answer citation fix; 39 new tests, live-verified 3 query shapes | 2026-08-01 | e49f59d | [260801-003-agentic-harness](./quick/260801-003-agentic-harness/) |
| 260801-004 | Agent-harness committee eval on cloud: 6 arms (agent×{deepseek,gptoss}×{v1,v2}, v2 rag + no-RAG baselines); v1 agent-deepseek 6.46 > rag 6.08; v2 rag 4.83 ≈ agent-ds 4.80 ≫ norag 2.94; failure analysis → category-filter refusal inflation, fixes queued | 2026-08-02 | (this push) | [260801-004-agent-eval-cloud](./quick/260801-004-agent-eval-cloud/) |

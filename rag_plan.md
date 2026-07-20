# RAG Pipeline Implementation Plan — Harel Insurance Corpus (Exercise 2, Stage 2)

**Status:** implementation-ready · **Date:** 2026-07-20
**Scope:** ingestion pipeline (corpus → persistent hybrid index) + query pipeline (question → grounded, cited answer). Config-driven module swapping via YAML. Two CLIs. Unit tests per phase + E2E system test.
**Non-goals:** FastAPI `/ask` wiring (Stage 3 — but the query pipeline's output type maps 1:1 onto `contract.py`'s `AskResponse`), agentic routing, fine-tuning (forbidden by exercise rules).

---

## 1. Overview & Architecture

Two pipelines share a config, a chunk-metadata contract, and a set of swappable module interfaces. Everything Hebrew-sensitive (parsing, tokenization, BM25) is gated by explicit validation steps because Hebrew/RTL handling is the project's top risk (all major open-source PDF parsers have documented RTL weaknesses — AI21's benchmark shows across-the-board degradation on Hebrew documents).

```
INGESTION (rag.cli.ingest — offline, run once per config)
┌──────────────────────────────────────────────────────────────────────────────┐
│ corpus/<cat>/files/*.pdf ─► Docling DocumentConverter ─► DoclingDocument     │
│                             (JSON cache keyed by file sha256; RTL canary     │
│                              gate on first 10 PDFs — HARD STOP on failure)   │
│ corpus/<cat>/pages/*.txt ─► UTF-8 read (no parser; page = null) ──────┐      │
│                                                                       ▼      │
│        Chunker (per_page default │ per_paragraph │ per_table)                │
│        every chunk: {file: category-relative path (NFC), page: int|null,     │
│                      category, chunk_id, text, source_url?}                  │
│              │                                                               │
│    ┌─────────┴──────────────┐                                                │
│    ▼                        ▼                                                │
│ Normalizer (Stanza he    Embedder (bge-m3 via                                │
│ lemma+surface union)     sentence-transformers)                              │
│    │                        │                                                │
│    ▼                        ▼                                                │
│ bm25s index (disk,       Qdrant local (path=…, HNSW,                         │
│ mmap, token lists)       payload = chunk metadata)                           │
│    └────────┬───────────────┘                                                │
│             ▼                                                                │
│ index manifest.json (config hash, model ids, chunk counts, file hashes)      │
└──────────────────────────────────────────────────────────────────────────────┘

QUERY (rag.cli.query — single question or interactive loop)
┌──────────────────────────────────────────────────────────────────────────────┐
│ question ──► [optional --category filter]                                    │
│   ├─► Embedder ──► Qdrant dense top-20 (payload filter: category)            │
│   ├─► Normalizer ──► bm25s top-20 (post-filter by category)                  │
│   ▼                                                                          │
│ RRF fusion (k=60) ─► top-20 candidates                                       │
│   ─► bge-reranker-v2-m3 CrossEncoder ─► relevance gate (sigmoid ≥ 0.35)      │
│   ├─ ≥1 chunk passes ─► context assembly (top-6, metadata headers)           │
│   │      ─► citation-mandating prompt ─► tf_client LLM                       │
│   │      ─► citation validator (every cited {file,page} ⊆ retrieved set)     │
│   │      ─► Answer{text, citations[], confidence, retrieved[]}               │
│   └─ none pass ─► "אין לי מספיק מידע במסמכים כדי לענות על כך" fallback       │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Load-bearing design rules** (from research — violating either silently breaks citations):
1. **Never chunk from Markdown.** Docling page provenance (`prov.page_no`) survives only through the `DoclingDocument` object / `export_to_dict()` JSON round-trip; `export_to_markdown()` destroys page numbers (docling discussion #1012). Citations are graded on `{file, page}`, so all chunkers consume `DoclingDocument`, never a Markdown intermediate.
2. **Hybrid = our own fusion, not `langchain-qdrant` HYBRID mode.** Its sparse side (`FastEmbedSparse` / `Qdrant/bm25`) has **no Hebrew stemmer** — Hebrew was removed from FastEmbed's supported-language list as a bug fix (fastembed #505). We run dense-only Qdrant + an in-process `bm25s` index over Stanza lemmas, fused with RRF in `rag/retrieve/fusion.py` (~15 lines).

## 2. Stack Summary

| Component | Default | Config alternatives | Pinned package | Rationale (one line) |
|---|---|---|---|---|
| Orchestration | LangChain (thin: `Document`, embeddings iface, vector store) | plain-Python behind same interfaces | `langchain~=1.3`, `langchain-core~=1.4` | Locked; 1.x line; `langchain-community` is archived — partner packages only |
| PDF parsing | Docling → DoclingDocument JSON | marker-pdf (per-file fallback; GPL-3.0 — verify before adding) | `docling~=2.113`, `langchain-docling~=2.0` | Best tables + native 1-based page provenance; RtL support since docling-parse 3.3.0; MinerU rejected (RTL "not usable") |
| TXT parsing | plain UTF-8 read | — | stdlib | Web-page dumps; `page=null` matches ground truth |
| Chunking | per_page | per_paragraph (Docling `HybridChunker`), per_table; size parameterized | ships with docling | Page chunks = trivially correct citations; bge-m3's 8192-token window tolerates full pages |
| Normalizer | Stanza he (`tokenize,mwt,pos,lemma`) | trankit, YAP — same `Normalizer` interface | `stanza~=1.14` | MWT expansion strips fused prefixes (ו/ב/ל/כ/מ/ש/ה) — this is what makes Hebrew BM25 work |
| Embeddings | `BAAI/bge-m3` (1024-dim, 8192 ctx, MIT) | `intfloat/multilingual-e5-large` (needs `query:`/`passage:` prefixes), Webiks KolZchut Hebrew embedder | `sentence-transformers~=5.6`, `langchain-huggingface~=1.2` | BGE-M3 paper shows Hebrew stability where other multilingual models collapse; no prefix footguns |
| Vector DB | Qdrant **local mode** (`QdrantClient(path=…)`) | Qdrant server (`url=`, same code path), Chroma, Milvus | `qdrant-client~=1.18`, `langchain-qdrant~=1.1` | Filterable HNSW (category filter), zero infra, identical API local/server |
| Keyword/BM25 | `bm25s` in-process (mmap persistence, pre-tokenized input) | Elasticsearch / OpenSearch + Hebrew analyzer (Docker) | `bm25s~=0.3.9` | Maintained (vs rank_bm25 dead since ~2022), ~7–500× faster, accepts Stanza lemma token lists directly |
| Fusion | RRF, k=60 | weighted-score fusion | own code | Standard, rank-based, no score-calibration coupling between backends |
| Reranker | `BAAI/bge-reranker-v2-m3` (Apache-2.0), CrossEncoder | jina-reranker-v2 (CC-BY-NC — license warning in config comment) | `sentence-transformers` CrossEncoder | Apache license, Hebrew coverage, calibrated sigmoid score doubles as the relevance gate |
| Generation | Token Factory via `tf_client.chat` | any TF model id via config | existing `litellm` | Locked; shared course key, per-call cost estimate |
| Config | YAML + pydantic validation | — | `pyyaml~=6.0`, existing `pydantic>=2` | Registry pattern: every phase is `{impl: name, params: {…}}` |
| Tests | pytest | — | `pytest~=8` | No test framework exists in repo yet; `-m "not slow"` for the quick loop |

Install (append to `requirements.txt` with these pins, then fresh resolve — venv has pydantic 2.13.4 / litellm 1.92.0 / fastapi 0.139.0, do **not** use `--no-deps`):

```bash
pip install "docling~=2.113" "langchain-docling~=2.0" "langchain~=1.3" \
  "langchain-qdrant~=1.1" "langchain-huggingface~=1.2" "qdrant-client~=1.18" \
  "sentence-transformers~=5.6" "bm25s~=0.3.9" "stanza~=1.14" "pyyaml~=6.0" "pytest~=8"
pip freeze > requirements.lock   # langchain-core minors have shipped regressions — lock
```

All 12 packages passed slopcheck legitimacy audit (2026-07-20, "12 OK") and were pinned from live PyPI queries. trankit/YAP/marker/ES are **not** installed by default — their adapters raise a clear `ImportError` message with install instructions if selected in config.

## 3. Repo Layout

```
rag/
  __init__.py
  types.py              # Chunk, RetrievedChunk, Answer, Citation (pydantic; Citation mirrors contract.py)
  config.py             # pydantic config models, YAML loader, per-phase impl registries
  parsing/
    __init__.py         # Parser protocol: parse(path) -> ParsedDoc; registry
    docling_parser.py   # DocumentConverter, DoclingDocument -> export_to_dict JSON
    txt_parser.py       # UTF-8 read, page=None, manifest.json URL lookup
    cache.py            # cache/parsed/<sha256>.json, keyed by file content hash
    canary.py           # RTL sanity gate (see §5 stage 2)
  chunking/
    __init__.py         # Chunker protocol: chunk(ParsedDoc) -> list[Chunk]; registry
    per_page.py         # group DoclingDocument items by prov.page_no
    per_paragraph.py    # docling HybridChunker(tokenizer=embed_model, max_tokens)
    per_table.py        # TableItem -> atomic markdown chunk + per_paragraph for prose
  normalize/
    __init__.py         # Normalizer protocol: tokens(text) -> list[str]; registry
    stanza_norm.py      # tokenize,mwt,pos,lemma; lemma+surface union; see §5 stage 4
    trankit_norm.py     # adapter stub (optional dep, ImportError with pip hint)
    yap_norm.py         # adapter stub (HTTP subprocess; optional)
  embed/
    __init__.py         # Embedder protocol: embed_docs/embed_query; registry
    st_embedder.py      # sentence-transformers; e5 prefix handling lives HERE, not call sites
  index/
    __init__.py
    dense.py            # VectorIndex protocol; impls: qdrant_local, qdrant_server, chroma, milvus
    sparse.py           # KeywordIndex protocol; impls: bm25s, elasticsearch, opensearch
    manifest.py         # write/verify index manifest (config hash, model ids, counts)
  retrieve/
    __init__.py
    fusion.py           # rrf(rankings, k=60) -> fused ranking by chunk_id
    rerank.py           # CrossEncoder scoring + relevance gate
    retriever.py        # orchestrates dense+sparse -> fuse -> rerank -> gate
  generate/
    prompts.py          # prompt variants (all citation-mandating), registry
    generator.py        # context assembly, tf_client call, one retry on citation failure
    citations.py        # parse SOURCES block, validate against retrieved metadata
  cli/
    ingest.py           # python -m rag.cli.ingest
    query.py            # python -m rag.cli.query
configs/
  default.yaml          # full annotated default config (§4)
  swap-example.yaml     # minimal override example (§4)
tests/
  conftest.py           # fixtures: mini corpus (2 real small PDFs + 2 TXTs, 2 categories)
  fixtures/mini_corpus/...
  test_config.py test_parsing.py test_chunking.py test_normalize.py
  test_index.py test_retrieve.py test_generate.py test_e2e.py
```

Interfaces are `Protocol` classes; each phase package's `__init__.py` holds a `REGISTRY: dict[str, factory]`. `rag/config.py` resolves `impl` names to factories and passes `params` through — adding a backend = one file + one registry entry, no call-site changes. LangChain imports stay inside adapter modules (community guidance: isolate LC behind your own interfaces; also keeps a plain-Python escape hatch).

## 4. YAML Configuration

Pydantic-validated. Every phase block is `{impl: <registry key>, params: {…}}`. Unknown keys are errors (catch typos). The ingestion CLI stamps `sha256(canonical_config_subset)` into the index manifest; the query CLI refuses to run against an index built with an incompatible embedder/chunker/normalizer (clear error telling you to re-ingest).

**`configs/default.yaml`** (full, annotated):

```yaml
corpus_dir: corpus            # 12 category dirs, each with files/ (PDF) and pages/ (TXT)
index_dir: rag_index/default  # all persistent artifacts live under here
cache_dir: cache/parsed       # DoclingDocument JSON cache (shared across configs)

parser:
  impl: docling               # alternatives: (marker fallback is per-file, not global)
  params:
    rtl_canary: true          # hard-stop gate on reversed Hebrew (see §5 stage 2)
    canary_sample: 10         # PDFs sampled for the gate, spread across categories

chunker:
  impl: per_page              # alternatives: per_paragraph, per_table
  params:
    max_tokens: 1800          # per_page: pages longer than this are split (token-counted
                              #   with the embedder tokenizer); bge-m3 tolerates full pages
    txt_max_tokens: 512       # TXT files have no pages -> paragraph-packed to this size
    # per_paragraph params: max_tokens: 512, merge_peers: true
    # per_table params: prose_max_tokens: 512 (non-table text falls back to per_paragraph)

normalizer:
  impl: stanza                # alternatives: trankit, yap (same interface; optional deps)
  params:
    package: default          # stanza he package; 'iahltwiki' scores better on modern
                              #   text (lemmas 92.5 vs 89.7) — worth an eval run
    index_surface_forms: true # index lemma+surface union (hedges ~10% lemma errors
                              #   on domain terms like הראל); lowercase Latin; strip nikud

embedder:
  impl: sentence_transformers
  params:
    model: BAAI/bge-m3        # alternatives: intfloat/multilingual-e5-large (adapter
                              #   auto-adds query:/passage: prefixes), Webiks KolZchut
    batch_size: 16            # CPU-only machine — embed in batches at ingestion

dense_index:
  impl: qdrant_local          # alternatives: qdrant_server (url:), chroma, milvus
  params: {}                  # qdrant_local path is derived: <index_dir>/qdrant
                              # NOTE: local mode = single-process lock; ingest and
                              # query CLIs cannot open it concurrently

sparse_index:
  impl: bm25s                 # alternatives: elasticsearch, opensearch (Docker, url:)
  params: {}                  # persisted mmap at <index_dir>/bm25

retrieval:
  dense_top_k: 20
  sparse_top_k: 20            # with a category filter, sparse fetches 60 and post-filters
  rrf_k: 60                   # standard RRF constant
  rerank:
    impl: bge                 # alternatives: jina (CC-BY-NC — non-commercial only!)
    params: {model: BAAI/bge-reranker-v2-m3, max_length: 512}
    top_n: 6                  # chunks passed to generation
    gate_threshold: 0.35      # sigmoid relevance gate; below for ALL candidates ->
                              #   "not enough information" fallback. Tune on dev set.

generation:
  model: deepseek-ai/DeepSeek-V4-Pro   # any Token Factory model id (via tf_client)
  prompt: grounded_cite       # alternatives: strict_extractive, few_shot_cite (§6 stage 6)
  max_tokens: 1024
  temperature: 0.2
  retry_on_citation_failure: true      # one retry with corrective system nudge
```

**`configs/swap-example.yaml`** (module swapping — Chroma + trankit + different prompt; everything omitted uses defaults via deep-merge over `default.yaml`):

```yaml
extends: configs/default.yaml
index_dir: rag_index/chroma-trankit
normalizer: {impl: trankit, params: {}}
dense_index: {impl: chroma, params: {}}
generation: {prompt: strict_extractive}
```

## 5. Ingestion Pipeline (stage-by-stage)

Entry: `rag/cli/ingest.py` → `run_ingestion(config)`. Stages run sequentially; each logs counts and timing. Re-running is incremental where cache allows.

**Stage 1 — Discover.**
*In:* `corpus_dir`. *Out:* `list[SourceFile{abs_path, rel_path, category, kind: pdf|txt, sha256, source_url}]`.
`rel_path` is the **category-relative POSIX path, NFC-normalized** (`unicodedata.normalize("NFC", …)`) — must byte-match `reference_questions.json`'s `file` field (e.g. `apartment/files/הודעה-על-תקופת-התיישנות.pdf`). `source_url` looked up from `corpus/manifest.json` (data only — never fetched; ground truth is anchored to the frozen snapshot). Skip non-PDF/TXT assets. *Failure:* missing corpus dir → exit with pointer to `get_corpus.py`.

**Stage 2 — Parse (with cache + RTL canary gate).**
*In:* SourceFiles. *Out:* `ParsedDoc{source, pages: …}` per file; PDFs via `docling_parser`, TXTs via `txt_parser` (whole text, `page=None`).
Cache: before parsing, check `cache/parsed/<sha256>.json`; on miss, run Docling `DocumentConverter` and persist `DoclingDocument.export_to_dict()`. Docling layout/table models on CPU = the slow path (budget **hours** for 350 PDFs); the cache means chunker/embedder experiments never re-parse. **RTL canary gate (blocking, first-run):** parse `canary_sample` PDFs spread across categories; assert canary tokens (ביטוח, הראל, פוליסה) appear and their reversals (e.g. חוטיב) do **not**; additionally spot-check one known ground-truth `{file,page}` (`apartment/files/הודעה-על-תקופת-התיישנות.pdf` p.1 must contain התיישנות and שלוש שנים) — this also verifies A1 (Docling `page_no` is 1-based) empirically. *Failure:* per-file parse error → log, record in manifest `failed_files`, continue. Canary failure → **hard stop** with per-file diagnosis; remediation ladder: marker-pdf fallback for the failing files → line-level BiDi repair pass → escalate to human if systemic (this is the project's #1 risk; do not build downstream on garbage text).

**Stage 3 — Chunk.**
*In:* ParsedDocs + chunker config. *Out:* `list[Chunk]` (schema in §8).
- `per_page` (default): iterate `DoclingDocument` items grouped by `prov.page_no`; one chunk per page; pages exceeding `max_tokens` split at paragraph boundaries (all fragments keep the same `page`). Simple, and citations are correct by construction.
- `per_paragraph`: Docling `HybridChunker(tokenizer=<embedder model id>, merge_peers=True)` — hierarchical structure-aware chunking with tokenizer-fitted split/merge; page taken from the chunk's `doc_items[].prov[].page_no` (first item's page).
- `per_table`: every `TableItem` becomes one **atomic** chunk serialized as Markdown with its section-heading context prepended (orphaned table cells are useless to both retrieval and the reader); non-table prose falls back to per_paragraph.
- TXT: paragraph-packed to `txt_max_tokens`; `page=None` always.
All strategies consume the DoclingDocument dict — **never** Markdown (rule §1.1). *Failure:* chunk with empty text → dropped with warning; invariant checks (§9) run here.

**Stage 4 — Normalize (for sparse indexing only).**
*In:* chunk texts. *Out:* `list[list[str]]` token lists, parallel to chunks.
`stanza.Pipeline('he', processors='tokenize,mwt,pos,lemma')`. MWT expands fused tokens (בבית → ב+בית) and lemmatization strips the prefixes — this is what un-fragments Hebrew BM25. Recipe: emit **union of lemma + surface form** per word; lowercase Latin tokens (mixed-script corpus: "dental-claim-form"); keep digits (₪ amounts, waiting periods are high-value BM25 tokens); normalize gershayim/geresh (״ ׳) and maqaf (־) to plain forms; strip nikud + punctuation. Normalized tokens exist **only** inside the sparse index — displayed/stored chunk text is always the original (MWT changes token counts; never mix normalized offsets with display text). First run calls `stanza.download('he')` — CLI prints a notice. Batch via `bulk_process` (CPU: minutes-to-tens-of-minutes for 571 docs — acceptable offline). *Failure:* Stanza error on a chunk → fall back to whitespace+punctuation tokenization for that chunk, log it.

**Stage 5 — Embed.**
*In:* chunk texts. *Out:* `float32[n_chunks, dim]`.
sentence-transformers, `batch_size` from config, normalized embeddings. The e5 `query:`/`passage:` prefix logic is inside `st_embedder` keyed on model name — call sites never know. *Failure:* OOM → halve batch and retry once.

**Stage 6 — Index dense + sparse.**
Dense: `QdrantVectorStore` over `QdrantClient(path=<index_dir>/qdrant)`; collection `chunks`; payload = full chunk metadata (category is a payload-indexed field → efficient filtered HNSW). Server mode: same code, `url=` instead of `path=` (locked requirement). Chroma/Milvus impls satisfy the same `VectorIndex` protocol.
Sparse: `bm25s.BM25()` over the Stage-4 token lists; `save(<index_dir>/bm25)` (mmap reload at query time); chunk ids stored alongside so fusion joins on a single `chunk_id` namespace.
*Failure:* partial index on crash → `index_dir` is written to a temp dir and atomically renamed on success.

**Stage 7 — Persist manifest.**
`<index_dir>/manifest.json`: config hash + resolved impl names, embedder/normalizer/chunker identity, chunk count per category, per-file sha256 + status (ok/failed/cached), canary result, timestamps. Query CLI validates compatibility at load (§6 stage 0).

## 6. Query Pipeline (stage-by-stage)

Entry: `rag/cli/query.py` → `QueryEngine(config)` loads once, then `answer(question, category=None)` per query. Stateless between queries.

**Stage 0 — Load & validate.** Open manifest; verify config compatibility (embedder/chunker/normalizer identity match — mismatch → error "re-ingest with this config or point at the matching index_dir"). Load Qdrant client, mmap bm25s, embedder, normalizer, reranker (lazy model loads, warm after first query). Note: Qdrant local single-process lock — cannot query while ingesting the same `index_dir`.

**Stage 1 — Normalize query.** Same `Normalizer` instance/recipe as ingestion (lemma+surface union). Symmetry is mandatory — an asymmetric normalizer silently zeroes BM25 recall.

**Stage 2 — Dual search.** Dense: embed query (`query:` prefix if e5) → Qdrant top-`dense_top_k`, with `models.Filter` on `category` payload when a filter is given. Sparse: bm25s top-`sparse_top_k`; with a category filter, fetch 3× and post-filter by chunk metadata (571-doc corpus — cheap).

**Stage 3 — RRF fusion.** `score(c) = Σ_r 1/(rrf_k + rank_r(c))` over the two rankings, joined on `chunk_id`; take top 20. Rank-based → no cross-backend score calibration needed.

**Stage 4 — Rerank + relevance gate.** CrossEncoder `(question, chunk_text)` pairs, `max_length=512` (query + chunk head first — Hebrew page chunks truncate). CPU cost ~1–3 s for 20 pairs (measured in task 7; shrink candidate set if worse). Keep top-`top_n`=6 with sigmoid score ≥ `gate_threshold` (0.35 start; **tune on dev set** — this is the "how do you know retrieval works, independent of generation" knob). **Gate fail (zero survivors):** skip generation entirely, return `Answer(text="אין לי מספיק מידע במסמכים כדי לענות על שאלה זו.", citations=[], confidence=0.0)` — zero LLM cost, and the evalharness treats refusals as non-hallucinations. This is the locked "retrieval validation before generation" step.

**Stage 5 — Context assembly.** Each surviving chunk rendered with a machine-readable header:
`[מקור: {file} | עמוד: {page|"-"} | תחום: {category}]` followed by original chunk text. Order: rerank score desc. Token budget guard (~6k tokens) trims the tail.

**Stage 6 — Generation.** Via `tf_client.chat(messages, model=…, quiet=False)` — cost estimate visible, shared-key etiquette. Prompt variants (registry in `generate/prompts.py`), **all** mandate citations and the fallback sentence, all instruct answering in the question's language and ending with a parseable sources block:

```
מקורות:
- file: apartment/files/הודעה-על-תקופת-התיישנות.pdf | page: 1
```

- `grounded_cite` (default): answer only from the provided sources; cite file+page for every factual claim; if the sources don't contain the answer, say so.
- `strict_extractive`: additionally require a short verbatim quote per claim (feeds `Citation.quote` — "optional but persuasive" per contract.py).
- `few_shot_cite`: `grounded_cite` + 2 worked Hebrew examples (one answerable with citation, one refusal).

**Stage 7 — Citation validation.** `generate/citations.py` parses the sources block (regex on the `- file: … | page: …` lines; tolerant of RTL punctuation, NFC-normalizes the parsed path). Validate: every cited `{file, page}` **must** exist in the retrieved chunk set — anything else is a fabricated citation. Invalid citations dropped; if **all** citations are invalid or the block is missing and `retry_on_citation_failure`, retry once with a corrective system message; still failing → keep answer text, attach the retrieved top-3 chunks' `{file,page}` as citations, flag `citation_fallback=true` in the answer metadata.

**Stage 8 — Response.** `Answer{text, citations: [{file, page, quote?}], category?, confidence: max_rerank_sigmoid, latency_ms, cost_estimate}` — field-compatible with `contract.py`'s `AskResponse` so Stage 3 wiring is a ~20-line adapter.

## 7. CLI Specs

**Ingestion** — `python -m rag.cli.ingest --config configs/default.yaml`

| Flag | Meaning |
|---|---|
| `--config PATH` | required; YAML config |
| `--categories apartment travel` | limit to listed categories (subset ingest for tests/E2E) |
| `--force-reparse` | ignore parse cache |
| `--skip-canary` | skip RTL gate (only after it has passed once for this corpus) |
| `--dry-run` | discover + parse-cache stats only, no indexing |

Examples: full ingest `python -m rag.cli.ingest --config configs/default.yaml`; E2E subset `python -m rag.cli.ingest --config configs/default.yaml --categories apartment travel`. Exit codes: 0 ok, 2 canary failure, 3 config error.

**Query** — single: `python -m rag.cli.query --config configs/default.yaml "מה תקופת ההתיישנות של תביעה?"`

| Flag | Meaning |
|---|---|
| `--config PATH` | required; must match the ingested index's manifest |
| `--category CAT` | restrict retrieval to one category |
| `--interactive` / `-i` | REPL mode |
| `--show-chunks` | print retrieved chunks + rerank scores (retrieval debugging) |
| `--json` | emit the full `Answer` as JSON (this is what evalharness/batch runners consume) |

Interactive semantics: prompt `שאלה> `; **each query is independent** (no conversation memory — Stage 3 concern); `exit`/`quit` (or Ctrl-D) terminates; empty line re-prompts; models stay loaded between queries so only the first is slow. Batch mode for evalharness: `--questions reference_questions.json --out answers.jsonl` runs all dev questions and writes the answers JSONL the existing harness scores.

## 8. Metadata & Citation Contract

**Chunk schema** (pydantic, `rag/types.py`):

```python
class Chunk(BaseModel):
    chunk_id: str        # "{file}#p{page}#c{n}" — stable, joins dense/sparse indexes
    file: str            # category-relative POSIX path, NFC — EXACTLY as in
                         # reference_questions.json (e.g. "apartment/files/חוברת-….pdf")
    page: int | None     # 1-based for PDFs; None for TXT (matches ground truth)
    category: str        # corpus dir name, e.g. "apartment"
    text: str            # ORIGINAL text (never normalized), tables as Markdown
    source_url: str | None  # from corpus/manifest.json (data only)
    chunker: str         # provenance: which strategy produced it
```

**Normalization rules:** `file` NFC-normalized at ingestion (macOS/tools can emit NFD; dev-set paths byte-compare), POSIX separators, no leading `./`. Invariants (unit-tested): PDF chunks `page ≥ 1`; TXT chunks `page is None`; `category` ∈ the 12 known dirs; `file` starts with `category/`.

**Citation validation vs `reference_questions.json`:** ground truth is `ground_truth_sources: [ {any_of: [{file, page}, …]}, … ]` — one `any_of` group per required fact; citing any member of a group scores; cross-document questions have two groups and need one hit in each. The **runtime** validator (§6 stage 7) checks citations ⊆ retrieved metadata (fabrication guard). The **eval** check (existing `evalharness/` citation scorer) compares system citations against `any_of` groups — exact match on NFC `file` string and `page` int/null. Because chunk `file`/`page` are built from the same corpus-relative convention, a correct retrieval → correct citation with no mapping layer.

## 9. Testing Plan

Framework: pytest (`pytest~=8`), new `tests/` tree, markers: `slow` (E2E, model downloads), `llm` (needs NEBIUS_API_KEY). Quick loop: `pytest -x -m "not slow and not llm"`. Fixture: `tests/fixtures/mini_corpus/` — 2 categories (apartment, travel), 2 small real PDFs (incl. `הודעה-על-תקופת-התיישנות.pdf` — its p.1 is a ground-truth anchor) + 2 TXTs, with a mini `manifest.json`.

**Per-phase unit tests (concrete cases):**
- `test_config.py`: default.yaml loads and validates; unknown key rejected; `extends:` deep-merge works; registry resolves every documented impl name; unpinned impl → helpful error.
- `test_parsing.py`: PDF → every text item carries `prov.page_no ≥ 1`; TXT → `page=None`; cache hit skips converter (converter mocked, assert not called); sha256 change → re-parse; **RTL canary**: passes on fixture PDF, fails on a synthetic reversed-Hebrew doc (`חוטיב` string); ground-truth anchor page contains התיישנות (validates A1: 1-based pages).
- `test_chunking.py`: per_page → one chunk per page, page order preserved; over-long page splits share the same `page`; per_paragraph → all chunks ≤ max_tokens (embedder tokenizer); per_table → table chunk is atomic, contains Markdown pipe rows and heading context; all invariants of §8; Hebrew filename survives into `file` NFC-byte-identical to the fixture ground truth.
- `test_normalize.py`: Hebrew fixtures — `בבית` tokens include lemma `בית`; `והפוליסות` yields `פוליסה`; surface form also present (union); `הראל` survives (surface hedge); Latin `Dental-Claim-Form` → `dental`, `claim`, `form`; `6,000 ₪` keeps `6,000`; gershayim `חו״ל` normalized; identical output for identical input across two pipeline instances (determinism); whitespace fallback triggers on induced Stanza failure.
- `test_index.py`: dense round-trip (index 10 chunks → query returns nearest with intact payload); category filter excludes other categories; bm25s save/load → identical scores; chunk_id join dense↔sparse consistent; manifest write→verify; incompatible manifest rejected.
- `test_retrieve.py`: RRF determinism + hand-computed 3-doc example; item ranked #1 in both lists wins fusion; gate: all-below-threshold → empty result (mock CrossEncoder); top_n truncation; **retrieval smoke check**: on the mini index, the question "תקופת התיישנות" retrieves the anchor chunk in top-3 (embeddings real, marked `slow`).
- `test_generate.py`: prompt variants all contain citation mandate + fallback instruction; SOURCES-block parser on well-formed / RTL-punctuated / missing-block outputs; validator rejects `{file,page}` not in retrieved set; fabricated-citation → retry path (LLM mocked); gate-fail path never calls the LLM (mock asserts).

**E2E system test** (`test_e2e.py`, `slow`+`llm`): ingest a 2-category subset (`--categories apartment travel`, real Docling parse of the mini corpus) → query CLI in `--json` mode with dev question `dev-02-apartment-easy` → assert: non-empty Hebrew answer, ≥1 citation, citation matches the question's `any_of` group `{file: "apartment/files/הודעה-על-תקופת-התיישנות.pdf", page: 1}`, latency recorded. Also one out-of-corpus question ("מה מזג האוויר?") → fallback text, zero citations. Pipelines run sequentially (Qdrant local lock).

**Evalharness integration:** `python -m rag.cli.query --config … --questions reference_questions.json --out rag_answers.jsonl` then `python -m evalharness.run --questions reference_questions.json --answers rag_answers.jsonl --out eval_results/rag-default`. Config swaps (embedder A/B: bge-m3 vs multilingual-e5-large vs Webiks; stanza `iahltwiki`; chunkers; prompts) are separate `index_dir`s + separate eval runs — this is the deciding instrument for every LOW-confidence model choice, compared against `eval_results/baseline`.

## 10. Implementation Task Breakdown

Ordered; each sized for a focused execution session. **Parallelizable:** after T1 lands (contracts), T2/T4/T8-prompts can run in parallel (disjoint files); T3 needs T2's ParsedDoc shape; T5 needs T3+T4; T7 needs T5; T9 needs T7+T8; T10 last.

| # | Task | Files | Action | Verify / Done |
|---|---|---|---|---|
| T1 | Scaffold + contracts + config | `rag/{__init__,types,config}.py`, all package `__init__.py` protocols+registries, `configs/*.yaml`, `tests/{conftest,test_config}.py`, `requirements.txt` pins, `pytest.ini` | Create package skeleton, pydantic Chunk/Answer/config models, registry pattern, both YAML examples, mini-corpus fixture dir | `pytest tests/test_config.py` green; `pip install -r requirements.txt` resolves cleanly; done when both configs load+validate |
| T2 | Parsing + cache + RTL canary | `rag/parsing/*`, `tests/test_parsing.py` | Docling adapter (DoclingDocument→dict, never Markdown), TXT reader, sha256 cache, canary module with ground-truth anchor check | `pytest tests/test_parsing.py`; **GATE:** run canary on 10 real corpus PDFs across categories — hard stop + escalation if reversed Hebrew; done when gate passes (or marker fallback documented for failing files) |
| T3 | Chunking (3 strategies) | `rag/chunking/*`, `tests/test_chunking.py` | per_page, per_paragraph (HybridChunker), per_table; §8 invariants enforced | `pytest tests/test_chunking.py`; done when all invariants hold on mini-corpus for all 3 strategies |
| T4 | Normalizer | `rag/normalize/*`, `tests/test_normalize.py` | Stanza adapter (lemma+surface union recipe), trankit/YAP ImportError stubs | `pytest tests/test_normalize.py` incl. all Hebrew fixture cases |
| T5 | Indexes + manifest | `rag/embed/*`, `rag/index/*`, `tests/test_index.py` | st_embedder (e5 prefixes inside), qdrant_local/qdrant_server dense impl (+ chroma/milvus stubs behind protocol), bm25s sparse impl, atomic manifest | `pytest tests/test_index.py`; done when round-trips + category filter + manifest verify pass |
| T6 | Ingestion CLI | `rag/cli/ingest.py` | Wire stages 1–7, flags per §7, timing/count logs, atomic index_dir | `python -m rag.cli.ingest --config configs/default.yaml --categories apartment travel` completes; manifest sane; done when subset index built |
| T7 | Retrieval + gate | `rag/retrieve/*`, `tests/test_retrieve.py` | RRF, CrossEncoder rerank, gate; measure CPU rerank latency, adjust candidate count if >3 s | `pytest tests/test_retrieve.py`; smoke check: anchor chunk in top-3 for its dev question |
| T8 | Generation + citations | `rag/generate/*`, `tests/test_generate.py` | 3 prompt variants, context assembly, tf_client call, SOURCES parser, validator, retry + fallback paths | `pytest tests/test_generate.py` (LLM mocked); one live smoke call via tf_client |
| T9 | Query CLI | `rag/cli/query.py` | QueryEngine, single/interactive/batch modes, `--json`, manifest compatibility check | Manual: interactive session answers dev-02 correctly with citation; `exit` quits; done when batch mode writes valid answers JSONL |
| T10 | E2E + eval + full ingest | `tests/test_e2e.py` | E2E test per §9; full-corpus ingest (measure Docling CPU hours on 10-doc sample first, then run); full dev-set run through evalharness vs baseline | `pytest -m slow` green; `eval_results/rag-default/report.md` exists and beats `eval_results/baseline` on relevance/hallucination/citation |

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Reversed/jumbled Hebrew from PDF extraction** (docling #1938 open; AI21: all parsers degrade on Hebrew) | Garbage index, wrong answers, wasted downstream work | T2 blocking canary gate before anything downstream; ladder: marker-pdf per-file fallback → BiDi line repair → human escalation if systemic. Born-digital PDFs (no OCR) are docling-parse's best case |
| **Page fidelity loss** (Markdown export destroys `prov.page_no`) | Citation metric → 0 | Architectural rule: chunk only from DoclingDocument JSON; unit invariant PDF `page≥1` / TXT `null`; ground-truth anchor test validates 1-based numbering (assumption A1) |
| **CPU-only machine** (no GPU) | Docling parse = hours for 350 PDFs; reranker 1–3 s/query | Parse cache keyed by sha256 (parse once, ever); measure on 10-doc sample before full run; batch embeddings; rerank set capped at 20, shrink if measured latency is worse; jina flash-attn path not viable anyway |
| **Mixed Hebrew/English + morphology in BM25** | Keyword recall collapse on fused prefixes or Latin terms | Stanza mwt+lemma with lemma+surface union; keep lowercased Latin + digits; normalize gershayim/maqaf; identical normalizer at ingest and query (symmetry unit-tested) |
| **Tables fragmented or diluted** | Medium/hard questions (limits, amounts) unanswerable | Tables atomic (one chunk per TableItem, Markdown + heading context) in per_table strategy; per_page keeps tables whole within their page; A/B via config + evalharness |
| **Embedder Hebrew ranking unknown** (no trustworthy Hebrew retrieval leaderboard) | Suboptimal dense recall | bge-m3 default (best published Hebrew-stability evidence, 8192 ctx, no prefixes); A/B multilingual-e5-large + Webiks via config swap, decided by evalharness citation accuracy on 48 dev questions |
| **Qdrant local single-process lock** | Concurrent ingest+query crash | Documented in config + CLI error message; E2E runs pipelines sequentially; server mode (url:) lifts the limit with zero code change |
| **LangChain churn** (community archived; core minors have regressed) | Breakage on install/upgrade | Thin LC usage behind own protocols; compatible-release pins + `requirements.lock`; partner packages only |
| **Shared TF key budget** | Burning the course pool | Gate-fail skips the LLM entirely; tf_client cost prints kept visible; E2E uses 2 questions, not 48 |

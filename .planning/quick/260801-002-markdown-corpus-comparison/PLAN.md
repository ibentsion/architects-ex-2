---
slug: markdown-corpus-comparison
date: 2026-08-01
status: planned
---

# Marker/DataLab markdown as an alternative to Docling

Ingest the `ilan3580/apex-ex2-harel-corpus-markdown` markdown rendering of the
same 350 PDFs, build a parallel index, and decide which parse the pipeline
should stand on.

## What the dataset actually contains (verified before planning)

- 923 files mirroring our corpus: 350 PDFs, **350 `.md`**, 221 `.txt`, `manifest.json`
- Markdown lives at `<category>/markdown-files/<same-stem>.md` — one per PDF
- **Pages are recoverable**: Marker's `--paginate_output` markers, `{N}------…`,
  0-indexed. So `page_no = N + 1`, and citations stay file+page as today.
- Tables come out with **correct row alignment** — the failure my BiDi repair
  could not touch. On `travel/…תעריפון-דרכון-first-class.pdf` p.3 Marker emits
  `| \$3.35 | 0-17 | החמרה של מצב רפואי קודם |` as one row, where Docling
  merged several rows into one scrambled cell.
- Hebrew word order is correct in the cells sampled.
- Marker also injects LLM image captions (`<p>Image: Icon of binoculars</p>`)
  and `<img …>` tags — noise that must be stripped or it lands in chunks.

## Design decisions

**1. Markdown is a parser variant, not a new document.** `SourceFile.rel_path`
stays `<category>/files/<name>.pdf` even though `abs_path` points at the `.md`.
Citations then have the same shape in both arms and remain comparable.
`sha256` is of the `.md` (correct cache keying — no collision with the Docling
entries).

**2. Synthetic DoclingDocument.** The markdown parser emits
`{"texts": [{"text": block, "prov": [{"page_no": n}]}], "tables": []}`. Every
existing chunker then works unchanged and page provenance survives, so no
chunker needs to learn about markdown.

**3. Hold the chunker constant: `per_page`.** `per_table` keys off Docling
`TableItem`s that markdown has no equivalent for, so it cannot be held
constant across arms. `per_page` is page-atomic, treats a page's markdown
(tables included) as one unit, and `rag_index/per_page-bgem3` already exists
as the Docling-side twin. Same chunker, same embedder (bge-m3), one variable:
the parse.

**4. The citation judge must see each arm's own text.** `evalharness/pages.py`
resolves a citation through the Docling parse cache. Left alone it would judge
markdown-arm citations against Docling's scrambled tables — understating the
arm under test and contradicting the module's own premise ("the judge sees
exactly the text the retriever indexed"). PageStore gains an optional markdown
page source, selected per run.

## Tasks

1. **Fetch** the 350 `.md` into `corpus/<category>/markdown-files/`; verify
   1:1 with the PDFs by stem; record counts. (`corpus/` is gitignored.)
2. **`rag/parsing/markdown_parser.py`** — split on `{N}------`, strip Marker
   image artifacts, emit the synthetic DoclingDocument. Register as
   `markdown` in `rag/parsing/REGISTRY`.
3. **Discovery** — `discover()` learns a document source (`pdf` | `markdown`);
   ingest selects it from the parser impl. TXT pages are shared by both arms.
4. **Tests** — page numbering is 1-based and matches the PDF's page count;
   image artifacts stripped; rel_path points at the PDF; chunkers produce
   correct `page` values; a real corpus file round-trips.
5. **Text-quality comparison** vs `pdftotext`, reusing the harness from the
   BiDi work: token recall, word-order (bigram) agreement, number recall —
   Docling+repair vs Marker, same 350 files.
6. **Config + ingest** — `configs/markdown-per_page-bgem3.yaml` →
   `rag_index/markdown-perpage-bgem3`; ingest on the GPU node.
7. **Eval** — same questions and judges as the post-repair run
   (gpt-oss-120b / Qwen3-235B / DeepSeek-V4-Pro; no GLM), markdown arm vs the
   Docling `per_page` arm, both judged in the same job.
8. **Summarize + recommend** markdown vs current parsed PDFs.

## Risks

- **Marker hallucination.** LLM-assisted parsing can invent text; the image
  captions prove an LLM is in the loop. The text-quality comparison against
  `pdftotext` (task 5) is the guard — a token-recall/precision gap in the
  wrong direction means invented content.
- **No page markers in some files** → those documents lose citations. Counted
  and reported, not silently dropped.
- `per_page` scored 6.15 vs `per_table` 5.71 on correctness historically, so
  the markdown arm's numbers are only comparable to the per_page Docling arm,
  not to the per_table production index.

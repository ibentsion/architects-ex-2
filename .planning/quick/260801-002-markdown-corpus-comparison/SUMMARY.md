---
slug: markdown-corpus-comparison
date: 2026-08-01
status: complete
---

# Marker/DataLab markdown vs Docling — result

## Delivered

- 350 markdown files installed at `corpus/<category>/markdown-files/` (1:1 with
  the PDFs by stem, verified both directions).
- `rag/parsing/markdown_parser.py` — Marker `{N}------` pages → synthetic
  `DoclingDocument`, so every chunker works unchanged. Registered as
  `markdown`; `discover()` gained a document-source selector.
- `evalharness` `--doc-source`: the citation judge is shown the arm that
  produced the answers.
- `configs/markdown-per_page-bgem3.yaml` → `rag_index/markdown-perpage-bgem3`,
  built on the GPU node.
- 23 new tests (337 pass).
- `cloud/markdown_arm_eval.sh` — both arms, same questions/judges/chunker/
  embedder/generator.

## The measurement problem, and how it was resolved

`pdftotext` cannot arbitrate: **Docling reads the same embedded text layer it
does**, so where that layer's glyph map is broken they are wrong *together*.
Confirmed by rendering the page — `travel/…שיר-לנוסע…` p.16 shows clauses
26.1–26.20 and ordinary commas, while pdftotext and Docling both report
26.0 / 26.9 / 47.84 and render commas as `0`. Marker, which reads the rendered
page, is right.

So parse quality was measured intrinsically, by whether numbered clauses come
out in sequence — no external reference needed:

| arm | clauses in order | clause pairs found |
|---|---|---|
| pdftotext | 34.5% | 9,256 |
| docling (repaired) | 32.6% | 9,268 |
| **markdown** | **40.5%** | **21,115** |

Cleaner in 49 files vs Docling's 5. Markdown also recovers 2.3× as many clause
numbers, i.e. Docling loses much of the numbering outright.

## End-to-end eval (48 questions, 3 judges, per_page + bge-m3 + gpt-oss-120b-low)

| metric | docling | markdown |
|---|---|---|
| correctness | **5.69** | 5.02 |
| completeness | **5.46** | 4.73 |
| hallucination rate | **15%** | 19% |
| citation accuracy | **61%** | 57% |
| refusal rate | **29%** | 31% |

Per question: 5 better, 9 worse, 34 unchanged.

**The two results disagree, and the disagreement is the finding.** Markdown is
the better *parse* and the worse *pipeline* — as configured.

## Verdict

Stay on Docling + BiDi repair. Keep the markdown parser: it fixes a class of
corruption Docling structurally cannot, including the exact clause that broke
`dev-06` (`לא תעלה על 2% מסכום הביטוח`, which Docling renders with the `2%`
stranded at the end — markdown answered it correctly, refusal → 10).

Caveats that keep this from being a clean loss for markdown:
- markdown was handicapped onto `per_page` for comparability; production runs
  `per_table`, which has no markdown equivalent.
- n=48, and 9-vs-5 churn is near the noise floor measured on this harness.
- markdown's losses were **not** missing content — the cited pages are present
  and ~50% *larger*; chunk counts are near-identical (318 vs 321). The
  retrieval mechanism behind those 3 correct→refusal flips is unexplained.

## Next step if pursued

Per-file routing rather than a wholesale swap: use markdown only for documents
whose text layer is provably corrupt (missing ToUnicode fonts + failing the
clause-ordering check — ~49 files), Docling elsewhere. That captures markdown's
wins without its unexplained retrieval losses.

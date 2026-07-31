"""Reference-dataset generation (v2).

Builds `reference_questions_v2.json`: 132 Hebrew customer questions covering
all 12 corpus categories, each written by an LLM from a real corpus page and
accepted only after passing gates that are checkable by machine, not by taste.

The formal spec lives in `refgen.schema`; `refgen.audit` re-checks any emitted
dataset against it. v1 (`reference_questions.json`) is held out as the
validation set and is never merged in — v2 may not reuse its pages or
paraphrase its questions.
"""

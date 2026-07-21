"""Stanza Hebrew normalizer (rag_plan.md §5 stage 4).

``stanza.Pipeline('he', processors='tokenize,mwt,pos,lemma')`` — MWT expansion
strips fused prefixes (בבית → ב+בית) and lemmatization un-fragments Hebrew
BM25. Recipe per word: emit the UNION of lemma + surface form (hedges ~10%
lemma errors on domain terms like הראל); lowercase Latin; keep digits;
normalize gershayim/geresh/maqaf to plain forms; strip nikud + punctuation.

Normalized tokens exist ONLY inside the sparse index — displayed/stored chunk
text is always the original (MWT changes token counts; never mix normalized
offsets with display text).

Failure mode: a Stanza error on a chunk falls back to whitespace+punctuation
tokenization for that chunk (logged; fallback-tokenized docs are NOT cached —
see rag/normalize/cache.py).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from rag.report import EVENT_STANZA_FALLBACK

logger = logging.getLogger(__name__)

#: Hebrew-specific punctuation normalized to plain ASCII forms BEFORE the
#: generic punctuation strip, so acronyms like חו״ל survive as חו"ל tokens.
_PLAIN_FORMS = str.maketrans({
    "״": '"',   # gershayim ״ -> "
    "׳": "'",   # geresh ׳ -> '
    "“": '"', "”": '"', "„": '"',  # curly double quotes
    "‘": "'", "’": "'",                  # curly single quotes
    "־": "-",   # maqaf ־ -> -
    "‐": "-", "‑": "-", "‒": "-",
    "–": "-", "—": "-",                  # unicode dashes
})

#: Nikud + Hebrew cantillation marks (U+0591–U+05C7 covers both).
_NIKUD_RE = re.compile(r"[֑-ׇ]")

#: Punctuation stripped from token EDGES. Internal '"' (acronyms: חו"ל) and
#: internal ',' / '.' in numbers (6,000 / 3.5) are kept.
_EDGE_PUNCT = "".join(
    chr(c) for c in range(0x21, 0x30)
) + ":;<=>?@[\\]^_`{|}~«»…•·"


def normalize_token(raw: str) -> list[str]:
    """Apply the full per-token recipe; hyphenated tokens split into parts
    (Dental-Claim-Form -> dental, claim, form). Returns [] for pure
    punctuation/empty tokens."""
    text = unicodedata.normalize("NFC", raw).translate(_PLAIN_FORMS)
    text = _NIKUD_RE.sub("", text).lower()
    out: list[str] = []
    for part in text.split("-"):
        part = part.strip(_EDGE_PUNCT)
        if part and not all(ch in _EDGE_PUNCT for ch in part):
            out.append(part)
    return out


def whitespace_tokens(text: str) -> list[str]:
    """Fallback tokenization: whitespace split + the same normalization
    recipe (no MWT/lemma — surface forms only)."""
    tokens: list[str] = []
    for raw in text.split():
        tokens.extend(normalize_token(raw))
    return tokens


#: Upper bound on the text length handed to Stanza in one piece. Serialized
#: Markdown tables produce huge punctuation-free "sentences"; Stanza's memory
#: use is superlinear in sentence length — a single 14.5K-char table chunk
#: peaked >8 GB (OOM-killed the ingest) while the same text split into
#: <=8K-char segments peaked at 1.9 GB with a near-identical token stream
#: (723 vs 725 tokens; measured 2026-07-21 on the shirbit travel policy PDF).
#: 4000 keeps a 2x safety margin. BM25 needs a token multiset, not sentence
#: structure, so segment boundaries are harmless.
MAX_SEGMENT_CHARS = 4000

#: Upper bound on the cumulative chars per ``bulk_process`` call. Bulk memory
#: also scales with total batch text: 24 segments / 71K chars in one call made
#: torch attempt a single 3.3 GB allocation (whole-doc batch of the same
#: policy PDF), while a 16K-char call peaked at 1.9 GB including the pipeline.
MAX_BULK_CHARS = 16000


def split_long_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split ``text`` into segments of at most ``max_chars``, preferring line
    boundaries; single lines longer than ``max_chars`` are hard-split at
    whitespace (or mid-token as a last resort)."""
    if len(text) <= max_chars:
        return [text]
    segments: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            segments.append("\n".join(current))
            current, current_len = [], 0

    for line in text.splitlines():
        while len(line) > max_chars:
            cut = line.rfind(" ", 1, max_chars)
            if cut <= 0:
                cut = max_chars
            flush()
            segments.append(line[:cut])
            line = line[cut:].lstrip()
        if current_len + len(line) + 1 > max_chars:
            flush()
        current.append(line)
        current_len += len(line) + 1
    flush()
    return segments or [text]


class StanzaNormalizer:
    """Satisfies the ``Normalizer`` protocol (``tokens``); adds
    ``tokens_batch_ex`` (batched via ``bulk_process``, per-chunk fallback
    flags) used by the token cache to skip caching fallback-tokenized docs."""

    def __init__(
        self,
        package: str = "default",
        index_surface_forms: bool = True,
        **params: Any,
    ) -> None:
        if params:
            raise TypeError(f"Unknown stanza normalizer params: {sorted(params)}")
        self.package = package
        self.index_surface_forms = index_surface_forms
        self._pipeline: Any = None

    # ------------------------------------------------------------------ #
    # Pipeline lifecycle (lazy; downloads the he model on first ever use)
    # ------------------------------------------------------------------ #

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            import stanza

            kwargs = dict(
                lang="he",
                processors="tokenize,mwt,pos,lemma",
                package=self.package,
                verbose=False,
            )
            try:
                self._pipeline = stanza.Pipeline(download_method=None, **kwargs)
            except Exception:
                print(
                    f"[rag.normalize] Stanza Hebrew model (package={self.package!r}) "
                    "not found locally — downloading (~250 MB, one-time)..."
                )
                stanza.download("he", package=self.package, verbose=False)
                self._pipeline = stanza.Pipeline(download_method=None, **kwargs)
        return self._pipeline

    # ------------------------------------------------------------------ #
    # Tokenization
    # ------------------------------------------------------------------ #

    def _doc_tokens(self, doc: Any) -> list[str]:
        """Per (pre-MWT) token: union of each word's lemma + word surface +
        the token-level surface (order-preserving, deduped within the token),
        each passed through the normalize recipe.

        The token-level surface matters: Stanza sometimes MWT-splits domain
        terms (הראל → ה+ראל), so word-level surfaces alone would lose the
        exact form ground-truth queries use — the token surface is the hedge.
        """
        tokens: list[str] = []
        for sentence in doc.sentences:
            for stoken in sentence.tokens:
                candidates = []
                for word in stoken.words:
                    candidates.append(word.lemma or word.text)
                    if self.index_surface_forms:
                        candidates.append(word.text)
                if self.index_surface_forms:
                    candidates.append(stoken.text)
                seen: set[str] = set()
                for candidate in candidates:
                    for token in normalize_token(candidate):
                        if token not in seen:
                            seen.add(token)
                            tokens.append(token)
        return tokens

    def tokens(self, text: str) -> list[str]:
        """Normalizer protocol: single-text tokenization (query side)."""
        return self.tokens_batch_ex([text])[0][0]

    def tokens_batch_ex(self, texts: list[str]) -> list[tuple[list[str], bool]]:
        """Batch tokenization via ``bulk_process``. Returns per text
        ``(tokens, used_fallback)`` — fallback docs must NOT be cached.

        Long texts are pre-split into ``MAX_SEGMENT_CHARS`` segments (memory
        bound — see :data:`MAX_SEGMENT_CHARS`); each text's segment token
        lists are concatenated back, so callers see one list per input text.
        """
        if not texts:
            return []
        pipeline = self._get_pipeline()
        segment_lists = [split_long_text(text) for text in texts]
        flat_segments = [seg for segments in segment_lists for seg in segments]

        # Bulk in sub-batches bounded by MAX_BULK_CHARS (memory guard).
        segment_results: list[tuple[list[str], bool]] = []
        batch: list[str] = []
        batch_chars = 0
        for segment in flat_segments:
            if batch and batch_chars + len(segment) > MAX_BULK_CHARS:
                segment_results.extend(self._bulk_tokens(pipeline, batch))
                batch, batch_chars = [], 0
            batch.append(segment)
            batch_chars += len(segment)
        if batch:
            segment_results.extend(self._bulk_tokens(pipeline, batch))

        # Reassemble: one (tokens, used_fallback) pair per input text.
        results: list[tuple[list[str], bool]] = []
        offset = 0
        for segments in segment_lists:
            per_text = segment_results[offset : offset + len(segments)]
            results.append(
                (
                    [token for tokens, _ in per_text for token in tokens],
                    any(fallback for _, fallback in per_text),
                )
            )
            offset += len(segments)
        return results

    def _bulk_tokens(
        self, pipeline: Any, segments: list[str]
    ) -> list[tuple[list[str], bool]]:
        """Tokenize one bounded batch of segments; bulk failure retries per
        segment, and a per-segment failure falls back to whitespace tokens
        (flagged so the doc is not token-cached)."""
        try:
            docs = pipeline.bulk_process(segments)
            return [(self._doc_tokens(doc), False) for doc in docs]
        except Exception as bulk_exc:
            logger.warning(
                "Stanza bulk_process failed (%s) — retrying per segment", bulk_exc
            )
        results: list[tuple[list[str], bool]] = []
        for segment in segments:
            try:
                results.append((self._doc_tokens(pipeline(segment)), False))
            except Exception as exc:
                logger.warning(
                    "Stanza failed on segment (%.60r): %s — whitespace fallback "
                    "(this chunk will not be token-cached)",
                    segment,
                    exc,
                    extra={
                        "rag_event": EVENT_STANZA_FALLBACK,
                        "rag_detail": {"error": str(exc), "segment_preview": segment[:120]},
                    },
                )
                results.append((whitespace_tokens(segment), True))
        return results

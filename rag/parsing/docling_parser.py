"""Docling PDF adapter (rag_plan.md §5 stage 2).

``DocumentConverter`` -> ``DoclingDocument`` -> ``export_to_dict()`` JSON.
NEVER ``export_to_markdown`` — Markdown destroys ``prov.page_no`` and with it
citation pages (docling discussion #1012; plan rule §1.1).

The first converter use downloads the docling layout/table models (~500 MB)
into the HF cache; CPU parsing is the slow path — budget hours for the full
350-PDF corpus. The read-through parse cache (``rag/parsing/cache.py``) means
each file is parsed once, ever.

Every parse then goes through the BiDi word-order repair
(``rag/parsing/rtl_repair.py``) — Docling emits Hebrew table cells in visual
(backwards) word order, so the repair is part of what "parsed" means here, not
an optional post-step. It is unconditional: reversed text is a defect, never a
mode. ``cache/parsed/`` therefore holds repaired documents; existing caches are
brought forward by ``python -m rag.cli.repair_cache``.
"""
from __future__ import annotations

import logging
from typing import Any

from rag.parsing import ParseError, ParsedDoc, SourceFile
from rag.parsing.rtl_repair import repair_docling

logger = logging.getLogger(__name__)


class DoclingParser:
    """PDF -> ParsedDoc carrying the DoclingDocument dict.

    ``rtl_canary`` / ``canary_sample`` are config params consumed by the
    ingestion CLI's canary orchestration (§5 stage 2); they live on the parser
    so the parser block carries its full config identity.
    """

    def __init__(self, rtl_canary: bool = True, canary_sample: int = 10) -> None:
        self.rtl_canary = rtl_canary
        self.canary_sample = canary_sample
        self._converter: Any = None  # lazy: creating it pulls in heavy imports/models

    def _get_converter(self) -> Any:
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def parse(self, source: SourceFile) -> ParsedDoc:
        if source.kind != "pdf":
            raise ValueError(f"DoclingParser only parses PDFs, got {source.kind}: {source.rel_path}")
        converter = self._get_converter()
        try:
            result = converter.convert(source.abs_path)
        except Exception as exc:  # noqa: BLE001 — per-file failures are recorded, not fatal
            raise ParseError(f"Docling failed on {source.rel_path}: {exc}") from exc
        doc_dict = result.document.export_to_dict()
        stats = repair_docling(doc_dict, source.abs_path)
        if stats.repaired:
            logger.debug(
                "BiDi repair on %s: %d/%d cells, %d/%d text items",
                source.rel_path,
                stats.cells_repaired,
                stats.cells_total,
                stats.texts_repaired,
                stats.texts_total,
            )
        return ParsedDoc(source=source, docling=doc_dict)

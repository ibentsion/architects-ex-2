"""TXT adapter (rag_plan.md §5 stage 2): plain UTF-8 read, ``page=None``.

TXT files are web-page dumps; the ground truth cites them with ``page: null``,
so no page structure is invented. ``source_url`` comes from the discovery
stage's manifest lookup (already on ``SourceFile``).
"""
from __future__ import annotations

from rag.parsing import ParsedDoc, SourceFile


class TxtParser:
    def parse(self, source: SourceFile) -> ParsedDoc:
        if source.kind != "txt":
            raise ValueError(f"TxtParser only parses TXT files, got {source.kind}: {source.rel_path}")
        return ParsedDoc(source=source, text=source.abs_path.read_text(encoding="utf-8"))

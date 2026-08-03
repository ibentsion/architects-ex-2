"""Discovery and loading of the repo's own answer / judgment / question files.

This is what the QA-History view browses. It is an allowlist of known globs,
not a filesystem browser (threat T-260803-02): ``.env``, ``venv/`` and
``rag_index/`` are unreachable because nothing matches them, and
:func:`load_pairs` refuses any id that discovery did not produce.

Read-only, always: nothing here writes to the repo.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from webapi import paths
from webapi.paths import resolve_in_repo
from webapi.schema import SupportPair, record_to_pair

logger = logging.getLogger(__name__)

#: Answer JSONLs written at the repo root by the query CLI / baseline runner.
ROOT_ANSWER_GLOBS = ("rag_answers*.jsonl", "team_1_results*.jsonl", "baseline_answers.jsonl")
#: Answer JSONLs written by the eval harness (these are the ones with traces).
EVAL_ANSWER_GLOB = "eval_results/**/answers/*.jsonl"
#: Judged runs. Nested up to depth 3, so the walk has to be recursive.
JUDGMENTS_GLOB = "eval_results/**/judgments.jsonl"

#: Question sources, in precedence order. v1/v2/v3 ids are disjoint; the
#: validation/holdout files are re-splits of v1+v2 and collide with identical
#: content, so first-wins is safe and silent.
QUESTION_SOURCES = (
    "reference_questions.json",
    "reference_questions_v2.json",
    "reference_questions_v3.json",
    "ref_q_validation_set_v1.jsonl",
    "ref_q_holdout_set_v1.jsonl",
)


class UnknownDataset(ValueError):
    """A dataset id that discovery did not produce."""


class DatasetInfo(BaseModel):
    """One selectable entry in the History view's picker."""

    id: str  # repo-relative POSIX path — also the load_pairs key
    label: str
    kind: Literal["answers", "judged"]
    n_pairs: int
    has_trace: bool
    has_judgment: bool
    questions_file: str | None = None


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file, skipping (and counting) lines that are not objects.

    These files are produced by long eval runs that can be killed mid-write, so
    a truncated last line is expected, not exceptional.
    """
    records: list[dict[str, Any]] = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue
            records.append(record)
    if skipped:
        logger.warning("%s: skipped %d malformed line(s)", path, skipped)
    return records


def _load_question_records(path: Path) -> list[dict[str, Any]]:
    """Question sources come in three shapes: ``{"questions": [...]}``, a bare
    JSON list, and JSONL."""
    if path.suffix == ".jsonl":
        return _iter_jsonl(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("%s: unreadable question source (%s)", path, exc)
        return []
    if isinstance(raw, dict):
        raw = raw.get("questions") or []
    return [r for r in raw if isinstance(r, dict) and r.get("id")]


def _count_lines(path: Path) -> int:
    """Non-empty line count without decoding 8 MB of Hebrew JSON."""
    count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            count += chunk.count(b"\n")
        handle.seek(0, 2)
        if handle.tell():
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                count += 1  # unterminated last line still holds a record
    return count


def _first_record(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                return record
    return {}


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #


class QuestionIndex:
    """Flat ``id -> question record`` across every reference set in the repo.

    NOT ONE answer JSONL carries its question text, so this join is the only
    way the History view can show what was asked.
    """

    def __init__(self, by_id: dict[str, dict[str, Any]]) -> None:
        self.by_id = by_id

    @classmethod
    def load(cls) -> QuestionIndex:
        by_id: dict[str, dict[str, Any]] = {}
        for name in QUESTION_SOURCES:
            path = paths.REPO_ROOT / name
            if not path.is_file():
                continue
            for record in _load_question_records(path):
                by_id.setdefault(str(record["id"]), record)
        return cls(by_id)

    def get(self, question_id: str) -> dict[str, Any] | None:
        return self.by_id.get(question_id)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _discover_answer_files() -> list[Path]:
    root = paths.REPO_ROOT
    found: list[Path] = []
    for pattern in ROOT_ANSWER_GLOBS:
        found.extend(p for p in root.glob(pattern) if p.is_file())
    found.extend(p for p in root.glob(EVAL_ANSWER_GLOB) if p.is_file())
    return sorted(set(found))


def _relative(path: Path) -> str:
    return path.relative_to(paths.REPO_ROOT).as_posix()


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def discover_datasets() -> list[DatasetInfo]:
    """Every answer file and judged run in the repo, newest first."""
    answer_files = _discover_answer_files()
    answers_by_name = {path.name: path for path in answer_files}

    datasets: list[tuple[float, DatasetInfo]] = []
    for path in answer_files:
        datasets.append((
            _mtime(path),
            DatasetInfo(
                id=_relative(path),
                label=path.stem,
                kind="answers",
                n_pairs=_count_lines(path),
                has_trace=bool(_first_record(path).get("trace")),
                has_judgment=False,
            ),
        ))

    for path in sorted(paths.REPO_ROOT.glob(JUDGMENTS_GLOB)):
        if not path.is_file():
            continue
        meta = _run_meta(path.parent)
        # `answers_file` is often an absolute path from another machine, or a
        # name that no longer exists. Join by basename against what we found;
        # never open the recorded path.
        recorded = meta.get("answers_file")
        answers_path = answers_by_name.get(Path(recorded).name) if recorded else None
        datasets.append((
            _mtime(path),
            DatasetInfo(
                id=_relative(path),
                label=_relative(path.parent),
                kind="judged",
                n_pairs=_count_lines(path),
                has_trace=bool(_first_record(answers_path).get("trace")) if answers_path else False,
                has_judgment=True,
                questions_file=meta.get("questions_file"),
            ),
        ))

    datasets.sort(key=lambda item: item[0], reverse=True)
    return [info for _mt, info in datasets]


def _run_meta(run_dir: Path) -> dict[str, Any]:
    metrics = run_dir / "metrics.json"
    if not metrics.is_file():
        return {}
    try:
        return json.loads(metrics.read_text(encoding="utf-8")).get("meta") or {}
    except (json.JSONDecodeError, OSError, AttributeError) as exc:
        logger.warning("%s: unreadable metrics.json (%s)", metrics, exc)
        return {}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _pair_id(record: dict[str, Any], stem: str, lineno: int) -> str:
    """Records normally carry an id; an arbitrary JSONL might not, and a pair
    still needs a stable handle for the UI's selection state."""
    return str(record.get("id") or f"{stem}#{lineno}")


def load_pairs(
    dataset_id: str, *, limit: int = 200, offset: int = 0
) -> tuple[int, list[SupportPair]]:
    """Load one dataset as wire pairs. Returns ``(total, page)``.

    Raises :class:`~webapi.paths.PathEscape` for a path outside the repo and
    :class:`UnknownDataset` for anything discovery did not offer.
    """
    path = resolve_in_repo(dataset_id)  # traversal dies here, before the allowlist
    info = next((d for d in discover_datasets() if d.id == dataset_id), None)
    if info is None:
        raise UnknownDataset(f"not a known dataset: {dataset_id!r}")

    questions = QuestionIndex.load()
    records = _iter_jsonl(path)
    judged = info.kind == "judged"
    answers_by_id = _judged_answers(path.parent) if judged else {}
    stem = path.parent.name if judged else path.stem

    pairs = []
    for lineno, record in enumerate(records[offset : offset + limit], start=offset + 1):
        pair_id = _pair_id(record, stem, lineno)
        pairs.append(
            record_to_pair(
                answers_by_id.get(pair_id, {}) if judged else record,
                pair_id=pair_id,
                question_record=questions.get(pair_id),
                judgment=record if judged else None,
            )
        )
    return len(records), pairs


def _judged_answers(run_dir: Path) -> dict[str, dict[str, Any]]:
    """The answers a judged run graded, joined by basename (see
    :func:`discover_datasets`). Empty when unresolvable — the run then renders
    judgments-only rather than failing."""
    recorded = _run_meta(run_dir).get("answers_file")
    if not recorded:
        return {}
    name = Path(recorded).name
    path = next((p for p in _discover_answer_files() if p.name == name), None)
    if path is None:
        logger.info("%s: answers file %s not in the repo — judgments only", run_dir, name)
        return {}
    return {
        _pair_id(record, path.stem, lineno): record
        for lineno, record in enumerate(_iter_jsonl(path), start=1)
    }

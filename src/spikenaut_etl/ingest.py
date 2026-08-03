"""Readers for the legacy free-form JSONL sources.

These files predate the versioned ``TelemetryEnvelope`` in
``rmems/Theseus-Quarry``. They stream, because ``neuromorphic_data.jsonl`` is
395 MB and holding it as a list of dicts costs several GB.

Parse failures are counted and reported, never skipped in silence.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

# Preserves the caller's concrete model type through read_validated. Returning
# the bare BaseModel would erase it, and every downstream `record.telemetry` /
# `record.timestamp` access would go unchecked -- which is a poor look for a
# pipeline whose whole argument is that declared schemas catch field errors.
M = TypeVar("M", bound=BaseModel)


@dataclass
class IngestStats:
    source: str
    n_lines: int = 0
    n_parsed: int = 0
    n_blank: int = 0
    n_json_errors: int = 0
    n_schema_errors: int = 0
    examples: list[str] = field(default_factory=list)

    def note_error(self, row: int, message: str) -> None:
        if len(self.examples) < 5:
            self.examples.append(f"row {row}: {message}")

    def render(self) -> str:
        parts = [f"{self.source}: {self.n_parsed}/{self.n_lines} parsed"]
        if self.n_blank:
            parts.append(f"{self.n_blank} blank")
        if self.n_json_errors:
            parts.append(f"{self.n_json_errors} malformed JSON")
        if self.n_schema_errors:
            parts.append(f"{self.n_schema_errors} schema violations")
        out = ", ".join(parts)
        if self.examples:
            out += "\n" + "\n".join(f"    {e}" for e in self.examples)
        return out


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(row_index, object)`` for each non-blank line. 0-indexed."""
    with path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            yield row, json.loads(line)


def read_validated(
    path: Path, model: type[M], stats: IngestStats
) -> Iterator[tuple[int, M]]:
    """Stream a source through its declared schema.

    ``extra="forbid"`` means an unexpected field raises here rather than being
    dropped downstream. That is the point: schema drift should be a decision, not
    an accident.
    """
    with path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            stats.n_lines += 1
            line = line.strip()
            if not line:
                stats.n_blank += 1
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                stats.n_json_errors += 1
                stats.note_error(row, f"malformed JSON: {exc}")
                continue
            try:
                record = model.model_validate(payload)
            except PydanticValidationError as exc:
                stats.n_schema_errors += 1
                stats.note_error(row, f"schema violation: {_terse(exc)}")
                continue
            stats.n_parsed += 1
            yield row, record


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _terse(exc: PydanticValidationError) -> str:
    errors = exc.errors()
    if not errors:  # pragma: no cover - defensive
        return str(exc)
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    return f"{loc}: {first.get('msg', '?')}" + (
        f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
    )

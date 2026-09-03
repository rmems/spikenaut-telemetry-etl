"""Readers for legacy free-form JSONL and Theseus-Quarry schema-v1 envelopes.

Legacy files predate the versioned ``TelemetryEnvelope`` in
``rmems/Theseus-Quarry``. They stream, because ``neuromorphic_data.jsonl`` is
395 MB and holding it as a list of dicts costs several GB.

Dispatch is per line on ``schema_version``:

* ``schema_version == 1`` → ``TelemetryEnvelope`` (tagged ``payload``).
* any other ``schema_version`` → raise; this reader does not guess.
* a v1-shaped line with no ``schema_version`` → raise.
* otherwise → the caller's legacy model.

Empty telemetry / empty payloads, unknown versions, unknown payload tags, and
extra fields all raise :class:`IngestError`. There is no warn-and-continue path
for those failures: a skipped v1 line is how a next collection run would
silently become empty published rows.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from .schemas import COLLECTOR_SCHEMA_VERSION, TelemetryEnvelope

# Preserves the caller's concrete model type through read_validated. Returning
# the bare BaseModel would erase it, and every downstream `record.telemetry` /
# `record.timestamp` access would go unchecked -- which is a poor look for a
# pipeline whose whole argument is that declared schemas catch field errors.
M = TypeVar("M", bound=BaseModel)

# Distinctive Theseus-Quarry envelope keys. Present together without
# schema_version means a v1-shaped line that forgot to declare its version.
_V1_SHAPE_KEYS = frozenset({"kind", "payload", "stem", "source"})


class IngestError(Exception):
    """Raised when a source line must not be ingested. Output must not be written."""


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
) -> Iterator[tuple[int, M | TelemetryEnvelope]]:
    """Stream a source through its declared schema, or schema v1.

    ``extra="forbid"`` means an unexpected field raises here rather than being
    dropped downstream. Unknown ``schema_version``, a v1-shaped line missing
    ``schema_version``, a wrong payload tag, extra fields, and an empty
    telemetry / payload object all raise :class:`IngestError` immediately.
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
                raise IngestError(
                    f"{path.name}:{row}: malformed JSON: {exc}; refusing to ingest"
                ) from exc
            if not isinstance(payload, dict):
                stats.n_schema_errors += 1
                raise IngestError(
                    f"{path.name}:{row}: JSONL line must be an object; refusing to ingest"
                )
            try:
                record = parse_record(payload, model, row=row, source=path.name)
            except IngestError as exc:
                if "schema violation" in str(exc):
                    stats.n_schema_errors += 1
                    stats.note_error(row, str(exc))
                raise
            stats.n_parsed += 1
            yield row, record


def parse_record(
    payload: dict[str, Any],
    model: type[M],
    *,
    row: int,
    source: str,
) -> M | TelemetryEnvelope:
    """Dispatch one decoded object to schema v1 or the legacy model."""
    if "schema_version" in payload:
        version = payload["schema_version"]
        if version != COLLECTOR_SCHEMA_VERSION:
            raise IngestError(
                f"{source}:{row}: unknown schema_version {version!r}; "
                f"this reader implements Theseus-Quarry schema v"
                f"{COLLECTOR_SCHEMA_VERSION} only and will not guess"
            )
        try:
            envelope = TelemetryEnvelope.model_validate(payload)
        except PydanticValidationError as exc:
            raise IngestError(
                f"{source}:{row}: schema violation: {_terse(exc)}; "
                "refusing to ingest"
            ) from exc
        _reject_empty_v1(envelope, row=row, source=source)
        return envelope

    if _v1_shaped(payload):
        raise IngestError(
            f"{source}:{row}: v1-shaped envelope is missing schema_version; "
            "refusing to guess"
        )

    try:
        record = model.model_validate(payload)
    except PydanticValidationError as exc:
        raise IngestError(
            f"{source}:{row}: schema violation: {_terse(exc)}; refusing to ingest"
        ) from exc
    _reject_empty_legacy(record, row=row, source=source)
    return record


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _v1_shaped(payload: dict[str, Any]) -> bool:
    return _V1_SHAPE_KEYS <= payload.keys()


def _reject_empty_legacy(record: BaseModel, *, row: int, source: str) -> None:
    telemetry = getattr(record, "telemetry", None)
    if telemetry is None:
        return
    values = telemetry.model_dump()
    if not values or all(v is None for v in values.values()):
        raise IngestError(
            f"{source}:{row}: empty telemetry payload; refusing to ingest"
        )


def _reject_empty_v1(record: TelemetryEnvelope, *, row: int, source: str) -> None:
    values = record.payload.model_dump()
    data = {k: v for k, v in values.items() if k != "type"}
    if not data or all(v is None for v in data.values()):
        raise IngestError(
            f"{source}:{row}: empty schema-v1 payload; refusing to ingest"
        )


def _terse(exc: PydanticValidationError) -> str:
    errors = exc.errors()
    if not errors:  # pragma: no cover - defensive
        return str(exc)
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    return f"{loc}: {first.get('msg', '?')}" + (
        f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
    )

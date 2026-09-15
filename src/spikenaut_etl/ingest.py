"""Readers for legacy free-form JSONL, published Vault Clean* JSONL, and
Theseus-Quarry schema-v1 envelopes.

Legacy files predate the versioned ``TelemetryEnvelope`` in
``rmems/Theseus-Quarry``. They stream, because ``neuromorphic_data.jsonl`` is
395 MB and holding it as a list of dicts costs several GB.

Dispatch is per line on ``schema_version``, then on shape:

* ``schema_version == 1`` → ``TelemetryEnvelope`` (tagged ``payload``).
* any other ``schema_version`` → raise; this reader does not guess.
* a v1-shaped line with no ``schema_version`` → raise.
* nested ``telemetry`` → the caller's legacy ``Raw*`` model.
* published Clean GPU / sparse CleanNodeSync / CleanQubicTick → re-validated
  without filling missing sensors (typed-null timestamps on coin-tagged
  harvest rows are kept, never filled).
* LIVE_COLUMNS (``sm_clock_mhz`` without ``gpu_clock_mhz``) → named contract
  unless ``--profile live-columns``. Never invent ``sm_clock_mhz`` from
  ``gpu_clock_mhz``.

Empty telemetry / empty payloads, unknown versions, unknown payload tags, and
extra fields all raise :class:`IngestError`. There is no warn-and-continue path
for those failures: a skipped v1 line is how a next collection run would
silently become empty published rows.

Malformed JSON is the exception: a truncated trailing line is ordinary for an
append-only writer. Those lines are counted and skipped so the rest of the
file can still be ingested.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from .contracts import (
    CONTRACT_LIVE_COLUMNS_REQUIRED,
    CONTRACT_MIXED_SHAPES,
    LiveColumnsRecord,
    PublishedNodeSync,
    PublishedQubicTick,
    ShapeProfile,
    live_columns_required_message,
    looks_like_clean_gpu,
    looks_like_clean_node_sync,
    looks_like_clean_qubic,
    looks_like_live_columns,
    looks_like_nested_raw,
    mixed_shapes_message,
)
from .schemas import (
    COLLECTOR_SCHEMA_VERSION,
    CleanGpuTelemetry,
    CleanNodeSync,
    CleanQubicTick,
    RawGpuRecord,
    RawNodeSyncRecord,
    RawQubicTick,
    TelemetryEnvelope,
)

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

    def __init__(self, message: str, *, kind: str = "ingest") -> None:
        super().__init__(message)
        self.kind = kind


class ContractError(IngestError):
    """Named producer/consumer contract mismatch. Output must not be written."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(f"[{code}] {message}", kind="contract")
        self.code = code


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


def read_validated(
    path: Path, model: type[M], stats: IngestStats
) -> Iterator[tuple[int, BaseModel]]:
    """Stream a source through its declared schema, or schema v1.

    ``extra="forbid"`` means an unexpected field raises here rather than being
    dropped downstream. Unknown ``schema_version``, a v1-shaped line missing
    ``schema_version``, a wrong payload tag, extra fields, and an empty
    telemetry / payload object all raise :class:`IngestError` immediately.
    A ``JSONDecodeError`` is counted and skipped so one truncated append does
    not discard the rest of the file.
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
            if not isinstance(payload, dict):
                stats.n_schema_errors += 1
                stats.note_error(row, "JSONL line must be an object")
                raise IngestError(
                    f"{path.name}:{row}: JSONL line must be an object; "
                    "refusing to ingest",
                    kind="schema",
                )
            try:
                record = parse_record(payload, model, row=row, source=path.name)
            except IngestError as exc:
                if exc.kind == "schema":
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
) -> BaseModel:
    """Dispatch one decoded object to schema v1 or the legacy model."""
    if "schema_version" in payload:
        version = payload["schema_version"]
        # bool is an int subclass; 1.0 == 1. Refuse both rather than coerce.
        if type(version) is not int or version != COLLECTOR_SCHEMA_VERSION:
            raise IngestError(
                f"{source}:{row}: unknown schema_version {version!r}; "
                f"this reader implements Theseus-Quarry schema v"
                f"{COLLECTOR_SCHEMA_VERSION} only and will not guess",
                kind="version",
            )
        try:
            envelope = TelemetryEnvelope.model_validate(payload)
        except PydanticValidationError as exc:
            raise IngestError(
                f"{source}:{row}: schema violation: {_terse(exc)}; refusing to ingest",
                kind="schema",
            ) from exc
        _reject_empty_v1(envelope, row=row, source=source)
        return envelope

    if _v1_shaped(payload):
        raise IngestError(
            f"{source}:{row}: v1-shaped envelope is missing schema_version; "
            "refusing to guess",
            kind="version",
        )

    record = _parse_legacy(payload, model, row=row, source=source)
    _reject_empty_legacy(record, row=row, source=source)
    return record


def _parse_legacy(
    payload: dict[str, Any],
    model: type[M],
    *,
    row: int,
    source: str,
) -> BaseModel:
    """Legacy collector JSONL, or the published Clean* / LIVE_COLUMNS shapes."""
    if model is RawGpuRecord:
        return _parse_gpu_payload(payload, row=row, source=source)
    if model is RawNodeSyncRecord:
        return _parse_node_sync_payload(payload, row=row, source=source)
    if model is RawQubicTick:
        return _parse_qubic_payload(payload, row=row, source=source)
    return _validate_model(payload, model, row=row, source=source)


def _parse_gpu_payload(payload: dict[str, Any], *, row: int, source: str) -> BaseModel:
    if looks_like_nested_raw(payload):
        return _validate_model(payload, RawGpuRecord, row=row, source=source)
    if looks_like_live_columns(payload):
        try:
            return LiveColumnsRecord.model_validate(payload)
        except PydanticValidationError as exc:
            raise ContractError(
                live_columns_required_message(source=source, row=row)
                + f" schema: {_terse(exc)}",
                code=CONTRACT_LIVE_COLUMNS_REQUIRED,
            ) from exc
    if looks_like_clean_gpu(payload):
        return _validate_model(payload, CleanGpuTelemetry, row=row, source=source)
    return _validate_model(payload, RawGpuRecord, row=row, source=source)


def _parse_node_sync_payload(
    payload: dict[str, Any], *, row: int, source: str
) -> BaseModel:
    if looks_like_nested_raw(payload):
        return _validate_model(payload, RawNodeSyncRecord, row=row, source=source)
    if looks_like_clean_node_sync(payload):
        return _validate_model(payload, PublishedNodeSync, row=row, source=source)
    return _validate_model(payload, RawNodeSyncRecord, row=row, source=source)


def _parse_qubic_payload(payload: dict[str, Any], *, row: int, source: str) -> BaseModel:
    if looks_like_clean_qubic(payload):
        return _validate_model(payload, PublishedQubicTick, row=row, source=source)
    return _validate_model(payload, RawQubicTick, row=row, source=source)


def _validate_model(
    payload: dict[str, Any],
    model: type[M],
    *,
    row: int,
    source: str,
) -> M:
    try:
        return model.model_validate(payload)
    except PydanticValidationError as exc:
        raise IngestError(
            f"{source}:{row}: schema violation: {_terse(exc)}; refusing to ingest",
            kind="schema",
        ) from exc


def shape_of(record: BaseModel) -> ShapeProfile:
    """Classify one parsed record as raw, published, or live-columns."""
    if isinstance(record, LiveColumnsRecord):
        return "live-columns"
    if isinstance(
        record,
        (
            CleanGpuTelemetry,
            CleanNodeSync,
            CleanQubicTick,
            PublishedNodeSync,
            PublishedQubicTick,
        ),
    ):
        return "published"
    return "raw"


def require_single_shape(
    *, source: str, row: int, previous: ShapeProfile | None, current: ShapeProfile
) -> ShapeProfile:
    """Refuse mixed collector / published / LIVE_COLUMNS rows in one file."""
    if previous is None:
        return current
    if previous != current:
        raise ContractError(
            mixed_shapes_message(
                source=source, row=row, previous=previous, current=current
            ),
            code=CONTRACT_MIXED_SHAPES,
        )
    return previous


def _v1_shaped(payload: dict[str, Any]) -> bool:
    return _V1_SHAPE_KEYS <= payload.keys()


def _reject_empty_legacy(record: BaseModel, *, row: int, source: str) -> None:
    telemetry = getattr(record, "telemetry", None)
    if telemetry is None:
        return
    values = telemetry.model_dump()
    if not values or all(v is None for v in values.values()):
        raise IngestError(
            f"{source}:{row}: empty telemetry payload; refusing to ingest",
            kind="empty",
        )


def _reject_empty_v1(record: TelemetryEnvelope, *, row: int, source: str) -> None:
    values = record.payload.model_dump()
    data = {k: v for k, v in values.items() if k != "type"}
    if not data or all(v is None for v in data.values()):
        raise IngestError(
            f"{source}:{row}: empty schema-v1 payload; refusing to ingest",
            kind="empty",
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

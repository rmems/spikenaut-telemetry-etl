"""Ingest helpers for ``rmems/gaming-telemetry`` (``system_telemetry_v1``).

Game-blind on purpose: the producer is multi-title hardware sensors only.
``session_label`` is capture-session hygiene for gating and splits, never a
Spikenaut axon. This mill does not fold a play session into mining Hub
``v3/state_telemetry`` episodes (``gpu-000000`` .. ``gpu-000198``).

Time is the source ``timestamp_ms`` clock. ``ts_utc`` is a millisecond-to-UTC
unit conversion, not a fabricated timestamp.

Layout (any one of):

* ``system_telemetry_v1.jsonl`` / ``.parquet``
* ``system_telemetry_v1/`` containing those files
* Hub checkout: ``{session}/train-*.parquet`` or ``{session}/train/*.parquet``
* a single session directory holding ``train-*.parquet``

One config is one play session. Multiple session shards in one input raise.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .ingest import IngestError, IngestStats, parse_record
from .schemas import CleanSystemTelemetry, RawSystemTelemetry, TelemetryEnvelope

try:
    import pyarrow.parquet as pq
except ImportError:  # optional extra; JSONL ingest stays pydantic-only
    pq = None

SOURCE_KEY = "system_telemetry_v1"
SOURCE_JSONL = "system_telemetry_v1.jsonl"
SOURCE_PARQUET = "system_telemetry_v1.parquet"

_SENSOR_COLUMNS = frozenset(RawSystemTelemetry.model_fields) - {
    "timestamp_ms",
    "session_label",
}


def discover_input(root: Path) -> Path | None:
    """Find a system_telemetry_v1 file or Hub-style checkout under ``root``."""
    for name in (SOURCE_JSONL, SOURCE_PARQUET):
        candidate = root / name
        if candidate.exists():
            return candidate
    explicit_dir = root / SOURCE_KEY
    if explicit_dir.is_dir():
        return explicit_dir
    if _hub_session_parquets(root) or _session_dir_parquets(root):
        return root
    return None


def iter_source_files(path: Path) -> list[Path]:
    """Resolve ``path`` to one session's parquet/JSONL files.

    Multiple Hub configs in one directory are refused: one config = one session.
    """
    if path.is_file():
        _require_supported_file(path)
        return [path]
    if not path.is_dir():
        raise IngestError(f"{path}: not a file or directory; refusing to ingest")

    explicit = [
        p
        for p in (path / SOURCE_JSONL, path / SOURCE_PARQUET)
        if p.exists() and p.is_file()
    ]
    if len(explicit) > 1:
        raise IngestError(
            f"{path}: both {SOURCE_JSONL} and {SOURCE_PARQUET} present; "
            "refusing to concatenate duplicate captures"
        )
    if explicit:
        return explicit

    grouped = _group_session_files(path)
    if not grouped:
        raise IngestError(
            f"{path}: no system_telemetry_v1 JSONL or parquet found; "
            "refusing to invent a path"
        )
    if len(grouped) > 1:
        sessions = ", ".join(sorted(grouped))
        raise IngestError(
            f"{path}: multiple play sessions in one input ({sessions}); "
            "one config = one session — pass a single session directory"
        )
    session_files = next(iter(grouped.values()))
    return sorted(session_files)


SystemTelemetryRecord = RawSystemTelemetry | CleanSystemTelemetry


def read_records(
    path: Path, stats: IngestStats
) -> Iterator[tuple[int, SystemTelemetryRecord]]:
    """Yield ``(row_index, record)`` from a file or session directory."""
    files = iter_source_files(path)
    row_index = 0
    for file_path in files:
        for record in _read_file(file_path, stats, row_index):
            yield row_index, record
            row_index += 1


def reject_empty_sensors(record: SystemTelemetryRecord, *, row: int, source: str) -> None:
    values = record.model_dump()
    sensors = {k: values[k] for k in _SENSOR_COLUMNS if k in values}
    if not sensors or all(v is None for v in sensors.values()):
        raise IngestError(
            f"{source}:{row}: empty system_telemetry payload; refusing to ingest",
            kind="empty",
        )


def _hub_session_parquets(root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for pattern in ("*/train-*.parquet", "*/train/*.parquet"):
        for file_path in root.glob(pattern):
            if not file_path.is_file():
                continue
            session = file_path.parent.name
            if session == "train":
                session = file_path.parent.parent.name
            grouped.setdefault(session, []).append(file_path)
    return grouped


def _session_dir_parquets(root: Path) -> list[Path]:
    files = [p for p in root.glob("train-*.parquet") if p.is_file()]
    train_dir = root / "train"
    if train_dir.is_dir():
        files.extend(p for p in train_dir.glob("*.parquet") if p.is_file())
    return files


def _group_session_files(path: Path) -> dict[str, list[Path]]:
    grouped = _hub_session_parquets(path)
    if grouped:
        return grouped
    session_files = _session_dir_parquets(path)
    if session_files:
        return {path.name: session_files}
    loose = [
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".jsonl", ".parquet"}
    ]
    if loose:
        return {path.name: loose}
    return {}


def _require_supported_file(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in {".jsonl", ".parquet"}:
        raise IngestError(
            f"{path.name}: unsupported system_telemetry_v1 suffix {suffix!r}; "
            "expected .jsonl or .parquet"
        )


def _read_file(
    path: Path, stats: IngestStats, row_offset: int
) -> Iterator[SystemTelemetryRecord]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _read_jsonl(path, stats, row_offset)
        return
    if suffix == ".parquet":
        yield from _read_parquet(path, stats, row_offset)
        return
    raise IngestError(f"{path.name}: unsupported system_telemetry_v1 suffix {suffix!r}")


def _read_jsonl(
    path: Path, stats: IngestStats, row_offset: int
) -> Iterator[SystemTelemetryRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for local_row, line in enumerate(handle):
            stats.n_lines += 1
            line = line.strip()
            if not line:
                stats.n_blank += 1
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                stats.n_json_errors += 1
                stats.note_error(row_offset + local_row, f"malformed JSON: {exc}")
                continue
            if not isinstance(payload, dict):
                stats.n_schema_errors += 1
                raise IngestError(
                    f"{path.name}:{local_row}: JSONL line must be an object; "
                    "refusing to ingest",
                    kind="schema",
                )
            yield _parse_row(payload, row=row_offset + local_row, source=path.name)
            stats.n_parsed += 1


def _read_parquet(
    path: Path, stats: IngestStats, row_offset: int
) -> Iterator[SystemTelemetryRecord]:
    if pq is None:
        raise IngestError(
            f"{path.name}: parquet ingest needs pyarrow; install "
            "spikenaut-telemetry-etl[dev] or [v3]"
        )
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise IngestError(f"{path.name}: cannot read parquet: {exc}") from exc
    for local_row, raw in enumerate(table.to_pylist()):
        stats.n_lines += 1
        payload = {key: _native(value) for key, value in raw.items()}
        yield _parse_row(payload, row=row_offset + local_row, source=path.name)
        stats.n_parsed += 1


def _parse_row(
    payload: dict[str, Any], *, row: int, source: str
) -> SystemTelemetryRecord:
    if "schema_version" in payload or _v1_shaped(payload):
        raise IngestError(
            f"{source}:{row}: Theseus-Quarry schema v1 envelope cannot be read "
            "as system_telemetry_v1; refusing to invent a mapping",
            kind="schema",
        )
    if "ts_utc" in payload:
        record = parse_record(payload, CleanSystemTelemetry, row=row, source=source)
    else:
        record = parse_record(payload, RawSystemTelemetry, row=row, source=source)
    if isinstance(record, TelemetryEnvelope):
        raise IngestError(
            f"{source}:{row}: Theseus-Quarry schema v1 envelope cannot be read "
            "as system_telemetry_v1; refusing to invent a mapping",
            kind="schema",
        )
    if isinstance(record, (RawSystemTelemetry, CleanSystemTelemetry)):
        return record
    raise IngestError(
        f"{source}:{row}: unexpected record type {type(record).__name__}; "
        "refusing to ingest as system_telemetry_v1",
        kind="schema",
    )


def _v1_shaped(payload: dict[str, Any]) -> bool:
    return {"kind", "payload", "stem", "source"} <= payload.keys()


def _native(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

"""Prepare completed gaming-telemetry sessions for anticipation experiments.

The transformation is deliberately strict: campaign assignments are immutable,
collector completion is proven by its sidecar, and every model example is built
from one uninterrupted causal segment.  Invalid frames remain in the output so
stateful consumers can observe the gap and reset at the following segment.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import directory_identity
from .artifacts import write_json as _write_json

SCHEMA_VERSION = "anticipation-prepared-v1"
FRAME_INTERVAL_MS = 100
MAX_SOURCE_AGE_MS = 200
MAX_TARGET_LATENESS_MS = 100
MAX_FRAME_COUNT = 100_000
HISTORY_OFFSETS_MS = (500, 1_000, 2_000, 5_000)
TARGET_OFFSETS_MS = (1_000, 5_000)
VALID_SPLITS = {"train", "validation", "test"}

FEATURE_MAP_ID = "anticipation-observed-gpu-v1"
FEATURE_MAP = [
    {"name": "vram_used", "unit": "MiB", "source": "memory_used_mb"},
    {"name": "gpu_power", "unit": "W", "source": "power_usage_mw"},
    {"name": "gpu_temperature", "unit": "C", "source": "temperature_c"},
    {"name": "graphics_clock", "unit": "MHz", "source": "graphics_clock_mhz"},
    {"name": "memory_clock", "unit": "MHz", "source": "memory_clock_mhz"},
]
TARGET_NAMES = [
    "temperature_delta_1s_c",
    "power_delta_1s_w",
    "temperature_delta_5s_c",
    "power_delta_5s_w",
]
PARQUET_COLUMNS = [
    "timestamp_ms",
    "session_label",
    *(item["source"] for item in FEATURE_MAP),
]
_BATCH_PATTERN = re.compile(
    r"^(?:gpu_telemetry_v2_|telemetry_)batch_([0-9]{1,10})\.parquet$"
)


class PreparationError(ValueError):
    """The campaign cannot be prepared without weakening its declared contract."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreparationError(f"{context} must be a JSON object")
    return value


def _load_campaign(
    campaign_path: Path, campaign_bytes: bytes
) -> tuple[dict[str, Any], int, str]:
    try:
        campaign_raw = json.loads(campaign_bytes.decode())
    except (OSError, ValueError, RecursionError) as exc:
        raise PreparationError(f"cannot read campaign {campaign_path}: {exc}") from exc
    campaign = _require_mapping(campaign_raw, "campaign")
    minimum = campaign.get("min_examples_per_session")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise PreparationError(
            "campaign min_examples_per_session must be a positive integer"
        )
    sessions = campaign.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise PreparationError("campaign sessions must be a non-empty list")
    seen: set[str] = set()
    for position, raw in enumerate(sessions):
        item = _require_mapping(raw, f"sessions[{position}]")
        required = {"session_id", "split", "path", "seed"}
        missing = sorted(required - item.keys())
        if missing:
            raise PreparationError(f"sessions[{position}] missing {', '.join(missing)}")
        session_id = item["session_id"]
        if not isinstance(session_id, str) or not session_id.strip():
            raise PreparationError(f"sessions[{position}].session_id must be non-empty")
        if session_id in seen:
            raise PreparationError(f"duplicate session_id {session_id}")
        seen.add(session_id)
        if not isinstance(item["split"], str) or item["split"] not in VALID_SPLITS:
            raise PreparationError(
                f"session {session_id} split must be train, validation, or test"
            )
        if not isinstance(item["seed"], int) or isinstance(item["seed"], bool):
            raise PreparationError(f"session {session_id} seed must be an integer")
        if not isinstance(item["path"], str) or not item["path"]:
            raise PreparationError(
                f"session {session_id} path must be a non-empty string"
            )
        if "\x00" in item["path"]:
            raise PreparationError(f"session {session_id} path contains a null byte")
    if not any(item["split"] == "train" for item in sessions):
        raise PreparationError("campaign needs at least one training session")
    return campaign, minimum, hashlib.sha256(campaign_bytes).hexdigest()


def _load_manifest(
    session_path: Path, expected_id: str
) -> tuple[dict[str, Any], Path, str]:
    manifest_path = session_path / "session_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_raw = json.loads(manifest_bytes.decode())
    except (OSError, ValueError, RecursionError) as exc:
        raise PreparationError(
            f"session {expected_id} manifest unavailable: {exc}"
        ) from exc
    manifest = _require_mapping(manifest_raw, str(manifest_path))
    schema_version = manifest.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise PreparationError(f"session {expected_id} manifest schema_version must be 1")
    if not isinstance(manifest.get("session_id"), str) or not manifest["session_id"]:
        raise PreparationError(f"session {expected_id} manifest session_id is incomplete")
    workload = manifest.get("workload")
    if (
        manifest.get("session_label") != expected_id
        or not isinstance(workload, dict)
        or workload.get("label") != expected_id
    ):
        raise PreparationError(
            f"session_id label mismatch: campaign has {expected_id}, "
            "manifest labels do not"
        )
    ended_at = manifest.get("ended_at_utc")
    if not isinstance(ended_at, str) or not ended_at:
        raise PreparationError(
            f"session {expected_id} ended_at_utc must be a valid UTC timestamp"
        )
    try:
        ended_at_parsed = datetime.fromisoformat(ended_at)
    except ValueError as exc:
        raise PreparationError(
            f"session {expected_id} ended_at_utc must be a valid UTC timestamp"
        ) from exc
    if ended_at_parsed.utcoffset() != timedelta(0):
        raise PreparationError(
            f"session {expected_id} ended_at_utc must be a valid UTC timestamp"
        )
    parquet_write_failures = manifest.get("parquet_write_failures")
    if (
        not isinstance(parquet_write_failures, int)
        or isinstance(parquet_write_failures, bool)
        or parquet_write_failures != 0
    ):
        raise PreparationError(f"session {expected_id} reports Parquet write failures")
    unclean_restart_count = manifest.get("unclean_restart_count")
    if (
        not isinstance(unclean_restart_count, int)
        or isinstance(unclean_restart_count, bool)
        or unclean_restart_count != 0
    ):
        raise PreparationError(f"session {expected_id} reports an unclean restart")
    restart_count = manifest.get("restart_count")
    if (
        not isinstance(restart_count, int)
        or isinstance(restart_count, bool)
        or restart_count != 0
    ):
        raise PreparationError(f"session {expected_id} is a restarted capture")
    if manifest.get("poll_interval_ms_requested") != FRAME_INTERVAL_MS:
        raise PreparationError(
            f"session {expected_id} did not request a 100 ms poll interval"
        )
    timing = manifest.get("timing")
    if not isinstance(timing, dict):
        raise PreparationError(f"session {expected_id} manifest timing is incomplete")
    if timing.get("poll_interval_ms_requested") != FRAME_INTERVAL_MS:
        raise PreparationError(
            f"session {expected_id} timing does not confirm 100 ms polling"
        )
    if not isinstance(timing.get("sample_count"), int) or timing["sample_count"] < 1:
        raise PreparationError(f"session {expected_id} timing sample_count is incomplete")
    if not isinstance(workload, dict) or workload.get("class") != "ai-compute":
        raise PreparationError(f"session {expected_id} workload class must be ai-compute")
    return manifest, manifest_path, hashlib.sha256(manifest_bytes).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _row_features(row: dict[str, Any]) -> list[float] | None:
    values = [_number(row.get(item["source"])) for item in FEATURE_MAP]
    if values[0] is None or values[0] < 0:
        return None
    if any(value is None or value <= 0 for value in values[1:]):
        return None
    result = [float(value) for value in values if value is not None]
    result[1] /= 1_000.0
    return result


def _read_rows(
    session_path: Path, session_id: str
) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    numbered_paths: list[tuple[int, Path]] = []
    for path in session_path.glob("*.parquet"):
        match = _BATCH_PATTERN.fullmatch(path.name)
        if match is None:
            raise PreparationError(
                f"session {session_id} has unexpected Parquet {path.name}"
            )
        numbered_paths.append((int(match.group(1)), path))
    numbered_paths.sort()
    parquet_paths = [path for _, path in numbered_paths]
    if not parquet_paths:
        raise PreparationError(f"session {session_id} has no Parquet batches")
    rows: list[dict[str, Any]] = []
    snapshots: list[tuple[Path, str]] = []
    ordinal = 0
    previous_timestamp: int | None = None
    for parquet_path in parquet_paths:
        try:
            parquet_bytes = parquet_path.read_bytes()
            table = pq.read_table(pa.BufferReader(parquet_bytes), columns=PARQUET_COLUMNS)
        except Exception as exc:  # pyarrow has several format/schema exception classes
            raise PreparationError(f"cannot read {parquet_path}: {exc}") from exc
        snapshots.append((parquet_path, hashlib.sha256(parquet_bytes).hexdigest()))
        for raw in table.to_pylist():
            if raw.get("session_label") != session_id:
                raise PreparationError(
                    f"session {session_id} row session_label does not match assignment"
                )
            timestamp = raw.get("timestamp_ms")
            if not isinstance(timestamp, int) or isinstance(timestamp, bool):
                raise PreparationError(
                    f"session {session_id} has a non-integer timestamp_ms"
                )
            if timestamp <= 0:
                raise PreparationError(
                    f"session {session_id} must have a positive timestamp_ms"
                )
            row = dict(raw)
            row["_ordinal"] = ordinal
            row["_clock_reversal"] = (
                previous_timestamp is not None and timestamp <= previous_timestamp
            )
            previous_timestamp = timestamp
            ordinal += 1
            rows.append(row)
    if any(row["_clock_reversal"] for row in rows):
        raise PreparationError(f"session {session_id} contains a clock reversal")
    return rows, snapshots


def _frames(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["timestamp_ms"], row["_ordinal"]))
    timestamps = [row["timestamp_ms"] for row in ordered]
    start = timestamps[0]
    end = timestamps[-1]
    grid_end = start + math.ceil((end - start) / FRAME_INTERVAL_MS) * FRAME_INTERVAL_MS
    frame_count = (grid_end - start) // FRAME_INTERVAL_MS + 1
    if frame_count > MAX_FRAME_COUNT:
        raise PreparationError(
            f"session frame span requires {frame_count} frames; "
            f"maximum is {MAX_FRAME_COUNT}"
        )
    forced_rejections: dict[int, set[str]] = {}
    previous_timestamp: int | None = None
    for row in rows:
        timestamp = row["timestamp_ms"]
        grid_timestamp = (
            start + math.ceil((timestamp - start) / FRAME_INTERVAL_MS) * FRAME_INTERVAL_MS
        )
        if _row_features(row) is None:
            forced_rejections.setdefault(grid_timestamp, set()).add("invalid_sensor")
        if (
            previous_timestamp is not None
            and timestamp - previous_timestamp > MAX_SOURCE_AGE_MS
        ):
            forced_rejections.setdefault(grid_timestamp, set()).add("source_gap")
        previous_timestamp = timestamp
    frames: list[dict[str, Any]] = []
    segment_id = 0
    after_invalid = False
    for frame_timestamp in range(start, grid_end + 1, FRAME_INTERVAL_MS):
        source_position = bisect_left(timestamps, frame_timestamp + 1) - 1
        source = ordered[source_position] if source_position >= 0 else None
        reasons: list[str] = []
        features: list[float] | None = None
        source_timestamp: int | None = None
        age: int | None = None
        if source is None:
            reasons.append("missing_source")
        else:
            source_timestamp = source["timestamp_ms"]
            age = frame_timestamp - source_timestamp
            if age < 0:
                reasons.append("future_source")
            elif age > MAX_SOURCE_AGE_MS:
                reasons.append("stale_source")
            if source["_clock_reversal"]:
                reasons.append("clock_reversal")
            features = _row_features(source)
            if features is None:
                reasons.append("invalid_sensor")
        reasons.extend(
            sorted(forced_rejections.get(frame_timestamp, set()) - set(reasons))
        )
        valid = not reasons
        if valid and after_invalid:
            segment_id += 1
            after_invalid = False
        elif not valid:
            after_invalid = True
        frames.append(
            {
                "timestamp_ms": frame_timestamp,
                "source_timestamp_ms": source_timestamp,
                "age_ms": age,
                "segment_id": segment_id,
                "valid": valid,
                "rejection_reasons": reasons,
                "x": features if valid else None,
            }
        )
    return frames


def _examples(
    frames: list[dict[str, Any]], rows: list[dict[str, Any]], rejections: Counter[str]
) -> list[dict[str, Any]]:
    frame_at = {frame["timestamp_ms"]: index for index, frame in enumerate(frames)}
    sources: list[tuple[int, list[float] | None, bool]] = []
    for row in sorted(rows, key=lambda item: (item["timestamp_ms"], item["_ordinal"])):
        sources.append((row["timestamp_ms"], _row_features(row), row["_clock_reversal"]))
    source_timestamps = [item[0] for item in sources]
    boundary_timestamps: list[int] = []
    previous_timestamp: int | None = None
    for row in rows:
        timestamp = row["timestamp_ms"]
        if _row_features(row) is None or (
            previous_timestamp is not None
            and timestamp - previous_timestamp > MAX_SOURCE_AGE_MS
        ):
            boundary_timestamps.append(timestamp)
        previous_timestamp = timestamp
    examples: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        if not frame["valid"]:
            rejections["current_invalid"] += 1
            continue
        history: list[int] = []
        for offset in HISTORY_OFFSETS_MS:
            history_index = frame_at.get(frame["timestamp_ms"] - offset)
            if history_index is None:
                rejections["history_missing"] += 1
                break
            history_frame = frames[history_index]
            if (
                not history_frame["valid"]
                or history_frame["segment_id"] != frame["segment_id"]
            ):
                rejections["history_gap"] += 1
                break
            history.append(history_index)
        if len(history) != len(HISTORY_OFFSETS_MS):
            continue
        targets: list[tuple[int, list[float] | None, bool]] = []
        for offset in TARGET_OFFSETS_MS:
            deadline = frame["timestamp_ms"] + offset
            position = bisect_left(source_timestamps, deadline)
            if position >= len(sources):
                rejections["target_missing"] += 1
                break
            candidate = sources[position]
            if candidate[0] - deadline > MAX_TARGET_LATENESS_MS:
                rejections["target_late"] += 1
                break
            if candidate[1] is None or candidate[2]:
                rejections["target_invalid"] += 1
                break
            if bisect_right(boundary_timestamps, candidate[0]) > bisect_right(
                boundary_timestamps, frame["timestamp_ms"]
            ):
                rejections["target_gap"] += 1
                break
            targets.append(candidate)
        if len(targets) != len(TARGET_OFFSETS_MS):
            continue
        current = frame["x"]
        assert current is not None
        one, five = targets
        assert one[1] is not None and five[1] is not None
        examples.append(
            {
                "frame_index": frame_index,
                "history_indices": history,
                "target_timestamps_ms": [one[0], five[0]],
                "y": [
                    one[1][2] - current[2],
                    one[1][1] - current[1],
                    five[1][2] - current[2],
                    five[1][1] - current[1],
                ],
            }
        )
    return examples


def _statistics(rows: list[list[float]], width: int, name: str) -> dict[str, list[Any]]:
    if not rows:
        raise PreparationError(f"training split has no values for {name} normalization")
    means: list[float] = []
    raw_std: list[float] = []
    count = len(rows)
    for column in range(width):
        values = [row[column] for row in rows]
        try:
            if not all(math.isfinite(value) for value in values):
                raise OverflowError
            scale = max(abs(value) for value in values)
            if scale == 0.0:
                mean = 0.0
                std = 0.0
            else:
                scaled = [value / scale for value in values]
                scaled_mean = math.fsum(scaled) / count
                mean = scale * scaled_mean
                std = scale * math.sqrt(
                    math.fsum((value - scaled_mean) ** 2 for value in scaled) / count
                )
        except OverflowError as exc:
            raise PreparationError(
                f"training split produced non-finite {name} normalization statistics"
            ) from exc
        if not math.isfinite(mean) or not math.isfinite(std):
            raise PreparationError(
                f"training split produced non-finite {name} normalization statistics"
            )
        means.append(mean)
        raw_std.append(std)
    constant = [value == 0.0 for value in raw_std]
    return {
        f"{name}_mean": means,
        f"{name}_std": [
            1.0 if is_constant else value
            for value, is_constant in zip(raw_std, constant, strict=True)
        ],
        f"{name}_raw_std": raw_std,
        f"{name}_constant": constant,
    }


def _out_of_training_range(
    sessions: list[dict[str, Any]], train_x: list[list[float]]
) -> dict[str, list[int]]:
    lower = [min(row[column] for row in train_x) for column in range(len(FEATURE_MAP))]
    upper = [max(row[column] for row in train_x) for column in range(len(FEATURE_MAP))]
    result: dict[str, list[int]] = {}
    for split in sorted(VALID_SPLITS):
        counts = [0] * len(FEATURE_MAP)
        for session in sessions:
            if session["split"] != split:
                continue
            for frame in session["frames"]:
                if not frame["valid"]:
                    continue
                for column, value in enumerate(frame["x"]):
                    if value < lower[column] or value > upper[column]:
                        counts[column] += 1
        result[split] = counts
    return result


def _prepare_campaign(
    campaign_path: Path,
    output_dir: Path,
    campaign: dict[str, Any],
    minimum: int,
    campaign_sha256: str,
) -> tuple[dict[str, Any], list[tuple[Path, frozenset[str]]], list[tuple[Path, str]]]:
    """Validate and prepare one immutable anticipation campaign.

    ``campaign_path`` contains the preassigned sessions and the predeclared
    minimum eligible-example count. ``output_dir/prepared.json`` is published
    only after every assigned session satisfies the same contract.
    """

    prepared_sessions: list[dict[str, Any]] = []
    provenance_sources: list[dict[str, Any]] = []
    total_rejections: Counter[str] = Counter()
    summaries: list[dict[str, Any]] = []
    deficiencies: list[str] = []
    source_snapshots: list[tuple[Path, str]] = []
    source_memberships: list[tuple[Path, frozenset[str]]] = []
    for item in campaign["sessions"]:
        session_id = item["session_id"]
        raw_path = Path(item["path"])
        session_path = (
            raw_path if raw_path.is_absolute() else campaign_path.parent / raw_path
        )
        try:
            collector_manifest, manifest_path, manifest_sha256 = _load_manifest(
                session_path, session_id
            )
            rows, parquet_snapshots = _read_rows(session_path, session_id)
            if collector_manifest["timing"]["sample_count"] != len(rows):
                raise PreparationError(
                    f"session {session_id} timing sample_count does not equal "
                    "persisted rows"
                )
            session_rejections: Counter[str] = Counter()
            frames = _frames(rows)
            examples = _examples(frames, rows, session_rejections)
            parquet_provenance = [
                {"path": path.name, "sha256": sha256}
                for path, sha256 in parquet_snapshots
            ]
            source_snapshots.append((manifest_path, manifest_sha256))
            source_snapshots.extend(parquet_snapshots)
            source_memberships.append(
                (session_path, frozenset(path.name for path, _ in parquet_snapshots))
            )
        except (PreparationError, OSError) as exc:
            details = {
                "session_summaries": summaries,
                "provenance": provenance_sources,
                "failed_session_id": session_id,
            }
            if isinstance(exc, PreparationError):
                details.update(exc.details)
            raise PreparationError(str(exc), details) from exc
        total_rejections.update(session_rejections)
        input_rejections: Counter[str] = Counter()
        for frame in frames:
            total_rejections.update(frame["rejection_reasons"])
            input_rejections.update(frame["rejection_reasons"])
        if len(examples) < minimum:
            deficiencies.append(
                f"{session_id} has {len(examples)} eligible examples; requires {minimum}"
            )
        prepared_sessions.append(
            {
                "session_id": session_id,
                "split": item["split"],
                "seed": item["seed"],
                "frames": frames,
                "examples": examples,
            }
        )
        summaries.append(
            {
                "session_id": session_id,
                "split": item["split"],
                "source_rows": len(rows),
                "frames": len(frames),
                "valid_frames": sum(frame["valid"] for frame in frames),
                "eligible_examples": len(examples),
                "rejections": dict(sorted(session_rejections.items())),
                "input_rejections": dict(sorted(input_rejections.items())),
                "input_rejection_rate": (
                    sum(not frame["valid"] for frame in frames) / len(frames)
                ),
            }
        )
        provenance_sources.append(
            {
                "session_id": session_id,
                "collector_session_id": collector_manifest["session_id"],
                "manifest_sha256": manifest_sha256,
                "parquet": parquet_provenance,
            }
        )
    if deficiencies:
        raise PreparationError(
            "; ".join(deficiencies),
            {"session_summaries": summaries, "provenance": provenance_sources},
        )

    train_x = [
        frame["x"]
        for session in prepared_sessions
        if session["split"] == "train"
        for frame in session["frames"]
        if frame["valid"]
    ]
    train_y = [
        example["y"]
        for session in prepared_sessions
        if session["split"] == "train"
        for example in session["examples"]
    ]
    normalization = {
        "fit_split": "train",
        **_statistics(train_x, len(FEATURE_MAP), "x"),
        **_statistics(train_y, len(TARGET_NAMES), "y"),
    }
    prepared = {
        "schema_version": SCHEMA_VERSION,
        "feature_map_id": FEATURE_MAP_ID,
        "feature_map": FEATURE_MAP,
        "target_names": TARGET_NAMES,
        "sessions": prepared_sessions,
        "normalization": normalization,
        "quality": {
            "schema_version": SCHEMA_VERSION,
            "criteria": {
                "frame_interval_ms": FRAME_INTERVAL_MS,
                "max_source_age_ms": MAX_SOURCE_AGE_MS,
                "max_target_lateness_ms": MAX_TARGET_LATENESS_MS,
                "history_offsets_ms": list(HISTORY_OFFSETS_MS),
                "target_offsets_ms": list(TARGET_OFFSETS_MS),
                "min_examples_per_session": minimum,
            },
            "rejections": dict(sorted(total_rejections.items())),
            "session_summaries": summaries,
            "out_of_training_range": _out_of_training_range(prepared_sessions, train_x),
        },
        "provenance": {
            "campaign_sha256": campaign_sha256,
            "sources": provenance_sources,
        },
    }
    _check_preparation_sources(source_memberships, source_snapshots)
    _write_json(output_dir / "prepared.json", prepared)
    return prepared, source_memberships, source_snapshots


def _check_preparation_sources(
    source_memberships: list[tuple[Path, frozenset[str]]],
    source_snapshots: list[tuple[Path, str]],
) -> None:
    for session_path, expected_names in source_memberships:
        current_names = frozenset(path.name for path in session_path.glob("*.parquet"))
        if current_names != expected_names:
            raise PreparationError(
                f"source changed during preparation: {session_path} Parquet membership"
            )
    for path, expected_sha256 in source_snapshots:
        try:
            current_sha256 = _sha256(path)
        except OSError as exc:
            raise PreparationError(f"source changed during preparation: {path}") from exc
        if current_sha256 != expected_sha256:
            raise PreparationError(f"source changed during preparation: {path}")
    for session_path, expected_names in source_memberships:
        current_names = frozenset(path.name for path in session_path.glob("*.parquet"))
        if current_names != expected_names:
            raise PreparationError(
                f"source changed during preparation: {session_path} Parquet membership"
            )


def _assignments(campaign_bytes: bytes) -> list[dict[str, Any]]:
    try:
        raw = json.loads(campaign_bytes.decode()).get("sessions", [])
    except OSError, ValueError, RecursionError, AttributeError:
        return []
    if not isinstance(raw, list):
        return []
    return [
        {key: item.get(key) for key in ("session_id", "split", "seed", "path")}
        for item in raw
        if isinstance(item, dict)
    ]


def _campaign_assignments(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: item.get(key) for key in ("session_id", "split", "seed", "path")}
        for item in campaign.get("sessions", [])
        if isinstance(item, dict)
    ]


def _remove_owned_preparation_outputs(output_dir: Path) -> None:
    for name in (
        "prepared.json",
        "quality-report.json",
        "manifest.json",
        "prepared.json.tmp",
        "quality-report.json.tmp",
        "manifest.json.tmp",
    ):
        path = output_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir() and name.endswith(".tmp"):
            try:
                path.rmdir()
            except OSError:
                pass


def prepare_campaign(campaign_path: Path | str, output_dir: Path | str) -> dict[str, Any]:
    """Prepare a campaign and always publish a versioned completion report."""

    supplied_campaign_path = Path(campaign_path)
    campaign_resolution_error: PreparationError | None = None
    try:
        campaign_path = supplied_campaign_path.resolve(strict=True)
    except FileNotFoundError:
        campaign_path = supplied_campaign_path.resolve()
    except OSError, RuntimeError:
        campaign_path = supplied_campaign_path.absolute()
        campaign_resolution_error = PreparationError(
            f"campaign path has a symlink loop: {campaign_path}"
        )
    output_dir = Path(output_dir).resolve()
    output_paths = [
        output_dir / name
        for name in (
            "prepared.json",
            "quality-report.json",
            "manifest.json",
            "prepared.json.tmp",
            "quality-report.json.tmp",
            "manifest.json.tmp",
        )
    ]
    resolution_error: PreparationError | None = None
    output_artifacts = set(output_paths)
    for path in output_paths:
        try:
            output_artifacts.add(path.resolve())
        except RuntimeError as exc:
            if resolution_error is None:
                resolution_error = PreparationError(
                    f"output artifact symlink loop: {exc}"
                )
    if campaign_path in output_artifacts:
        raise PreparationError(f"campaign {campaign_path} collides with output artifact")
    campaign_read_error: PreparationError | None = None
    try:
        campaign_bytes = campaign_path.read_bytes()
    except OSError as exc:
        campaign_bytes = b""
        campaign_read_error = PreparationError(
            f"cannot read campaign {campaign_path}: {exc}"
        )
    # Guard source directories before the error handler can clean or publish outputs.
    for item in _assignments(campaign_bytes):
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
            continue
        session_path = Path(raw_path)
        if not session_path.is_absolute():
            session_path = campaign_path.parent / session_path
        if output_dir.is_relative_to(session_path) or session_path.is_relative_to(
            output_dir
        ):
            raise PreparationError(
                f"output directory overlaps session source: {session_path}"
            )
        try:
            session_path = session_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            if resolution_error is None:
                resolution_error = PreparationError(
                    f"cannot resolve session source {session_path}: {exc}"
                )
            continue
        if output_dir.is_relative_to(session_path) or session_path.is_relative_to(
            output_dir
        ):
            raise PreparationError(
                f"output directory overlaps session source: {session_path}"
            )
    assignments: list[dict[str, Any]] = []
    try:
        if resolution_error is not None:
            raise resolution_error
        if campaign_resolution_error is not None:
            raise campaign_resolution_error
        if campaign_read_error is not None:
            raise campaign_read_error
        staging_symlinks = [path for path in output_paths if path.is_symlink()]
        if staging_symlinks:
            raise PreparationError(
                f"output staging path is a symlink: {staging_symlinks[0]}"
            )
        hardlinked_staging = [
            path
            for path in output_paths
            if path.is_file() and not path.is_symlink() and path.stat().st_nlink > 1
        ]
        if hardlinked_staging:
            raise PreparationError(
                f"output staging path is multiply linked: {hardlinked_staging[0]}"
            )
        campaign, minimum, campaign_sha256 = _load_campaign(campaign_path, campaign_bytes)
        assignments = _campaign_assignments(campaign)
        output_dir.mkdir(parents=True, exist_ok=True)
        publication_identity = directory_identity(output_dir)
        (output_dir / "manifest.json").unlink(missing_ok=True)
        prepared, source_memberships, source_snapshots = _prepare_campaign(
            campaign_path,
            output_dir,
            campaign,
            minimum,
            campaign_sha256,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "assignments": assignments,
            "prepared_sha256": _sha256(output_dir / "prepared.json"),
            "session_counts": prepared["quality"]["session_summaries"],
            "sources": prepared["provenance"]["sources"],
        }
        _write_json(output_dir / "quality-report.json", prepared["quality"])
        if directory_identity(output_dir) != publication_identity:
            raise PreparationError("publication directory changed during preparation")
        if _sha256(output_dir / "prepared.json") != manifest["prepared_sha256"]:
            raise PreparationError("prepared artifact changed during publication")
        _check_preparation_sources(source_memberships, source_snapshots)
        if _sha256(output_dir / "prepared.json") != manifest["prepared_sha256"]:
            raise PreparationError("prepared artifact changed during publication")
        if directory_identity(output_dir) != publication_identity:
            raise PreparationError("publication directory changed during preparation")
        _write_json(output_dir / "manifest.json", manifest)
        if directory_identity(output_dir) != publication_identity:
            raise PreparationError("publication directory changed during preparation")
    except (OSError, PreparationError) as exc:
        if not assignments:
            assignments = _assignments(campaign_bytes)
        incomplete = {
            "schema_version": SCHEMA_VERSION,
            "status": "incomplete",
            "assignments": assignments,
            "failure_reasons": [str(exc)],
            **getattr(exc, "details", {}),
        }
        _remove_owned_preparation_outputs(output_dir)
        _write_json(output_dir / "quality-report.json", incomplete)
        _write_json(output_dir / "manifest.json", incomplete)
        raise
    return prepared

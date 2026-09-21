"""Fail-closed audit and additive forecasting view for historical v3 telemetry."""

from __future__ import annotations

import errno
import hashlib
import math
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import (
    clean_artifacts,
    directory_identity,
    ensure_directory,
    pin_directory,
    pinned_publication,
    write_json,
    write_parquet,
)
from .v3_build import OUTCOMES_SCHEMA, PROPOSALS_SCHEMA, STATE_SCHEMA

AUDIT_VERSION = "1.0.0"
VIEW_ID = "v3-forecast-eligible-v1"
HORIZON_SAMPLES = 64
EPISODE_LEN = 4096
SPLITS = ("train", "validation", "test")
SHARD_GLOB = "*.parquet"
SENSOR_COLUMNS = ("gpu_temp_c", "power_w", "sm_clock_mhz", "mem_clock_mhz")
ZERO_SUSPECT_COLUMNS = SENSOR_COLUMNS


class AuditError(RuntimeError):
    """The source corpus violates a structural audit invariant."""


@dataclass(frozen=True)
class AuditReport:
    audit_version: str
    view_id: str
    horizon_samples: int
    splits: dict[str, dict[str, Any]]
    integrity: dict[str, Any]
    status: str = "complete"
    failure: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_version": self.audit_version,
            "view_id": self.view_id,
            "horizon_samples": self.horizon_samples,
            "splits": self.splits,
            "integrity": self.integrity,
            "status": self.status,
            "failure": self.failure,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path, output_root: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    lines = result.stdout.strip().splitlines()
    if len(lines) != 2 or Path(lines[0]).resolve() != path.resolve():
        return None
    pathspecs = ["--", "."]
    if output_root is not None and output_root.is_relative_to(path):
        relative_output = output_root.relative_to(path).as_posix()
        pathspecs.append(f":(top,exclude,literal){relative_output}")
    commands = (
        ["status", "--porcelain", "--untracked-files=all", "--ignored=no"],
        ["ls-files", "--others", "--ignored", "--exclude-standard"],
    )
    for command in commands:
        try:
            status = subprocess.run(
                ["git", "-C", str(path), *command, *pathspecs],
                check=True,
                capture_output=True,
                text=True,
            )
        except OSError, subprocess.CalledProcessError:
            return None
        if status.stdout.strip():
            return None
    return lines[1] or None


def _retained_revision(
    dataset_root: Path, v3_root: Path, output_root: Path, expected: str | None
) -> str | None:
    if expected is None:
        return None
    for config in ("state_telemetry", "outcomes", "action_proposals"):
        directory = v3_root / config
        if any(
            path.resolve() != path for path in (directory, *directory.glob(SHARD_GLOB))
        ):
            return None
    if _git_head(dataset_root, output_root) == expected:
        return expected
    return None


def _key_columns(table: pa.Table, label: str) -> list[tuple[str, int]]:
    raw_keys = list(
        zip(
            table.column("episode_id").to_pylist(),
            table.column("step_idx").to_pylist(),
            strict=True,
        )
    )
    keys: list[tuple[str, int]] = []
    for episode_id, step_idx in raw_keys:
        _episode_number(episode_id)
        if not isinstance(step_idx, int) or isinstance(step_idx, bool):
            raise AuditError(f"{label} step_idx must be an integer")
        if step_idx < 0 or step_idx >= EPISODE_LEN:
            raise AuditError(
                f"{label} step_idx outside 0..{EPISODE_LEN - 1}: "
                f"{(episode_id, step_idx)!r}"
            )
        keys.append((episode_id, step_idx))
    return keys


def _require_columns(table: pa.Table, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if name not in table.column_names]
    if missing:
        raise AuditError(f"{label} missing required columns: {', '.join(missing)}")


def _require_numeric_columns(table: pa.Table, names: tuple[str, ...], label: str) -> None:
    for name in names:
        field_type = table.schema.field(name).type
        if not (pa.types.is_integer(field_type) or pa.types.is_floating(field_type)):
            raise AuditError(f"{label} {name} must have a numeric Arrow type")


def _assert_unique(keys: list[tuple[str, int]], label: str) -> None:
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise AuditError(f"{label} has duplicate join keys; first={duplicates[0]!r}")


def _episode_number(episode_id: object) -> int:
    if not isinstance(episode_id, str) or not episode_id:
        raise AuditError("episode_id must be a non-empty string")
    prefix, separator, value = episode_id.rpartition("-")
    if not separator or prefix != "gpu" or re.fullmatch(r"[0-9]+", value) is None:
        raise AuditError(
            f"cannot recover source row_index from episode_id {episode_id!r}"
        )
    if len(value) != 6:
        raise AuditError(f"noncanonical episode_id {episode_id!r}")
    try:
        episode_number = int(value)
    except ValueError as exc:
        raise AuditError(
            f"cannot recover source row_index from episode_id {episode_id!r}"
        ) from exc
    if episode_id != f"gpu-{episode_number:06d}":
        raise AuditError(f"noncanonical episode_id {episode_id!r}")
    return episode_number


def _source_index(key: tuple[str, int]) -> int:
    episode_id, step_idx = key
    if step_idx < 0 or step_idx >= EPISODE_LEN:
        raise AuditError(f"step_idx outside 0..{EPISODE_LEN - 1}: {key!r}")
    return _episode_number(episode_id) * EPISODE_LEN + step_idx


def _sensor_reason(name: str, value: Any) -> str | None:
    if value is None:
        return f"missing_{name}"
    try:
        finite = math.isfinite(float(value))
    except TypeError, ValueError:
        finite = False
    if not finite:
        return f"non_finite_{name}"
    if name in ZERO_SUSPECT_COLUMNS and float(value) == 0.0:
        return f"zero_{name}"
    return None


def _distribution(column: pa.ChunkedArray) -> dict[str, int | float | None]:
    values = column.to_pylist()
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    mean = math.fsum(value / len(finite) for value in finite) if finite else None
    if mean is not None and not math.isfinite(mean):
        raise AuditError("numeric distribution mean is non-finite")
    return {
        "count": len(values),
        "null_count": sum(value is None for value in values),
        "non_finite_count": sum(
            value is not None and not math.isfinite(float(value)) for value in values
        ),
        "zero_count": sum(
            value is not None and math.isfinite(float(value)) and float(value) == 0.0
            for value in values
        ),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "mean": mean,
    }


def _source_root(source_dir: Path) -> tuple[Path, Path, bool]:
    try:
        source_dir = source_dir.resolve(strict=True)
    except FileNotFoundError:
        source_dir = source_dir.resolve()
    except (OSError, RuntimeError, UnicodeError) as exc:
        if isinstance(exc, UnicodeError) or (
            isinstance(exc, OSError) and exc.errno != errno.ELOOP
        ):
            raise AuditError(f"cannot resolve source path {source_dir!r}: {exc}") from exc
        raise AuditError(f"source path has a symlink loop: {source_dir}") from exc
    v3 = source_dir / "v3"
    if (v3 / "state_telemetry").is_dir():
        return source_dir, v3, True
    if (source_dir / "state_telemetry").is_dir():
        return source_dir.parent, source_dir, False
    raise AuditError(f"cannot find v3 state_telemetry under {source_dir}")


def _read_source_table(
    path: Path, expected_hashes: dict[Path, str], *, columns: list[str] | None = None
) -> pa.Table:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hashes.get(path):
        raise AuditError(f"source changed before parsing: {path}")
    return pq.read_table(pa.BufferReader(data), columns=columns)


def _load_split(
    v3_root: Path, split: str, expected_hashes: dict[Path, str]
) -> tuple[Path, Path, pa.Table, pa.Table]:
    state_path = v3_root / "state_telemetry" / f"{split}-00000.parquet"
    outcome_path = v3_root / "outcomes" / f"{split}-00000.parquet"
    for path in (state_path, outcome_path):
        if not path.is_file():
            raise AuditError(f"missing source shard {path}")
    try:
        state = _read_source_table(state_path, expected_hashes)
        outcomes = _read_source_table(outcome_path, expected_hashes)
    except (OSError, pa.ArrowException) as exc:
        raise AuditError(f"cannot read {split} source shards: {exc}") from exc
    _require_columns(
        state,
        ("episode_id", "step_idx", "ts_utc", *SENSOR_COLUMNS),
        f"state_telemetry/{split}",
    )
    _require_columns(
        outcomes,
        ("episode_id", "step_idx", "reward", "d_gpu_temp_c"),
        f"outcomes/{split}",
    )
    for table, config in ((state, "state_telemetry"), (outcomes, "outcomes")):
        _require_columns(table, ("schema_version",), f"{config}/{split}")
        if any(
            version != "3.0.0" for version in table.column("schema_version").to_pylist()
        ):
            raise AuditError(f"{config}/{split} schema_version must be 3.0.0")
    _require_numeric_columns(state, SENSOR_COLUMNS, f"state_telemetry/{split}")
    _require_numeric_columns(outcomes, ("reward", "d_gpu_temp_c"), f"outcomes/{split}")
    for table, schema in ((state, STATE_SCHEMA), (outcomes, OUTCOMES_SCHEMA)):
        unexpected = set(table.column_names) - set(schema.names)
        if unexpected:
            raise AuditError(f"unexpected source fields: {sorted(unexpected)}")
        for field in schema:
            if (
                field.name not in table.column_names
                or table.schema.field(field.name).type != field.type
            ):
                raise AuditError(
                    f"source schema field missing or wrong type: {field.name}"
                )
        if not table.schema.equals(schema, check_metadata=False):
            raise AuditError(
                "source schema must use the canonical field order and nullability"
            )
    if state.num_rows == 0 or outcomes.num_rows == 0:
        raise AuditError(f"{split} source split is empty")
    return state_path, outcome_path, state, outcomes


def _numeric_sensor_columns(state: pa.Table) -> tuple[str, ...]:
    alignment_or_metadata = {"ts_utc", "step_idx", "window_ms"}
    return tuple(
        field.name
        for field in state.schema
        if field.name not in alignment_or_metadata
        and (pa.types.is_integer(field.type) or pa.types.is_floating(field.type))
    )


def _load_actions(
    v3_root: Path,
    split: str,
    state_keys: list[tuple[str, int]],
    expected_hashes: dict[Path, str],
) -> tuple[pa.Table, list[tuple[str, int]], int]:
    path = v3_root / "action_proposals" / f"{split}-00000.parquet"
    if not path.is_file():
        raise AuditError(f"missing source shard {path}")
    try:
        table = _read_source_table(path, expected_hashes)
        unexpected = set(table.column_names) - set(PROPOSALS_SCHEMA.names)
        if unexpected:
            raise AuditError(f"unexpected action proposal fields: {sorted(unexpected)}")
        for field in PROPOSALS_SCHEMA:
            if (
                field.name not in table.column_names
                or table.schema.field(field.name).type != field.type
            ):
                raise AuditError(f"action proposal field type mismatch: {field.name}")
        canonical_schema = table.schema.equals(
            PROPOSALS_SCHEMA, check_metadata=False
        )
        _require_columns(
            table,
            (
                "episode_id",
                "step_idx",
                "proposed_action",
                "teacher_action",
                "schema_version",
            ),
            f"action_proposals/{split}",
        )
    except (OSError, pa.ArrowException) as exc:
        raise AuditError(f"cannot read {split} action-proposal shard: {exc}") from exc
    if any(version != "3.0.0" for version in table.column("schema_version").to_pylist()):
        raise AuditError(f"action_proposals/{split} schema_version must be 3.0.0")
    if not canonical_schema:
        raise AuditError(
            "action proposals must use the canonical schema order and nullability"
        )
    proposal_keys = _key_columns(table, f"action_proposals/{split}")
    _assert_unique(proposal_keys, f"action_proposals/{split}")
    state_key_set = set(state_keys)
    orphan_keys = set(proposal_keys) - state_key_set
    if orphan_keys:
        first_orphan = next(iter(orphan_keys))
        raise AuditError(
            f"action_proposals/{split} has orphan join keys; first={first_orphan!r}"
        )
    observed_keys = {
        key
        for key, proposed, teacher in zip(
            proposal_keys,
            table.column("proposed_action").to_pylist(),
            table.column("teacher_action").to_pylist(),
            strict=True,
        )
        if proposed is not None or teacher is not None
    }
    observed = len(observed_keys)
    return table, proposal_keys, len(state_keys) - observed


def _guard_output_path(
    dataset_root: Path,
    v3_root: Path,
    output_root: Path,
    source_is_dataset_root: bool,
) -> None:
    resolved_v3 = v3_root.resolve()
    overlaps = (
        output_root == resolved_v3
        or output_root.is_relative_to(resolved_v3)
        or resolved_v3.is_relative_to(output_root)
        or output_root == dataset_root
        or output_root == v3_root
        or output_root.is_relative_to(v3_root)
        or (source_is_dataset_root and output_root.is_relative_to(dataset_root))
        or dataset_root.is_relative_to(output_root)
    )
    if overlaps:
        raise AuditError(f"output directory would overlap source corpus: {output_root}")
    _guard_source_members(v3_root, output_root)


def _guard_source_members(v3_root: Path, output_root: Path) -> None:
    for config in ("state_telemetry", "outcomes", "action_proposals"):
        directory = v3_root / config
        for source in (directory, *directory.glob(SHARD_GLOB)):
            try:
                resolved = source.resolve()
            except (OSError, RuntimeError, UnicodeError) as exc:
                raise AuditError(f"cannot resolve source member {source!r}") from exc
            if resolved.is_relative_to(output_root) or output_root.is_relative_to(
                resolved
            ):
                raise AuditError(
                    f"output directory would overlap source member: {source}"
                )


def _clean_owned_outputs(output_root: Path) -> None:
    try:
        clean_artifacts(
            output_root, ("manifest.json", "audit-report.json", "exclusions.parquet")
        )
        view_root = output_root / VIEW_ID
        if view_root.is_symlink():
            raise AuditError(f"audit view directory must not be a symlink: {view_root}")
        if view_root.exists():
            clean_artifacts(view_root)
    except OSError as exc:
        raise AuditError(f"cannot safely clean audit outputs: {exc}") from exc


def _available_source_hashes(
    dataset_root: Path, v3_root: Path, *, tolerate_unreadable: bool = False
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    memberships: dict[Path, list[Path]] = {}
    expected_names = {f"{split}-00000.parquet" for split in SPLITS}
    for config in ("state_telemetry", "outcomes", "action_proposals"):
        directory = v3_root / config
        paths = sorted(directory.glob(SHARD_GLOB))
        memberships[directory] = paths
        if not directory.is_dir():
            continue
        unexpected = [path.name for path in paths if path.name not in expected_names]
        if unexpected and not tolerate_unreadable:
            raise AuditError(
                f"{directory.name} has unexpected source shards: {', '.join(unexpected)}"
            )
        for path in paths:
            if path.name not in expected_names or not path.is_file():
                continue
            try:
                hashes[path.relative_to(dataset_root).as_posix()] = _sha256(path)
            except OSError:
                if not tolerate_unreadable:
                    raise
        final_paths = sorted(directory.glob(SHARD_GLOB))
        if not tolerate_unreadable and [path.name for path in final_paths] != [
            path.name for path in paths
        ]:
            raise AuditError(
                f"{directory.name} source shard membership changed during hashing"
            )
    if not tolerate_unreadable:
        for directory, paths in memberships.items():
            if sorted(directory.glob(SHARD_GLOB)) != paths:
                raise AuditError(
                    f"{directory.name} source shard membership changed during hashing"
                )
    return dict(sorted(hashes.items()))


def _append_output_columns(
    state: pa.Table,
    outcomes: pa.Table,
    state_indices: list[int],
    outcome_indices: list[int],
    split: str,
    source_indices: list[int],
    target_steps: list[int],
) -> pa.Table:
    reserved = {
        "source_split",
        "source_row_index",
        "target_step_idx",
        "forecast_horizon_samples",
        "d_gpu_temp_c_64",
    }
    collisions = reserved.intersection(state.column_names + outcomes.column_names)
    if collisions:
        raise AuditError(f"generated view column name collision: {sorted(collisions)}")
    selected = state.take(pa.array(state_indices, type=pa.int64()))
    chosen_outcomes = outcomes.take(pa.array(outcome_indices, type=pa.int64()))
    for name in outcomes.column_names:
        if name in {"episode_id", "step_idx", "ts_utc", "schema_version"}:
            continue
        if name in selected.column_names:
            raise AuditError(f"state/outcome column name collision: {name}")
        selected = selected.append_column(name, chosen_outcomes.column(name))
    n = len(state_indices)
    selected = selected.append_column(
        "source_split", pa.array([split] * n, type=pa.string())
    )
    selected = selected.append_column(
        "source_row_index", pa.array(source_indices, type=pa.int64())
    )
    selected = selected.append_column(
        "target_step_idx", pa.array(target_steps, type=pa.int32())
    )
    selected = selected.append_column(
        "forecast_horizon_samples", pa.array([HORIZON_SAMPLES] * n, type=pa.int32())
    )
    selected = selected.append_column(
        "d_gpu_temp_c_64", chosen_outcomes.column("d_gpu_temp_c")
    )
    return selected


def _append_action_columns(
    selected: pa.Table,
    actions: pa.Table,
    action_keys: list[tuple[str, int]],
    eligible_keys: list[tuple[str, int]],
) -> pa.Table:
    lookup = {key: index for index, key in enumerate(action_keys)}
    indices = pa.array([lookup.get(key) for key in eligible_keys], type=pa.int64())
    matched = actions.take(indices)
    for name in actions.column_names:
        if name in {"episode_id", "step_idx", "ts_utc", "schema_version"}:
            continue
        if name in selected.column_names:
            raise AuditError(f"action/output column name collision: {name}")
        selected = selected.append_column(name, matched.column(name))
    return selected


type LoadedSplit = tuple[Path, Path, pa.Table, pa.Table]
type RowKey = tuple[str, int]


def _load_validated_splits(
    v3_root: Path, expected_hashes: dict[Path, str]
) -> tuple[dict[str, LoadedSplit], dict[str, set[str]]]:
    loaded: dict[str, LoadedSplit] = {}
    episode_split: dict[str, str] = {}
    episodes_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        loaded[split] = _load_split(v3_root, split, expected_hashes)
        _, _, state, outcomes = loaded[split]
        state_keys = _key_columns(state, f"state_telemetry/{split}")
        outcome_keys = _key_columns(outcomes, f"outcomes/{split}")
        _assert_unique(state_keys, f"state_telemetry/{split}")
        _assert_unique(outcome_keys, f"outcomes/{split}")
        if set(state_keys) != set(outcome_keys):
            missing_outcomes = len(set(state_keys) - set(outcome_keys))
            orphan_outcomes = len(set(outcome_keys) - set(state_keys))
            raise AuditError(
                f"{split} state/outcome join keys differ: "
                f"missing_outcomes={missing_outcomes}, orphan_outcomes={orphan_outcomes}"
            )
        state_timestamps = dict(
            zip(state_keys, state.column("ts_utc").to_pylist(), strict=True)
        )
        outcome_timestamps = dict(
            zip(outcome_keys, outcomes.column("ts_utc").to_pylist(), strict=True)
        )
        if state_timestamps != outcome_timestamps:
            raise AuditError(f"{split} state/outcome timestamps differ")
        episodes_by_split[split] = {episode_id for episode_id, _ in state_keys}
        for episode_id in episodes_by_split[split]:
            previous = episode_split.setdefault(episode_id, split)
            if previous != split:
                raise AuditError(
                    f"episode {episode_id!r} belongs to multiple splits: "
                    f"{previous}, {split}"
                )
    return loaded, episodes_by_split


def _window_reasons(
    episode_id: str,
    step_idx: int,
    state_index: dict[RowKey, int],
    base_reasons: dict[RowKey, list[str]],
) -> list[str]:
    reasons: list[str] = []
    for offset in range(HORIZON_SAMPLES + 1):
        window_key = (episode_id, step_idx + offset)
        if window_key not in state_index:
            reasons.append("window_has_index_gap")
            continue
        if offset == 0:
            continue
        for reason in base_reasons[window_key]:
            window_reason = f"window_contains_{reason}"
            if window_reason not in reasons:
                reasons.append(window_reason)
    return reasons


def _delta_reasons(current_temp: Any, target_temp: Any, recorded: Any) -> list[str]:
    reasons: list[str] = []
    if (
        current_temp is not None
        and target_temp is not None
        and math.isfinite(float(current_temp))
        and math.isfinite(float(target_temp))
    ):
        expected = float(target_temp) - float(current_temp)
        if recorded is None or not math.isfinite(float(recorded)):
            reasons.append("missing_or_non_finite_outcome_delta")
        elif not math.isclose(float(recorded), expected, rel_tol=1e-6, abs_tol=1e-5):
            reasons.append("outcome_delta_mismatch")

    return reasons


def _eligibility_reasons(
    index: int,
    key: RowKey,
    state_index: dict[RowKey, int],
    outcome_index: dict[RowKey, int],
    temperatures: list[Any],
    deltas: list[Any],
    base_reasons: dict[RowKey, list[str]],
) -> list[str]:
    episode_id, step_idx = key
    target_key = (episode_id, step_idx + HORIZON_SAMPLES)
    reasons = list(base_reasons[key])
    if target_key not in state_index:
        reasons.append("missing_exact_64_sample_target")
    else:
        reasons.extend(_window_reasons(episode_id, step_idx, state_index, base_reasons))
        current_temp = temperatures[index]
        target_temp = temperatures[state_index[target_key]]
        recorded = deltas[outcome_index[key]]
        reasons.extend(_delta_reasons(current_temp, target_temp, recorded))
    return sorted(set(reasons))


def _evaluate_split(
    v3_root: Path,
    split: str,
    state: pa.Table,
    outcomes: pa.Table,
    expected_hashes: dict[Path, str],
) -> tuple[dict[str, Any], pa.Table, list[dict[str, object]]]:
    exclusion_rows: list[dict[str, object]] = []
    state_keys = _key_columns(state, f"state_telemetry/{split}")
    outcome_keys = _key_columns(outcomes, f"outcomes/{split}")
    state_index = {key: index for index, key in enumerate(state_keys)}
    outcome_index = {key: index for index, key in enumerate(outcome_keys)}
    sensor_values = {name: state.column(name).to_pylist() for name in SENSOR_COLUMNS}
    deltas = outcomes.column("d_gpu_temp_c").to_pylist()

    base_reasons: dict[tuple[str, int], list[str]] = {}
    for index, key in enumerate(state_keys):
        reasons = [
            reason
            for name in SENSOR_COLUMNS
            if (reason := _sensor_reason(name, sensor_values[name][index])) is not None
        ]
        base_reasons[key] = reasons

    eligible_state_indices: list[int] = []
    eligible_outcome_indices: list[int] = []
    eligible_source_indices: list[int] = []
    eligible_target_steps: list[int] = []
    reason_counts: Counter[str] = Counter()

    for index, key in enumerate(state_keys):
        episode_id, step_idx = key
        reasons = _eligibility_reasons(
            index,
            key,
            state_index,
            outcome_index,
            sensor_values["gpu_temp_c"],
            deltas,
            base_reasons,
        )
        if reasons:
            reason_counts.update(reasons)
            exclusion_rows.append(
                {
                    "source_split": split,
                    "episode_id": episode_id,
                    "step_idx": step_idx,
                    "source_row_index": _source_index(key),
                    "reasons": reasons,
                }
            )
            continue
        eligible_state_indices.append(index)
        eligible_outcome_indices.append(outcome_index[key])
        eligible_source_indices.append(_source_index(key))
        eligible_target_steps.append(step_idx + HORIZON_SAMPLES)

    actions, action_keys, missing_actions = _load_actions(
        v3_root, split, state_keys, expected_hashes
    )
    state_times = dict(zip(state_keys, state.column("ts_utc").to_pylist(), strict=True))
    if any(
        state_times[key] != timestamp
        for key, timestamp in zip(
            action_keys, actions.column("ts_utc").to_pylist(), strict=True
        )
    ):
        raise AuditError(f"{split} state/action timestamps differ")
    eligible_table = _append_output_columns(
        state,
        outcomes,
        eligible_state_indices,
        eligible_outcome_indices,
        split,
        eligible_source_indices,
        eligible_target_steps,
    )
    eligible_table = _append_action_columns(
        eligible_table,
        actions,
        action_keys,
        [state_keys[index] for index in eligible_state_indices],
    )
    distributions = {
        name: _distribution(state.column(name)) for name in _numeric_sensor_columns(state)
    }
    split_report = {
        "source_rows": state.num_rows,
        "eligible_rows": len(eligible_state_indices),
        "excluded_rows": state.num_rows - len(eligible_state_indices),
        "exclusion_reasons": dict(sorted(reason_counts.items())),
        "numeric_sensor_columns": distributions,
        "missing_fields": {
            "timestamp_missing_count": state.column("ts_utc").null_count,
            "reward_missing_count": outcomes.column("reward").null_count,
            "action_label_missing_count": missing_actions,
        },
        **{name: distributions[name] for name in SENSOR_COLUMNS},
    }

    return split_report, eligible_table, exclusion_rows


@dataclass(frozen=True)
class _AuditSources:
    dataset_root: Path
    v3_root: Path
    hashes: dict[str, str]
    git_commit: str | None


@dataclass(frozen=True)
class _AuditOutputs:
    report: AuditReport
    eligible_tables: dict[str, pa.Table]
    exclusions: list[dict[str, object]]


def _verify_audit_directory(path: Path, identity: tuple[int, int]) -> None:
    if directory_identity(path) != identity:
        raise AuditError("publication directory changed during audit")


def _publish_audit(
    sources: _AuditSources,
    outputs: _AuditOutputs,
    output_root: Path,
    publication_identity: tuple[int, int],
) -> None:
    dataset_root, v3_root = sources.dataset_root, sources.v3_root
    source_hashes, source_git_commit = sources.hashes, sources.git_commit
    report, eligible_tables = outputs.report, outputs.eligible_tables
    exclusion_rows = outputs.exclusions
    final_source_hashes = _available_source_hashes(dataset_root, v3_root)
    if final_source_hashes != source_hashes:
        raise AuditError("source shards changed during audit")
    _guard_output_path(dataset_root.resolve(), v3_root.resolve(), output_root, False)
    ensure_directory(output_root)
    view_root = output_root / VIEW_ID
    ensure_directory(view_root)
    view_identity = directory_identity(view_root)
    for split, table in eligible_tables.items():
        write_parquet(view_root / f"{split}-00000.parquet", table)
    exclusion_schema = pa.schema(
        [
            pa.field("source_split", pa.string()),
            pa.field("episode_id", pa.string()),
            pa.field("step_idx", pa.int32()),
            pa.field("source_row_index", pa.int64()),
            pa.field("reasons", pa.list_(pa.string())),
        ]
    )
    write_parquet(
        output_root / "exclusions.parquet",
        pa.Table.from_pylist(exclusion_rows, schema=exclusion_schema),
    )
    manifest: dict[str, Any] = {
        "status": "complete",
        "audit_version": AUDIT_VERSION,
        "view_id": VIEW_ID,
        "horizon_samples": HORIZON_SAMPLES,
        "source_git_commit": source_git_commit,
        "source_files": dict(sorted(source_hashes.items())),
        "outputs": {
            f"{VIEW_ID}/{split}-00000.parquet": _sha256(
                view_root / f"{split}-00000.parquet"
            )
            for split in SPLITS
        }
        | {"exclusions.parquet": _sha256(output_root / "exclusions.parquet")},
    }
    write_json(output_root / "audit-report.json", report.to_dict())
    _verify_audit_directory(output_root, publication_identity)
    manifest["outputs"]["audit-report.json"] = _sha256(output_root / "audit-report.json")
    manifest["source_git_commit"] = _retained_revision(
        dataset_root, v3_root, output_root, source_git_commit
    )
    _verify_audit_directory(output_root, publication_identity)
    _verify_audit_directory(view_root, view_identity)
    if _available_source_hashes(dataset_root, v3_root) != source_hashes:
        raise AuditError("source shards changed during publication")
    _verify_audit_directory(output_root, publication_identity)
    _check_output_snapshot(output_root, view_root, manifest["outputs"])
    write_json(output_root / "manifest.json", manifest)
    _verify_audit_directory(output_root, publication_identity)
    _verify_audit_directory(view_root, view_identity)


def _audit_v3_impl(
    dataset_root: Path, v3_root: Path, output_root: Path, source_git_commit: str | None
) -> AuditReport:
    publication_identity = directory_identity(output_root)
    source_hashes = _available_source_hashes(dataset_root, v3_root)
    source_git_commit = _retained_revision(
        dataset_root, v3_root, output_root, source_git_commit
    )
    expected_hashes = {
        dataset_root / name: digest for name, digest in source_hashes.items()
    }
    loaded, episodes_by_split = _load_validated_splits(v3_root, expected_hashes)
    split_reports: dict[str, dict[str, Any]] = {}
    eligible_tables: dict[str, pa.Table] = {}
    exclusion_rows: list[dict[str, object]] = []
    for split in SPLITS:
        _, _, state, outcomes = loaded[split]
        split_report, table, exclusions = _evaluate_split(
            v3_root, split, state, outcomes, expected_hashes
        )
        split_reports[split] = split_report
        eligible_tables[split] = table
        exclusion_rows.extend(exclusions)
    integrity = {
        "duplicate_keys": "passed",
        "state_outcome_joins": "passed",
        "episode_split_membership": "passed",
        "episodes_by_split": dict(
            sorted(
                (split, len(episodes)) for split, episodes in episodes_by_split.items()
            )
        ),
        "overlapping_episode_count": 0,
    }
    report = AuditReport(
        audit_version=AUDIT_VERSION,
        view_id=VIEW_ID,
        horizon_samples=HORIZON_SAMPLES,
        splits=split_reports,
        integrity=integrity,
    )
    _publish_audit(
        _AuditSources(dataset_root, v3_root, source_hashes, source_git_commit),
        _AuditOutputs(report, eligible_tables, exclusion_rows),
        output_root,
        publication_identity,
    )
    return report


def _check_output_snapshot(
    output_root: Path, view_root: Path, expected: dict[str, str]
) -> None:
    names = {f"{split}-00000.parquet" for split in SPLITS}
    if {path.name for path in view_root.glob(SHARD_GLOB)} != names:
        raise AuditError("output shard membership changed during publication")
    if any(_sha256(output_root / name) != digest for name, digest in expected.items()):
        raise AuditError("output artifacts changed during publication")
    if {path.name for path in view_root.glob(SHARD_GLOB)} != names:
        raise AuditError("output shard membership changed during publication")


def _write_incomplete_evidence(
    output_root: Path,
    error: AuditError,
    dataset_root: Path | None,
    v3_root: Path | None,
    source_git_commit: str | None = None,
) -> None:
    failure = {"category": "structural_integrity", "reason": str(error)}
    incomplete_report = {
        "audit_version": AUDIT_VERSION,
        "view_id": VIEW_ID,
        "horizon_samples": HORIZON_SAMPLES,
        "status": "incomplete",
        "failure": failure,
        "integrity": {"status": "failed"},
        "splits": {},
    }
    incomplete_manifest = {
        "audit_version": AUDIT_VERSION,
        "view_id": VIEW_ID,
        "horizon_samples": HORIZON_SAMPLES,
        "status": "incomplete",
        "failure": failure,
        "source_git_commit": source_git_commit,
        "source_files": (
            _available_source_hashes(dataset_root, v3_root, tolerate_unreadable=True)
            if dataset_root is not None and v3_root is not None
            else {}
        ),
        "outputs": {},
    }
    if dataset_root is not None and v3_root is not None:
        incomplete_manifest["source_git_commit"] = _retained_revision(
            dataset_root, v3_root, output_root, source_git_commit
        )
    write_json(output_root / "audit-report.json", incomplete_report)
    write_json(output_root / "manifest.json", incomplete_manifest)


def _guard_supplied_source(source_path: Path, output_root: Path) -> None:
    try:
        provisional_source = source_path.resolve()
    except OSError, RuntimeError, UnicodeError:
        provisional_source = source_path.absolute()
    if output_root.is_relative_to(
        provisional_source
    ) or provisional_source.is_relative_to(output_root):
        raise AuditError(
            f"output directory would overlap supplied source path: {output_root}"
        )
    try:
        os.fsencode(provisional_source)
    except UnicodeError:
        # No filesystem entry can exist at an unencodable path.
        return
    for root in (provisional_source, provisional_source / "v3"):
        _guard_source_members(root, output_root)


def _pin_audit_view(
    output_root: Path, dataset_root: Path, v3_root: Path, source_git_commit: str | None
) -> None:
    try:
        view_root = output_root / VIEW_ID
        if view_root.is_symlink():
            raise OSError(f"audit view directory must not be a symlink: {view_root}")
        pin_directory(view_root)
    except OSError as exc:
        error = AuditError(f"cannot safely pin audit view: {exc}")
        clean_artifacts(
            output_root, ("manifest.json", "audit-report.json", "exclusions.parquet")
        )
        _write_incomplete_evidence(
            output_root, error, dataset_root, v3_root, source_git_commit
        )
        raise error from exc


def audit_v3(source_dir: str | Path, output_dir: str | Path) -> AuditReport:
    """Audit historical v3 and write a provenance-bound additive eligible view.

    Structural corruption (duplicate keys, mismatched joins, or cross-split
    episode membership) fails closed after recording an incomplete report and
    manifest. Row-level defects are written with exact reasons. Source files
    are read only and the output is forbidden from overlapping their tree.
    If the output root is replaced, abort without writing into its replacement.
    """
    source_path = Path(source_dir)
    try:
        output_root = Path(output_dir).resolve()
    except (RuntimeError, UnicodeError) as exc:
        raise AuditError(f"cannot resolve output path: {output_dir!r}") from exc
    initial_identity = directory_identity(output_root) if output_root.is_dir() else None
    try:
        dataset_root, v3_root, source_is_dataset_root = _source_root(source_path)
    except AuditError as exc:
        _guard_supplied_source(source_path, output_root)
        ensure_directory(output_root)
        with pinned_publication(output_root, initial_identity):
            _guard_supplied_source(source_path, output_root)
            try:
                _clean_owned_outputs(output_root)
            except AuditError:
                # A view symlink must not prevent publishing safe root-level evidence.
                pass
            _write_incomplete_evidence(output_root, exc, None, None)

        raise

    _guard_output_path(dataset_root, v3_root, output_root, source_is_dataset_root)
    source_git_commit = _git_head(dataset_root, output_root)
    with pinned_publication(output_root, initial_identity):
        _guard_output_path(
            dataset_root.resolve(), v3_root.resolve(), output_root, source_is_dataset_root
        )
        _pin_audit_view(output_root, dataset_root, v3_root, source_git_commit)
        try:
            _guard_output_path(
                dataset_root.resolve(),
                v3_root.resolve(),
                output_root,
                source_is_dataset_root,
            )
        except AuditError as exc:
            # A substituted view may now hold source shards. Only replace root evidence.
            clean_artifacts(
                output_root, ("manifest.json", "audit-report.json", "exclusions.parquet")
            )
            _write_incomplete_evidence(
                output_root, exc, dataset_root, v3_root, source_git_commit
            )
            raise
        try:
            ensure_directory(output_root)
            _clean_owned_outputs(output_root)
            return _audit_v3_impl(dataset_root, v3_root, output_root, source_git_commit)
        except (AuditError, OSError, pa.ArrowException) as raw_error:
            audit_error = (
                raw_error
                if isinstance(raw_error, AuditError)
                else AuditError(f"cannot write audit outputs: {raw_error}")
            )
            try:
                _clean_owned_outputs(output_root)
            except AuditError:
                # Preserve an unsafe view symlink without following it; the JSON
                # evidence files live directly under the already-guarded root.
                pass
            try:
                _write_incomplete_evidence(
                    output_root, audit_error, dataset_root, v3_root, source_git_commit
                )
            except OSError as publication_error:
                # A replaced output root is no longer ours, even for error evidence.
                raise audit_error from publication_error
            if audit_error is raw_error:
                raise
            raise audit_error from raw_error

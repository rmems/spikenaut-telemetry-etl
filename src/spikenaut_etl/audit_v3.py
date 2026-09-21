"""Fail-closed audit and additive forecasting view for historical v3 telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

AUDIT_VERSION = "1.0.0"
VIEW_ID = "v3-forecast-eligible-v1"
HORIZON_SAMPLES = 64
EPISODE_LEN = 4096
SPLITS = ("train", "validation", "test")
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


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return result.stdout.strip() or None


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
    if not separator or prefix != "gpu" or not value.isdigit():
        raise AuditError(
            f"cannot recover source row_index from episode_id {episode_id!r}"
        )
    episode_number = int(value)
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
        "mean": sum(finite) / len(finite) if finite else None,
    }


def _source_root(source_dir: Path) -> tuple[Path, Path, bool]:
    source_dir = source_dir.resolve()
    v3 = source_dir / "v3"
    if (v3 / "state_telemetry").is_dir():
        return source_dir, v3, True
    if (source_dir / "state_telemetry").is_dir():
        return source_dir.parent, source_dir, False
    raise AuditError(f"cannot find v3 state_telemetry under {source_dir}")


def _load_split(v3_root: Path, split: str) -> tuple[Path, Path, pa.Table, pa.Table]:
    state_path = v3_root / "state_telemetry" / f"{split}-00000.parquet"
    outcome_path = v3_root / "outcomes" / f"{split}-00000.parquet"
    expected_names = {f"{name}-00000.parquet" for name in SPLITS}
    for directory in (state_path.parent, outcome_path.parent):
        unexpected = sorted(
            path.name
            for path in directory.glob("*.parquet")
            if path.name not in expected_names
        )
        if unexpected:
            raise AuditError(
                f"{directory.name} has unexpected source shards: {', '.join(unexpected)}"
            )
    for path in (state_path, outcome_path):
        if not path.is_file():
            raise AuditError(f"missing source shard {path}")
    try:
        state = pq.read_table(state_path)
        outcomes = pq.read_table(outcome_path)
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
    _require_numeric_columns(state, SENSOR_COLUMNS, f"state_telemetry/{split}")
    _require_numeric_columns(outcomes, ("d_gpu_temp_c",), f"outcomes/{split}")
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


def _action_label_counts(
    v3_root: Path, split: str, state_keys: list[tuple[str, int]]
) -> tuple[int, int]:
    path = v3_root / "action_proposals" / f"{split}-00000.parquet"
    if not path.is_file():
        return len(state_keys), 0
    try:
        table = pq.read_table(
            path,
            columns=["episode_id", "step_idx", "proposed_action", "teacher_action"],
        )
    except (OSError, pa.ArrowException) as exc:
        raise AuditError(f"cannot read {split} action-proposal shard: {exc}") from exc
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
    return len(state_keys) - observed, observed


def _guard_output_path(
    dataset_root: Path,
    v3_root: Path,
    output_root: Path,
    source_is_dataset_root: bool,
) -> None:
    overlaps = (
        output_root == dataset_root
        or output_root == v3_root
        or output_root.is_relative_to(v3_root)
        or (source_is_dataset_root and output_root.is_relative_to(dataset_root))
        or dataset_root.is_relative_to(output_root)
    )
    if overlaps:
        raise AuditError(f"output directory would overlap source corpus: {output_root}")


def _clean_owned_outputs(output_root: Path) -> None:
    for name in ("manifest.json", "audit-report.json", "exclusions.parquet"):
        path = output_root / name
        if path.is_file():
            path.unlink()
    view_root = output_root / VIEW_ID
    if view_root.is_dir():
        for path in view_root.rglob("*.parquet"):
            if path.is_file():
                path.unlink()


def _available_source_hashes(dataset_root: Path, v3_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for config in ("state_telemetry", "outcomes", "action_proposals"):
        for split in SPLITS:
            path = v3_root / config / f"{split}-00000.parquet"
            if path.is_file():
                hashes[path.relative_to(dataset_root).as_posix()] = _sha256(path)
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
    selected = state.take(pa.array(state_indices, type=pa.int64()))
    chosen_outcomes = outcomes.take(pa.array(outcome_indices, type=pa.int64()))
    for name in outcomes.column_names:
        if name in {"episode_id", "step_idx", "ts_utc", "schema_version"}:
            continue
        if name in selected.column_names:
            continue
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


def _audit_v3_impl(dataset_root: Path, v3_root: Path, output_root: Path) -> AuditReport:
    loaded: dict[str, tuple[Path, Path, pa.Table, pa.Table]] = {}
    episode_split: dict[str, str] = {}
    episodes_by_split: dict[str, set[str]] = {}
    source_hashes = _available_source_hashes(dataset_root, v3_root)
    for split in SPLITS:
        loaded[split] = _load_split(v3_root, split)
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
        episodes_by_split[split] = {episode_id for episode_id, _ in state_keys}
        for episode_id in episodes_by_split[split]:
            previous = episode_split.setdefault(episode_id, split)
            if previous != split:
                raise AuditError(
                    f"episode {episode_id!r} belongs to multiple splits: "
                    f"{previous}, {split}"
                )
    split_reports: dict[str, dict[str, Any]] = {}
    eligible_tables: dict[str, pa.Table] = {}
    exclusion_rows: list[dict[str, object]] = []

    for split in SPLITS:
        _, _, state, outcomes = loaded[split]
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
                if (reason := _sensor_reason(name, sensor_values[name][index]))
                is not None
            ]
            base_reasons[key] = reasons

        eligible_state_indices: list[int] = []
        eligible_outcome_indices: list[int] = []
        eligible_source_indices: list[int] = []
        eligible_target_steps: list[int] = []
        reason_counts: Counter[str] = Counter()

        for index, key in enumerate(state_keys):
            episode_id, step_idx = key
            target_key = (episode_id, step_idx + HORIZON_SAMPLES)
            reasons = list(base_reasons[key])
            if target_key not in state_index:
                reasons.append("missing_exact_64_sample_target")
            else:
                for offset in range(HORIZON_SAMPLES + 1):
                    window_key = (episode_id, step_idx + offset)
                    if window_key not in state_index:
                        reasons.append("window_has_index_gap")
                        break
                    if offset == 0:
                        continue
                    for reason in base_reasons[window_key]:
                        window_reason = f"window_contains_{reason}"
                        if window_reason not in reasons:
                            reasons.append(window_reason)
                current_temp = sensor_values["gpu_temp_c"][index]
                target_temp = sensor_values["gpu_temp_c"][state_index[target_key]]
                recorded = deltas[outcome_index[key]]
                if (
                    current_temp is not None
                    and target_temp is not None
                    and math.isfinite(float(current_temp))
                    and math.isfinite(float(target_temp))
                ):
                    expected = float(target_temp) - float(current_temp)
                    if recorded is None or not math.isfinite(float(recorded)):
                        reasons.append("missing_or_non_finite_outcome_delta")
                    elif not math.isclose(
                        float(recorded), expected, rel_tol=1e-6, abs_tol=1e-5
                    ):
                        reasons.append("outcome_delta_mismatch")

            reasons = sorted(set(reasons))
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

        eligible_tables[split] = _append_output_columns(
            state,
            outcomes,
            eligible_state_indices,
            eligible_outcome_indices,
            split,
            eligible_source_indices,
            eligible_target_steps,
        )
        split_reports[split] = {
            "source_rows": state.num_rows,
            "eligible_rows": len(eligible_state_indices),
            "excluded_rows": state.num_rows - len(eligible_state_indices),
            "exclusion_reasons": dict(sorted(reason_counts.items())),
            "numeric_sensor_columns": {
                name: _distribution(state.column(name))
                for name in _numeric_sensor_columns(state)
            },
            "missing_fields": {
                "timestamp_missing_count": state.column("ts_utc").null_count,
                "reward_missing_count": outcomes.column("reward").null_count,
                "action_label_missing_count": _action_label_counts(
                    v3_root, split, state_keys
                )[0],
            },
            **{name: _distribution(state.column(name)) for name in SENSOR_COLUMNS},
        }

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
    output_root.mkdir(parents=True, exist_ok=True)
    view_root = output_root / VIEW_ID
    view_root.mkdir(parents=True, exist_ok=True)
    for split, table in eligible_tables.items():
        pq.write_table(table, view_root / f"{split}-00000.parquet")
    exclusion_schema = pa.schema(
        [
            pa.field("source_split", pa.string()),
            pa.field("episode_id", pa.string()),
            pa.field("step_idx", pa.int32()),
            pa.field("source_row_index", pa.int64()),
            pa.field("reasons", pa.list_(pa.string())),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(exclusion_rows, schema=exclusion_schema),
        output_root / "exclusions.parquet",
    )
    manifest = {
        "status": "complete",
        "audit_version": AUDIT_VERSION,
        "view_id": VIEW_ID,
        "horizon_samples": HORIZON_SAMPLES,
        "source_git_commit": _git_head(dataset_root),
        "source_files": dict(sorted(source_hashes.items())),
        "outputs": {
            f"{VIEW_ID}/{split}-00000.parquet": _sha256(
                view_root / f"{split}-00000.parquet"
            )
            for split in SPLITS
        }
        | {"exclusions.parquet": _sha256(output_root / "exclusions.parquet")},
    }
    (output_root / "audit-report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return report


def _write_incomplete_evidence(
    output_root: Path,
    error: AuditError,
    dataset_root: Path | None,
    v3_root: Path | None,
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
        "source_git_commit": _git_head(dataset_root) if dataset_root else None,
        "source_files": (
            _available_source_hashes(dataset_root, v3_root)
            if dataset_root is not None and v3_root is not None
            else {}
        ),
        "outputs": {},
    }
    (output_root / "audit-report.json").write_text(
        json.dumps(incomplete_report, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "manifest.json").write_text(
        json.dumps(incomplete_manifest, indent=2, sort_keys=True) + "\n"
    )


def audit_v3(source_dir: str | Path, output_dir: str | Path) -> AuditReport:
    """Audit historical v3 and write a provenance-bound additive eligible view.

    Structural corruption (duplicate keys, mismatched joins, or cross-split
    episode membership) fails closed after recording an incomplete report and
    manifest. Row-level defects are written with exact reasons. Source files
    are read only and the output is forbidden from overlapping their tree.
    """
    output_root = Path(output_dir).resolve()
    try:
        dataset_root, v3_root, source_is_dataset_root = _source_root(Path(source_dir))
    except AuditError as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        _clean_owned_outputs(output_root)
        _write_incomplete_evidence(output_root, exc, None, None)
        raise

    _guard_output_path(dataset_root, v3_root, output_root, source_is_dataset_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _clean_owned_outputs(output_root)
    try:
        return _audit_v3_impl(dataset_root, v3_root, output_root)
    except (AuditError, OSError, pa.ArrowException) as raw_error:
        audit_error = (
            raw_error
            if isinstance(raw_error, AuditError)
            else AuditError(f"cannot write audit outputs: {raw_error}")
        )
        _clean_owned_outputs(output_root)
        _write_incomplete_evidence(output_root, audit_error, dataset_root, v3_root)
        if audit_error is raw_error:
            raise
        raise audit_error from raw_error

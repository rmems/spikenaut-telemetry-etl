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

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_version": self.audit_version,
            "view_id": self.view_id,
            "horizon_samples": self.horizon_samples,
            "splits": self.splits,
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
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _key_columns(table: pa.Table) -> list[tuple[str, int]]:
    return list(
        zip(
            table.column("episode_id").to_pylist(),
            table.column("step_idx").to_pylist(),
            strict=True,
        )
    )


def _require_columns(table: pa.Table, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if name not in table.column_names]
    if missing:
        raise AuditError(f"{label} missing required columns: {', '.join(missing)}")


def _assert_unique(keys: list[tuple[str, int]], label: str) -> None:
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise AuditError(f"{label} has duplicate join keys; first={duplicates[0]!r}")


def _episode_number(episode_id: str) -> int:
    prefix, separator, value = episode_id.rpartition("-")
    if not separator or prefix != "gpu" or not value.isdigit():
        raise AuditError(
            f"cannot recover source row_index from episode_id {episode_id!r}"
        )
    return int(value)


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
    except (TypeError, ValueError):
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


def _source_root(source_dir: Path) -> tuple[Path, Path]:
    source_dir = source_dir.resolve()
    v3 = source_dir / "v3"
    if (v3 / "state_telemetry").is_dir():
        return source_dir, v3
    if (source_dir / "state_telemetry").is_dir():
        return source_dir.parent, source_dir
    raise AuditError(f"cannot find v3 state_telemetry under {source_dir}")


def _load_split(v3_root: Path, split: str) -> tuple[Path, Path, pa.Table, pa.Table]:
    state_path = v3_root / "state_telemetry" / f"{split}-00000.parquet"
    outcome_path = v3_root / "outcomes" / f"{split}-00000.parquet"
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
        ("episode_id", "step_idx", *SENSOR_COLUMNS),
        f"state_telemetry/{split}",
    )
    _require_columns(
        outcomes,
        ("episode_id", "step_idx", "reward", "d_gpu_temp_c"),
        f"outcomes/{split}",
    )
    return state_path, outcome_path, state, outcomes


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
    selected = selected.append_column("source_split", pa.array([split] * n))
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


def audit_v3(source_dir: str | Path, output_dir: str | Path) -> AuditReport:
    """Audit historical v3 and write a provenance-bound additive eligible view.

    Structural corruption (duplicate keys, mismatched joins, or cross-split
    episode membership) fails closed. Row-level sensor and target defects are
    retained in ``exclusions.parquet`` with exact reasons and omitted from the
    additive eligible view. Source files are read only.
    """
    dataset_root, v3_root = _source_root(Path(source_dir))
    output_root = Path(output_dir).resolve()

    loaded: dict[str, tuple[Path, Path, pa.Table, pa.Table]] = {}
    episode_split: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for split in SPLITS:
        loaded[split] = _load_split(v3_root, split)
        state_path, outcome_path, state, outcomes = loaded[split]
        state_keys = _key_columns(state)
        outcome_keys = _key_columns(outcomes)
        _assert_unique(state_keys, f"state_telemetry/{split}")
        _assert_unique(outcome_keys, f"outcomes/{split}")
        if set(state_keys) != set(outcome_keys):
            missing_outcomes = len(set(state_keys) - set(outcome_keys))
            orphan_outcomes = len(set(outcome_keys) - set(state_keys))
            raise AuditError(
                f"{split} state/outcome join keys differ: "
                f"missing_outcomes={missing_outcomes}, orphan_outcomes={orphan_outcomes}"
            )
        for episode_id, _ in state_keys:
            previous = episode_split.setdefault(episode_id, split)
            if previous != split:
                raise AuditError(
                    f"episode {episode_id!r} belongs to multiple splits: "
                    f"{previous}, {split}"
                )
        for path in (state_path, outcome_path):
            relative = path.relative_to(dataset_root).as_posix()
            source_hashes[relative] = _sha256(path)

    split_reports: dict[str, dict[str, Any]] = {}
    eligible_tables: dict[str, pa.Table] = {}
    exclusion_rows: list[dict[str, object]] = []

    for split in SPLITS:
        _, _, state, outcomes = loaded[split]
        state_keys = _key_columns(state)
        outcome_keys = _key_columns(outcomes)
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
            **{name: _distribution(state.column(name)) for name in SENSOR_COLUMNS},
        }

    report = AuditReport(
        audit_version=AUDIT_VERSION,
        view_id=VIEW_ID,
        horizon_samples=HORIZON_SAMPLES,
        splits=split_reports,
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

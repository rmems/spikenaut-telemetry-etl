from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spikenaut_etl.audit_v3 import AuditError, audit_v3
from spikenaut_etl.v3_build import OUTCOMES_SCHEMA, STATE_SCHEMA

SPLITS = ("train", "validation", "test")


def _state_table(
    episode: str, steps: list[int], *, zero_at: int | None = None
) -> pa.Table:
    rows = []
    for step in steps:
        row = {field.name: None for field in STATE_SCHEMA}
        row.update(
            episode_id=episode,
            step_idx=step,
            schema_version="3.0.0",
            mem_util_pct=float(20 + step % 5),
            power_w=0.0 if step == zero_at else float(100 + step),
            gpu_temp_c=float(40 + step),
            sm_clock_mhz=1500.0,
            mem_clock_mhz=9000.0,
            synthetic=False,
        )
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=STATE_SCHEMA)


def _outcomes_table(
    episode: str,
    steps: list[int],
    *,
    wrong_delta_at: int | None = None,
) -> pa.Table:
    rows = []
    present = set(steps)
    for step in steps:
        row = {field.name: None for field in OUTCOMES_SCHEMA}
        delta = 64.0 if step + 64 in present else None
        if step == wrong_delta_at:
            delta = 63.0
        row.update(
            episode_id=episode,
            step_idx=step,
            schema_version="3.0.0",
            d_gpu_temp_c=delta,
            discount=1.0,
            is_terminal=False,
            is_first=step == min(steps),
            is_last=step == max(steps),
        )
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=OUTCOMES_SCHEMA)


def _write_corpus(
    root: Path,
    *,
    steps: list[int] | None = None,
    zero_at: int | None = None,
    wrong_delta_at: int | None = None,
) -> None:
    steps = steps or list(range(66))
    for split_index, split in enumerate(SPLITS):
        episode = f"gpu-{split_index:06d}"
        state_dir = root / "v3" / "state_telemetry"
        outcomes_dir = root / "v3" / "outcomes"
        state_dir.mkdir(parents=True, exist_ok=True)
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            _state_table(episode, steps, zero_at=zero_at if split == "train" else None),
            state_dir / f"{split}-00000.parquet",
        )
        pq.write_table(
            _outcomes_table(
                episode,
                steps,
                wrong_delta_at=wrong_delta_at if split == "train" else None,
            ),
            outcomes_dir / f"{split}-00000.parquet",
        )


def test_audit_preserves_rows_splits_indices_and_missing_fields(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)

    report = audit_v3(source, output)

    eligible = pq.read_table(output / "v3-forecast-eligible-v1" / "train-00000.parquet")
    assert eligible.column("source_split").to_pylist() == ["train", "train"]
    assert eligible.column("source_row_index").to_pylist() == [0, 1]
    assert eligible.column("step_idx").to_pylist() == [0, 1]
    assert eligible.column("target_step_idx").to_pylist() == [64, 65]
    assert eligible.column("d_gpu_temp_c_64").to_pylist() == [64.0, 64.0]
    assert eligible.column("reward").null_count == 2
    assert report.splits["train"]["source_rows"] == 66
    assert report.splits["train"]["eligible_rows"] == 2
    assert report.splits["train"]["gpu_temp_c"]["min"] == 40.0
    assert report.splits["train"]["gpu_temp_c"]["max"] == 105.0
    assert report.splits["train"]["numeric_sensor_columns"]["mem_util_pct"] == {
        "count": 66,
        "null_count": 0,
        "non_finite_count": 0,
        "zero_count": 0,
        "min": 20.0,
        "max": 24.0,
        "mean": pytest.approx(21.9696969697),
    }
    assert report.splits["train"]["numeric_sensor_columns"]["cpu_temp_c"] == {
        "count": 66,
        "null_count": 66,
        "non_finite_count": 0,
        "zero_count": 0,
        "min": None,
        "max": None,
        "mean": None,
    }
    assert report.integrity == {
        "duplicate_keys": "passed",
        "state_outcome_joins": "passed",
        "episode_split_membership": "passed",
        "episodes_by_split": {"test": 1, "train": 1, "validation": 1},
        "overlapping_episode_count": 0,
    }
    assert report.splits["train"]["missing_fields"] == {
        "action_label_missing_count": 66,
        "reward_missing_count": 66,
        "timestamp_missing_count": 66,
    }

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["view_id"] == "v3-forecast-eligible-v1"
    assert manifest["horizon_samples"] == 64
    state_path = source / "v3" / "state_telemetry" / "train-00000.parquet"
    expected_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert (
        manifest["source_files"]["v3/state_telemetry/train-00000.parquet"]
        == expected_hash
    )


def test_suspicious_reading_excludes_every_64_sample_window_that_contains_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source, zero_at=32)

    report = audit_v3(source, output)

    eligible = pq.read_table(output / "v3-forecast-eligible-v1" / "train-00000.parquet")
    assert eligible.num_rows == 0
    exclusions = pq.read_table(output / "exclusions.parquet").to_pylist()
    row_zero = next(
        row
        for row in exclusions
        if row["source_split"] == "train" and row["step_idx"] == 0
    )
    assert row_zero["reasons"] == ["window_contains_zero_power_w"]
    assert report.splits["train"]["exclusion_reasons"] == {
        "missing_exact_64_sample_target": 64,
        "window_contains_zero_power_w": 2,
        "zero_power_w": 1,
    }


def test_gap_is_not_compressed_into_a_false_64_sample_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source, steps=[0, *range(2, 67)])

    audit_v3(source, output)

    eligible = pq.read_table(output / "v3-forecast-eligible-v1" / "train-00000.parquet")
    # Step 0 has a real +64 endpoint, but its 64-sample window contains the
    # missing step 1 and is excluded. Step 2 targets the exact source step 66;
    # the 65th retained row is never allowed to stand in for another index.
    assert eligible.column("step_idx").to_pylist() == [2]
    assert eligible.column("target_step_idx").to_pylist() == [66]
    assert eligible.column("source_row_index").to_pylist() == [2]
    exclusions = pq.read_table(output / "exclusions.parquet").to_pylist()
    row_zero = next(
        row
        for row in exclusions
        if row["source_split"] == "train" and row["step_idx"] == 0
    )
    assert row_zero["reasons"] == ["window_has_index_gap"]


@pytest.mark.parametrize("duplicate_side", ["state", "outcomes"])
def test_duplicate_join_keys_fail_closed(tmp_path: Path, duplicate_side: str) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    path = (
        source
        / "v3"
        / ("state_telemetry" if duplicate_side == "state" else "outcomes")
        / "train-00000.parquet"
    )
    table = pq.read_table(path)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), path)

    output = tmp_path / "audit"
    with pytest.raises(AuditError, match="duplicate"):
        audit_v3(source, output)
    report = json.loads((output / "audit-report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["status"] == "incomplete"
    assert report["failure"]["category"] == "structural_integrity"
    assert "duplicate" in report["failure"]["reason"]
    assert manifest["status"] == "incomplete"
    assert manifest["outputs"] == {}


def test_state_outcome_join_mismatch_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    path = source / "v3" / "outcomes" / "train-00000.parquet"
    pq.write_table(pq.read_table(path).slice(0, 65), path)

    with pytest.raises(AuditError, match="join keys"):
        audit_v3(source, tmp_path / "audit")


def test_episode_cannot_belong_to_two_splits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    train_state = pq.read_table(source / "v3" / "state_telemetry" / "train-00000.parquet")
    train_outcomes = pq.read_table(source / "v3" / "outcomes" / "train-00000.parquet")
    pq.write_table(
        train_state,
        source / "v3" / "state_telemetry" / "validation-00000.parquet",
    )
    pq.write_table(
        train_outcomes,
        source / "v3" / "outcomes" / "validation-00000.parquet",
    )

    with pytest.raises(AuditError, match="multiple splits"):
        audit_v3(source, tmp_path / "audit")


def test_wrong_existing_64_sample_outcome_is_excluded_and_reported(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source, wrong_delta_at=0)

    report = audit_v3(source, output)

    eligible = pq.read_table(output / "v3-forecast-eligible-v1" / "train-00000.parquet")
    assert eligible.column("step_idx").to_pylist() == [1]
    assert report.splits["train"]["exclusion_reasons"]["outcome_delta_mismatch"] == 1


def test_non_finite_sensor_is_excluded(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    path = source / "v3" / "state_telemetry" / "train-00000.parquet"
    table = pq.read_table(path)
    values = table.column("gpu_temp_c").to_pylist()
    values[10] = float("nan")
    table = table.set_column(
        table.schema.get_field_index("gpu_temp_c"),
        "gpu_temp_c",
        pa.array(values, type=pa.float32()),
    )
    pq.write_table(table, path)

    report = audit_v3(source, output)

    assert (
        report.splits["train"]["exclusion_reasons"][
            "window_contains_non_finite_gpu_temp_c"
        ]
        == 2
    )
    assert report.splits["train"]["exclusion_reasons"]["non_finite_gpu_temp_c"] == 1


@pytest.mark.parametrize("relative_output", [".", "v3", "v3/state_telemetry/audit"])
def test_output_cannot_overlap_source_tree(tmp_path: Path, relative_output: str) -> None:
    source = tmp_path / "source"
    _write_corpus(source)

    with pytest.raises(AuditError, match="overlap source"):
        audit_v3(source, source / relative_output)

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import spikenaut_etl.audit_v3 as audit_module
from spikenaut_etl.audit_v3 import AuditError, audit_v3
from spikenaut_etl.cli import main
from spikenaut_etl.v3_build import OUTCOMES_SCHEMA, PROPOSALS_SCHEMA, STATE_SCHEMA

SPLITS = ("train", "validation", "test")


def test_cli_reports_audit_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_audit(*_args: object, **_kwargs: object) -> None:
        raise OSError("source filesystem unavailable")

    monkeypatch.setattr(audit_module, "audit_v3", fail_audit)

    assert (
        main(
            [
                "audit-v3",
                "--input",
                str(tmp_path / "v3"),
                "--output",
                str(tmp_path / "audit"),
            ]
        )
        == 1
    )
    assert capsys.readouterr().err == "audit-v3 failed: source filesystem unavailable\n"


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
        proposal_dir = root / "v3" / "action_proposals"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            PROPOSALS_SCHEMA.empty_table(), proposal_dir / f"{split}-00000.parquet"
        )
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
    assert eligible.schema.field("source_split").type == pa.string()
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


def test_missing_source_root_writes_incomplete_evidence(tmp_path: Path) -> None:
    output = tmp_path / "audit"

    with pytest.raises(AuditError, match="cannot find v3 state_telemetry"):
        audit_v3(tmp_path / "missing-source", output)

    report = json.loads((output / "audit-report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["status"] == "incomplete"
    assert manifest["status"] == "incomplete"
    assert manifest["source_files"] == {}


def test_cyclic_source_path_writes_incomplete_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source-loop"
    source.symlink_to(source, target_is_directory=True)
    output = tmp_path / "audit"

    with pytest.raises(AuditError, match="source path has a symlink loop"):
        audit_v3(source, output)

    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_invalid_source_output_overlap_is_rejected_before_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid-source"
    source.mkdir()
    sentinels = {
        "manifest.json": b"source manifest",
        "audit-report.json": b"source audit report",
        "exclusions.parquet": b"source exclusions",
    }
    for name, content in sentinels.items():
        (source / name).write_bytes(content)

    with pytest.raises(AuditError, match="overlap supplied source path"):
        audit_v3(source, source)

    assert {name: (source / name).read_bytes() for name in sentinels} == sentinels


def test_invalid_nested_source_cannot_clean_ancestor_output(tmp_path: Path) -> None:
    output = tmp_path / "corpus"
    source = output / "invalid-source"
    source.mkdir(parents=True)
    sentinels = {
        "manifest.json": b"source manifest",
        "audit-report.json": b"source audit report",
        "exclusions.parquet": b"source exclusions",
    }
    for name, content in sentinels.items():
        (output / name).write_bytes(content)

    with pytest.raises(AuditError, match="overlap supplied source path"):
        audit_v3(source, output)

    assert {name: (output / name).read_bytes() for name in sentinels} == sentinels


def test_missing_timestamp_column_fails_closed_with_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    path = source / "v3" / "state_telemetry" / "train-00000.parquet"
    table = pq.read_table(path).drop_columns(["ts_utc"])
    pq.write_table(table, path)

    with pytest.raises(AuditError, match="missing required columns: ts_utc"):
        audit_v3(source, output)

    report = json.loads((output / "audit-report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["status"] == "incomplete"
    assert manifest["status"] == "incomplete"


def test_required_sensor_columns_must_have_numeric_arrow_types(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    path = source / "v3" / "state_telemetry" / "train-00000.parquet"
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("gpu_temp_c"),
        "gpu_temp_c",
        pa.array(["42.0"] * table.num_rows, type=pa.string()),
    )
    pq.write_table(table, path)

    with pytest.raises(AuditError, match="gpu_temp_c must have a numeric Arrow type"):
        audit_v3(source, output)

    assert (
        json.loads((output / "audit-report.json").read_text())["status"] == "incomplete"
    )


def test_unexpected_source_shard_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    source_shard = source / "v3" / "state_telemetry" / "train-00000.parquet"
    extra_shard = source / "v3" / "state_telemetry" / "train-00001.parquet"
    extra_shard.write_bytes(source_shard.read_bytes())

    with pytest.raises(AuditError, match="unexpected source shards"):
        audit_v3(source, output)

    assert (
        json.loads((output / "audit-report.json").read_text())["status"] == "incomplete"
    )


def test_unexpected_action_proposal_shard_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    proposal_dir = source / "v3" / "action_proposals"
    proposal_dir.mkdir(exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_id": "gpu-000000",
                    "step_idx": 0,
                    "schema_version": "3.0.0",
                    "proposed_action": "hold",
                    "teacher_action": None,
                }
            ]
        ),
        proposal_dir / "train-00001.parquet",
    )

    with pytest.raises(AuditError, match="unexpected source shards"):
        audit_v3(source, output)

    assert (
        json.loads((output / "audit-report.json").read_text())["status"] == "incomplete"
    )


def test_rebuild_removes_every_stale_owned_view_shard(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    stale = output / "v3-forecast-eligible-v1" / "train-00001.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale shard")

    audit_v3(source, output)

    assert not stale.exists()


def test_symlinked_view_directory_is_rejected_without_touching_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    target = source / "v3" / "state_telemetry"
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in target.glob("*.parquet")
    }
    output.mkdir()
    (output / "v3-forecast-eligible-v1").symlink_to(target, target_is_directory=True)

    with pytest.raises(AuditError, match="must not be a symlink"):
        audit_v3(source, output)

    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in target.glob("*.parquet")
    }
    assert after == before
    assert (
        json.loads((output / "audit-report.json").read_text())["status"] == "incomplete"
    )


@pytest.mark.parametrize(
    "artifact_name", ["manifest.json", "audit-report.json", "exclusions.parquet"]
)
def test_dangling_root_artifact_symlink_is_removed_without_following(
    tmp_path: Path, artifact_name: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    outside = tmp_path / "outside" / artifact_name
    _write_corpus(source)
    output.mkdir()
    (output / artifact_name).symlink_to(outside)

    audit_v3(source, output)

    assert not outside.exists()
    assert not (output / artifact_name).is_symlink()
    assert (output / artifact_name).is_file()


def test_dangling_view_shard_symlink_is_removed_without_following(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    outside = tmp_path / "outside" / "train-00000.parquet"
    _write_corpus(source)
    shard = output / "v3-forecast-eligible-v1" / "train-00000.parquet"
    shard.parent.mkdir(parents=True)
    shard.symlink_to(outside)

    audit_v3(source, output)

    assert not outside.exists()
    assert not shard.is_symlink()
    assert shard.is_file()


def test_incomplete_evidence_tolerates_unreadable_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    original_sha256 = audit_module._sha256

    def fail_train_hash(path: Path) -> str:
        if path.name == "train-00000.parquet":
            raise OSError("source disappeared")
        return original_sha256(path)

    monkeypatch.setattr(audit_module, "_sha256", fail_train_hash)

    with pytest.raises(AuditError, match="cannot write audit outputs"):
        audit_v3(source, output)

    report = json.loads((output / "audit-report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["status"] == "incomplete"
    assert manifest["status"] == "incomplete"
    assert not any(
        name.endswith("train-00000.parquet") for name in manifest["source_files"]
    )


def test_source_hash_change_during_audit_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    original_hashes = audit_module._available_source_hashes
    calls = 0

    def changed_hashes(
        dataset_root: Path, v3_root: Path, *, tolerate_unreadable: bool = False
    ) -> dict[str, str]:
        nonlocal calls
        hashes = original_hashes(
            dataset_root, v3_root, tolerate_unreadable=tolerate_unreadable
        )
        calls += 1
        if calls == 2:
            first = next(iter(hashes))
            hashes[first] = "0" * 64
        return hashes

    monkeypatch.setattr(audit_module, "_available_source_hashes", changed_hashes)

    with pytest.raises(AuditError, match="source shards changed during audit"):
        audit_v3(source, output)

    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_late_unexpected_source_shard_fails_final_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    original_hashes = audit_module._available_source_hashes
    calls = 0

    def add_shard_before_final_snapshot(
        dataset_root: Path, v3_root: Path, *, tolerate_unreadable: bool = False
    ) -> dict[str, str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            canonical = v3_root / "state_telemetry" / "train-00000.parquet"
            (canonical.parent / "train-00001.parquet").write_bytes(canonical.read_bytes())
        return original_hashes(
            dataset_root, v3_root, tolerate_unreadable=tolerate_unreadable
        )

    monkeypatch.setattr(
        audit_module, "_available_source_hashes", add_shard_before_final_snapshot
    )

    with pytest.raises(AuditError, match="unexpected source shards"):
        audit_v3(source, output)

    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_source_shard_membership_is_rechecked_after_directory_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    original_sha256 = audit_module._sha256
    inserted = False

    def add_shard_during_hash(path: Path) -> str:
        nonlocal inserted
        digest = original_sha256(path)
        if not inserted and path.parent.name == "state_telemetry":
            inserted = True
            (path.parent / "train-00001.parquet").write_bytes(path.read_bytes())
        return digest

    monkeypatch.setattr(audit_module, "_sha256", add_shard_during_hash)

    with pytest.raises(AuditError, match="membership changed during hashing"):
        audit_v3(source, output)

    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


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


def test_noncanonical_episode_identifier_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    for config in ("state_telemetry", "outcomes"):
        path = source / "v3" / config / "validation-00000.parquet"
        table = pq.read_table(path)
        aliases = pa.array(["gpu-0"] * table.num_rows, type=pa.string())
        table = table.set_column(
            table.schema.get_field_index("episode_id"), "episode_id", aliases
        )
        pq.write_table(table, path)

    with pytest.raises(AuditError, match="noncanonical episode_id"):
        audit_v3(source, output)

    report = json.loads((output / "audit-report.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["status"] == "incomplete"
    assert manifest["status"] == "incomplete"


@pytest.mark.parametrize(
    ("episode_id", "match"),
    [
        ("gpu-²", "cannot recover source row_index"),
        ("gpu-" + "9" * 5_000, "noncanonical"),
    ],
)
def test_unparseable_digit_episode_identifier_fails_closed(
    tmp_path: Path, episode_id: str, match: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    for config in ("state_telemetry", "outcomes"):
        path = source / "v3" / config / "validation-00000.parquet"
        table = pq.read_table(path)
        values = pa.array([episode_id] * table.num_rows, type=pa.string())
        table = table.set_column(
            table.schema.get_field_index("episode_id"), "episode_id", values
        )
        pq.write_table(table, path)

    with pytest.raises(AuditError, match=match):
        audit_v3(source, output)

    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_null_episode_identifier_fails_closed_with_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    for config in ("state_telemetry", "outcomes"):
        path = source / "v3" / config / "validation-00000.parquet"
        table = pq.read_table(path)
        nulls = pa.nulls(table.num_rows, type=pa.string())
        table = table.set_column(
            table.schema.get_field_index("episode_id"), "episode_id", nulls
        )
        pq.write_table(table, path)

    with pytest.raises(AuditError, match="episode_id must be a non-empty string"):
        audit_v3(source, output)

    assert (
        json.loads((output / "audit-report.json").read_text())["status"] == "incomplete"
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_null_step_index_fails_closed_with_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    for config in ("state_telemetry", "outcomes"):
        path = source / "v3" / config / "validation-00000.parquet"
        table = pq.read_table(path)
        values = table.column("step_idx").to_pylist()
        values[0] = None
        table = table.set_column(
            table.schema.get_field_index("step_idx"),
            "step_idx",
            pa.array(values, type=pa.int32()),
        )
        pq.write_table(table, path)

    with pytest.raises(AuditError, match="step_idx must be an integer"):
        audit_v3(source, output)

    assert (
        json.loads((output / "audit-report.json").read_text())["status"] == "incomplete"
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize("proposal_defect", ["duplicate", "orphan"])
def test_action_proposal_keys_must_be_unique_and_match_state(
    tmp_path: Path, proposal_defect: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    proposal_dir = source / "v3" / "action_proposals"
    proposal_dir.mkdir(exist_ok=True)
    rows = [
        {
            "episode_id": "gpu-000000",
            "step_idx": 0,
            "schema_version": "3.0.0",
            "proposed_action": "hold",
            "teacher_action": None,
        }
    ]
    rows.append(
        dict(rows[0])
        if proposal_defect == "duplicate"
        else {
            "episode_id": "gpu-999999",
            "step_idx": 0,
            "schema_version": "3.0.0",
            "proposed_action": "hold",
            "teacher_action": None,
        }
    )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PROPOSALS_SCHEMA),
        proposal_dir / "train-00000.parquet",
    )

    expected = "duplicate" if proposal_defect == "duplicate" else "orphan"
    with pytest.raises(AuditError, match=expected):
        audit_v3(source, output)

    assert (
        json.loads((output / "audit-report.json").read_text())["status"] == "incomplete"
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_empty_required_split_fails_closed_with_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    for config in ("state_telemetry", "outcomes"):
        path = source / "v3" / config / "validation-00000.parquet"
        pq.write_table(pq.read_table(path).slice(0, 0), path)

    with pytest.raises(AuditError, match="validation source split is empty"):
        audit_v3(source, output)

    assert (
        json.loads((output / "audit-report.json").read_text())["status"] == "incomplete"
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


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


def test_window_gap_does_not_hide_later_sensor_rejection(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    steps = [0, *range(2, 66)]
    _write_corpus(source, steps=steps, zero_at=2)

    report = audit_v3(source, output)

    reasons = report.splits["train"]["exclusion_reasons"]
    assert reasons["window_has_index_gap"] >= 1
    assert reasons["window_contains_zero_power_w"] >= 1


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


def test_distribution_mean_is_overflow_resistant() -> None:
    # Exercise float64 aggregation directly; published v3 sensor fields are float32.
    result = audit_module._distribution(pa.array([1e308] * 66, type=pa.float64()))
    assert result["mean"] == 1e308


@pytest.mark.parametrize(
    "relative_output", [".", "v3", "v3/state_telemetry/audit", "full_data/audit"]
)
def test_output_cannot_overlap_source_tree(tmp_path: Path, relative_output: str) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    output = source / relative_output
    output.mkdir(parents=True, exist_ok=True)
    sentinel = output / "manifest.json"
    sentinel.write_text("source-owned evidence")

    with pytest.raises(AuditError, match="overlap source"):
        audit_v3(source, output)

    assert sentinel.read_text() == "source-owned evidence"


@pytest.mark.parametrize("location", ["target", "child", "parent"])
def test_symlinked_v3_target_cannot_overlap_output(tmp_path: Path, location: str) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    target = tmp_path / "external" / "v3"
    target.parent.mkdir()
    (source / "v3").rename(target)
    (source / "v3").symlink_to(target, target_is_directory=True)
    output = {"target": target, "child": target / "audit", "parent": target.parent}[
        location
    ]
    output.mkdir(exist_ok=True)
    sentinel = output / "manifest.json"
    sentinel.write_text("source sentinel")
    with pytest.raises(AuditError, match="overlap"):
        audit_v3(source, output)
    assert sentinel.read_text() == "source sentinel"


def test_missing_action_shard_is_incomplete(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    (source / "v3/action_proposals/train-00000.parquet").unlink()
    output = tmp_path / "audit"
    with pytest.raises(AuditError, match="missing source shard"):
        audit_v3(source, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize("config", ["state_telemetry", "outcomes", "action_proposals"])
@pytest.mark.parametrize("version", ["missing", "4.0.0", None])
def test_unsupported_source_schema_is_incomplete(
    tmp_path: Path, config: str, version: str | None
) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    path = source / "v3" / config / "train-00000.parquet"
    table = pq.read_table(path).drop(["schema_version"])
    if config == "action_proposals":
        table = pa.Table.from_pylist(
            [
                {
                    "episode_id": "gpu-000000",
                    "step_idx": 0,
                    "proposed_action": None,
                    "teacher_action": None,
                }
            ],
            schema=PROPOSALS_SCHEMA,
        ).drop(["schema_version"])
    if version != "missing":
        table = table.append_column(
            "schema_version", pa.array([version] * table.num_rows, type=pa.string())
        )
    pq.write_table(table, path)
    output = tmp_path / "audit"
    with pytest.raises(AuditError, match="schema_version"):
        audit_v3(source, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_string_reward_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    path = source / "v3/outcomes/train-00000.parquet"
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("reward"),
        "reward",
        pa.array(["1.0"] * table.num_rows),
    )
    pq.write_table(table, path)
    with pytest.raises(AuditError, match="reward"):
        audit_v3(source, tmp_path / "audit")


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_audit_publication_does_not_follow_replacement_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link_kind: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    sentinel = tmp_path / "source-sentinel.json"
    sentinel.write_text("source sentinel")
    original = audit_module._clean_owned_outputs

    def replace_after_cleanup(root: Path) -> None:
        original(root)
        destination = root / "manifest.json"
        if link_kind == "symlink":
            destination.symlink_to(sentinel)
        else:
            destination.hardlink_to(sentinel)

    monkeypatch.setattr(audit_module, "_clean_owned_outputs", replace_after_cleanup)
    audit_v3(source, output)
    assert sentinel.read_text() == "source sentinel"


def test_audit_parses_the_same_bytes_it_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    target = source / "v3/state_telemetry/train-00000.parquet"
    original_bytes = target.read_bytes()
    read_table = pq.read_table
    changed = read_table(target)
    changed = changed.set_column(
        changed.schema.get_field_index("mem_util_pct"),
        "mem_util_pct",
        pa.array([77.0] * changed.num_rows),
    )

    def replace_while_reading(path: object, *args: object, **kwargs: object) -> pa.Table:
        if path == target:
            pq.write_table(changed, target)
            try:
                return read_table(path, *args, **kwargs)
            finally:
                target.write_bytes(original_bytes)
        return read_table(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", replace_while_reading)
    audit_v3(source, output)
    view = read_table(output / "v3-forecast-eligible-v1/train-00000.parquet")
    assert view.column("mem_util_pct").to_pylist() == [20.0, 21.0]


def test_missing_source_with_view_symlink_writes_incomplete_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit"
    output.mkdir()
    target = tmp_path / "untouched"
    target.mkdir()
    sentinel = target / "manifest.json"
    sentinel.write_text("preserve")
    view = output / "v3-forecast-eligible-v1"
    view.symlink_to(target, target_is_directory=True)
    (output / "manifest.json").write_text('{"status": "complete"}')
    with pytest.raises(AuditError, match="cannot find v3 state_telemetry"):
        audit_v3(tmp_path / "missing-source", output)
    assert view.is_symlink()
    assert sentinel.read_text() == "preserve"
    for name in ("manifest.json", "audit-report.json"):
        assert json.loads((output / name).read_text())["status"] == "incomplete"


def test_hash_pass_rechecks_earlier_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    original = audit_module._sha256
    added = source / "v3/state_telemetry/train-00001.parquet"

    def add_while_hashing_later_directory(path: Path) -> str:
        digest = original(path)
        if path.parent.name == "outcomes" and not added.exists():
            added.write_bytes(b"new shard")
        return digest

    monkeypatch.setattr(audit_module, "_sha256", add_while_hashing_later_directory)
    with pytest.raises(AuditError, match="membership changed"):
        audit_module._available_source_hashes(source, source / "v3")


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
@pytest.mark.parametrize(
    "artifact", ["v3-forecast-eligible-v1/train-00000.parquet", "exclusions.parquet"]
)
def test_parquet_publication_replaces_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link_kind: str, artifact: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    sentinel = tmp_path / "source-sentinel.parquet"
    sentinel.write_bytes(b"preserve source bytes")
    original = audit_module._clean_owned_outputs

    def replace_after_cleanup(root: Path) -> None:
        original(root)
        destination = root / artifact
        destination.parent.mkdir(parents=True, exist_ok=True)
        if link_kind == "symlink":
            destination.symlink_to(sentinel)
        else:
            destination.hardlink_to(sentinel)

    monkeypatch.setattr(audit_module, "_clean_owned_outputs", replace_after_cleanup)
    audit_v3(source, output)
    assert sentinel.read_bytes() == b"preserve source bytes"


def test_publication_rejects_replaced_view_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    target = source / "v3/state_telemetry"
    before = {p.name: p.read_bytes() for p in target.glob("*.parquet")}
    original = audit_module._clean_owned_outputs
    replaced = False

    def swap_after_cleanup(root: Path) -> None:
        nonlocal replaced
        original(root)
        if not replaced:
            view = root / "v3-forecast-eligible-v1"
            if view.exists():
                view.rmdir()
            view.symlink_to(target, target_is_directory=True)
            replaced = True

    monkeypatch.setattr(audit_module, "_clean_owned_outputs", swap_after_cleanup)
    with pytest.raises(AuditError, match="cannot write audit outputs"):
        audit_v3(source, output)
    assert {p.name: p.read_bytes() for p in target.glob("*.parquet")} == before
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_source_change_during_publication_prevents_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    target = source / "v3/state_telemetry/train-00000.parquet"
    original = audit_module.write_parquet

    def change_after_output(path: Path, table: pa.Table) -> None:
        original(path, table)
        target.write_bytes(target.read_bytes() + b"changed")

    monkeypatch.setattr(audit_module, "write_parquet", change_after_output)
    with pytest.raises(AuditError, match="source shards changed during publication"):
        audit_v3(source, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_directory_swap_during_replace_keeps_source_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spikenaut_etl import artifacts

    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    target = source / "v3/state_telemetry"
    before = {p.name: p.read_bytes() for p in target.glob("*.parquet")}
    original = artifacts.os.replace
    swapped = False

    def swap_before_replace(src: str, dst: str, **kwargs: int) -> None:
        nonlocal swapped
        if dst.endswith(".parquet") and not swapped:
            view = output / "v3-forecast-eligible-v1"
            view.rename(output / "detached-view")
            view.symlink_to(target, target_is_directory=True)
            swapped = True
        original(src, dst, **kwargs)

    monkeypatch.setattr(artifacts.os, "replace", swap_before_replace)
    with pytest.raises(AuditError, match="directory changed during publication"):
        audit_v3(source, output)
    assert {p.name: p.read_bytes() for p in target.glob("*.parquet")} == before
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_eligible_view_preserves_action_labels_and_typed_missing_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    actions = pa.Table.from_pylist(
        [
            {
                "episode_id": "gpu-000000",
                "step_idx": 1,
                "schema_version": "3.0.0",
                "proposed_action": "hold",
                "teacher_action": "cool",
            }
        ],
        schema=PROPOSALS_SCHEMA,
    )
    pq.write_table(actions, source / "v3/action_proposals/train-00000.parquet")
    audit_v3(source, output)
    view = pq.read_table(output / "v3-forecast-eligible-v1/train-00000.parquet")
    assert view.column("proposed_action").to_pylist() == [None, "hold"]
    assert view.column("teacher_action").to_pylist() == [None, "cool"]
    assert (
        view.schema.field("proposed_action").type
        == PROPOSALS_SCHEMA.field("proposed_action").type
    )


def test_state_outcome_column_collision_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    path = source / "v3/state_telemetry/train-00000.parquet"
    state = pq.read_table(path)
    pq.write_table(state.append_column("reward", pa.array([42.0] * state.num_rows)), path)
    with pytest.raises(AuditError, match="column.*collision"):
        audit_v3(source, tmp_path / "audit")


def test_audit_output_generation_cannot_change_between_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    original = audit_module.write_json

    def swap_after_report(path: Path, value: object) -> None:
        original(path, value)
        if path.name == "audit-report.json" and not (tmp_path / "detached").exists():
            output.rename(tmp_path / "detached")
            output.mkdir()

    monkeypatch.setattr(audit_module, "write_json", swap_after_report)
    with pytest.raises(AuditError, match="publication directory changed"):
        audit_v3(source, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_audit_output_generation_cannot_change_at_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    original = audit_module.write_json

    def swap_before_manifest(path: Path, value: object) -> None:
        if path.name == "manifest.json" and not (tmp_path / "detached").exists():
            output.rename(tmp_path / "detached")
            output.mkdir()
        original(path, value)

    monkeypatch.setattr(audit_module, "write_json", swap_before_manifest)
    with pytest.raises(AuditError, match="publication directory changed"):
        audit_v3(source, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize(
    "name", ["proposed_action", "teacher_action", "label_confidence"]
)
def test_action_proposal_types_match_published_schema(tmp_path: Path, name: str) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    path = source / "v3/action_proposals/train-00000.parquet"
    table = pq.read_table(path)
    index = table.schema.get_field_index(name)
    table = table.set_column(index, name, pa.array([], type=pa.int64()))
    pq.write_table(table, path)
    with pytest.raises(AuditError, match="proposal.*type"):
        audit_v3(source, tmp_path / "audit")


@pytest.mark.parametrize(
    "name",
    [
        "source_split",
        "source_row_index",
        "target_step_idx",
        "forecast_horizon_samples",
        "d_gpu_temp_c_64",
    ],
)
def test_generated_column_names_are_reserved(tmp_path: Path, name: str) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    path = source / "v3/state_telemetry/train-00000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.append_column(name, pa.array([1] * table.num_rows)), path)
    with pytest.raises(AuditError, match="generated.*collision"):
        audit_v3(source, tmp_path / "audit")


@pytest.mark.parametrize("mutation", ["replace", "extra"])
def test_output_snapshot_rechecked_after_source_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    original = audit_module._available_source_hashes

    def mutate_after_hash(*args: object, **kwargs: object) -> dict[str, str]:
        result = original(*args, **kwargs)
        if (output / "audit-report.json").exists():
            view = output / audit_module.VIEW_ID
            target = view / (
                "train-00000.parquet" if mutation == "replace" else "extra.parquet"
            )
            target.write_bytes(b"changed")
        return result

    monkeypatch.setattr(audit_module, "_available_source_hashes", mutate_after_hash)
    with pytest.raises(AuditError, match="output.*changed"):
        audit_v3(source, output)


def test_extra_action_proposal_field_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    path = source / "v3/action_proposals/train-00000.parquet"
    table = pq.read_table(path)
    pq.write_table(
        table.append_column("unexpected", pa.array([], type=pa.string())), path
    )
    with pytest.raises(AuditError, match="unexpected action proposal fields"):
        audit_v3(source, tmp_path / "audit")


def test_cleanup_cannot_follow_replaced_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "audit"
    _write_corpus(source)
    output.mkdir()
    sentinel = source / "manifest.json"
    sentinel.write_text("source sentinel")
    original = audit_module._guard_output_path

    def swap_after_guard(*args: object) -> None:
        original(*args)
        output.rename(tmp_path / "detached")
        output.symlink_to(source, target_is_directory=True)

    monkeypatch.setattr(audit_module, "_guard_output_path", swap_after_guard)
    with pytest.raises((AuditError, OSError)):
        audit_v3(source, output)
    assert sentinel.read_text() == "source sentinel"


@pytest.mark.parametrize(
    "config,field,mutation",
    [
        ("state_telemetry", "synthetic", "missing"),
        ("state_telemetry", "step_idx", "type"),
        ("outcomes", "d_tokens_per_s", "type"),
    ],
)
def test_all_state_outcome_fields_match_schema(
    tmp_path: Path, config: str, field: str, mutation: str
) -> None:
    source = tmp_path / "source"
    _write_corpus(source)
    path = source / "v3" / config / "train-00000.parquet"
    table = pq.read_table(path)
    if mutation == "missing":
        table = table.drop([field])
    else:
        table = table.set_column(
            table.schema.get_field_index(field), field, pa.array([1.0] * table.num_rows)
        )
    pq.write_table(table, path)
    with pytest.raises(AuditError, match="schema field"):
        audit_v3(source, tmp_path / "audit")

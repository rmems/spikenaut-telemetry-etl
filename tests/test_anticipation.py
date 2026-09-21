from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import spikenaut_etl.anticipation as anticipation
from spikenaut_etl.anticipation import PreparationError, _statistics, prepare_campaign
from spikenaut_etl.cli import main


def _rows(start: int = 1_000_000, count: int = 111) -> list[dict[str, int]]:
    return [
        {
            "timestamp_ms": start + index * 100,
            "power_usage_mw": 100_000 + index * 1_000,
            "temperature_c": 40 + index,
            "graphics_clock_mhz": 2_000 + index,
            "memory_clock_mhz": 10_000 + index,
            "memory_used_mb": 4_000 + index,
        }
        for index in range(count)
    ]


def _session(root: Path, session_id: str, rows: list[dict[str, int]]) -> Path:
    path = root / session_id
    path.mkdir()
    labeled_rows = [dict(row, session_label=session_id) for row in rows]
    pq.write_table(
        pa.Table.from_pylist(labeled_rows), path / "gpu_telemetry_v2_batch_0.parquet"
    )
    manifest = {
        "schema_version": 1,
        "session_id": session_id,
        "session_label": session_id,
        "started_at_utc": "2026-09-20T00:00:00Z",
        "run_started_at_utc": "2026-09-20T00:00:00Z",
        "ended_at_utc": "2026-09-20T00:02:30Z",
        "poll_interval_ms_requested": 100,
        "collector_version": "0.4.0",
        "git_commit": "a" * 40,
        "restart_count": 0,
        "unclean_restart_count": 0,
        "host": {"gpu_name": "fixture", "driver": "fixture", "cpu_model": "fixture"},
        "workload": {"class": "ai-compute", "label": session_id},
        "parquet_write_failures": 0,
        "timing": {
            "scope": "latest_process",
            "poll_interval_ms_requested": 100,
            "sample_count": len(rows),
            "observed_interval_ms": {"p50": 100.0, "p95": 100.0, "max": 100.0},
            "late_sample_count": 0,
            "skipped_tick_estimate": 0,
            "elapsed_basis": "monotonic",
            "row_timestamp_basis": "utc_wall_clock",
        },
        "prior_runs": [],
    }
    (path / "session_manifest.json").write_text(json.dumps(manifest))
    return path


def _campaign(
    tmp_path: Path, sessions: list[tuple[str, str, list[dict[str, int]]]]
) -> Path:
    entries = []
    for seed, (session_id, split, rows) in enumerate(sessions, start=1):
        path = _session(tmp_path, session_id, rows)
        entries.append(
            {"session_id": session_id, "split": split, "path": str(path), "seed": seed}
        )
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"min_examples_per_session": 1, "sessions": entries}))
    return campaign


def test_builds_causal_histories_and_first_future_targets(tmp_path: Path) -> None:
    rows = _rows()
    # The first observation at/after the +1 s deadline is 50 ms late and must be used.
    rows[61]["timestamp_ms"] += 50
    prepared = prepare_campaign(
        _campaign(tmp_path, [("session-01", "train", rows)]), tmp_path / "out"
    )

    session = prepared["sessions"][0]
    example = next(
        example for example in session["examples"] if example["frame_index"] == 51
    )
    assert example["history_indices"] == [46, 41, 31, 1]
    assert example["target_timestamps_ms"] == [1_006_150, 1_010_100]
    # Current row is index 51: temp 91 C / power 151 W.
    assert example["y"] == [10.0, 10.0, 50.0, 50.0]
    assert max(example["history_indices"]) < example["frame_index"]
    assert (tmp_path / "out" / "prepared.json").exists()


def test_rejects_future_observation_more_than_100_ms_after_deadline(
    tmp_path: Path,
) -> None:
    rows = _rows(count=180)
    del rows[61]
    rows[61]["timestamp_ms"] += 1
    prepared = prepare_campaign(
        _campaign(tmp_path, [("session-01", "train", rows)]), tmp_path / "out"
    )
    session = prepared["sessions"][0]

    assert not any(example["frame_index"] == 51 for example in session["examples"])
    assert prepared["quality"]["rejections"]["target_late"] >= 1


def test_first_future_observation_cannot_be_skipped_when_its_sensor_is_invalid(
    tmp_path: Path,
) -> None:
    rows = _rows(count=180)
    rows[61]["power_usage_mw"] = 0
    prepared = prepare_campaign(
        _campaign(tmp_path, [("session-01", "train", rows)]), tmp_path / "out"
    )

    assert not any(
        example["frame_index"] == 51 for example in prepared["sessions"][0]["examples"]
    )
    assert prepared["quality"]["rejections"]["target_invalid"] >= 1


def test_later_invalid_read_does_not_retroactively_poison_valid_target(
    tmp_path: Path,
) -> None:
    rows = _rows(count=180)
    rows[101]["timestamp_ms"] += 20
    invalid = dict(rows[101])
    invalid["timestamp_ms"] += 30
    invalid["temperature_c"] = 0
    rows.insert(102, invalid)

    prepared = prepare_campaign(
        _campaign(tmp_path, [("session-01", "train", rows)]), tmp_path / "out"
    )

    example = next(
        item for item in prepared["sessions"][0]["examples"] if item["frame_index"] == 51
    )
    assert example["target_timestamps_ms"][1] == 1_010_120


def test_invalid_gap_splits_segments_and_blocks_history_and_targets(
    tmp_path: Path,
) -> None:
    rows = _rows(count=230)
    rows[70]["temperature_c"] = 0
    prepared = prepare_campaign(
        _campaign(tmp_path, [("session-01", "train", rows)]), tmp_path / "out"
    )
    session = prepared["sessions"][0]

    assert session["frames"][70]["valid"] is False
    assert "invalid_sensor" in session["frames"][70]["rejection_reasons"]
    assert session["frames"][69]["segment_id"] != session["frames"][71]["segment_id"]
    assert not any(70 in example["history_indices"] for example in session["examples"])
    assert not any(65 <= example["frame_index"] < 121 for example in session["examples"])


def test_invalid_raw_read_between_ticks_cannot_disappear(tmp_path: Path) -> None:
    rows = _rows(count=180)
    invalid = dict(rows[0])
    invalid["timestamp_ms"] += 50
    invalid["temperature_c"] = 0
    rows.insert(1, invalid)

    prepared = prepare_campaign(
        _campaign(tmp_path, [("session-01", "train", rows)]), tmp_path / "out"
    )

    frame = prepared["sessions"][0]["frames"][1]
    assert frame["valid"] is False
    assert "invalid_sensor" in frame["rejection_reasons"]


def test_trailing_off_grid_invalid_read_is_retained(tmp_path: Path) -> None:
    rows = _rows(count=180)
    invalid = dict(rows[-1])
    invalid["timestamp_ms"] += 50
    invalid["temperature_c"] = 0
    rows.append(invalid)

    prepared = prepare_campaign(
        _campaign(tmp_path, [("session-01", "train", rows)]), tmp_path / "out"
    )

    frame = prepared["sessions"][0]["frames"][-1]
    assert frame["timestamp_ms"] == invalid["timestamp_ms"] + 50
    assert frame["valid"] is False
    assert "invalid_sensor" in frame["rejection_reasons"]


def test_extreme_timestamp_gap_fails_before_frame_expansion(tmp_path: Path) -> None:
    rows = _rows(count=111)
    rows[-1]["timestamp_ms"] = rows[0]["timestamp_ms"] + 100_000 * 100
    campaign = _campaign(tmp_path, [("session-01", "train", rows)])

    with pytest.raises(PreparationError, match="maximum is 100000"):
        prepare_campaign(campaign, tmp_path / "out")

    assert (
        json.loads((tmp_path / "out" / "manifest.json").read_text())["status"]
        == "incomplete"
    )


def test_raw_gap_over_200ms_breaks_segment_even_when_next_row_hits_grid(
    tmp_path: Path,
) -> None:
    rows = _rows(count=180)
    del rows[1:3]

    prepared = prepare_campaign(
        _campaign(tmp_path, [("session-01", "train", rows)]), tmp_path / "out"
    )

    frame = prepared["sessions"][0]["frames"][3]
    assert frame["valid"] is False
    assert "source_gap" in frame["rejection_reasons"]


def test_clock_reversal_fails_closed_before_timestamp_reordering(tmp_path: Path) -> None:
    rows = _rows(count=250)
    rows[100]["timestamp_ms"] = rows[99]["timestamp_ms"] - 1
    with pytest.raises(PreparationError, match="clock reversal"):
        prepare_campaign(
            _campaign(tmp_path, [("session-01", "train", rows)]), tmp_path / "out"
        )


def test_stale_gap_is_retained_and_resets_segment(tmp_path: Path) -> None:
    rows = _rows(count=230)
    del rows[70:74]
    prepared = prepare_campaign(
        _campaign(tmp_path, [("session-01", "train", rows)]), tmp_path / "out"
    )
    frames = prepared["sessions"][0]["frames"]

    assert any("stale_source" in frame["rejection_reasons"] for frame in frames)
    assert len({frame["segment_id"] for frame in frames}) >= 2


def test_campaign_id_matches_collector_labels_and_preserves_actual_id(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    manifest_path = tmp_path / "session-01" / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["session_id"] = "session-01_20260920T123456Z"
    manifest_path.write_text(json.dumps(manifest))

    prepared = prepare_campaign(campaign, tmp_path / "out")

    assert prepared["sessions"][0]["session_id"] == "session-01"
    assert (
        prepared["provenance"]["sources"][0]["collector_session_id"]
        == "session-01_20260920T123456Z"
    )


def test_batches_are_read_in_numeric_order_and_zero_vram_is_valid(tmp_path: Path) -> None:
    rows = _rows(count=180)
    rows[0]["memory_used_mb"] = 0
    campaign = _campaign(tmp_path, [("session-01", "train", rows)])
    session_path = tmp_path / "session-01"
    (session_path / "gpu_telemetry_v2_batch_0.parquet").unlink()
    labeled_rows = [dict(row, session_label="session-01") for row in rows]
    pq.write_table(
        pa.Table.from_pylist(labeled_rows[:90]),
        session_path / "gpu_telemetry_v2_batch_2.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(labeled_rows[90:]),
        session_path / "gpu_telemetry_v2_batch_10.parquet",
    )

    prepared = prepare_campaign(campaign, tmp_path / "out")

    assert prepared["sessions"][0]["frames"][0]["valid"] is True
    assert prepared["sessions"][0]["frames"][0]["x"][0] == 0.0


def test_normalization_uses_training_only_and_records_constants(tmp_path: Path) -> None:
    train = _rows(count=121)
    validation = _rows(start=2_000_000, count=121)
    for row in train:
        row["memory_used_mb"] = 4_096
    for row in validation:
        row["memory_used_mb"] = 99_999
        row["temperature_c"] += 10_000

    prepared = prepare_campaign(
        _campaign(
            tmp_path,
            [("session-01", "train", train), ("session-07", "validation", validation)],
        ),
        tmp_path / "out",
    )
    normalization = prepared["normalization"]

    assert normalization["fit_split"] == "train"
    assert normalization["x_mean"][0] == 4_096.0
    assert normalization["x_raw_std"][0] == 0.0
    assert normalization["x_std"][0] == 1.0
    assert normalization["x_constant"] == [True, False, False, False, False]
    assert normalization["x_mean"][2] < 1_000
    assert prepared["quality"]["out_of_training_range"]["validation"][0] > 0
    assert prepared["quality"]["out_of_training_range"]["validation"][2] > 0


def test_normalization_statistics_are_overflow_resistant_and_finite() -> None:
    statistics = _statistics([[1.7e308], [-1.7e308], [1.7e308]], 1, "x")

    assert all(math.isfinite(value) for value in statistics["x_mean"])
    assert all(math.isfinite(value) for value in statistics["x_std"])
    assert all(math.isfinite(value) for value in statistics["x_raw_std"])

    with pytest.raises(PreparationError, match="non-finite x normalization"):
        _statistics([[math.inf]], 1, "x")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda manifest: manifest.pop("ended_at_utc"), "ended_at_utc"),
        (
            lambda manifest: manifest.__setitem__("ended_at_utc", "not-a-timestamp"),
            "valid UTC timestamp",
        ),
        (
            lambda manifest: manifest.__setitem__("ended_at_utc", True),
            "valid UTC timestamp",
        ),
        (
            lambda manifest: manifest.__setitem__("parquet_write_failures", 1),
            "write failures",
        ),
        (
            lambda manifest: manifest.__setitem__("poll_interval_ms_requested", 200),
            "100 ms",
        ),
        (
            lambda manifest: manifest.__setitem__("unclean_restart_count", 1),
            "unclean restart",
        ),
        (lambda manifest: manifest.__setitem__("restart_count", 1), "restarted capture"),
        (
            lambda manifest: manifest["timing"].__setitem__("sample_count", 1),
            "sample_count",
        ),
        (lambda manifest: manifest.__setitem__("session_label", "wrong"), "session_id"),
    ],
)
def test_manifest_completeness_fails_closed(tmp_path: Path, mutate, match: str) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    manifest_path = tmp_path / "session-01" / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(PreparationError, match=match):
        prepare_campaign(campaign, tmp_path / "out")


def test_preserves_assignments_and_rejects_duplicate_ids_or_bad_splits(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    body = json.loads(campaign.read_text())
    body["sessions"].append({**body["sessions"][0], "split": "test"})
    campaign.write_text(json.dumps(body))
    with pytest.raises(PreparationError, match="duplicate session_id"):
        prepare_campaign(campaign, tmp_path / "out")

    body["sessions"] = [body["sessions"][0] | {"split": "holdout"}]
    campaign.write_text(json.dumps(body))
    with pytest.raises(PreparationError, match="split"):
        prepare_campaign(campaign, tmp_path / "out")

    body["sessions"] = [body["sessions"][0] | {"split": []}]
    campaign.write_text(json.dumps(body))
    output = tmp_path / "out"
    (output / "prepared.json").write_text("stale complete result")
    with pytest.raises(PreparationError, match="split"):
        prepare_campaign(campaign, output)
    assert not (output / "prepared.json").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize(
    "artifact_name",
    [
        "prepared.json",
        "quality-report.json",
        "manifest.json",
        "prepared.json.tmp",
        "quality-report.json.tmp",
        "manifest.json.tmp",
    ],
)
def test_campaign_cannot_collide_with_output_artifacts(
    tmp_path: Path, artifact_name: str
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    output = tmp_path / "out"
    output.mkdir()
    colliding_campaign = output / artifact_name
    original = campaign.read_text()
    colliding_campaign.write_text(original)

    with pytest.raises(PreparationError, match="collides with output artifact"):
        prepare_campaign(colliding_campaign, output)

    assert colliding_campaign.read_text() == original


def test_symlinked_staging_path_cannot_overwrite_campaign(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    original = campaign.read_bytes()
    output = tmp_path / "out"
    output.mkdir()
    (output / "manifest.json.tmp").symlink_to(campaign)

    with pytest.raises(PreparationError, match="collides with output artifact"):
        prepare_campaign(campaign, output)

    assert campaign.read_bytes() == original


def test_noncampaign_output_symlink_routes_through_incomplete_cleanup(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    output = tmp_path / "out"
    output.mkdir()
    (output / "prepared.json").write_text("stale complete result")
    outside = tmp_path / "outside.json"
    (output / "quality-report.json.tmp").symlink_to(outside)

    with pytest.raises(PreparationError, match="staging path is a symlink"):
        prepare_campaign(campaign, output)

    assert not outside.exists()
    assert not (output / "prepared.json").exists()
    assert (
        json.loads((output / "quality-report.json").read_text())["status"] == "incomplete"
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_cyclic_output_symlink_routes_through_incomplete_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    output = tmp_path / "out"
    output.mkdir()
    loop = output / "prepared.json"
    loop.symlink_to(loop)
    real_resolve = Path.resolve

    def reject_loop(path: Path, *args: object, **kwargs: object) -> Path:
        if path == loop:
            raise RuntimeError("Symlink loop")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", reject_loop)

    with pytest.raises(PreparationError, match="symlink loop"):
        prepare_campaign(campaign, output)

    assert not loop.is_symlink()
    assert (
        json.loads((output / "quality-report.json").read_text())["status"] == "incomplete"
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_cyclic_campaign_path_routes_through_incomplete_cleanup(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign-loop.json"
    campaign.symlink_to(campaign)
    output = tmp_path / "out"
    output.mkdir()
    (output / "prepared.json").write_text("stale complete result")

    with pytest.raises(PreparationError, match="campaign path has a symlink loop"):
        prepare_campaign(campaign, output)

    assert not (output / "prepared.json").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_hardlinked_staging_path_cannot_truncate_campaign(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    original = campaign.read_bytes()
    output = tmp_path / "out"
    output.mkdir()
    os.link(campaign, output / "prepared.json.tmp")

    with pytest.raises(PreparationError, match="multiply linked"):
        prepare_campaign(campaign, output)

    assert campaign.read_bytes() == original
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_collision_is_preserved_when_another_output_artifact_is_cyclic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    output = tmp_path / "out"
    output.mkdir()
    campaign = output / "manifest.json"
    original = source_campaign.read_bytes()
    campaign.write_bytes(original)
    loop = output / "prepared.json"
    loop.symlink_to(loop)
    real_resolve = Path.resolve

    def reject_loop(path: Path, *args: object, **kwargs: object) -> Path:
        if path == loop:
            raise RuntimeError("Symlink loop")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", reject_loop)

    with pytest.raises(PreparationError, match="collides with output artifact"):
        prepare_campaign(campaign, output)

    assert campaign.read_bytes() == original
    assert loop.is_symlink()


def test_boolean_collector_schema_version_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    manifest_path = tmp_path / "session-01" / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = True
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "out"

    with pytest.raises(PreparationError, match="schema_version must be 1"):
        prepare_campaign(campaign, output)

    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("parquet_write_failures", "write failures"),
        ("unclean_restart_count", "unclean restart"),
        ("restart_count", "restarted capture"),
    ],
)
def test_boolean_collector_completion_counters_fail_closed(
    tmp_path: Path, field: str, match: str
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    manifest_path = tmp_path / "session-01" / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = False
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "out"

    with pytest.raises(PreparationError, match=match):
        prepare_campaign(campaign, output)

    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_batch_suffix_is_bounded_before_integer_conversion(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    oversized = tmp_path / "session-01" / f"gpu_telemetry_v2_batch_{'9' * 100}.parquet"
    oversized.write_bytes(b"not read")

    with pytest.raises(PreparationError, match="unexpected Parquet"):
        prepare_campaign(campaign, tmp_path / "out")

    assert (
        json.loads((tmp_path / "out" / "manifest.json").read_text())["status"]
        == "incomplete"
    )


def test_late_parquet_batch_fails_final_membership_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    session = tmp_path / "session-01"
    canonical = session / "gpu_telemetry_v2_batch_0.parquet"
    original = anticipation._out_of_training_range

    def add_batch(*args: object, **kwargs: object) -> dict[str, object]:
        (session / "gpu_telemetry_v2_batch_1.parquet").write_bytes(canonical.read_bytes())
        return original(*args, **kwargs)

    monkeypatch.setattr(anticipation, "_out_of_training_range", add_batch)

    with pytest.raises(PreparationError, match="Parquet membership"):
        prepare_campaign(campaign, tmp_path / "out")

    assert (
        json.loads((tmp_path / "out" / "manifest.json").read_text())["status"]
        == "incomplete"
    )


def test_parquet_membership_is_rechecked_after_final_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    session = tmp_path / "session-01"
    canonical = session / "gpu_telemetry_v2_batch_0.parquet"
    original_sha256 = anticipation._sha256
    inserted = False

    def add_batch_during_hash(path: Path) -> str:
        nonlocal inserted
        digest = original_sha256(path)
        if not inserted:
            inserted = True
            (session / "gpu_telemetry_v2_batch_1.parquet").write_bytes(
                canonical.read_bytes()
            )
        return digest

    monkeypatch.setattr(anticipation, "_sha256", add_batch_during_hash)

    with pytest.raises(PreparationError, match="Parquet membership"):
        prepare_campaign(campaign, tmp_path / "out")

    assert (
        json.loads((tmp_path / "out" / "manifest.json").read_text())["status"]
        == "incomplete"
    )


def test_final_report_write_failure_removes_complete_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    output = tmp_path / "out"
    original_write = anticipation._write_json
    failed = False

    def fail_first_manifest(path: Path, value: object) -> None:
        nonlocal failed
        if path.name == "manifest.json" and not failed:
            failed = True
            raise OSError("blocked final manifest")
        original_write(path, value)

    monkeypatch.setattr(anticipation, "_write_json", fail_first_manifest)

    with pytest.raises(OSError, match="blocked final manifest"):
        prepare_campaign(campaign, output)

    assert not (output / "prepared.json").exists()
    assert (
        json.loads((output / "quality-report.json").read_text())["status"] == "incomplete"
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_campaign_hash_and_assignments_use_loaded_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    original = campaign.read_bytes()
    original_read_rows = anticipation._read_rows

    def replace_campaign_after_load(session_path: Path, session_id: str):
        campaign.write_text(json.dumps({"min_examples_per_session": 1, "sessions": []}))
        return original_read_rows(session_path, session_id)

    monkeypatch.setattr(anticipation, "_read_rows", replace_campaign_after_load)

    prepared = prepare_campaign(campaign, tmp_path / "out")
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())

    assert (
        prepared["provenance"]["campaign_sha256"]
        == __import__("hashlib").sha256(original).hexdigest()
    )
    assert manifest["assignments"][0]["session_id"] == "session-01"


def test_session_sources_must_match_bytes_used_for_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    original_read_rows = anticipation._read_rows

    def replace_parquet_after_read(session_path: Path, session_id: str):
        rows, snapshots = original_read_rows(session_path, session_id)
        snapshots[0][0].write_bytes(snapshots[0][0].read_bytes() + b"changed")
        return rows, snapshots

    monkeypatch.setattr(anticipation, "_read_rows", replace_parquet_after_read)

    with pytest.raises(PreparationError, match="source changed during preparation"):
        prepare_campaign(campaign, tmp_path / "out")

    assert not (tmp_path / "out" / "prepared.json").exists()
    assert (
        json.loads((tmp_path / "out" / "manifest.json").read_text())["status"]
        == "incomplete"
    )


def test_invalid_utf8_campaign_writes_incomplete_reports(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign.json"
    campaign.write_bytes(b"\xff\xfe")
    output = tmp_path / "out"
    output.mkdir()
    (output / "prepared.json").write_text("stale complete result")

    with pytest.raises(PreparationError, match="cannot read campaign"):
        prepare_campaign(campaign, output)

    assert not (output / "prepared.json").exists()
    assert (
        json.loads((output / "quality-report.json").read_text())["status"] == "incomplete"
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_oversized_campaign_integer_writes_incomplete_reports(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign.json"
    campaign.write_text('{"min_examples_per_session":' + "9" * 5000 + ',"sessions":[]}')
    output = tmp_path / "out"
    output.mkdir()
    (output / "prepared.json").write_text("stale complete result")

    with pytest.raises(PreparationError, match="cannot read campaign"):
        prepare_campaign(campaign, output)

    assert not (output / "prepared.json").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_invalid_utf8_session_manifest_writes_incomplete_reports(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    (tmp_path / "session-01" / "session_manifest.json").write_bytes(b"\xff\xfe")
    output = tmp_path / "out"

    with pytest.raises(PreparationError, match="manifest unavailable"):
        prepare_campaign(campaign, output)

    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_oversized_manifest_integer_writes_incomplete_reports(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    manifest_path = tmp_path / "session-01" / "session_manifest.json"
    original = manifest_path.read_text().rstrip()
    manifest_path.write_text(original[:-1] + ',"oversized":' + "9" * 5000 + "}")
    output = tmp_path / "out"
    output.mkdir()
    (output / "prepared.json").write_text("stale complete result")

    with pytest.raises(PreparationError, match="manifest unavailable"):
        prepare_campaign(campaign, output)

    assert not (output / "prepared.json").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_predeclared_minimum_keeps_deficient_session_visible_and_fails(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    body = json.loads(campaign.read_text())
    body["min_examples_per_session"] = 500
    campaign.write_text(json.dumps(body))

    with pytest.raises(PreparationError, match="session-01.*500"):
        prepare_campaign(campaign, tmp_path / "out")
    failure_manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    quality = json.loads((tmp_path / "out" / "quality-report.json").read_text())
    assert failure_manifest["status"] == "incomplete"
    assert failure_manifest["assignments"][0]["session_id"] == "session-01"
    assert failure_manifest["assignments"][0]["path"] == str(tmp_path / "session-01")
    assert "requires 500" in quality["failure_reasons"][0]
    assert quality["session_summaries"][0]["eligible_examples"] == 11
    assert quality["provenance"][0]["manifest_sha256"]
    assert not (tmp_path / "out" / "prepared.json").exists()


def test_cli_registers_prepare_anticipation(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    output = tmp_path / "prepared"

    assert (
        main(["prepare-anticipation", "--input", str(campaign), "--output", str(output)])
        == 0
    )
    assert (output / "prepared.json").exists()
    prepared = json.loads((output / "prepared.json").read_text())
    assert prepared["feature_map_id"] == "anticipation-observed-gpu-v1"
    assert prepared["feature_map"][0]["unit"] == "MiB"
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["session_counts"][0]["source_rows"] == 111
    assert manifest["sources"][0]["manifest_sha256"]


def test_cli_reports_prepare_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_prepare(*_args: object, **_kwargs: object) -> None:
        raise OSError("output filesystem unavailable")

    monkeypatch.setattr(anticipation, "prepare_campaign", fail_prepare)

    assert (
        main(
            [
                "prepare-anticipation",
                "--input",
                str(tmp_path / "campaign.json"),
                "--output",
                str(tmp_path / "prepared"),
            ]
        )
        == 1
    )
    assert (
        capsys.readouterr().err
        == "prepare-anticipation failed: output filesystem unavailable\n"
    )


def test_malformed_path_still_writes_incomplete_reports(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "min_examples_per_session": 1,
                "sessions": [
                    {
                        "session_id": "session-01",
                        "split": "train",
                        "path": None,
                        "seed": 1,
                    }
                ],
            }
        )
    )

    with pytest.raises(PreparationError, match="path"):
        prepare_campaign(campaign, tmp_path / "out")
    assert (
        json.loads((tmp_path / "out" / "manifest.json").read_text())["status"]
        == "incomplete"
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("session_id", "   ", "session_id must be non-empty"),
        ("path", "bad\x00path", "path contains a null byte"),
    ],
)
def test_invalid_session_identity_or_path_writes_incomplete_reports(
    tmp_path: Path, field: str, value: str, match: str
) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    body = json.loads(campaign.read_text())
    body["sessions"][0][field] = value
    campaign.write_text(json.dumps(body))

    with pytest.raises(PreparationError, match=match):
        prepare_campaign(campaign, tmp_path / "out")

    assert (
        json.loads((tmp_path / "out" / "manifest.json").read_text())["status"]
        == "incomplete"
    )


def test_nonpositive_source_timestamp_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows(start=0))])

    with pytest.raises(PreparationError, match="positive timestamp_ms"):
        prepare_campaign(campaign, tmp_path / "out")

    assert (
        json.loads((tmp_path / "out" / "manifest.json").read_text())["status"]
        == "incomplete"
    )


def test_non_list_sessions_remove_stale_prepared_and_write_incomplete_reports(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"min_examples_per_session": 1, "sessions": None}))
    output = tmp_path / "out"
    output.mkdir()
    (output / "prepared.json").write_text("stale complete result")

    with pytest.raises(PreparationError, match="sessions must be a non-empty list"):
        prepare_campaign(campaign, output)

    assert not (output / "prepared.json").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"
    assert (
        json.loads((output / "quality-report.json").read_text())["status"] == "incomplete"
    )


def test_parquet_rows_must_match_assigned_session_label(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    parquet = tmp_path / "session-01" / "gpu_telemetry_v2_batch_0.parquet"
    table = pq.read_table(parquet)
    labels = table.column("session_label").to_pylist()
    labels[10] = "foreign-session"
    table = table.set_column(
        table.schema.get_field_index("session_label"),
        "session_label",
        pa.array(labels, type=pa.string()),
    )
    pq.write_table(table, parquet)

    with pytest.raises(PreparationError, match="row session_label"):
        prepare_campaign(campaign, tmp_path / "out")

    assert (
        json.loads((tmp_path / "out" / "manifest.json").read_text())["status"]
        == "incomplete"
    )


def test_later_session_failure_preserves_prior_verified_evidence(tmp_path: Path) -> None:
    campaign = _campaign(
        tmp_path,
        [
            ("session-01", "train", _rows()),
            ("session-02", "train", _rows(start=2_000_000)),
        ],
    )
    second_manifest = tmp_path / "session-02" / "session_manifest.json"
    body = json.loads(second_manifest.read_text())
    body["ended_at_utc"] = None
    second_manifest.write_text(json.dumps(body))

    with pytest.raises(PreparationError, match="session-02.*ended_at_utc"):
        prepare_campaign(campaign, tmp_path / "out")

    quality = json.loads((tmp_path / "out" / "quality-report.json").read_text())
    assert [item["session_id"] for item in quality["assignments"]] == [
        "session-01",
        "session-02",
    ]
    assert quality["session_summaries"][0]["session_id"] == "session-01"
    assert quality["provenance"][0]["session_id"] == "session-01"
    assert quality["failed_session_id"] == "session-02"


def test_recursive_manifest_writes_incomplete_reports(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, [("session-01", "train", _rows())])
    (tmp_path / "session-01/session_manifest.json").write_text(
        "[" * 100000 + "0" + "]" * 100000
    )
    output = tmp_path / "out"
    output.mkdir()
    (output / "prepared.json").write_text("stale complete result")
    with pytest.raises(PreparationError, match="manifest unavailable"):
        prepare_campaign(campaign, output)
    assert not (output / "prepared.json").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"

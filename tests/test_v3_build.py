"""Tests for the v3 trajectory builder.

Run against a small synthetic v2 GPU capture, with the episode length and
outcome horizon passed explicitly so the full-size constants stay out of the
test's way. What is pinned here is the *shape contract* of the published v3
tree: schemas, episode/split geometry, embargo gaps, RLDS flags, and the
refusal to fabricate or degenerate.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spikenaut_etl import v3_build
from spikenaut_etl.v3_build import (
    OUTCOMES_SCHEMA,
    PROPOSALS_SCHEMA,
    SAFETY_SCHEMA,
    SCHEMA_VERSION,
    STATE_SCHEMA,
    BuildError,
    build_action_proposals,
    build_outcomes,
    build_state_telemetry,
    build_v3,
    episode_id_str,
    split_episodes,
)

V2_GPU_COLUMNS = [
    "vddcr_gfx_v",
    "vram_temp_c",
    "gpu_temp_c",
    "power_w",
    "gpu_clock_mhz",
    "mem_clock_mhz",
    "clock_mhz",
    "fan_speed_pct",
    "mem_util_pct",
    "qubic_tick_trace",
    "qubic_tick_rate",
    "qubic_epoch_progress",
]


def write_v2_aux(root) -> None:
    """Minimal mining/hft/qubic files so build_v3's v2_parquet stage runs."""
    full_data = root / "full_data"
    full_data.mkdir(parents=True, exist_ok=True)
    with (full_data / "node_sync_harvest.jsonl").open("w") as f:
        for i in range(10):
            f.write(
                json.dumps(
                    {
                        "hashrate_mh": 1.0 + i,
                        "timestamp": f"2026-03-19T12:00:{i:02d}" if i % 2 else None,
                        "blockchain": None if i % 2 else "dynex",
                    }
                )
                + "\n"
            )
    with (full_data / "ghost_market_log.jsonl").open("w") as f:
        for i in range(10):
            f.write(
                json.dumps(
                    {
                        "timestamp": f"2026-03-11T18:22:{i:02d}",
                        "step": i,
                        "action": "observe",
                    }
                )
                + "\n"
            )
    with (full_data / "qubic_ticks_snn.jsonl").open("w") as f:
        for i in range(10):
            f.write(
                json.dumps(
                    {"timestamp": f"2026-03-20T08:55:{i:02d}+00:00", "tick": i}
                )
                + "\n"
            )


def write_v2_gpu(root, n_rows: int, clock_mismatch: bool = False) -> None:
    """A tiny but non-degenerate stand-in for neuromorphic_data.jsonl."""
    full_data = root / "full_data"
    full_data.mkdir(parents=True, exist_ok=True)
    with (full_data / "neuromorphic_data.jsonl").open("w") as f:
        for i in range(n_rows):
            clock = 2000.0 + (i % 7) * 100.0
            record = {
                "vddcr_gfx_v": 0.7 + (i % 5) * 0.01,
                "vram_temp_c": 40.0 + (i % 11),
                "gpu_temp_c": 30.0 + (i % 13),
                "power_w": 100.0 + (i % 17),
                "gpu_clock_mhz": clock,
                "mem_clock_mhz": 10000.0 + (i % 3) * 100.0,
                "clock_mhz": clock + (1.0 if clock_mismatch else 0.0),
                "fan_speed_pct": float(i % 100),
                "mem_util_pct": float(i % 50),
                "qubic_tick_trace": (i % 10) / 10.0,
                "qubic_tick_rate": 0.0,
                "qubic_epoch_progress": (i % 100) / 100.0,
                "row_index": i,
            }
            f.write(json.dumps(record) + "\n")


# Shrunk geometry so a few hundred rows exercise every split.
EP_LEN = 10
HORIZON = 3


# --------------------------------------------------------------------------- #
# Split geometry
# --------------------------------------------------------------------------- #


def test_split_blocks_are_chronological_and_embargoed():
    layout = split_episodes(20)
    assert layout.train == range(0, 14)
    assert layout.validation == range(15, 18)
    assert layout.test == range(19, 20)
    assert layout.embargo == (14, 18)
    # No episode is in two places; embargo episodes are in none.
    seen: set[int] = set()
    for block in (layout.train, layout.validation, layout.test, layout.embargo):
        for episode in block:
            assert episode not in seen
            seen.add(episode)
    assert seen == set(range(20))


def test_split_of_covers_every_episode():
    layout = split_episodes(20)
    names = [layout.split_of(e) for e in range(20)]
    assert names.count(None) == 2
    assert names.index("validation") > max(
        i for i, n in enumerate(names) if n == "train"
    )
    assert names.index("test") > max(
        i for i, n in enumerate(names) if n == "validation"
    )


def test_too_few_episodes_refuse_to_split():
    with pytest.raises(BuildError):
        split_episodes(4)


def test_full_size_geometry_matches_the_real_capture():
    """The layout the production build will publish for 813,973 rows."""
    layout = split_episodes(199)
    assert layout.train == range(0, 139)
    assert layout.embargo == (139, 169)
    assert layout.validation == range(140, 169)
    assert layout.test == range(170, 199)


# --------------------------------------------------------------------------- #
# Full build on a synthetic capture
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One full build shared by the read-only assertions below."""
    root = tmp_path_factory.mktemp("dataset_repo")
    write_v2_gpu(root, n_rows=205)  # 21 episodes, last one short
    write_v2_aux(root)
    report = build_v3(root, episode_len=EP_LEN, horizon=HORIZON)
    return root, report


def test_build_writes_every_config(built):
    root, report = built
    v3 = root / "v3"
    for config, splits in {
        "gpu_telemetry": ["full"],
        "qubic_signals": ["full"],
        "state_telemetry": ["train", "validation", "test"],
        "action_proposals": ["train", "validation", "test"],
        "safety_filter_log": ["train", "validation", "test"],
        "outcomes": ["train", "validation", "test"],
        "encoding_params": ["train"],
    }.items():
        for split in splits:
            assert (v3 / config / f"{split}-00000.parquet").exists()
    for v2_config in ("gpu_telemetry", "mining", "hft", "qubic_ticks"):
        assert (root / "v2_parquet" / v2_config / "train-00000.parquet").exists()
        assert report.configs[v2_config]["train"] > 0
    assert report.schema_version == SCHEMA_VERSION
    assert (v3 / "build_report.json").exists()


def test_v2_files_are_untouched(built):
    root, _ = built
    assert (root / "full_data" / "neuromorphic_data.jsonl").exists()


def test_gpu_v3_drops_qubic_and_duplicate_clock(built):
    root, _ = built
    table = pq.read_table(root / "v3" / "gpu_telemetry" / "full-00000.parquet")
    names = set(table.column_names)
    assert "qubic_tick_trace" not in names
    assert "clock_mhz" not in names
    assert "gpu_clock_mhz" in names
    assert table.column("ts_utc").null_count == table.num_rows
    assert table.column("ts_synthetic_offset_ms").null_count == table.num_rows
    assert table.column("schema_version")[0].as_py() == SCHEMA_VERSION


def test_qubic_signals_carry_only_qubic_columns(built):
    root, _ = built
    table = pq.read_table(root / "v3" / "qubic_signals" / "full-00000.parquet")
    assert set(table.column_names) == {
        "row_index",
        "ts_utc",
        "ts_synthetic_offset_ms",
        "schema_version",
        "qubic_tick_trace",
        "qubic_tick_rate",
        "qubic_epoch_progress",
    }


def test_state_schema_and_backfill(built):
    root, _ = built
    table = pq.read_table(root / "v3" / "state_telemetry" / "train-00000.parquet")
    assert table.schema.equals(STATE_SCHEMA)
    # Real signals came through; absent signals are null, not zero.
    assert table.column("gpu_temp_c").null_count == 0
    assert table.column("sm_clock_mhz").null_count == 0
    assert table.column("throttle_reasons").null_count == table.num_rows
    assert table.column("regime").null_count == table.num_rows
    # Measured data, so synthetic is a real False on every row.
    assert table.column("synthetic").to_pylist() == [False] * table.num_rows


def test_no_episode_spans_two_splits(built):
    root, _ = built
    by_split = {}
    for split in ("train", "validation", "test"):
        table = pq.read_table(
            root / "v3" / "state_telemetry" / f"{split}-00000.parquet"
        )
        by_split[split] = set(table.column("episode_id").to_pylist())
    assert not by_split["train"] & by_split["validation"]
    assert not by_split["train"] & by_split["test"]
    assert not by_split["validation"] & by_split["test"]


def test_embargo_episodes_are_published_nowhere(built):
    root, report = built
    published: set[str] = set()
    for split in ("train", "validation", "test"):
        table = pq.read_table(
            root / "v3" / "state_telemetry" / f"{split}-00000.parquet"
        )
        published |= set(table.column("episode_id").to_pylist())
    assert report.embargo_episodes
    assert not published & set(report.embargo_episodes)


def test_backfill_emits_zero_teacher_labels(built):
    """v2 rows carry no throttle mask or ECC counters: labeling must refuse."""
    root, _ = built
    for split in ("train", "validation", "test"):
        table = pq.read_table(
            root / "v3" / "action_proposals" / f"{split}-00000.parquet"
        )
        assert table.num_rows == 0
        assert table.schema.equals(PROPOSALS_SCHEMA)


def test_safety_filter_log_is_empty_but_typed(built):
    root, _ = built
    table = pq.read_table(
        root / "v3" / "safety_filter_log" / "train-00000.parquet"
    )
    assert table.num_rows == 0
    assert table.schema.equals(SAFETY_SCHEMA)


def test_encoding_params_min_max_fit_on_train_only(built):
    root, _ = built
    encoding = pq.read_table(root / "v3" / "encoding_params" / "train-00000.parquet")
    train = pq.read_table(root / "v3" / "state_telemetry" / "train-00000.parquet")
    rows = {r["feature_name"]: r for r in encoding.to_pylist()}
    temps = [t for t in train.column("gpu_temp_c").to_pylist() if t is not None]
    assert rows["gpu_temp_c"]["min"] == pytest.approx(min(temps))
    assert rows["gpu_temp_c"]["max"] == pytest.approx(max(temps))
    assert rows["gpu_temp_c"]["encoder_type"] == "rate"
    assert rows["gpu_temp_c"]["q8_8_scale"] == 256
    # Features with no v2 source have honest null ranges, not invented ones.
    assert rows["ecc_dbe_vol"]["min"] is None
    assert rows["ecc_dbe_vol"]["encoder_type"] == "delta"
    assert rows["ecc_dbe_vol"]["threshold"] == 1.0


# --------------------------------------------------------------------------- #
# Outcomes: horizon deltas and RLDS flags
# --------------------------------------------------------------------------- #


def test_outcomes_delta_and_rlds_flags(tmp_path):
    write_v2_gpu(tmp_path, n_rows=205)
    gpu = v3_build._read_gpu_v2(tmp_path)
    state = build_state_telemetry(gpu, episode_len=EP_LEN)
    outcomes = build_outcomes(state, episode_len=EP_LEN, horizon=HORIZON)
    assert outcomes.schema.equals(OUTCOMES_SCHEMA)

    rows = outcomes.to_pylist()
    temps = state.column("gpu_temp_c").to_pylist()
    # In-episode delta is the real forward difference...
    assert rows[0]["d_gpu_temp_c"] == pytest.approx(temps[3] - temps[0])
    # ...and the last `horizon` steps of an episode are null, never guessed.
    assert rows[9]["d_gpu_temp_c"] is None
    assert rows[8]["d_gpu_temp_c"] is None
    assert rows[7]["d_gpu_temp_c"] is None
    assert rows[6]["d_gpu_temp_c"] is not None

    assert rows[0]["is_first"] and not rows[0]["is_last"]
    assert rows[9]["is_last"] and not rows[9]["is_first"]
    # Episodes end by windowing (truncation), not termination.
    assert all(r["is_terminal"] is False for r in rows)
    assert all(r["discount"] == pytest.approx(1.0) for r in rows)
    # The short final episode still closes cleanly.
    assert rows[-1]["is_last"]
    assert rows[-1]["episode_id"] == episode_id_str(20)


def test_teacher_labels_appear_when_signals_exist(built):
    """Same builder path, but with core signals present: labels flow."""
    root, _ = built
    train = pq.read_table(root / "v3" / "state_telemetry" / "train-00000.parquet")
    n = train.num_rows
    train = train.set_column(
        train.schema.get_field_index("throttle_reasons"),
        "throttle_reasons",
        pa.array([0x40] * n, type=pa.int64()),
    )
    train = train.set_column(
        train.schema.get_field_index("ecc_dbe_vol"),
        "ecc_dbe_vol",
        pa.array([0] * n, type=pa.int64()),
    )
    proposals = build_action_proposals(train)
    assert proposals.num_rows == n
    first = proposals.to_pylist()[0]
    assert first["teacher_action"] == "throttle_clocks"
    assert first["teacher_action_id"] == 1
    assert first["teacher_rule_id"] == "TR-003-HW-THERMAL"
    assert first["label_source"] == "teacher_rule"
    assert first["label_confidence"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_clock_duplicate_assumption_is_guarded(tmp_path):
    write_v2_gpu(tmp_path, n_rows=205, clock_mismatch=True)
    write_v2_aux(tmp_path)
    with pytest.raises(BuildError, match="clock_mhz"):
        build_v3(tmp_path, episode_len=EP_LEN, horizon=HORIZON)


def test_missing_source_is_a_build_error(tmp_path):
    with pytest.raises(BuildError, match="missing v2 source"):
        build_v3(tmp_path)


def test_unreadable_source_is_a_build_error(tmp_path):
    write_v2_aux(tmp_path)
    (tmp_path / "full_data" / "neuromorphic_data.jsonl").write_text("{not json at all\n")
    with pytest.raises(BuildError, match="cannot read"):
        build_v3(tmp_path)


def test_aux_source_failure_does_not_replace_v3(tmp_path):
    """A bad mining/hft/qubic JSONL must fail before v3/ is cleaned or written."""
    write_v2_gpu(tmp_path, n_rows=205)
    write_v2_aux(tmp_path)
    build_v3(tmp_path, episode_len=EP_LEN, horizon=HORIZON)
    report = (tmp_path / "v3" / "build_report.json").read_text()
    shard = (tmp_path / "v3" / "gpu_telemetry" / "full-00000.parquet").read_bytes()
    (tmp_path / "full_data" / "node_sync_harvest.jsonl").write_text("{not json\n")
    with pytest.raises(BuildError):
        build_v3(tmp_path, episode_len=EP_LEN, horizon=HORIZON)
    assert (tmp_path / "v3" / "build_report.json").read_text() == report
    assert (
        tmp_path / "v3" / "gpu_telemetry" / "full-00000.parquet"
    ).read_bytes() == shard


def test_missing_aux_source_fails_before_any_write(tmp_path):
    """A v2 source the late conversion stage would need must fail the build
    up front, leaving no partial v3 tree behind."""
    write_v2_gpu(tmp_path, n_rows=205)
    with pytest.raises(BuildError, match="missing v2 source"):
        build_v3(tmp_path, episode_len=EP_LEN, horizon=HORIZON)
    assert not (tmp_path / "v3").exists()
    assert not (tmp_path / "v2_parquet").exists()


def test_rebuild_removes_stale_owned_shards(tmp_path):
    write_v2_gpu(tmp_path, n_rows=205)
    write_v2_aux(tmp_path)
    build_v3(tmp_path, episode_len=EP_LEN, horizon=HORIZON)
    stale = tmp_path / "v3" / "state_telemetry" / "train-00001.parquet"
    stale.write_bytes(b"stale shard from an older build")
    foreign = tmp_path / "v3" / "notes.txt"
    foreign.write_text("not builder-owned; must survive")
    build_v3(tmp_path, episode_len=EP_LEN, horizon=HORIZON)
    assert not stale.exists()
    assert foreign.exists()


def test_gpu_v3_and_qubic_signals_row_counts_match_source(built):
    root, report = built
    assert report.configs["gpu_telemetry_v3"]["full"] == 205
    assert report.configs["qubic_signals"]["full"] == 205
    total_state = sum(report.configs["state_telemetry"].values())
    # 21 episodes of 10 (last short), minus two 10-row embargo episodes.
    assert total_state == 205 - 20

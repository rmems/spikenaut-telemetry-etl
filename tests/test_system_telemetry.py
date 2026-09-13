"""system_telemetry_v1 source: gaming-telemetry sensors, game-blind.

Covers load (JSONL + Hub parquet layout), allow_constant hardware invariants,
fail-loud zeros-as-real, and the prohibition on folding a play session into
mining Hub v3/state_telemetry episodes.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError as PydanticValidationError

from spikenaut_etl import clean, timestamps, v3_build
from spikenaut_etl.cli import main
from spikenaut_etl.ingest import IngestError
from spikenaut_etl.pipeline import SOURCES, run_all, run_source
from spikenaut_etl.schemas import (
    SYSTEM_TELEMETRY_FEATURE_AXONS,
    SYSTEM_TELEMETRY_HARDWARE_INVARIANTS,
    SYSTEM_TELEMETRY_HYGIENE_COLUMNS,
    CleanSystemTelemetry,
    RawSystemTelemetry,
)
from spikenaut_etl.system_telemetry import SOURCE_KEY, discover_input
from spikenaut_etl.v2_parquet import V2_SOURCES
from spikenaut_etl.validate import check_all

FIXTURES = Path(__file__).parent / "fixtures"
CORRUPT = FIXTURES / "corrupt"
SOURCE_FIXTURE = FIXTURES / "system_telemetry_v1.jsonl"


def _first_source_row() -> dict:
    return json.loads(SOURCE_FIXTURE.read_text().splitlines()[0])


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _spec():
    return next(s for s in SOURCES if s.key == SOURCE_KEY)


# --------------------------------------------------------------------------- #
# Load path
# --------------------------------------------------------------------------- #


_INT_SENSOR_FIELDS = (
    "power_usage_mw",
    "temperature_c",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "pcie_rx_kbps",
    "pcie_tx_kbps",
    "pstate",
    "throttle_reasons_bitmask",
    "fan_speed_perc",
    "memory_used_mb",
    "memory_total_mb",
    "encoder_util_perc",
    "decoder_util_perc",
)

_FLOAT_SENSOR_FIELDS = (
    "cpu_tctl_c",
    "cpu_ccd1_c",
    "cpu_ccd2_c",
    "cpu_package_power_w",
)

_BOOL_REJECT_FIELDS = ("timestamp_ms", *_INT_SENSOR_FIELDS, *_FLOAT_SENSOR_FIELDS)

_LEGACY_JSONL = (
    "neuromorphic_data.jsonl",
    "node_sync_harvest.jsonl",
    "qubic_ticks.jsonl",
    "ghost_market_log.jsonl",
)


def test_source_is_registered_and_named_system_telemetry_v1():
    keys = [s.key for s in SOURCES]
    assert SOURCE_KEY in keys
    assert keys == [
        "neuromorphic_data",
        "node_sync_harvest",
        "qubic_ticks_snn",
        "ghost_market_log",
        "system_telemetry_v1",
    ]
    assert {s.key for s in SOURCES if s.optional} == {SOURCE_KEY}


def test_jsonl_load_derives_ts_utc_from_timestamp_ms():
    result = clean.clean_system_telemetry(SOURCE_FIXTURE)
    assert result.ingest.n_parsed == 12
    assert len(result.rows) == 12
    row = result.rows[0]
    assert row["timestamp_ms"] == 1775992159648
    moment = timestamps.from_epoch_ms(row["timestamp_ms"])
    assert row["ts_utc"] == moment.isoformat()
    assert moment == datetime(2026, 4, 12, 11, 9, 19, 648000, tzinfo=timezone.utc)
    assert result.epochs[0] == pytest.approx(moment.timestamp())
    assert {r["session_label"] for r in result.rows} == {"lab"}


def test_timestamp_ms_must_not_go_through_parse_as_seconds():
    ms = 1775992159648
    utc = timestamps.from_epoch_ms(ms)
    assert utc.year == 2026
    # parse() treats a bare int as epoch *seconds*. That path cannot represent
    # this millisecond clock. Do not match the ValueError text; CPython does
    # not guarantee it across versions.
    with pytest.raises(ValueError):
        timestamps.parse(ms)


def test_hub_parquet_layout_loads(tmp_path):
    rows = [json.loads(line) for line in SOURCE_FIXTURE.read_text().splitlines() if line]
    parquet_path = tmp_path / "re4" / "train-00000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    discovered = discover_input(tmp_path)
    assert discovered == tmp_path
    result = clean.clean_system_telemetry(tmp_path)
    assert len(result.rows) == 12
    assert result.rows[0]["timestamp_ms"] == rows[0]["timestamp_ms"]


def test_pipeline_accepts_jsonl_fixture(tmp_path):
    outcome = run_source(_spec(), FIXTURES, tmp_path / "out", tmp_path / "reports")
    assert outcome.ok, outcome.rendered
    assert outcome.output_path is not None
    assert outcome.output_path.name == "system_telemetry_v1.jsonl"
    assert not list((tmp_path / "out").glob("samples/**/*"))


def test_missing_optional_system_telemetry_is_successful_skip(tmp_path):
    outcome = run_source(_spec(), tmp_path, tmp_path / "out", tmp_path / "reports")
    assert outcome.ok
    assert outcome.output_path is None
    assert outcome.rendered.startswith("SKIP  system_telemetry_v1:")


def test_missing_required_source_is_failed_skip(tmp_path):
    spec = next(s for s in SOURCES if s.key == "neuromorphic_data")
    outcome = run_source(spec, tmp_path, tmp_path / "out", tmp_path / "reports")
    assert not spec.optional
    assert not outcome.ok
    assert outcome.rendered.startswith("SKIP  neuromorphic_data:")


@pytest.mark.parametrize("command", ["validate", "report", "clean"])
def test_four_source_tree_without_gaming_telemetry_exits_zero(command, tmp_path):
    for name in _LEGACY_JSONL:
        shutil.copy(FIXTURES / name, tmp_path / name)
    argv = [command, "--input", str(tmp_path), "--reports", str(tmp_path / "reports")]
    if command == "clean":
        argv.extend(["--output", str(tmp_path / "out")])
    assert main(argv) == 0
    outcomes = run_all(
        tmp_path, tmp_path / "out", tmp_path / "reports", write_output=False
    )
    by_key = {o.key: o for o in outcomes}
    assert by_key[SOURCE_KEY].ok
    assert by_key[SOURCE_KEY].rendered.startswith("SKIP")
    assert all(outcome.ok for outcome in outcomes)


@pytest.mark.parametrize("command", ["validate", "report", "clean"])
def test_explicit_only_missing_system_telemetry_exits_nonzero(command, tmp_path):
    argv = [
        command,
        "--input",
        str(tmp_path),
        "--reports",
        str(tmp_path / "reports"),
        "--only",
        SOURCE_KEY,
    ]
    if command == "clean":
        argv.extend(["--output", str(tmp_path / "out")])
    assert main(argv) == 1
    outcomes = run_all(
        tmp_path,
        tmp_path / "out",
        tmp_path / "reports",
        only=[SOURCE_KEY],
        write_output=False,
    )
    assert len(outcomes) == 1
    assert not outcomes[0].ok
    assert outcomes[0].rendered.startswith("SKIP  system_telemetry_v1:")


def test_only_including_optional_source_fails_when_absent(tmp_path):
    for name in _LEGACY_JSONL:
        shutil.copy(FIXTURES / name, tmp_path / name)
    argv = [
        "validate",
        "--input",
        str(tmp_path),
        "--reports",
        str(tmp_path / "reports"),
        "--only",
        "ghost_market_log",
        SOURCE_KEY,
    ]
    assert main(argv) == 1
    outcomes = run_all(
        tmp_path,
        tmp_path / "out",
        tmp_path / "reports",
        only=["ghost_market_log", SOURCE_KEY],
        write_output=False,
    )
    by_key = {o.key: o for o in outcomes}
    assert by_key["ghost_market_log"].ok
    assert not by_key[SOURCE_KEY].ok
    assert by_key[SOURCE_KEY].rendered.startswith("SKIP  system_telemetry_v1:")


def test_multiple_sessions_in_one_input_fail_loud(tmp_path):
    rows = [json.loads(line) for line in SOURCE_FIXTURE.read_text().splitlines() if line]
    table = pa.Table.from_pylist(rows)
    for session in ("re4", "other"):
        path = tmp_path / session / "train-00000.parquet"
        path.parent.mkdir()
        pq.write_table(table, path)
    with pytest.raises(IngestError, match="multiple play sessions"):
        clean.clean_system_telemetry(tmp_path)


def test_jsonl_and_parquet_together_are_refused(tmp_path):
    rows = [json.loads(line) for line in SOURCE_FIXTURE.read_text().splitlines() if line]
    _write_jsonl(tmp_path / "system_telemetry_v1.jsonl", rows)
    pq.write_table(
        pa.Table.from_pylist(rows), tmp_path / "system_telemetry_v1.parquet"
    )
    with pytest.raises(IngestError, match="duplicate captures"):
        clean.clean_system_telemetry(tmp_path)


def test_mixed_session_labels_fail_loud(tmp_path):
    a = _first_source_row()
    b = _varied(a, 1)
    b["session_label"] = "other"
    path = _write_jsonl(tmp_path / "mixed.jsonl", [a, b])
    with pytest.raises(IngestError, match="mixed session_label"):
        clean.clean_system_telemetry(path)


def test_overflow_timestamp_ms_is_ingest_error_not_crash(tmp_path):
    row = _first_source_row()
    row["timestamp_ms"] = 10**18
    path = _write_jsonl(tmp_path / "system_telemetry_v1.jsonl", [row])
    with pytest.raises(IngestError, match="not a convertible UTC clock"):
        clean.clean_system_telemetry(path)
    outcome = run_source(
        _spec(), tmp_path, tmp_path / "out", tmp_path / "reports"
    )
    assert not outcome.ok
    assert outcome.output_path is None


def test_corrupt_parquet_is_ingest_error_not_crash(tmp_path):
    path = tmp_path / "system_telemetry_v1.parquet"
    path.write_bytes(b"not a parquet file")
    with pytest.raises(IngestError, match="cannot read parquet"):
        clean.clean_system_telemetry(path)


# --------------------------------------------------------------------------- #
# Game-blind / session_label hygiene
# --------------------------------------------------------------------------- #


def test_session_label_is_hygiene_not_a_feature_axon():
    assert "session_label" in SYSTEM_TELEMETRY_HYGIENE_COLUMNS
    assert "session_label" in SYSTEM_TELEMETRY_HARDWARE_INVARIANTS
    assert "session_label" in CleanSystemTelemetry.model_fields
    assert "session_label" not in SYSTEM_TELEMETRY_FEATURE_AXONS
    spec = _spec()
    assert "session_label" in spec.gates.allow_constant
    assert "session_label" in spec.gates.identity_columns


# --------------------------------------------------------------------------- #
# allow_constant gates
# --------------------------------------------------------------------------- #


def test_default_gateconfig_fails_hardware_invariants():
    result = clean.clean_system_telemetry(SOURCE_FIXTURE)
    validation = check_all(
        SOURCE_KEY,
        result.rows,
        n_in=result.n_in,
        timestamps=result.epochs,
    )
    assert not validation.ok
    constants = {
        f.detail.split()[1].strip("'")
        for f in validation.failures
        if f.gate == "no_constant_columns"
    }
    assert constants == {
        "session_label",
        "memory_total_mb",
        "encoder_util_perc",
        "decoder_util_perc",
    }


def test_allow_constant_spec_passes_timestamp_gates():
    result = clean.clean_system_telemetry(SOURCE_FIXTURE)
    spec = _spec()
    validation = check_all(
        SOURCE_KEY,
        result.rows,
        n_in=result.n_in,
        config=spec.gates,
        expected_columns=set(result.rows[0]),
        timestamps=result.epochs,
    )
    assert validation.ok, validation.render()
    assert all(r["encoder_util_perc"] == 0 for r in result.rows)
    assert all(r["decoder_util_perc"] == 0 for r in result.rows)
    assert all(r["memory_total_mb"] == 16303 for r in result.rows)


# --------------------------------------------------------------------------- #
# Fail-loud: zeros-as-real / missing not coerced to 0
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", _BOOL_REJECT_FIELDS)
@pytest.mark.parametrize("flag", (True, False))
def test_bool_is_rejected_on_sensor_fields(field, flag):
    row = _first_source_row()
    row[field] = flag
    with pytest.raises(PydanticValidationError, match="not bool"):
        RawSystemTelemetry.model_validate(row)


@pytest.mark.parametrize("field", _BOOL_REJECT_FIELDS)
def test_bool_on_sensors_is_ingest_error(field, tmp_path):
    row = _first_source_row()
    row[field] = True
    path = _write_jsonl(tmp_path / "bool.jsonl", [row])
    with pytest.raises(IngestError, match="not bool"):
        clean.clean_system_telemetry(path)


def test_unavailable_zero_power_is_refused():
    with pytest.raises(IngestError, match="power_usage_mw is 0"):
        clean.clean_system_telemetry(CORRUPT / "system_telemetry_zero_power.jsonl")


@pytest.mark.parametrize(
    "field",
    [
        "temperature_c",
        "graphics_clock_mhz",
        "memory_total_mb",
        "cpu_tctl_c",
    ],
)
def test_unavailable_zero_sensors_are_refused(field, tmp_path):
    row = _first_source_row()
    row[field] = 0
    path = _write_jsonl(tmp_path / "bad.jsonl", [row])
    with pytest.raises(IngestError, match=f"{field} is 0"):
        clean.clean_system_telemetry(path)


def test_missing_sensor_stays_null_not_zero(tmp_path):
    row = _first_source_row()
    missing = dict(row)
    del missing["power_usage_mw"]
    path = _write_jsonl(
        tmp_path / "missing.jsonl",
        [missing, _varied(row, 1), _varied(row, 2)],
    )
    result = clean.clean_system_telemetry(path)
    assert "power_usage_mw" in result.rows[0]
    assert result.rows[0]["power_usage_mw"] is None
    assert result.rows[0]["power_usage_mw"] != 0
    assert result.rows[1]["power_usage_mw"] != 0
    assert result.rows[1]["power_usage_mw"] is not None


def test_empty_sensors_fail_loud(tmp_path):
    row = {"timestamp_ms": 1775992159648, "session_label": "lab"}
    path = _write_jsonl(tmp_path / "empty.jsonl", [row])
    with pytest.raises(IngestError, match="empty system_telemetry payload"):
        clean.clean_system_telemetry(path)


def test_theseus_quarry_envelope_is_not_mapped(tmp_path):
    envelope = {
        "schema_version": 1,
        "timestamp": "2026-09-03T08:00:00Z",
        "source": "collector",
        "kind": "host_hw",
        "stem": "host_hw",
        "payload": {"type": "host_hw", "cpu_tctl_c": 70.0},
    }
    path = _write_jsonl(tmp_path / "v1.jsonl", [envelope])
    with pytest.raises(IngestError, match="Theseus-Quarry"):
        clean.clean_system_telemetry(path)


def test_encoder_decoder_zero_is_observed_not_missing():
    result = clean.clean_system_telemetry(SOURCE_FIXTURE)
    assert {r["encoder_util_perc"] for r in result.rows} == {0}
    assert {r["decoder_util_perc"] for r in result.rows} == {0}


# --------------------------------------------------------------------------- #
# No mining v3 mix
# --------------------------------------------------------------------------- #


def test_system_telemetry_is_not_a_v3_or_v2_mining_source():
    assert v3_build.GPU_SOURCE == "full_data/neuromorphic_data.jsonl"
    assert v3_build.EPISODE_PREFIX == "gpu"
    assert SOURCE_KEY not in V2_SOURCES
    assert all(SOURCE_KEY not in rel for rel in V2_SOURCES.values())
    spec = _spec()
    assert spec.output == "full_data/system_telemetry_v1.jsonl"
    assert spec.output != v3_build.GPU_SOURCE


def test_v3_build_ignores_a_play_session_sitting_in_the_dataset_root(tmp_path):
    _write_v2_gpu(tmp_path, n_rows=205)
    _write_v2_aux(tmp_path)
    (tmp_path / "full_data" / "system_telemetry_v1.jsonl").write_bytes(
        SOURCE_FIXTURE.read_bytes()
    )
    v3_build.build_v3(tmp_path, episode_len=10, horizon=3)
    table = pq.read_table(tmp_path / "v3" / "state_telemetry" / "train-00000.parquet")
    assert "session_label" not in table.column_names
    ids = set(table.column("episode_id").to_pylist())
    assert ids
    assert all(eid.startswith("gpu-") for eid in ids)
    assert "gpu-000199" not in ids
    assert table.num_rows < 95893


def _write_v2_aux(root: Path) -> None:
    full_data = root / "full_data"
    full_data.mkdir(parents=True, exist_ok=True)
    with (full_data / "node_sync_harvest.jsonl").open("w") as handle:
        for i in range(10):
            handle.write(
                json.dumps(
                    {
                        "hashrate_mh": 1.0 + i,
                        "timestamp": f"2026-03-19T12:00:{i:02d}" if i % 2 else None,
                        "blockchain": None if i % 2 else "dynex",
                    }
                )
                + "\n"
            )
    with (full_data / "ghost_market_log.jsonl").open("w") as handle:
        for i in range(10):
            handle.write(
                json.dumps(
                    {
                        "timestamp": f"2026-03-11T18:22:{i:02d}",
                        "step": i,
                        "action": "observe",
                    }
                )
                + "\n"
            )
    with (full_data / "qubic_ticks_snn.jsonl").open("w") as handle:
        for i in range(10):
            handle.write(
                json.dumps({"timestamp": f"2026-03-20T08:55:{i:02d}+00:00", "tick": i})
                + "\n"
            )


def _write_v2_gpu(root: Path, n_rows: int) -> None:
    full_data = root / "full_data"
    full_data.mkdir(parents=True, exist_ok=True)
    with (full_data / "neuromorphic_data.jsonl").open("w") as handle:
        for i in range(n_rows):
            clock = 2000.0 + (i % 7) * 100.0
            record = {
                "vddcr_gfx_v": 0.7 + (i % 5) * 0.01,
                "vram_temp_c": 40.0 + (i % 11),
                "gpu_temp_c": 30.0 + (i % 13),
                "power_w": 100.0 + (i % 17),
                "gpu_clock_mhz": clock,
                "mem_clock_mhz": 10000.0 + (i % 3) * 100.0,
                "clock_mhz": clock,
                "fan_speed_pct": float(i % 100),
                "mem_util_pct": float(i % 50),
                "qubic_tick_trace": (i % 10) / 10.0,
                "qubic_tick_rate": 0.0,
                "qubic_epoch_progress": (i % 100) / 100.0,
                "row_index": i,
            }
            handle.write(json.dumps(record) + "\n")


def _varied(row: dict, i: int) -> dict:
    """A second row so a one-row missing-sensor file is not empty-degenerate."""
    out = dict(row)
    out["timestamp_ms"] = row["timestamp_ms"] + 58
    out["power_usage_mw"] = 261000 + i
    out["temperature_c"] = 50 + i
    out["graphics_clock_mhz"] = 2800 + i
    out["memory_clock_mhz"] = 15000 + i
    out["pcie_rx_kbps"] = 910000 + i
    out["pcie_tx_kbps"] = 81000 + i
    out["pstate"] = 0
    out["throttle_reasons_bitmask"] = 4
    out["fan_speed_perc"] = 80
    out["memory_used_mb"] = 13000 + i
    out["cpu_tctl_c"] = 75.0 + i
    out["cpu_ccd1_c"] = 66.0 + i
    out["cpu_ccd2_c"] = 55.0 + i
    out["cpu_package_power_w"] = 160.0 + i
    return out

"""Published Vault full_data shapes vs LiveStimAdapter LIVE_COLUMNS.

Reproduces the 2026-09-14 host-smoke SUMMARY: ingest of published Clean GPU
failed ``telemetry: Field required``, and coin-tagged harvest rows failed
``timestamp`` wanting ``str``. This mill must accept those published shapes
without inventing zeros or clocks, and must name the v3 LIVE_COLUMNS path when
v2 GPU JSONL is offered as live-bank input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spikenaut_etl import clean, timestamps
from spikenaut_etl.cli import build_parser, main
from spikenaut_etl.contracts import (
    CONTRACT_LIVE_COLUMNS_REQUIRED,
    CONTRACT_MIXED_SHAPES,
    LIVE_COLUMNS,
    LIVE_COLUMNS_VAULT_GLOB,
)
from spikenaut_etl.ingest import ContractError, IngestError, parse_record
from spikenaut_etl.pipeline import SOURCES, run_source, write_jsonl
from spikenaut_etl.schemas import CleanGpuTelemetry, RawGpuRecord, RawNodeSyncRecord

FIXTURES = Path(__file__).parent / "fixtures"
CORRUPT = FIXTURES / "corrupt"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _live_row(i: int) -> dict:
    return {
        "episode_id": f"gpu-{i:06d}",
        "mem_util_pct": 10.0 + i,
        "power_w": 80.0 + i,
        "gpu_temp_c": 40.0 + (i % 7),
        "sm_clock_mhz": 180.0 + 10 * i,
        "mem_clock_mhz": 405.0 + 20 * i,
    }


# --------------------------------------------------------------------------- #
# Cleaned GPU JSONL (smoke: telemetry Field required)
# --------------------------------------------------------------------------- #


def test_published_clean_gpu_roundtrip_passes(tmp_path):
    """Already-flattened Clean GPU is a published-shape profile, not raw ingest."""
    first = clean.clean_gpu_telemetry(FIXTURES / "neuromorphic_data.jsonl")
    assert first.profile == "raw"
    assert first.rows, "fixture must produce published rows"
    assert "telemetry" not in first.rows[0]
    assert "gpu_clock_mhz" in first.rows[0]
    assert "sm_clock_mhz" not in first.rows[0]

    published = tmp_path / "neuromorphic_data.jsonl"
    write_jsonl(first.rows, published)
    second = clean.clean_gpu_telemetry(published)
    assert second.profile == "published"
    assert second.rows == first.rows
    assert all("sm_clock_mhz" not in row for row in second.rows)

    spec = next(s for s in SOURCES if s.key == "neuromorphic_data")
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    (source_dir / spec.filename).write_bytes(published.read_bytes())
    outcome = run_source(spec, source_dir, tmp_path / "out", tmp_path / "reports")
    assert outcome.ok, outcome.rendered
    assert "CONTRACT_LIVE_COLUMNS_REQUIRED" not in outcome.rendered


def test_published_gpu_parse_does_not_require_nested_telemetry():
    row = CleanGpuTelemetry(
        row_index=0,
        vddcr_gfx_v=0.7,
        vram_temp_c=40.0,
        gpu_temp_c=31.0,
        power_w=23.4,
        gpu_clock_mhz=1207.0,
        mem_clock_mhz=7001.0,
        clock_mhz=1207.0,
        fan_speed_pct=0.0,
        mem_util_pct=1.0,
        qubic_tick_trace=0.3,
        qubic_tick_rate=1.0,
        qubic_epoch_progress=1.0,
    ).model_dump()
    parsed = parse_record(row, RawGpuRecord, row=0, source="neuromorphic_data.jsonl")
    assert isinstance(parsed, CleanGpuTelemetry)
    assert parsed.gpu_clock_mhz == 1207.0


def test_empty_telemetry_still_fails_loud():
    with pytest.raises(IngestError, match="empty telemetry payload"):
        clean.clean_gpu_telemetry(CORRUPT / "all_empty_telemetry.jsonl")


def test_extra_field_on_published_gpu_still_fails_loud(tmp_path):
    first = clean.clean_gpu_telemetry(FIXTURES / "neuromorphic_data.jsonl")
    bad = dict(first.rows[0], unexpected=True)
    path = _write_jsonl(tmp_path / "neuromorphic_data.jsonl", [bad])
    with pytest.raises(IngestError, match="schema violation"):
        clean.clean_gpu_telemetry(path)


# --------------------------------------------------------------------------- #
# LIVE_COLUMNS contract (smoke: v2 has gpu_clock_mhz, live bank wants sm_clock)
# --------------------------------------------------------------------------- #


def test_auto_profile_names_v3_path_for_live_columns_file(tmp_path):
    spec = next(s for s in SOURCES if s.key == "neuromorphic_data")
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    _write_jsonl(source_dir / spec.filename, [_live_row(i) for i in range(8)])
    outcome = run_source(spec, source_dir, tmp_path / "out", tmp_path / "reports")
    assert not outcome.ok
    assert CONTRACT_LIVE_COLUMNS_REQUIRED in outcome.rendered
    assert LIVE_COLUMNS_VAULT_GLOB in outcome.rendered
    assert "gpu_clock_mhz" in outcome.rendered
    assert outcome.output_path is None


def test_live_columns_profile_accepts_sm_clock_projection(tmp_path):
    spec = next(s for s in SOURCES if s.key == "neuromorphic_data")
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    rows = [_live_row(i) for i in range(20)]
    _write_jsonl(source_dir / spec.filename, rows)
    outcome = run_source(
        spec,
        source_dir,
        tmp_path / "out",
        tmp_path / "reports",
        write_output=False,
        profile="live-columns",
    )
    assert outcome.ok, outcome.rendered
    assert all(col in rows[0] for col in LIVE_COLUMNS)


def test_live_columns_profile_refuses_v2_gpu_clock_without_inventing(tmp_path):
    first = clean.clean_gpu_telemetry(FIXTURES / "neuromorphic_data.jsonl")
    spec = next(s for s in SOURCES if s.key == "neuromorphic_data")
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    write_jsonl(first.rows, source_dir / spec.filename)
    outcome = run_source(
        spec,
        source_dir,
        tmp_path / "out",
        tmp_path / "reports",
        profile="live-columns",
    )
    assert not outcome.ok
    assert CONTRACT_LIVE_COLUMNS_REQUIRED in outcome.rendered
    assert LIVE_COLUMNS_VAULT_GLOB in outcome.rendered
    assert "Refusing to invent sm_clock_mhz" in outcome.rendered
    assert outcome.output_path is None
    cleaned = json.loads((source_dir / spec.filename).read_text().splitlines()[0])
    assert "gpu_clock_mhz" in cleaned
    assert "sm_clock_mhz" not in cleaned


def test_live_columns_does_not_zero_fill_missing_sensor(tmp_path):
    row = _live_row(0)
    del row["sm_clock_mhz"]
    path = _write_jsonl(tmp_path / "neuromorphic_data.jsonl", [row])
    with pytest.raises((ContractError, IngestError)):
        clean.clean_gpu_telemetry(path)


def test_mixed_raw_and_published_gpu_fails_named(tmp_path):
    raw = json.loads((FIXTURES / "neuromorphic_data.jsonl").read_text().splitlines()[0])
    published = clean.clean_gpu_telemetry(FIXTURES / "neuromorphic_data.jsonl").rows[0]
    path = _write_jsonl(tmp_path / "neuromorphic_data.jsonl", [raw, published])
    with pytest.raises(ContractError, match=CONTRACT_MIXED_SHAPES):
        clean.clean_gpu_telemetry(path)


# --------------------------------------------------------------------------- #
# Null timestamp on coin-tagged published harvest rows
# --------------------------------------------------------------------------- #


def test_published_node_sync_typed_null_timestamp_roundtrip(tmp_path):
    first = clean.clean_node_sync(FIXTURES / "node_sync_harvest.jsonl")
    null_ts = [r for r in first.rows if r.get("timestamp") is None]
    assert null_ts, "fixture must include coin-tagged rows with typed-null timestamp"
    assert all(r.get("blockchain") for r in null_ts)

    published = tmp_path / "node_sync_harvest.jsonl"
    write_jsonl(first.rows, published)
    second = clean.clean_node_sync(published)
    assert second.profile == "published"
    second_null = [r for r in second.rows if r.get("timestamp") is None]
    assert len(second_null) == len(null_ts)
    assert [r.get("timestamp") for r in second.rows] == [
        r.get("timestamp") for r in first.rows
    ], "typed-null timestamps must round-trip; clocks must not be invented"

    spec = next(s for s in SOURCES if s.key == "node_sync_harvest")
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    (source_dir / spec.filename).write_bytes(published.read_bytes())
    outcome = run_source(spec, source_dir, tmp_path / "out", tmp_path / "reports")
    assert outcome.ok, outcome.rendered


def test_raw_nested_null_timestamp_fails_loud_not_invented(tmp_path):
    line = json.loads((FIXTURES / "node_sync_harvest.jsonl").read_text().splitlines()[0])
    line["timestamp"] = None
    path = _write_jsonl(tmp_path / "node_sync_harvest.jsonl", [line])
    with pytest.raises(IngestError, match="schema violation"):
        clean.clean_node_sync(path)
    with pytest.raises(IngestError):
        parse_record(line, RawNodeSyncRecord, row=0, source="node_sync_harvest.jsonl")


def test_published_null_timestamp_is_not_dropped(tmp_path):
    first = clean.clean_node_sync(FIXTURES / "node_sync_harvest.jsonl")
    coin_rows = [r for r in first.rows if r.get("blockchain") == "dynex"]
    dated = [r for r in first.rows if r.get("timestamp")][:20]
    mixed = coin_rows[:20] + dated
    path = _write_jsonl(tmp_path / "node_sync_harvest.jsonl", mixed)
    result = clean.clean_node_sync(path)
    assert result.profile == "published"
    assert len(result.rows) == len(mixed)
    assert timestamps.UNPARSEABLE not in result.quarantine.counts()
    kept_null = sum(1 for r in result.rows if r.get("timestamp") is None)
    assert kept_null == 20


def test_all_zero_harvest_still_rejected(tmp_path):
    """Typed-null policy must not weaken the all-zero fail-loud gate."""
    spec = next(s for s in SOURCES if s.key == "node_sync_harvest")
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    (source_dir / spec.filename).write_bytes(
        (CORRUPT / "all_zero_telemetry.jsonl").read_bytes()
    )
    outcome = run_source(spec, source_dir, tmp_path / "out", tmp_path / "reports")
    assert not outcome.ok
    assert outcome.output_path is None


# --------------------------------------------------------------------------- #
# Vault layout discovery + CLI
# --------------------------------------------------------------------------- #


def test_validate_discovers_published_files_under_full_data(tmp_path):
    gpu = clean.clean_gpu_telemetry(FIXTURES / "neuromorphic_data.jsonl")
    harvest = clean.clean_node_sync(FIXTURES / "node_sync_harvest.jsonl")
    vault = tmp_path / "vault"
    full_data = vault / "full_data"
    full_data.mkdir(parents=True)
    write_jsonl(gpu.rows, full_data / "neuromorphic_data.jsonl")
    write_jsonl(harvest.rows, full_data / "node_sync_harvest.jsonl")

    gpu_spec = next(s for s in SOURCES if s.key == "neuromorphic_data")
    harvest_spec = next(s for s in SOURCES if s.key == "node_sync_harvest")
    gpu_out = run_source(gpu_spec, vault, tmp_path / "out", tmp_path / "rg")
    harvest_out = run_source(harvest_spec, vault, tmp_path / "out", tmp_path / "rh")
    assert gpu_out.ok, gpu_out.rendered
    assert harvest_out.ok, harvest_out.rendered


def test_cli_live_columns_profile_on_v2_gpu_exits_1(tmp_path):
    gpu = clean.clean_gpu_telemetry(FIXTURES / "neuromorphic_data.jsonl")
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    write_jsonl(gpu.rows, source_dir / "neuromorphic_data.jsonl")
    code = main(
        [
            "validate",
            "--input",
            str(source_dir),
            "--only",
            "neuromorphic_data",
            "--profile",
            "live-columns",
            "--reports",
            str(tmp_path / "reports"),
        ]
    )
    assert code == 1


def test_cli_help_names_vault_v3_live_columns():
    parser_help = build_parser().format_help()
    assert "v3/state_telemetry" in parser_help
    assert "live-columns" in parser_help
    assert "sm_clock" in parser_help

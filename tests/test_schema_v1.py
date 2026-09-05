"""Theseus-Quarry schema-v1 reader and fail-loud ingest gates.

Reproduces issue #4: main had no reader for ``TelemetryEnvelope``. Legacy
``read_validated`` treated v1 lines as ``Raw*`` schema violations and continued,
which is how a next collection run would silently become empty published rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spikenaut_etl import clean
from spikenaut_etl.ingest import IngestError, IngestStats, parse_record, read_validated
from spikenaut_etl.pipeline import SOURCES, run_source
from spikenaut_etl.schemas import (
    CLEAN_COLUMNS,
    COLLECTOR_SCHEMA_VERSION,
    PayloadGpuSched,
    PayloadMinerPerf,
    PayloadNodeHealth,
    RawGpuRecord,
    RawNodeSyncRecord,
    TelemetryEnvelope,
)
from spikenaut_etl.v1 import hashrate_to_mh, map_envelope_to_node_sync

SCHEMA_V1 = Path(__file__).parent / "fixtures" / "schema_v1"
CORRUPT = Path(__file__).parent / "fixtures" / "corrupt"


def _first_line(path: Path) -> dict:
    return json.loads(path.read_text().splitlines()[0])


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Empty / degenerate rows fail at ingest, not after a successful parse
# --------------------------------------------------------------------------- #


def test_empty_telemetry_fixture_fails_loud_at_ingest():
    with pytest.raises(IngestError, match="empty telemetry payload"):
        clean.clean_gpu_telemetry(CORRUPT / "all_empty_telemetry.jsonl")


def test_empty_telemetry_writes_nothing(tmp_path):
    spec = next(s for s in SOURCES if s.key == "neuromorphic_data")
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / spec.filename).write_bytes(
        (CORRUPT / "all_empty_telemetry.jsonl").read_bytes()
    )
    outcome = run_source(spec, source_dir, tmp_path / "out", tmp_path / "reports")
    assert not outcome.ok
    assert outcome.output_path is None
    assert not list((tmp_path / "out").rglob("*.jsonl"))
    assert "empty telemetry payload" in outcome.rendered
    assert outcome.report_path is not None and outcome.report_path.exists()


def test_missing_payload_columns_on_v1_status_fail_loud(tmp_path):
    """status has a message but no Clean* overlap -- refuse, do not invent."""
    line = {
        "schema_version": 1,
        "timestamp": "2026-09-03T08:00:00Z",
        "source": "collector",
        "kind": "status",
        "stem": "status_telemetry",
        "payload": {"type": "status", "message": "idle"},
    }
    path = _write_jsonl(tmp_path / "node_sync_harvest.jsonl", [line])
    with pytest.raises(IngestError, match="cannot map onto CleanNodeSync"):
        clean.clean_node_sync(path)


# --------------------------------------------------------------------------- #
# Valid v1 envelopes parse and map onto stable Clean* columns
# --------------------------------------------------------------------------- #


def test_rfc3339_z_timestamp_from_collector_parses():
    from spikenaut_etl import timestamps

    parsed = timestamps.parse("2026-09-03T08:00:00.123456Z")
    assert parsed.ok and parsed.moment is not None
    assert parsed.moment.year == 2026 and parsed.moment.tzinfo is not None


def test_rfc3339_lowercase_z_timestamp_parses():
    from spikenaut_etl import timestamps

    parsed = timestamps.parse("2026-09-03T08:00:00.123456z")
    assert parsed.ok and parsed.moment is not None
    assert parsed.moment.tzinfo is not None
    assert parsed.epoch == timestamps.parse("2026-09-03T08:00:00.123456Z").epoch


def test_collector_schema_version_is_one():
    assert COLLECTOR_SCHEMA_VERSION == 1


def test_miner_perf_envelope_parses():
    env = TelemetryEnvelope.model_validate(_first_line(SCHEMA_V1 / "miner_perf.jsonl"))
    assert env.schema_version == 1
    assert env.kind == "miner_perf"
    assert isinstance(env.payload, PayloadMinerPerf)
    assert env.payload.coin == "kaspa"
    assert env.payload.hashrate_unit == "H/s"


def test_node_health_and_gpu_sched_envelopes_parse():
    health = TelemetryEnvelope.model_validate(
        _first_line(SCHEMA_V1 / "node_health.jsonl")
    )
    gpu = TelemetryEnvelope.model_validate(_first_line(SCHEMA_V1 / "gpu_sched.jsonl"))
    assert isinstance(health.payload, PayloadNodeHealth)
    assert health.payload.height == 919876
    assert isinstance(gpu.payload, PayloadGpuSched)
    assert gpu.payload.decision == "hold"


def test_hashrate_units_convert_onto_hashrate_mh():
    assert hashrate_to_mh(1.25e9, "H/s") == pytest.approx(1250.0)
    assert hashrate_to_mh(1.31, "GH/s") == pytest.approx(1310.0)
    assert hashrate_to_mh(1280.0, "MH/s") == pytest.approx(1280.0)
    with pytest.raises(IngestError, match="unknown hashrate_unit"):
        hashrate_to_mh(1.0, "sol/s")


def test_miner_perf_maps_to_stable_node_sync_columns():
    result = clean.clean_node_sync(SCHEMA_V1 / "miner_perf.jsonl")
    assert result.ingest.n_parsed == 8
    assert len(result.rows) == 8
    assert {row["blockchain"] for row in result.rows} == {"kaspa"}
    assert result.coin_counts["kaspa"] == 8
    assert result.rows[0]["hashrate_mh"] == pytest.approx(1250.0)
    published = CLEAN_COLUMNS["node_sync_harvest"]
    for row in result.rows:
        extra = set(row) - published
        assert not extra, extra


def test_node_health_maps_to_stable_node_sync_columns():
    result = clean.clean_node_sync(SCHEMA_V1 / "node_health.jsonl")
    assert len(result.rows) == 8
    heights = {row.get("block_height") for row in result.rows}
    assert 919876 in heights
    assert 46075040 in heights, "qubic tick must map onto block_height"
    epochs = {row.get("chain_epoch") for row in result.rows if "chain_epoch" in row}
    assert epochs == {205}
    published = CLEAN_COLUMNS["node_sync_harvest"]
    for row in result.rows:
        assert set(row) <= published
    assert result.epochs and len(result.epochs) == 8


def test_node_health_tick_maps_to_block_height():
    env = TelemetryEnvelope.model_validate(
        {
            "schema_version": 1,
            "timestamp": "2026-09-03T08:01:18.400000Z",
            "source": "collector",
            "kind": "node_health",
            "stem": "qubic_telemetry",
            "payload": {
                "type": "node_health",
                "coin": "qubic",
                "epoch": 205,
                "tick": 46075040,
                "hashrate_mh": 1.02,
            },
        }
    )
    row = map_envelope_to_node_sync(env)
    assert row["block_height"] == 46075040
    assert row["chain_epoch"] == 205
    assert "tick" not in row


def test_node_health_speed_hs_maps_to_hashrate_mh():
    env = TelemetryEnvelope.model_validate(
        {
            "schema_version": 1,
            "timestamp": "2026-09-03T08:03:00Z",
            "source": "collector",
            "kind": "node_health",
            "stem": "dynex_telemetry",
            "payload": {
                "type": "node_health",
                "coin": "dynex",
                "height": 919900,
                "speed_hs": 1_250_000,
            },
        }
    )
    row = map_envelope_to_node_sync(env)
    assert row["hashrate_mh"] == pytest.approx(1.25)
    assert row["block_height"] == 919900


def test_node_health_height_and_tick_together_fail_loud():
    env = TelemetryEnvelope.model_validate(
        {
            "schema_version": 1,
            "timestamp": "2026-09-03T08:03:00Z",
            "source": "collector",
            "kind": "node_health",
            "stem": "mixed_telemetry",
            "payload": {
                "type": "node_health",
                "coin": "dynex",
                "height": 919900,
                "tick": 46075040,
            },
        }
    )
    with pytest.raises(IngestError, match="both height and tick"):
        map_envelope_to_node_sync(env)


def test_gpu_sched_is_rejected_as_sparse_for_build_v3():
    """gpu_sched cannot fill CleanGpuTelemetry without inventing clocks/qubic."""
    with pytest.raises(IngestError, match="cannot fill CleanGpuTelemetry"):
        clean.clean_gpu_telemetry(SCHEMA_V1 / "gpu_sched.jsonl")


def test_pipeline_accepts_v1_miner_perf_and_rejects_gpu_sched(tmp_path):
    node_spec = next(s for s in SOURCES if s.key == "node_sync_harvest")
    gpu_spec = next(s for s in SOURCES if s.key == "neuromorphic_data")

    node_in = tmp_path / "node"
    node_in.mkdir()
    (node_in / node_spec.filename).write_bytes(
        (SCHEMA_V1 / "miner_perf.jsonl").read_bytes()
    )
    node_out = run_source(node_spec, node_in, tmp_path / "nout", tmp_path / "nrep")
    assert node_out.ok, node_out.rendered
    assert node_out.output_path is not None and node_out.output_path.exists()

    gpu_in = tmp_path / "gpu"
    gpu_in.mkdir()
    (gpu_in / gpu_spec.filename).write_bytes((SCHEMA_V1 / "gpu_sched.jsonl").read_bytes())
    gpu_out = run_source(gpu_spec, gpu_in, tmp_path / "gout", tmp_path / "grep")
    assert not gpu_out.ok
    assert gpu_out.output_path is None
    assert not list((tmp_path / "gout").rglob("*.jsonl"))
    assert "gpu_sched" in gpu_out.rendered
    assert gpu_out.report_path is not None and gpu_out.report_path.exists()


def test_mixed_miner_perf_and_node_health_parse(tmp_path):
    mixed = (SCHEMA_V1 / "miner_perf.jsonl").read_text() + (
        SCHEMA_V1 / "node_health.jsonl"
    ).read_text()
    path = tmp_path / "mixed.jsonl"
    path.write_text(mixed)
    result = clean.clean_node_sync(path)
    assert result.ingest.n_parsed == 16
    coins = {row["blockchain"] for row in result.rows}
    assert coins == {"kaspa", "dynex", "qubic"}


# --------------------------------------------------------------------------- #
# Wrong / unknown schema fails loud and writes nothing
# --------------------------------------------------------------------------- #


def test_unknown_schema_version_fails_loud():
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    line["schema_version"] = 99
    with pytest.raises(IngestError, match="unknown schema_version 99"):
        parse_record(line, RawNodeSyncRecord, row=0, source="test.jsonl")


def test_missing_schema_version_on_v1_shaped_line_fails_loud():
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    del line["schema_version"]
    with pytest.raises(IngestError, match="missing schema_version"):
        parse_record(line, RawNodeSyncRecord, row=0, source="test.jsonl")


def test_wrong_payload_tag_fails_loud():
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    line["payload"] = {"type": "not_a_real_variant", "coin": "kaspa"}
    with pytest.raises(IngestError, match="schema violation"):
        parse_record(line, RawNodeSyncRecord, row=0, source="test.jsonl")


def test_extra_forbidden_field_fails_loud():
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    line["wallet"] = "do-not-ingest"
    with pytest.raises(IngestError, match="schema violation"):
        parse_record(line, RawNodeSyncRecord, row=0, source="test.jsonl")


def test_extra_field_on_legacy_line_fails_loud():
    line = {"telemetry": {"gpu_temp_c": 40.0}, "unexpected": True}
    with pytest.raises(IngestError, match="schema violation"):
        parse_record(line, RawGpuRecord, row=0, source="test.jsonl")


def test_kind_payload_mismatch_fails_loud():
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    line["kind"] = "gpu_sched"
    with pytest.raises(IngestError, match="does not match payload.type"):
        parse_record(line, RawNodeSyncRecord, row=0, source="test.jsonl")


def test_unknown_schema_version_writes_nothing(tmp_path):
    spec = next(s for s in SOURCES if s.key == "node_sync_harvest")
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    line["schema_version"] = 2
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    _write_jsonl(source_dir / spec.filename, [line])
    outcome = run_source(spec, source_dir, tmp_path / "out", tmp_path / "reports")
    assert not outcome.ok
    assert outcome.output_path is None
    assert not list((tmp_path / "out").rglob("*.jsonl"))
    assert "unknown schema_version" in outcome.rendered


def test_read_validated_raises_on_first_bad_v1_line(tmp_path):
    good = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    bad = dict(good, schema_version=99)
    path = _write_jsonl(tmp_path / "mixed.jsonl", [good, bad])
    stats = IngestStats(source="mixed")
    with pytest.raises(IngestError, match="unknown schema_version"):
        list(read_validated(path, RawNodeSyncRecord, stats))
    assert stats.n_parsed == 1
    assert stats.n_schema_errors == 0, "version errors raise before a schema count"


def test_coin_tag_timestamp_rejected_on_v1(tmp_path):
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    line["timestamp"] = "dynex:919876"
    path = _write_jsonl(tmp_path / "node_sync_harvest.jsonl", [line])
    with pytest.raises(IngestError, match="not an RFC3339 datetime"):
        clean.clean_node_sync(path)


def test_gpu_v1_coin_tag_timestamp_rejected(tmp_path):
    line = _first_line(SCHEMA_V1 / "gpu_sched.jsonl")
    line["timestamp"] = "dynex:919876"
    path = _write_jsonl(tmp_path / "neuromorphic_data.jsonl", [line])
    with pytest.raises(IngestError, match="not an RFC3339 datetime"):
        clean.clean_gpu_telemetry(path)


def test_v1_timestamps_feed_ordering_gate(tmp_path):
    first = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    later = dict(first)
    later["timestamp"] = "2026-09-03T10:00:00.000000Z"
    earlier = dict(first)
    earlier["timestamp"] = "2026-09-02T08:00:00.000000Z"
    # Vary hashrate so the file is not degenerate if ordering were skipped.
    later["payload"] = dict(first["payload"], hashrate=1.4e9)
    earlier["payload"] = dict(first["payload"], hashrate=1.1e9)
    path = _write_jsonl(tmp_path / "node_sync_harvest.jsonl", [first, later, earlier])
    spec = next(s for s in SOURCES if s.key == "node_sync_harvest")
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    (source_dir / spec.filename).write_bytes(path.read_bytes())
    outcome = run_source(spec, source_dir, tmp_path / "out", tmp_path / "reports")
    assert not outcome.ok
    assert "timestamps_ordered" in outcome.rendered or "backwards" in outcome.rendered
    assert not list((tmp_path / "out").rglob("*.jsonl"))


@pytest.mark.parametrize("token", [True, 1.0, "1"])
def test_non_integer_schema_version_fails_loud(token):
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    line["schema_version"] = token
    with pytest.raises(IngestError, match="unknown schema_version"):
        parse_record(line, RawNodeSyncRecord, row=0, source="test.jsonl")


@pytest.mark.parametrize("token", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_payload_value_fails_loud(token):
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    line["payload"] = dict(line["payload"], hashrate=token)
    with pytest.raises(IngestError, match="schema violation"):
        parse_record(line, RawNodeSyncRecord, row=0, source="test.jsonl")


def test_truncated_json_line_is_skipped_not_aborted(tmp_path):
    """A truncated append-only line is counted; later valid rows still parse."""
    good = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    later = dict(good)
    later["payload"] = dict(good["payload"], hashrate=1.4e9)
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps(good)
        + "\n"
        + '{"schema_version":1,"kind":\n'
        + json.dumps(later)
        + "\n",
        encoding="utf-8",
    )
    stats = IngestStats(source="mixed")
    rows = list(read_validated(path, RawNodeSyncRecord, stats))
    assert len(rows) == 2
    assert stats.n_parsed == 2
    assert stats.n_json_errors == 1
    assert stats.n_schema_errors == 0
    assert any("malformed JSON" in example for example in stats.examples)


def test_truncated_json_does_not_abort_clean(tmp_path):
    good = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    later = dict(good)
    later["payload"] = dict(good["payload"], hashrate=1.4e9)
    path = tmp_path / "node_sync_harvest.jsonl"
    path.write_text(
        json.dumps(good)
        + "\n"
        + '{"schema_version":1,"kind":\n'
        + json.dumps(later)
        + "\n",
        encoding="utf-8",
    )
    result = clean.clean_node_sync(path)
    assert result.ingest.n_json_errors == 1
    assert len(result.rows) == 2


def test_non_dict_line_notes_error_before_raise(tmp_path):
    path = tmp_path / "arr.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    stats = IngestStats(source="arr")
    with pytest.raises(IngestError, match="must be an object"):
        list(read_validated(path, RawNodeSyncRecord, stats))
    assert stats.n_schema_errors == 1
    assert stats.examples and "must be an object" in stats.examples[0]


def test_empty_coin_is_rejected():
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    line["payload"] = dict(line["payload"], coin="   ")
    env = TelemetryEnvelope.model_validate(line)
    with pytest.raises(IngestError, match="empty coin"):
        map_envelope_to_node_sync(env, source="test.jsonl", row=3)
    with pytest.raises(IngestError, match="test.jsonl:3"):
        map_envelope_to_node_sync(env, source="test.jsonl", row=3)


def test_coin_is_lowercased():
    line = _first_line(SCHEMA_V1 / "miner_perf.jsonl")
    line["payload"] = dict(line["payload"], coin="KASPA")
    env = TelemetryEnvelope.model_validate(line)
    row = map_envelope_to_node_sync(env)
    assert row["blockchain"] == "kaspa"


def test_gpu_sched_error_includes_source_row():
    with pytest.raises(IngestError, match=r"gpu_sched\.jsonl:0:"):
        clean.clean_gpu_telemetry(SCHEMA_V1 / "gpu_sched.jsonl")


def test_pipeline_undecodable_file_fails_without_writing(tmp_path):
    spec = next(s for s in SOURCES if s.key == "node_sync_harvest")
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / spec.filename).write_bytes(b"\xff\xfe not utf-8\n")
    outcome = run_source(spec, source_dir, tmp_path / "out", tmp_path / "reports")
    assert not outcome.ok
    assert outcome.output_path is None
    assert not list((tmp_path / "out").rglob("*.jsonl"))
    assert "UnicodeDecodeError" in outcome.rendered
    assert outcome.report_path is not None and outcome.report_path.exists()

"""Regression tests against the defects that actually shipped.

Each test reproduces a specific failure found in the Hugging Face dataset
``rmems/Spikenaut-SNN-Telemetry`` as published before 2026-08-03, and asserts
that this pipeline refuses it. If any of these ever pass silently, the pipeline
has regressed to the behaviour it was built to replace.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from spikenaut_etl import clean, timestamps
from spikenaut_etl.pipeline import SOURCES, run_source
from spikenaut_etl.validate import (
    GateConfig,
    ValidationError,
    assert_publishable,
    check_all,
    gate_distinct_rows,
    gate_no_constant_columns,
    gate_timestamps_not_fabricated,
)

CORRUPT = Path(__file__).parent / "fixtures" / "corrupt"
FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# Defect 1: 813,973 identical {"telemetry":{}} records
# --------------------------------------------------------------------------- #


def test_all_empty_telemetry_is_rejected(tmp_path):
    """The shipped neuromorphic_data.jsonl must not survive validation."""
    result = clean.clean_gpu_telemetry(CORRUPT / "all_empty_telemetry.jsonl")
    validation = check_all(
        "all_empty",
        result.rows,
        n_in=result.n_in,
        config=GateConfig(require_timestamp_jitter=False),
    )
    assert not validation.ok
    with pytest.raises(ValidationError):
        assert_publishable(validation)


def test_no_output_is_written_when_a_gate_fails(tmp_path):
    """A failed run must leave the previous artifact untouched."""
    spec = next(s for s in SOURCES if s.key == "node_sync_harvest")
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / spec.filename).write_bytes(
        (CORRUPT / "all_zero_telemetry.jsonl").read_bytes()
    )

    outcome = run_source(spec, source_dir, tmp_path / "out", tmp_path / "reports")

    assert not outcome.ok
    assert outcome.output_path is None
    assert not list((tmp_path / "out").rglob("*.jsonl")), (
        "no output may be written on failure"
    )
    assert outcome.report_path is not None and outcome.report_path.exists(), (
        "a failed run must still leave a report explaining why"
    )


def test_distinct_row_gate_catches_repeated_line():
    rows = [{"telemetry_empty": True} for _ in range(1000)]
    failures = gate_distinct_rows(rows, min_ratio=0.01)
    assert failures, "1 distinct row in 1000 must fail"
    assert "degenerate" in str(failures[0])


# --------------------------------------------------------------------------- #
# Defect 2: every numeric field written as 0.0 for 120,314 rows
# --------------------------------------------------------------------------- #


def test_all_zero_telemetry_is_rejected(tmp_path):
    result = clean.clean_node_sync(CORRUPT / "all_zero_telemetry.jsonl")
    validation = check_all(
        "all_zero",
        result.rows,
        n_in=result.n_in,
        config=GateConfig(allow_constant=frozenset({"timestamp", "blockchain"})),
        timestamps=result.epochs or None,
    )
    assert not validation.ok
    with pytest.raises(ValidationError):
        assert_publishable(validation)


def test_constant_column_gate_names_the_offender():
    rows = [{"power_w": 0.0, "gpu_temp_c": float(i)} for i in range(100)]
    failures = gate_no_constant_columns(rows)
    assert len(failures) == 1
    assert "power_w" in str(failures[0])


# --------------------------------------------------------------------------- #
# Defect 3: 114,250 real timestamps overwritten with a fixed 10s ramp
# --------------------------------------------------------------------------- #


def test_fabricated_timestamps_are_rejected():
    result = clean.clean_node_sync(CORRUPT / "fabricated_timestamps.jsonl")
    # Telemetry here is genuinely varied, so only the fabrication gate should fire.
    validation = check_all(
        "fabricated",
        result.rows,
        n_in=result.n_in,
        config=GateConfig(allow_constant=frozenset({"blockchain"})),
        timestamps=result.epochs,
    )
    gates = {f.gate for f in validation.failures}
    assert "timestamps_not_fabricated" in gates, (
        f"fabrication gate did not fire; failures were {validation.failures}"
    )


def test_uniform_spacing_is_flagged():
    assert gate_timestamps_not_fabricated([float(i * 10) for i in range(100)])


def test_real_jitter_passes():
    uneven = [0.0, 1.0, 2.0, 11.0, 12.0, 21.0, 22.0, 23.0]
    assert gate_timestamps_not_fabricated(uneven) == []


# --------------------------------------------------------------------------- #
# Defect 4: the ISO-8601 check that missed space-separated timestamps
# --------------------------------------------------------------------------- #


def test_space_separated_timestamp_parses():
    """`occursin("T", ts)` returned false for this form and it was overwritten."""
    parsed = timestamps.parse("2026-03-19 11:55:13.132")
    assert parsed.ok and parsed.moment is not None
    assert parsed.moment.year == 2026 and parsed.moment.minute == 55


@pytest.mark.parametrize(
    "raw",
    [
        "2026-03-19 11:55:13.132",
        "2026-03-19 11:55:13",
        "2026-03-11T18:22:37.433458521+00:00",
        "2026-03-11T18:22:37",
        "1773996924",
        1773996924,
    ],
)
def test_real_timestamp_forms_all_parse(raw):
    assert timestamps.parse(raw).ok, f"{raw!r} must parse"


def test_unparseable_timestamp_yields_none_not_a_guess():
    parsed = timestamps.parse("not-a-timestamp")
    assert not parsed.ok
    assert parsed.moment is None and parsed.iso is None


# --------------------------------------------------------------------------- #
# Defect 5: coin attribution discarded, written as ""
# --------------------------------------------------------------------------- #


def test_coin_tag_is_decomposed_not_coerced():
    parsed = timestamps.parse("dynex:919876")
    assert parsed.coin == "dynex"
    assert parsed.height == 919876
    assert parsed.moment is None, "a block height is not a clock"


def test_node_sync_recovers_coin_attribution():
    """The fixture is stratified 70/70/70 across the three timestamp forms."""
    result = clean.clean_node_sync(FIXTURES / "node_sync_harvest.jsonl")
    assert result.coin_counts["dynex"] == 70, "coin:height form"
    assert result.coin_counts["qubic"] == 70, "coin:epoch:tick form"
    assert result.coin_counts["__unattributed__"] == 70, "datetime form"
    assert len(result.quarantine) == 0, "every real timestamp form must parse"


def test_qubic_epoch_tick_form_is_fully_decomposed():
    """'qubic:204:46075040' carries both an epoch and a tick; keep both."""
    parsed = timestamps.parse("qubic:204:46075040")
    assert parsed.coin == "qubic"
    assert parsed.epoch_number == 204
    assert parsed.height == 46075040


def test_unattributed_rows_are_null_never_empty_string():
    result = clean.clean_node_sync(FIXTURES / "node_sync_harvest.jsonl")
    values = {row.get("blockchain") for row in result.rows}
    assert "" not in values, "empty string is what the broken pipeline wrote"
    assert None in values and "dynex" in values


# --------------------------------------------------------------------------- #
# Defect 6: real data must still pass, or the gates are useless
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key,filename",
    [
        ("neuromorphic_data", "neuromorphic_data.jsonl"),
        ("node_sync_harvest", "node_sync_harvest.jsonl"),
        ("qubic_ticks_snn", "qubic_ticks.jsonl"),
        ("ghost_market_log", "ghost_market_log.jsonl"),
    ],
)
def test_real_fixtures_pass_all_gates(key, filename, tmp_path):
    spec = next(s for s in SOURCES if s.key == key)
    outcome = run_source(spec, FIXTURES, tmp_path, tmp_path / "reports")
    assert outcome.ok, f"real data must pass:\n{outcome.rendered}"
    assert outcome.output_path is not None and outcome.output_path.exists()


def test_real_node_sync_preserves_actual_timestamps():
    """Output timestamps must trace to the source, not a generated sequence."""
    result = clean.clean_node_sync(FIXTURES / "node_sync_harvest.jsonl")
    stamped = [r["timestamp"] for r in result.rows if r.get("timestamp")]
    assert stamped, "datetime-form rows must retain their timestamp"
    deltas = {round(b - a, 6) for a, b in pairwise(result.epochs)}
    assert len(deltas) > 1, "real capture is not evenly spaced"

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
from spikenaut_etl.ingest import IngestError
from spikenaut_etl.pipeline import SOURCES, run_source
from spikenaut_etl.validate import (
    GateConfig,
    ValidationError,
    assert_publishable,
    check_all,
    gate_distinct_rows,
    gate_no_constant_columns,
    gate_timestamps_not_fabricated,
    gate_timestamps_ordered,
)

CORRUPT = Path(__file__).parent / "fixtures" / "corrupt"
FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# Defect 1: 813,973 identical {"telemetry":{}} records
# --------------------------------------------------------------------------- #


def test_all_empty_telemetry_is_rejected(tmp_path):
    """The shipped neuromorphic_data.jsonl must not survive ingest or validation.

    ``{"telemetry":{}}`` used to parse because every RawGpuTelemetry field is
    optional, then satisfy distinctness via ``row_index``. Ingest now refuses
    the empty payload immediately; the validate gates remain as a backstop.
    """
    with pytest.raises(IngestError, match="empty telemetry payload"):
        clean.clean_gpu_telemetry(CORRUPT / "all_empty_telemetry.jsonl")

    # Belt: if an empty-after-clean frame ever reached the gates, still refuse.
    # ≥101 rows of identity-only payloads trip both payload_columns_remain
    # (no data columns) and distinct_rows (1 payload / 101 < 0.01). Fifty
    # rows only trip the column gate (ratio 0.02 ≥ min_distinct_ratio).
    empty_after_clean = [{"row_index": i} for i in range(101)]
    validation = check_all(
        "all_empty",
        empty_after_clean,
        n_in=101,
        config=GateConfig(require_timestamp_jitter=False),
    )
    assert not validation.ok
    gates = {f.gate for f in validation.failures}
    assert "payload_columns_remain" in gates
    assert "distinct_rows" in gates
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


def test_nanosecond_precision_is_truncated_not_rejected():
    """ghost_market_log timestamps carry 9 fractional digits.

    Python's %f accepts at most 6 and datetime cannot represent finer than
    microseconds. Truncating is correct; rejecting would quarantine all 31,573
    rows of the one healthy file in the dataset.
    """
    parsed = timestamps.parse("2026-03-11T18:22:37.433458521+00:00")
    assert parsed.ok, "nanosecond timestamps must parse"
    assert parsed.moment is not None
    assert parsed.moment.microsecond == 433458


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
    """The fixture is stratified 100 per form across the timestamp forms."""
    result = clean.clean_node_sync(FIXTURES / "node_sync_harvest.jsonl")
    assert result.coin_counts["dynex"] == 100, "coin:height form"
    assert result.coin_counts["qubic"] == 100, "coin:epoch:tick form"
    assert result.coin_counts["__unattributed__"] == 100, "datetime form"
    assert timestamps.UNPARSEABLE not in result.quarantine.counts(), (
        "every real timestamp form must parse"
    )


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


# --------------------------------------------------------------------------- #
# Defect 7: 12 placeholder rows published in the 2026-08-03 rebuild
# --------------------------------------------------------------------------- #


def test_trailing_placeholder_rows_are_quarantined():
    """The 12 rows appended ~21h before the record they follow must not ship.

    They are the only rows in the source carrying a UTC offset and hold two
    distinct (power_w, gpu_temp_c) pairs between them. beads sv-l43 flagged them
    in July as placeholders that "teach the SNN a false idle pattern"; the
    2026-08-03 rebuild restored them along with the 20 rows the old broken
    heuristic had trimmed.
    """
    result = clean.clean_node_sync(FIXTURES / "node_sync_harvest.jsonl")

    reasons = result.quarantine.counts()
    assert reasons.get(timestamps.OUT_OF_ORDER) == 12, reasons
    assert timestamps.UNPARSEABLE not in reasons, "all real forms must still parse"

    # 312 fixture rows in, 12 quarantined.
    assert len(result.rows) == 300
    assert not any(
        r["timestamp"] and r["timestamp"].endswith(("-05:00", "+00:00"))
        for r in result.rows
        if r.get("timestamp")
    ), "no offset-bearing placeholder row may survive"


def test_surviving_rows_are_in_time_order():
    result = clean.clean_node_sync(FIXTURES / "node_sync_harvest.jsonl")
    assert gate_timestamps_ordered(result.epochs, 60.0) == []


def test_ordering_gate_catches_a_backward_jump():
    """Belt and braces: whatever quarantine misses, the gate must reject."""
    ordered = [0.0, 1.0, 2.0, 3.0]
    assert gate_timestamps_ordered(ordered, 60.0) == []

    reversed_tail = [0.0, 1.0, 2.0, 3.0, -80000.0]
    failures = gate_timestamps_ordered(reversed_tail, 60.0)
    assert failures and "backwards" in str(failures[0])


def test_ordering_gate_tolerates_duplicate_seconds():
    """qubic_ticks has 235 records sharing a second with their predecessor.

    Strict monotonicity would reject a healthy 1 Hz capture, so the gate uses a
    tolerance. This is the case that stops it from becoming the next heuristic
    that fires on good data.
    """
    jittery = [0.0, 1.0, 1.0, 0.9, 2.0, 3.0, 2.5, 4.0]
    assert gate_timestamps_ordered(jittery, 60.0) == []


def test_trailing_disorder_rule_refuses_to_eat_a_large_run():
    """A big out-of-order run is a real problem, not stray appended records.

    Quarantining it silently would repeat the original sin: clean_datasets.jl
    trimmed 20 good rows because a value heuristic misfired. Anything large is
    left in place for the gate to reject loudly.
    """
    stamped = [(i, float(i)) for i in range(100)]
    stamped += [(100 + i, -50000.0 + i) for i in range(50)]  # 33% of the file
    assert clean.find_trailing_disorder(stamped, total_rows=150) == []


def test_trailing_disorder_rule_ignores_a_mid_file_break():
    """Only a run reaching the end of the file qualifies."""
    stamped = [(0, 0.0), (1, 1.0), (2, -90000.0), (3, 5.0), (4, 6.0)]
    assert clean.find_trailing_disorder(stamped, total_rows=5) == []


def test_published_samples_are_draws_not_prefixes(tmp_path):
    """Samples must not be byte-exact prefixes of their parent file.

    Every published ``*_SAMPLE_*.jsonl`` was: ``head -100`` and ``head -1000`` of
    the full file. That double-counted rows whenever a sample config was loaded
    alongside its parent, and made the sample unrepresentative -- ``ch2_mode`` is
    constant across the first 200 rows of ``ghost_market_log.jsonl`` but varies
    over the file.
    """
    spec = next(s for s in SOURCES if s.key == "ghost_market_log")
    outcome = run_source(spec, FIXTURES, tmp_path, tmp_path / "reports")
    assert outcome.ok
    assert outcome.sample_paths, "a sample must be produced"

    full = (tmp_path / spec.output).read_text().splitlines()
    checked = 0
    for sample_path in outcome.sample_paths:
        sample = sample_path.read_text().splitlines()
        assert set(sample) <= set(full), "sample rows must come from the parent"
        if len(sample) >= len(full):
            # Requested size exceeds the population, so the "sample" is the whole
            # file. Nothing to assert about ordering.
            continue
        assert sample != full[: len(sample)], (
            f"{sample_path.name} is a prefix of its parent file"
        )
        checked += 1
    assert checked, "fixture too small to exercise sampling"


def test_samples_are_reproducible(tmp_path):
    """The same input must yield byte-identical samples across runs."""
    spec = next(s for s in SOURCES if s.key == "ghost_market_log")
    first = run_source(spec, FIXTURES, tmp_path / "a", tmp_path / "ra")
    second = run_source(spec, FIXTURES, tmp_path / "b", tmp_path / "rb")
    for left, right in zip(first.sample_paths, second.sample_paths, strict=True):
        assert left.read_bytes() == right.read_bytes()


def test_real_node_sync_preserves_actual_timestamps():
    """Output timestamps must trace to the source, not a generated sequence."""
    result = clean.clean_node_sync(FIXTURES / "node_sync_harvest.jsonl")
    stamped = [r["timestamp"] for r in result.rows if r.get("timestamp")]
    assert stamped, "datetime-form rows must retain their timestamp"
    deltas = {round(b - a, 6) for a, b in pairwise(result.epochs)}
    assert len(deltas) > 1, "real capture is not evenly spaced"

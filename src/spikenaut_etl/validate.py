"""Fail-loud validation gates.

Every gate here exists because a real defect shipped past its absence. A pipeline
that cannot detect its own degenerate output will publish 813,973 identical empty
records and report success -- that is exactly what happened to the revision of
``rmems/Spikenaut-SNN-Telemetry`` published before 2026-08-03.

The contract is deliberately blunt: :func:`assert_publishable` raises and nothing
is written. There is no "warn and continue" path, because the pipeline this
replaces effectively had one.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any


class ValidationError(Exception):
    """Raised when a dataset fails a gate. Output must not be written."""


@dataclass
class GateFailure:
    gate: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.gate}] {self.detail}"


@dataclass
class GateConfig:
    """Thresholds for :func:`check_all`.

    Attributes:
        min_distinct_ratio: Minimum ``distinct_rows / total_rows``. The corrupt
            ``neuromorphic_data.jsonl`` scored ``1/813973`` = 1.2e-6.
        max_row_delta_ratio: Permitted drift between input and output row counts.
            Cleaning may drop rows, but silently losing a third of a file is a bug.
        allow_constant: Columns exempt from the constant-column gate, for fields
            that are legitimately invariant (e.g. a literal ``blockchain`` tag on a
            single-chain file). Everything else that never varies is dropped as a
            dead column *before* validation, so reaching this gate means something
            unexpected collapsed.
        require_timestamp_jitter: Enforce the fabrication gate. Disable only for
            sources genuinely sampled on a fixed interval.
        max_backward_jump_seconds: How far time may run backwards between adjacent
            records before the sequence is treated as disordered. Not zero:
            ``qubic_ticks`` has 235 records sharing a whole second with their
            predecessor, which is ordinary for a 1 Hz capture. A 21-hour reversal
            is not.
    """

    min_distinct_ratio: float = 0.01
    max_row_delta_ratio: float = 0.05
    allow_constant: frozenset[str] = frozenset()
    require_timestamp_jitter: bool = True
    max_backward_jump_seconds: float = 60.0
    identity_columns: frozenset[str] = frozenset(
        {"row_index", "timestamp", "blockchain", "block_height", "tick", "epoch", "step"}
    )
    min_payload_columns: int = 1


# --------------------------------------------------------------------------- #
# Individual gates
# --------------------------------------------------------------------------- #


def gate_not_empty(rows: Sequence[dict[str, Any]]) -> list[GateFailure]:
    if not rows:
        return [GateFailure("not_empty", "output has zero rows")]
    return []


def gate_payload_columns_remain(
    rows: Sequence[dict[str, Any]], identity: Iterable[str], minimum: int
) -> list[GateFailure]:
    """Require columns that carry data, not just position or labels.

    Dead-column removal is what makes this gate necessary. Feeding the shipped
    ``{"telemetry":{}}`` file through cleaning leaves every payload field null,
    so all of them are dropped as dead -- and the surviving ``row_index`` is
    unique per row, which would satisfy the distinctness gate and let entirely
    empty data through. Position is not information.
    """
    if not rows:
        return []
    payload = _columns(rows) - set(identity)
    if len(payload) < minimum:
        return [
            GateFailure(
                "payload_columns_remain",
                f"only {len(payload)} non-identity column(s) survived cleaning "
                f"({sorted(payload)}); every payload field was dead",
            )
        ]
    return []


def gate_distinct_rows(
    rows: Sequence[dict[str, Any]], min_ratio: float, identity: Iterable[str] = ()
) -> list[GateFailure]:
    """Catch degenerate output: many rows carrying one repeated value.

    This is the gate that would have stopped ``{"telemetry":{}}`` x 813,973.

    Distinctness is measured over payload columns only. Counting an index or a
    timestamp would make any file trivially "distinct" regardless of whether it
    carries data.
    """
    if not rows:
        return []
    skip = set(identity)
    seen = {
        repr(sorted((k, _hashable(v)) for k, v in r.items() if k not in skip))
        for r in rows
    }
    ratio = len(seen) / len(rows)
    if ratio < min_ratio:
        return [
            GateFailure(
                "distinct_rows",
                f"only {len(seen)} distinct payload(s) across {len(rows)} rows "
                f"(ratio {ratio:.2e} < {min_ratio}); output is degenerate",
            )
        ]
    return []


def gate_no_constant_columns(
    rows: Sequence[dict[str, Any]], allow: Iterable[str] = ()
) -> list[GateFailure]:
    """Every surviving column must carry signal.

    Dead columns are dropped upstream by :mod:`spikenaut_etl.clean`. A constant
    column *here* means a field collapsed during cleaning -- the signature of the
    all-zero ``node_sync_harvest.jsonl``, where 13 numeric fields were written as
    ``0.0`` for all 120,314 rows.
    """
    if not rows:
        return []
    allow = set(allow)
    failures: list[GateFailure] = []
    for col in _columns(rows):
        if col in allow:
            continue
        values = {_hashable(r.get(col)) for r in rows}
        if len(values) == 1:
            only = next(iter(values))
            failures.append(
                GateFailure(
                    "no_constant_columns",
                    f"column {col!r} is {only!r} in all {len(rows)} rows",
                )
            )
    return failures


def gate_no_all_null_columns(rows: Sequence[dict[str, Any]]) -> list[GateFailure]:
    if not rows:
        return []
    failures: list[GateFailure] = []
    for col in _columns(rows):
        if all(r.get(col) is None for r in rows):
            failures.append(
                GateFailure("no_all_null_columns", f"column {col!r} is null in all rows")
            )
    return failures


def gate_row_count_delta(
    n_in: int, n_out: int, max_ratio: float
) -> list[GateFailure]:
    if n_in == 0:
        return []
    delta = abs(n_in - n_out) / n_in
    if delta > max_ratio:
        return [
            GateFailure(
                "row_count_delta",
                f"{n_in} rows in, {n_out} out (delta {delta:.1%} > {max_ratio:.1%})",
            )
        ]
    return []


def gate_timestamps_not_fabricated(
    timestamps: Sequence[float],
) -> list[GateFailure]:
    """Reject perfectly uniform spacing -- the signature of generated timestamps.

    The prior pipeline replaced 114,250 real timestamps with
    ``2026-03-20T00:00:00 + 10s * index`` because its ISO-8601 check tested for a
    literal ``T`` and the real values used a space separator. The result was a
    sequence with zero variance in its deltas. Real capture never looks like that.
    """
    if len(timestamps) < 3:
        return []
    deltas = [b - a for a, b in pairwise(timestamps)]
    distinct = {round(d, 6) for d in deltas}
    if len(distinct) == 1:
        return [
            GateFailure(
                "timestamps_not_fabricated",
                f"all {len(deltas)} inter-record gaps are exactly "
                f"{next(iter(distinct))}s -- timestamps look generated, not observed",
            )
        ]
    stdev = statistics.pstdev(deltas)
    mean = statistics.fmean(deltas)
    if mean > 0 and stdev / mean < 1e-9:
        return [
            GateFailure(
                "timestamps_not_fabricated",
                f"inter-record gaps have effectively zero variance "
                f"(mean {mean:.6f}s, stdev {stdev:.2e}) -- timestamps look generated",
            )
        ]
    return []


def find_backward_jumps(
    timestamps: Sequence[float], tolerance: float
) -> list[tuple[int, float]]:
    """Indices where time runs backwards by more than ``tolerance`` seconds.

    Returns ``(index, seconds_backwards)`` for each break, where ``index`` is the
    position of the *first* record of the out-of-order run.
    """
    breaks: list[tuple[int, float]] = []
    for i, (a, b) in enumerate(pairwise(timestamps), start=1):
        if a - b > tolerance:
            breaks.append((i, a - b))
    return breaks


def gate_timestamps_ordered(
    timestamps: Sequence[float], tolerance: float
) -> list[GateFailure]:
    """Reject a capture whose records are not in time order.

    Appended records that predate everything before them did not come from the
    same run. ``node_sync_harvest.jsonl`` ended with 12 such rows -- the only ones
    in the file carrying a UTC offset, holding two distinct
    ``(power_w, gpu_temp_c)`` pairs between them -- sitting ~21 hours before the
    record they follow. They were placeholders, and they shipped, because the
    existing fabrication gate looks for *uniform spacing* and 12 rows in 120,334
    move no distribution-based check.

    Cleaning quarantines a small trailing run like that. This gate is what catches
    anything the quarantine did not, so disorder can never reach a published file
    silently.
    """
    if len(timestamps) < 2:
        return []
    breaks = find_backward_jumps(timestamps, tolerance)
    if not breaks:
        return []
    index, seconds = breaks[0]
    detail = (
        f"time runs backwards {seconds:,.0f}s at record {index} "
        f"(tolerance {tolerance}s)"
    )
    if len(breaks) > 1:
        detail += f"; {len(breaks)} such breaks total"
    return [GateFailure("timestamps_ordered", detail)]


def gate_schema_matches(
    rows: Sequence[dict[str, Any]], expected: Iterable[str]
) -> list[GateFailure]:
    if not rows:
        return []
    expected_set = set(expected)
    actual = _columns(rows)
    missing = expected_set - actual
    extra = actual - expected_set
    failures: list[GateFailure] = []
    if missing:
        failures.append(
            GateFailure("schema_matches", f"declared columns absent: {sorted(missing)}")
        )
    if extra:
        failures.append(
            GateFailure("schema_matches", f"undeclared columns present: {sorted(extra)}")
        )
    return failures


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class ValidationResult:
    name: str
    n_in: int
    n_out: int
    failures: list[GateFailure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        head = f"{self.name}: {self.n_in} -> {self.n_out} rows"
        if self.ok:
            return f"PASS  {head}"
        lines = [f"FAIL  {head}"]
        lines += [f"        {f}" for f in self.failures]
        return "\n".join(lines)


def check_all(
    name: str,
    rows: Sequence[dict[str, Any]],
    *,
    n_in: int,
    config: GateConfig | None = None,
    expected_columns: Iterable[str] | None = None,
    timestamps: Sequence[float] | None = None,
) -> ValidationResult:
    """Run every applicable gate and collect all failures.

    Gates are not short-circuited: a caller fixing one problem should be able to
    see the rest in the same run.
    """
    cfg = config or GateConfig()
    failures: list[GateFailure] = []
    failures += gate_not_empty(rows)
    failures += gate_payload_columns_remain(
        rows, cfg.identity_columns, cfg.min_payload_columns
    )
    failures += gate_distinct_rows(rows, cfg.min_distinct_ratio, cfg.identity_columns)
    failures += gate_no_constant_columns(rows, cfg.allow_constant)
    failures += gate_no_all_null_columns(rows)
    failures += gate_row_count_delta(n_in, len(rows), cfg.max_row_delta_ratio)
    if expected_columns is not None:
        failures += gate_schema_matches(rows, expected_columns)
    if timestamps is not None:
        failures += gate_timestamps_ordered(timestamps, cfg.max_backward_jump_seconds)
        if cfg.require_timestamp_jitter:
            failures += gate_timestamps_not_fabricated(timestamps)
    return ValidationResult(name=name, n_in=n_in, n_out=len(rows), failures=failures)


def assert_publishable(result: ValidationResult) -> None:
    """Raise unless ``result`` passed every gate.

    Call this immediately before writing output. There is intentionally no
    override flag.
    """
    if not result.ok:
        detail = "\n".join(f"  {f}" for f in result.failures)
        raise ValidationError(
            f"{result.name} failed {len(result.failures)} gate(s); refusing to "
            f"write output:\n{detail}"
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _columns(rows: Sequence[dict[str, Any]]) -> set[str]:
    cols: set[str] = set()
    for r in rows:
        cols.update(r)
    return cols


def _hashable(value: Any) -> Any:
    """Make a JSON value usable in a set, and collapse NaN to one identity."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, dict):
        return repr(sorted(value.items()))
    if isinstance(value, list):
        return repr(value)
    return value


def column_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Counter]:
    """Distinct-value counts per column, for reporting."""
    stats: dict[str, Counter] = {}
    for col in sorted(_columns(rows)):
        stats[col] = Counter(_hashable(r.get(col)) for r in rows)
    return stats

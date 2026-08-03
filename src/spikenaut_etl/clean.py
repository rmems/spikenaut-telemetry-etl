"""Cleaning transforms: legacy free-form JSONL -> declared flat records.

Dead columns are *measured*, not assumed. The prior pipeline hardcoded its
``DEAD_COLS_*`` sets; when the data shifted, the lists would have silently
mismatched. :func:`find_dead_columns` recomputes them per run and
:func:`report_dead_column_drift` flags any divergence from what was expected, so
a change in the source is visible instead of absorbed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import timestamps
from .ingest import IngestStats, read_validated
from .schemas import (
    RawGpuRecord,
    RawNodeSyncRecord,
    RawQubicTick,
    RawTradingLog,
)

# Columns expected to be dead, from profiling the recovered originals on
# 2026-08-03. Divergence is reported, not silently applied.
EXPECTED_DEAD_GPU = frozenset(
    {
        "hashrate_mh",
        "rejected_shares",
        "ocean_intel",
        "qu_price_usd",
        "power_z_score",
        "temp_z_score",
        "clock_z_score",
        "solver_steps",
        "solver_chips",
        "complexity",
        "joules_per_step",
        "kaspa_hashrate_mh",
        "kaspa_power_w",
        "kaspa_gpu_temp_c",
        "monero_hashrate_h",
        "monero_power_w",
        "monero_cpu_temp_c",
    }
)

EXPECTED_DEAD_NODE_SYNC = frozenset(
    {
        "clock_mhz",
        "clock_z_score",
        "fan_speed_pct",
        "gpu_clock_mhz",
        "mem_clock_mhz",
        "mem_util_pct",
        "ocean_intel",
        "power_z_score",
        "qu_price_usd",
        "rejected_shares",
        "temp_z_score",
        "vddcr_gfx_v",
        "vram_temp_c",
    }
)


@dataclass
class CleanResult:
    name: str
    rows: list[dict[str, Any]]
    n_in: int
    ingest: IngestStats
    quarantine: timestamps.QuarantineLog
    dead_columns: set[str] = field(default_factory=set)
    dead_column_drift: list[str] = field(default_factory=list)
    coin_counts: Counter = field(default_factory=Counter)
    epochs: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Dead-column detection
# --------------------------------------------------------------------------- #


def find_dead_columns(rows: Sequence[dict[str, Any]]) -> set[str]:
    """Columns carrying at most one distinct non-null value across ``rows``.

    Dead columns are dropped before validation, so the constant-column gate only
    fires on fields that collapsed *during* cleaning.

    "At most one distinct value" deliberately ignores how many rows are null. A
    column that is absent from most rows and identical wherever it does appear --
    the ``kaspa_*`` / ``monero_*`` drift fields are exactly this -- distinguishes
    nothing and would otherwise survive into the published schema as a column of
    nulls punctuated by one repeated constant.
    """
    if not rows:
        return set()
    seen: dict[str, set[Any]] = defaultdict(set)
    nulls: Counter = Counter()
    for row in rows:
        for key, value in row.items():
            if value is None:
                nulls[key] += 1
            else:
                seen[key].add(
                    value if not isinstance(value, (dict, list)) else repr(value)
                )
    columns = set(seen) | set(nulls)
    return {col for col in columns if len(seen.get(col, ())) <= 1}


def report_dead_column_drift(
    observed: Iterable[str], expected: Iterable[str]
) -> list[str]:
    observed_set, expected_set = set(observed), set(expected)
    messages = []
    for col in sorted(expected_set - observed_set):
        messages.append(f"{col!r} was expected dead but now carries signal")
    for col in sorted(observed_set - expected_set):
        messages.append(f"{col!r} is newly dead (was expected to carry signal)")
    return messages


def drop_columns(
    rows: Sequence[dict[str, Any]], columns: Iterable[str]
) -> list[dict[str, Any]]:
    drop = set(columns)
    return [{k: v for k, v in row.items() if k not in drop} for row in rows]


# --------------------------------------------------------------------------- #
# Per-source cleaners
# --------------------------------------------------------------------------- #


def clean_gpu_telemetry(path: Path) -> CleanResult:
    """``neuromorphic_data.jsonl`` -> flat GPU telemetry.

    The source carries no timestamp field at all, so output is positional
    (``row_index``). That is a genuine limitation of the capture and is documented
    rather than concealed behind a generated clock.
    """
    stats = IngestStats(source="neuromorphic_data")
    quarantine = timestamps.QuarantineLog(source="neuromorphic_data")
    raw: list[dict[str, Any]] = []
    for row, record in read_validated(path, RawGpuRecord, stats):
        payload = record.telemetry.model_dump()
        payload["row_index"] = row
        raw.append(payload)

    dead = find_dead_columns(raw) - {"row_index"}
    drift = report_dead_column_drift(dead, EXPECTED_DEAD_GPU)
    rows = drop_columns(raw, dead)

    return CleanResult(
        name="neuromorphic_data",
        rows=rows,
        n_in=stats.n_lines,
        ingest=stats,
        quarantine=quarantine,
        dead_columns=dead,
        dead_column_drift=drift,
    )


def clean_node_sync(path: Path) -> CleanResult:
    """``node_sync_harvest.jsonl`` -> flat mining telemetry.

    Preserves real timestamps and recovers chain attribution from the
    ``coin:height`` form. Rows without a chain label get ``blockchain=None``,
    never ``""`` -- the prior output wrote the empty string for all 120,314 rows
    and the card then advertised a per-coin breakdown that did not exist.
    """
    stats = IngestStats(source="node_sync_harvest")
    quarantine = timestamps.QuarantineLog(source="node_sync_harvest")
    raw: list[dict[str, Any]] = []
    coins: Counter = Counter()
    epochs: list[float] = []

    for row, record in read_validated(path, RawNodeSyncRecord, stats):
        parsed = timestamps.parse(record.timestamp)
        if not parsed.ok:
            quarantine.record(row, record.timestamp)
            continue

        payload = record.telemetry.model_dump()
        payload["timestamp"] = parsed.iso
        payload["blockchain"] = parsed.coin
        payload["block_height"] = parsed.height
        payload["chain_epoch"] = parsed.epoch_number
        coins[parsed.coin or "__unattributed__"] += 1
        if parsed.epoch is not None:
            epochs.append(parsed.epoch)
        raw.append(payload)

    identity = {"timestamp", "blockchain", "block_height", "chain_epoch"}
    dead = find_dead_columns(raw) - identity
    drift = report_dead_column_drift(dead, EXPECTED_DEAD_NODE_SYNC)
    rows = drop_columns(raw, dead)

    return CleanResult(
        name="node_sync_harvest",
        rows=rows,
        n_in=stats.n_lines,
        ingest=stats,
        quarantine=quarantine,
        dead_columns=dead,
        dead_column_drift=drift,
        coin_counts=coins,
        epochs=epochs,
    )


def clean_qubic_ticks(path: Path) -> CleanResult:
    """``qubic_ticks.jsonl`` -> SNN-format ticks with derived columns labelled.

    The published ``qubic_ticks_snn.jsonl`` presented four columns as hardware
    telemetry. They are a fixed function of ``tick_rate`` (16 distinct values
    across 27,430 rows). The mapping is preserved for continuity with existing
    consumers, but every derived column now carries a ``_derived`` suffix.
    """
    stats = IngestStats(source="qubic_ticks")
    quarantine = timestamps.QuarantineLog(source="qubic_ticks")
    records: list[tuple[int, RawQubicTick]] = list(
        read_validated(path, RawQubicTick, stats)  # type: ignore[arg-type]
    )
    if not records:
        return CleanResult(
            name="qubic_ticks_snn",
            rows=[],
            n_in=stats.n_lines,
            ingest=stats,
            quarantine=quarantine,
        )

    rates = [r.tick_rate for _, r in records]
    ticks = [r.tick for _, r in records]
    rate_min, rate_max = min(rates), max(rates)
    tick_min, tick_max = min(ticks), max(ticks)
    rate_span = (rate_max - rate_min) or 1.0
    tick_span = (tick_max - tick_min) or 1

    rows: list[dict[str, Any]] = []
    epochs: list[float] = []
    for _, rec in records:
        parsed = timestamps.parse(rec.ts)
        norm_rate = (rec.tick_rate - rate_min) / rate_span
        rows.append(
            {
                "timestamp": parsed.iso,
                "tick": rec.tick,
                "epoch": rec.epoch,
                "tick_rate": rec.tick_rate,
                "epoch_progress": rec.epoch_progress,
                "qubic_tick_trace": (rec.tick - tick_min) / tick_span,
                "hashrate_mh_derived": round(1.0 + norm_rate, 6),
                "power_w_derived": round(300.0 + 100.0 * norm_rate, 6),
                "gpu_temp_c_derived": round(60.0 + 15.0 * norm_rate, 6),
                "reward_hint_derived": round(norm_rate, 6),
            }
        )
        if parsed.epoch is not None:
            epochs.append(parsed.epoch)

    dead = find_dead_columns(rows) - {"timestamp", "tick"}
    rows = drop_columns(rows, dead)

    return CleanResult(
        name="qubic_ticks_snn",
        rows=rows,
        n_in=stats.n_lines,
        ingest=stats,
        quarantine=quarantine,
        dead_columns=dead,
        epochs=epochs,
    )


def clean_trading_log(path: Path) -> CleanResult:
    """``ghost_market_log.jsonl`` -> passthrough with validation.

    This file is already healthy: one schema, no nulls, no constant columns. It
    is still parsed through its declared model so a future regression is caught.
    """
    stats = IngestStats(source="ghost_market_log")
    quarantine = timestamps.QuarantineLog(source="ghost_market_log")
    rows: list[dict[str, Any]] = []
    epochs: list[float] = []

    for row, record in read_validated(path, RawTradingLog, stats):
        parsed = timestamps.parse(record.timestamp)
        if not parsed.ok:
            quarantine.record(row, record.timestamp)
            continue
        rows.append(record.model_dump())
        if parsed.epoch is not None:
            epochs.append(parsed.epoch)

    return CleanResult(
        name="ghost_market_log",
        rows=rows,
        n_in=stats.n_lines,
        ingest=stats,
        quarantine=quarantine,
        epochs=epochs,
    )

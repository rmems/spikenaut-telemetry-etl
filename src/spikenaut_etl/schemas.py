"""Declared record schemas.

Every model sets ``extra="forbid"``. Schema drift becomes a loud
``ValidationError`` at parse time instead of a column that silently vanishes --
which is how the ``kaspa_*`` / ``monero_*`` fields in ``neuromorphic_data.jsonl``
were dropped without anyone noticing.

Field names track ``TelemetryEnvelope`` in ``rmems/Theseus-Quarry``
(``crates/mining-telemetry-core/src/schema.rs``) so schema-v1 collections can flow
through this pipeline unchanged once the legacy backlog is cleared.

Two rules apply throughout:

1. **No silent defaults on required fields.** A missing value raises. The prior
   pipeline's ``get(dict, "hashrate_mh", 0.0)`` returned ``0.0`` for 120,314 rows
   because it looked up a String key in a Symbol-keyed dict, and nothing complained.
2. **Optional means genuinely absent**, never "we could not read it".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Schema version of the *cleaned output*, independent of Theseus-Quarry's
# collection schema version. Bump on any breaking change to published columns.
OUTPUT_SCHEMA_VERSION = 2


class StrictRecord(BaseModel):
    """Base: reject unknown fields, forbid coercion surprises."""

    model_config = ConfigDict(extra="forbid", strict=False, frozen=True)


# --------------------------------------------------------------------------- #
# Source: neuromorphic_data.jsonl  (GPU telemetry, legacy {"telemetry": {...}})
# --------------------------------------------------------------------------- #


class RawGpuTelemetry(StrictRecord):
    """Inner payload of a legacy ``neuromorphic_data.jsonl`` line.

    All fields optional because the legacy writer omitted rather than nulled, and
    later rows carry ``kaspa_*`` / ``monero_*`` drift fields. Declaring them here
    is what keeps them from being silently discarded.
    """

    vddcr_gfx_v: float | None = None
    vram_temp_c: float | None = None
    gpu_temp_c: float | None = None
    hashrate_mh: float | None = None
    power_w: float | None = None
    gpu_clock_mhz: float | None = None
    mem_clock_mhz: float | None = None
    clock_mhz: float | None = None
    fan_speed_pct: float | None = None
    mem_util_pct: float | None = None
    rejected_shares: int | None = None
    qubic_tick_trace: float | None = None
    qubic_tick_rate: float | None = None
    qubic_epoch_progress: float | None = None
    qu_price_usd: float | None = None
    ocean_intel: float | None = None
    power_z_score: float | None = None
    temp_z_score: float | None = None
    clock_z_score: float | None = None
    solver_steps: float | None = None
    solver_chips: float | None = None
    complexity: float | None = None
    joules_per_step: float | None = None
    # Schema drift observed in the tail of the 395 MB original.
    kaspa_hashrate_mh: float | None = None
    kaspa_power_w: float | None = None
    kaspa_gpu_temp_c: float | None = None
    monero_hashrate_h: float | None = None
    monero_power_w: float | None = None
    monero_cpu_temp_c: float | None = None


class RawGpuRecord(StrictRecord):
    telemetry: RawGpuTelemetry


# --------------------------------------------------------------------------- #
# Source: node_sync_harvest.jsonl  (mining telemetry, legacy nested)
# --------------------------------------------------------------------------- #


class RawNodeSyncTelemetry(StrictRecord):
    hashrate_mh: float | None = None
    power_w: float | None = None
    gpu_temp_c: float | None = None
    reward_hint: float | None = None
    qubic_tick_trace: float | None = None
    qubic_tick_rate: float | None = None
    qubic_epoch_progress: float | None = None
    vddcr_gfx_v: float | None = None
    vram_temp_c: float | None = None
    gpu_clock_mhz: float | None = None
    mem_clock_mhz: float | None = None
    clock_mhz: float | None = None
    fan_speed_pct: float | None = None
    mem_util_pct: float | None = None
    rejected_shares: int | None = None
    ocean_intel: float | None = None
    qu_price_usd: float | None = None
    power_z_score: float | None = None
    temp_z_score: float | None = None
    clock_z_score: float | None = None


class RawNodeSyncRecord(StrictRecord):
    # Either an ISO-ish datetime ("2026-03-19 11:55:13.132") or a "coin:height"
    # tag ("dynex:919876"). Both are real and must survive -- see
    # spikenaut_etl.timestamps.
    timestamp: str
    telemetry: RawNodeSyncTelemetry


# --------------------------------------------------------------------------- #
# Source: qubic_ticks.jsonl
# --------------------------------------------------------------------------- #


class RawQubicTick(StrictRecord):
    ts: int
    tick: int
    epoch: int
    tick_rate: float
    epoch_progress: float


# --------------------------------------------------------------------------- #
# Source: ghost_market_log.jsonl  (already flat and healthy)
# --------------------------------------------------------------------------- #


class RawTradingLog(StrictRecord):
    timestamp: str
    step: int
    action: Literal["buy", "sell", "observe"]
    asset: str
    price_usd: float
    quantity: float
    trade_value_usdt: float
    realized_pnl_usdt: float
    balance_usdt: float
    balance_dnx: float
    balance_sol: float
    balance_render: float
    balance_asi: float
    balance_near: float
    balance_btc: float
    balance_pepe: float
    render_qty: float
    asi_qty: float
    near_qty: float
    btc_qty: float
    pepe_qty: float
    cumulative_pnl: float
    portfolio_value: float
    ch2_mode: str
    reason: str


# --------------------------------------------------------------------------- #
# Cleaned output
# --------------------------------------------------------------------------- #


class CleanGpuTelemetry(StrictRecord):
    """Flat GPU telemetry. Only the 12 fields that carry signal in the source.

    ``neuromorphic_data.jsonl`` has no timestamp of any kind -- the legacy writer
    never emitted one. Records are therefore ordered but not time-located, which
    is a real limitation of the source and is stated in the published card rather
    than papered over with a synthesized clock.
    """

    row_index: int = Field(description="Position in the source file; not a timestamp.")
    vddcr_gfx_v: float
    vram_temp_c: float
    gpu_temp_c: float
    power_w: float
    gpu_clock_mhz: float
    mem_clock_mhz: float
    clock_mhz: float
    fan_speed_pct: float
    mem_util_pct: float
    qubic_tick_trace: float
    qubic_tick_rate: float
    qubic_epoch_progress: float


class CleanNodeSync(StrictRecord):
    """Flat mining telemetry with real timestamps and honest coin attribution."""

    timestamp: str | None = Field(
        default=None,
        description="ISO-8601 when the source carried a datetime; None when the "
        "source carried a coin:height tag instead. Never synthesized.",
    )
    blockchain: str | None = Field(
        default=None,
        description="Parsed from a 'coin:height' timestamp prefix. None -- never "
        "the empty string -- when the source carries no chain label.",
    )
    block_height: int | None = None
    chain_epoch: int | None = Field(
        default=None,
        description="Present only for the 'coin:epoch:tick' timestamp form "
        "(qubic). None for 'coin:height' and datetime forms.",
    )
    hashrate_mh: float
    power_w: float
    gpu_temp_c: float
    reward_hint: float
    qubic_tick_trace: float
    qubic_tick_rate: float
    qubic_epoch_progress: float


class CleanQubicTick(StrictRecord):
    """Qubic ticks with derived columns named as such.

    The published ``qubic_ticks_snn.jsonl`` presented ``hashrate_mh``, ``power_w``
    and ``gpu_temp_c`` as GPU telemetry. They are a fixed function of
    ``tick_rate`` -- 16 distinct values across 27,430 rows, with
    ``power_w / hashrate_mh`` pinned at 210.9. They are kept for continuity but
    suffixed ``_derived`` so no consumer mistakes them for measurements.
    """

    timestamp: str
    tick: int
    epoch: int
    tick_rate: float
    epoch_progress: float
    qubic_tick_trace: float
    hashrate_mh_derived: float
    power_w_derived: float
    gpu_temp_c_derived: float
    reward_hint_derived: float


# Column contracts consumed by the schema gate in :mod:`spikenaut_etl.validate`.
CLEAN_COLUMNS: dict[str, frozenset[str]] = {
    "neuromorphic_data": frozenset(CleanGpuTelemetry.model_fields),
    "node_sync_harvest": frozenset(CleanNodeSync.model_fields),
    "qubic_ticks_snn": frozenset(CleanQubicTick.model_fields),
    "ghost_market_log": frozenset(RawTradingLog.model_fields),
}

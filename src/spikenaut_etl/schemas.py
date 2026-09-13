"""Declared record schemas.

Every model sets ``extra="forbid"``. Schema drift becomes a loud
``ValidationError`` at parse time instead of a column that silently vanishes --
which is how the ``kaspa_*`` / ``monero_*`` fields in ``neuromorphic_data.jsonl``
were dropped without anyone noticing.

Field names track ``TelemetryEnvelope`` in ``rmems/Theseus-Quarry``
(``crates/mining-telemetry-core/src/schema.rs``). Ingest dispatches on
``schema_version`` and maps v1 ``kind`` / tagged ``payload`` onto the ``Clean*``
contracts below so published columns stay stable.

Two rules apply throughout:

1. **No silent defaults on required fields.** A missing value raises. The prior
   pipeline's ``get(dict, "hashrate_mh", 0.0)`` returned ``0.0`` for 120,314 rows
   because it looked up a String key in a Symbol-keyed dict, and nothing complained.
2. **Optional means genuinely absent**, never "we could not read it".
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

# Schema version of the *cleaned output*, independent of Theseus-Quarry's
# collection schema version. Bump on any breaking change to published columns.
OUTPUT_SCHEMA_VERSION = 2

# Theseus-Quarry ``SCHEMA_VERSION`` in mining-telemetry-core. Break-free: this
# reader implements v1 only and refuses to guess at any other version.
COLLECTOR_SCHEMA_VERSION = 1

RecordKind = Literal[
    "miner_perf",
    "node_health",
    "host_hw",
    "status",
    "gpu_sched",
    "rotation",
]


class StrictRecord(BaseModel):
    """Base: reject unknown fields, forbid coercion surprises."""

    model_config = ConfigDict(
        extra="forbid", strict=False, frozen=True, allow_inf_nan=False
    )


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
# Source: rmems/gaming-telemetry  (system_telemetry_v1, game-agnostic sensors)
# --------------------------------------------------------------------------- #
# Producer contract: missing NVML/hwmon/RAPL reads are null, never 0. A literal
# 0 on the fields in UNAVAILABLE_ZERO_FIELDS is a failed read written as a
# measurement -- refuse it rather than teach sensor failure as hardware.
# encoder/decoder util may legitimately be 0 for a whole session.
# session_label is ETL session hygiene (GateConfig.allow_constant / splits).
# It is not a Spikenaut axon and must not be published as a training feature.


UNAVAILABLE_ZERO_FIELDS = frozenset(
    {
        "power_usage_mw",
        "temperature_c",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "memory_total_mb",
        "cpu_tctl_c",
        "cpu_ccd1_c",
        "cpu_ccd2_c",
        "cpu_package_power_w",
    }
)

SYSTEM_TELEMETRY_HARDWARE_INVARIANTS = frozenset(
    {
        "session_label",
        "memory_total_mb",
        "encoder_util_perc",
        "decoder_util_perc",
    }
)

SYSTEM_TELEMETRY_HYGIENE_COLUMNS = frozenset({"session_label"})


class RawSystemTelemetry(StrictRecord):
    """One ``system_telemetry_v1`` sensor row (parquet or JSONL).

    Field names match ``rmems/gaming-telemetry``. Extra fields are forbidden so
    a collector schema bump is loud. ``session_label`` identifies the capture
    session for gating; it is not game identity for training.
    """

    timestamp_ms: int
    power_usage_mw: int | None = None
    temperature_c: int | None = None
    graphics_clock_mhz: int | None = None
    memory_clock_mhz: int | None = None
    pcie_rx_kbps: int | None = None
    pcie_tx_kbps: int | None = None
    pstate: int | None = None
    throttle_reasons_bitmask: int | None = None
    fan_speed_perc: int | None = None
    memory_used_mb: int | None = None
    memory_total_mb: int | None = None
    encoder_util_perc: int | None = None
    decoder_util_perc: int | None = None
    cpu_tctl_c: float | None = None
    cpu_ccd1_c: float | None = None
    cpu_ccd2_c: float | None = None
    cpu_package_power_w: float | None = None
    session_label: str

    @field_validator(
        "timestamp_ms",
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
        "cpu_tctl_c",
        "cpu_ccd1_c",
        "cpu_ccd2_c",
        "cpu_package_power_w",
        mode="before",
    )
    @classmethod
    def _reject_bool_sensor_fields(cls, value: object, info: ValidationInfo) -> object:
        # strict=False otherwise coerces True->1 / False->0 (and True->1.0)
        # before after-validators.
        if type(value) is bool:
            raise ValueError(
                f"{info.field_name} must be a number, not bool; "
                "refusing to coerce True/False onto a sensor field"
            )
        return value

    @field_validator("timestamp_ms")
    @classmethod
    def _positive_epoch_ms(cls, value: int) -> int:
        if type(value) is bool or value <= 0:
            raise ValueError(
                "timestamp_ms must be a positive epoch millisecond count; "
                "refusing to treat a missing clock as time"
            )
        return value

    @field_validator("session_label")
    @classmethod
    def _nonempty_session_label(cls, value: str) -> str:
        label = value.strip()
        if not label:
            raise ValueError(
                "session_label is ETL session hygiene and must be non-empty"
            )
        return label

    @model_validator(mode="after")
    def _reject_unavailable_zeros(self) -> RawSystemTelemetry:
        for name in sorted(UNAVAILABLE_ZERO_FIELDS):
            value = getattr(self, name)
            if value is None:
                continue
            if value == 0:
                raise ValueError(
                    f"{name} is 0; unavailable NVML/hwmon/RAPL reads must be "
                    "null, not a zero measurement"
                )
        return self


class CleanSystemTelemetry(StrictRecord):
    """Flat system telemetry with ``ts_utc`` derived from ``timestamp_ms``.

    ``ts_utc`` is a unit conversion of the source clock, not a synthesized
    timestamp. ``session_label`` is kept for session splits and
    ``GateConfig.allow_constant``; it is not a Spikenaut feature axon.
    """

    timestamp_ms: int
    ts_utc: str
    power_usage_mw: int | None = None
    temperature_c: int | None = None
    graphics_clock_mhz: int | None = None
    memory_clock_mhz: int | None = None
    pcie_rx_kbps: int | None = None
    pcie_tx_kbps: int | None = None
    pstate: int | None = None
    throttle_reasons_bitmask: int | None = None
    fan_speed_perc: int | None = None
    memory_used_mb: int | None = None
    memory_total_mb: int | None = None
    encoder_util_perc: int | None = None
    decoder_util_perc: int | None = None
    cpu_tctl_c: float | None = None
    cpu_ccd1_c: float | None = None
    cpu_ccd2_c: float | None = None
    cpu_package_power_w: float | None = None
    session_label: str


SYSTEM_TELEMETRY_FEATURE_AXONS = frozenset(CleanSystemTelemetry.model_fields) - (
    SYSTEM_TELEMETRY_HYGIENE_COLUMNS
)


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
# Theseus-Quarry schema v1 (TelemetryEnvelope + tagged TelemetryPayload)
# --------------------------------------------------------------------------- #
# Shape is taken from crates/mining-telemetry-core/src/schema.rs. extra=forbid
# so an unknown field or payload tag is a loud ValidationError, not a drop.


class PayloadMinerPerf(StrictRecord):
    type: Literal["miner_perf"]
    coin: str
    hashrate: float
    hashrate_unit: str
    shares_accepted: int | None = None
    shares_rejected: int | None = None
    is_active: bool
    uptime_seconds: int | None = None


class PayloadNodeHealth(StrictRecord):
    type: Literal["node_health"]
    coin: str
    height: int | None = None
    target_height: int | None = None
    tick: int | None = None
    epoch: int | None = None
    active: bool | None = None
    speed_hs: int | None = None
    threads: int | None = None
    hashrate_mh: float | None = None


class PayloadHostHw(StrictRecord):
    type: Literal["host_hw"]
    cpu_tctl_c: float | None = None
    cpu_ccd1_c: float | None = None
    cpu_ccd2_c: float | None = None
    cpu_package_power_w: float | None = None


class PayloadStatus(StrictRecord):
    type: Literal["status"]
    message: str


class PayloadGpuSched(StrictRecord):
    type: Literal["gpu_sched"]
    decision: str
    vram_used_mb: int
    vram_total_mb: int
    gpu_temp_c: float
    power_w: float
    transition_count: int


class PayloadRotation(StrictRecord):
    type: Literal["rotation"]
    kind: str
    from_algo: str | None = None
    to_algo: str | None = None
    market_age_secs: float


TelemetryPayload = Annotated[
    PayloadMinerPerf
    | PayloadNodeHealth
    | PayloadHostHw
    | PayloadStatus
    | PayloadGpuSched
    | PayloadRotation,
    Field(discriminator="type"),
]


class TelemetryEnvelope(StrictRecord):
    """Theseus-Quarry durable JSONL line (``SCHEMA_VERSION = 1``)."""

    schema_version: int = Field(strict=True)
    timestamp: str
    source: str
    kind: RecordKind
    host: str | None = None
    run_id: str | None = None
    stem: str
    payload: TelemetryPayload

    @field_validator("schema_version")
    @classmethod
    def _only_v1(cls, value: int) -> int:
        if value != COLLECTOR_SCHEMA_VERSION:
            raise ValueError(
                f"unknown schema_version {value!r}; this reader implements "
                f"Theseus-Quarry schema v{COLLECTOR_SCHEMA_VERSION} only"
            )
        return value

    @model_validator(mode="after")
    def _kind_matches_payload(self) -> TelemetryEnvelope:
        payload_type = self.payload.type
        if self.kind != payload_type:
            raise ValueError(
                f"kind {self.kind!r} does not match payload.type {payload_type!r}"
            )
        return self


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
    "system_telemetry_v1": frozenset(CleanSystemTelemetry.model_fields),
}

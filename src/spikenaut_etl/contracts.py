"""Named producer/consumer contracts for published Vault shapes.

The 2026-09-14 Spikenaut host smoke ran ``spikenaut-etl validate`` against
published Vault ``full_data`` JSONL and got fail-loud 2/5. That was ingest
refusing *already-cleaned* rows, not a new collection defect:

* ``neuromorphic_data`` is Clean GPU (``row_index`` + sensors, ``gpu_clock_mhz``),
  not ``RawGpuRecord { telemetry: {...} }``.
* ``node_sync_harvest`` coin-tagged rows have ``timestamp: null`` (the typed-null
  this pipeline already publishes), not a raw ``coin:height`` string.
* LiveStimAdapter still needs the v3 ``LIVE_COLUMNS`` projection
  (``sm_clock_mhz`` after ``STATE_BACKFILL``). v2 GPU JSONL must not be renamed
  or zero-filled into that map.

Errors that cite these constants are named so a consumer can tell "wrong Vault
path" from "corrupt schema". True schema corruption still raises a plain ingest
error.
"""

from __future__ import annotations

from typing import Any, Literal

from .schemas import StrictRecord

# Spikenaut-SNN ``encode::LIVE_COLUMNS`` / ``tools/hamming_const.py`` order.
# Axon 3 is ``sm_clock_mhz``, filled only by v3 STATE_BACKFILL from v2
# ``gpu_clock_mhz``. This mill never invents that mapping at ingest time.
LIVE_COLUMNS: tuple[str, ...] = (
    "mem_util_pct",
    "power_w",
    "gpu_temp_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
)

LIVE_COLUMNS_OPTIONAL = frozenset({"episode_id"})
LIVE_COLUMNS_ALLOWED = frozenset(LIVE_COLUMNS) | LIVE_COLUMNS_OPTIONAL

# Dataset-repo / ShipOfTheseus Vault path. Split shards are the live-bank input.
LIVE_COLUMNS_VAULT_PATH = "v3/state_telemetry"
LIVE_COLUMNS_VAULT_GLOB = "v3/state_telemetry/{train,validation,test}-00000.parquet"

CONTRACT_LIVE_COLUMNS_REQUIRED = "CONTRACT_LIVE_COLUMNS_REQUIRED"
CONTRACT_MIXED_SHAPES = "CONTRACT_MIXED_SHAPES"
CONTRACT_PROFILE_MISMATCH = "CONTRACT_PROFILE_MISMATCH"

IngestProfile = Literal["auto", "raw", "published", "live-columns"]
ShapeProfile = Literal["raw", "published", "live-columns"]


class LiveColumnsRecord(StrictRecord):
    """One LiveStimAdapter JSONL row (forbidden sensors already stripped).

    This is **not** published ``full_data/neuromorphic_data.jsonl``. Extra
    state_telemetry columns are forbidden so a full parquet dump cannot silently
    become training input.
    """

    mem_util_pct: float
    power_w: float
    gpu_temp_c: float
    sm_clock_mhz: float
    mem_clock_mhz: float
    episode_id: str | None = None


def live_columns_required_message(*, source: str, row: int | None = None) -> str:
    """Point at the v3 parquet projection; refuse gpu_clock → sm_clock invent."""
    loc = f"{source}:{row}: " if row is not None else f"{source}: "
    return (
        f"{loc}LiveStimAdapter / LIVE_COLUMNS input is Vault "
        f"{LIVE_COLUMNS_VAULT_GLOB} (sm_clock_mhz after STATE_BACKFILL). "
        f"Published v2 GPU JSONL (full_data/neuromorphic_data.jsonl) carries "
        f"gpu_clock_mhz, not sm_clock_mhz. Refusing to invent sm_clock_mhz "
        f"from gpu_clock_mhz or to zero-fill missing live sensors. "
        f"Spikenaut live-bank columns in order: {list(LIVE_COLUMNS)}."
    )


def mixed_shapes_message(*, source: str, row: int, previous: str, current: str) -> str:
    return (
        f"{source}:{row}: mixed {previous!r} and {current!r} shapes in one file; "
        "refusing to guess which contract applies"
    )


def looks_like_live_columns(payload: dict[str, Any]) -> bool:
    """True when the row is a LIVE_COLUMNS projection, not Clean GPU.

    Clean GPU has ``gpu_clock_mhz`` and never ``sm_clock_mhz``. Live-bank rows
    have the opposite. Presence of ``sm_clock_mhz`` without ``gpu_clock_mhz``
    is the discriminator — we do not treat a v2 clock as an SM clock.
    """
    return "sm_clock_mhz" in payload and "gpu_clock_mhz" not in payload


def looks_like_nested_raw(payload: dict[str, Any]) -> bool:
    return "telemetry" in payload


def looks_like_clean_gpu(payload: dict[str, Any]) -> bool:
    return (
        not looks_like_nested_raw(payload)
        and not looks_like_live_columns(payload)
        and ("row_index" in payload or "gpu_clock_mhz" in payload)
    )


def looks_like_clean_node_sync(payload: dict[str, Any]) -> bool:
    return not looks_like_nested_raw(payload) and (
        "blockchain" in payload
        or "hashrate_mh" in payload
        or "block_height" in payload
        or "chain_epoch" in payload
    )


def looks_like_clean_qubic(payload: dict[str, Any]) -> bool:
    return "hashrate_mh_derived" in payload or (
        "timestamp" in payload and "ts" not in payload and "tick" in payload
    )

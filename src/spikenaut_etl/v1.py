"""Map Theseus-Quarry schema-v1 envelopes onto existing Clean* columns.

Only fields that already exist on the published contracts are emitted. Missing
payload columns stay absent -- never filled with ``0.0`` or ``""``. A kind that
has no overlap with the target contract raises rather than inventing columns.
"""

from __future__ import annotations

from typing import Any

from .ingest import IngestError
from .schemas import (
    PayloadGpuSched,
    PayloadMinerPerf,
    PayloadNodeHealth,
    TelemetryEnvelope,
)

# Units documented on TelemetryPayload::MinerPerf in schema.rs, plus the
# adjacent SI prefixes a collector already emits as H/s (bzminer).
_HASHRATE_TO_MH: dict[str, float] = {
    "H/S": 1e-6,
    "KH/S": 1e-3,
    "MH/S": 1.0,
    "GH/S": 1e3,
    "TH/S": 1e6,
}


def hashrate_to_mh(hashrate: float, unit: str) -> float:
    """Convert a miner_perf hashrate into the published ``hashrate_mh`` column."""
    key = unit.strip().upper()
    scale = _HASHRATE_TO_MH.get(key)
    if scale is None:
        raise IngestError(
            f"unknown hashrate_unit {unit!r}; refusing to guess a conversion "
            f"onto hashrate_mh"
        )
    return hashrate * scale


def map_envelope_to_gpu(envelope: TelemetryEnvelope, row_index: int) -> dict[str, Any]:
    """Map a v1 envelope onto ``CleanGpuTelemetry`` columns that the payload carries."""
    payload = envelope.payload
    if isinstance(payload, PayloadGpuSched):
        return {
            "row_index": row_index,
            "gpu_temp_c": float(payload.gpu_temp_c),
            "power_w": float(payload.power_w),
        }
    raise IngestError(
        f"kind {envelope.kind!r} cannot map onto CleanGpuTelemetry; "
        "refusing to invent columns"
    )


def map_envelope_to_node_sync(envelope: TelemetryEnvelope) -> dict[str, Any]:
    """Map a v1 envelope onto ``CleanNodeSync`` columns that the payload carries.

    Null identity fields are omitted so an all-null ``block_height`` does not
    survive into the published row and trip ``no_all_null_columns``.
    """
    payload = envelope.payload
    if isinstance(payload, PayloadMinerPerf):
        return {
            "timestamp": envelope.timestamp,
            "blockchain": payload.coin,
            "hashrate_mh": hashrate_to_mh(payload.hashrate, payload.hashrate_unit),
        }
    if isinstance(payload, PayloadNodeHealth):
        row: dict[str, Any] = {
            "timestamp": envelope.timestamp,
            "blockchain": payload.coin,
        }
        if payload.height is not None:
            row["block_height"] = payload.height
        if payload.epoch is not None:
            row["chain_epoch"] = payload.epoch
        if payload.hashrate_mh is not None:
            row["hashrate_mh"] = payload.hashrate_mh
        return row
    raise IngestError(
        f"kind {envelope.kind!r} cannot map onto CleanNodeSync; "
        "refusing to invent columns"
    )

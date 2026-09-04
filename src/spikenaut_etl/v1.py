"""Map Theseus-Quarry schema-v1 envelopes onto existing Clean* columns.

Only fields that already exist on the published contracts are emitted. Missing
payload columns stay absent -- never filled with ``0.0`` or ``""``. A producer
field that overlaps a Clean* column is mapped or the row is refused; a kind
that cannot fill a published contract without inventing measurements raises.
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
    """Map a v1 envelope onto the full ``CleanGpuTelemetry`` contract.

    ``gpu_sched`` carries temp/power only. ``v3_build.py`` then requires
    ``clock_mhz``, ``gpu_clock_mhz``, and ``QUBIC_COLUMNS`` on the published
    GPU file. Filling those from nothing would be fabrication; publishing the
    sparse triple would crash build-v3. Refuse until a collector kind can
    satisfy the contract with real measurements. ``row_index`` is accepted
    only so the signature stays aligned with the cleaner.
    """
    del row_index
    payload = envelope.payload
    if isinstance(payload, PayloadGpuSched):
        raise IngestError(
            "kind 'gpu_sched' cannot fill CleanGpuTelemetry "
            "(clock_mhz, gpu_clock_mhz, QUBIC_COLUMNS) without inventing "
            "measurements; refusing sparse GPU rows that would break build-v3"
        )
    raise IngestError(
        f"kind {envelope.kind!r} cannot map onto CleanGpuTelemetry; "
        "refusing to invent columns"
    )


def map_envelope_to_node_sync(envelope: TelemetryEnvelope) -> dict[str, Any]:
    """Map a v1 envelope onto ``CleanNodeSync`` columns that the payload carries.

    Null identity fields are omitted so an all-null ``block_height`` does not
    survive into the published row and trip ``no_all_null_columns``.

    ``node_health.tick`` maps onto ``block_height`` when ``height`` is absent
    (legacy ``coin:epoch:tick``). ``speed_hs`` maps onto ``hashrate_mh``
    (÷ 1e6) when ``hashrate_mh`` is absent. Both-present collisions raise
    rather than silently dropping one of the producer fields.
    """
    payload = envelope.payload
    if isinstance(payload, PayloadMinerPerf):
        return {
            "timestamp": envelope.timestamp,
            "blockchain": payload.coin,
            "hashrate_mh": hashrate_to_mh(payload.hashrate, payload.hashrate_unit),
        }
    if isinstance(payload, PayloadNodeHealth):
        if payload.height is not None and payload.tick is not None:
            raise IngestError(
                "node_health carries both height and tick; "
                "CleanNodeSync.block_height cannot represent both; "
                "refusing to drop tick"
            )
        row: dict[str, Any] = {
            "timestamp": envelope.timestamp,
            "blockchain": payload.coin,
        }
        if payload.height is not None:
            row["block_height"] = payload.height
        elif payload.tick is not None:
            row["block_height"] = payload.tick
        if payload.epoch is not None:
            row["chain_epoch"] = payload.epoch
        if payload.hashrate_mh is not None and payload.speed_hs is not None:
            converted = payload.speed_hs / 1e6
            if converted != payload.hashrate_mh:
                raise IngestError(
                    "node_health carries both hashrate_mh and speed_hs; "
                    "they disagree after H/s → MH/s conversion; refusing to "
                    "drop a perf signal"
                )
            row["hashrate_mh"] = payload.hashrate_mh
        elif payload.hashrate_mh is not None:
            row["hashrate_mh"] = payload.hashrate_mh
        elif payload.speed_hs is not None:
            row["hashrate_mh"] = payload.speed_hs / 1e6
        return row
    raise IngestError(
        f"kind {envelope.kind!r} cannot map onto CleanNodeSync; "
        "refusing to invent columns"
    )

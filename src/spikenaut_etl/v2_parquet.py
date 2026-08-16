"""Parquet conversions of the published v2 JSONL, for config serving.

Why this exists: a script-less Hugging Face dataset gets **one** packaged
builder for every config, inferred from the first config's data files
(``datasets/load.py``, ``HubDatasetModuleFactoryWithoutScript`` — it reads
``next(iter(metadata_configs.values()))["data_files"]``). The moment the v3
Parquet configs joined the v2 JSONL configs in one card, the viewer applied
the JSON builder to Parquet bytes and every v3 config failed with
``JSON parse error: Invalid value. in row 0``. Mixed formats cannot work; the
v3 contract mandates Parquet; therefore the v2 *configs* are served from
Parquet conversions while the original JSONL files stay in ``full_data/``,
untouched and canonical.

Fidelity contract — the conversion is *defined* as "exactly what
``load_dataset('json', ...)`` yields from the published JSONL, written to
Parquet". It is produced with the ``datasets`` library itself and verified by
reloading: row counts, first/last rows, and features must match. One known,
allowed representation change: Parquet has no seconds-resolution timestamps,
so ``timestamp[s]`` columns come back as ``timestamp[ms]`` — same values,
finer unit. Anything else fails the build.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import datasets

from .v3_build import BuildError

# config name -> published v2 JSONL (relative to the dataset repo root).
V2_SOURCES: dict[str, str] = {
    "gpu_telemetry": "full_data/neuromorphic_data.jsonl",
    "mining": "full_data/node_sync_harvest.jsonl",
    "hft": "full_data/ghost_market_log.jsonl",
    "qubic_ticks": "full_data/qubic_ticks_snn.jsonl",
}

OUTPUT_DIR = "v2_parquet"


@dataclass(frozen=True)
class ConversionResult:
    config: str
    rows: int
    path: Path


def _features_equivalent(
    json_features: datasets.Features, parquet_features: datasets.Features
) -> bool:
    """Equal, allowing only the documented timestamp[s] -> timestamp[ms] shift."""
    if json_features == parquet_features:
        return True
    if set(json_features) != set(parquet_features):
        return False
    for name, json_feature in json_features.items():
        parquet_feature = parquet_features[name]
        if json_feature == parquet_feature:
            continue
        allowed = (
            isinstance(json_feature, datasets.Value)
            and isinstance(parquet_feature, datasets.Value)
            and json_feature.dtype == "timestamp[s]"
            and parquet_feature.dtype == "timestamp[ms]"
        )
        if not allowed:
            return False
    return True


def convert_source(dataset_root: Path, config: str, out_root: Path) -> ConversionResult:
    if config not in V2_SOURCES:
        raise BuildError(
            f"unknown v2 config {config!r}; expected one of {sorted(V2_SOURCES)}"
        )
    source = dataset_root / V2_SOURCES[config]
    if not source.exists():
        raise BuildError(f"missing v2 source {source}")
    try:
        loaded = datasets.load_dataset("json", data_files=str(source), split="train")
    except Exception as exc:  # noqa: BLE001 -- empty/corrupt sources surface as
        # arbitrary loader internals (even StopIteration); a failed read of one
        # named input is an expected failure mode and must exit as BuildError.
        raise BuildError(f"cannot load {source}: {exc!r}") from exc
    if loaded.num_rows == 0:
        raise BuildError(f"{source} is empty")

    directory = out_root / OUTPUT_DIR / config
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.parquet"):
        stale.unlink()
    path = directory / "train-00000.parquet"
    loaded.to_parquet(str(path))

    # Verify the fidelity contract by reloading what was just written.
    reloaded = datasets.load_dataset("parquet", data_files=str(path), split="train")
    if reloaded.num_rows != loaded.num_rows:
        raise BuildError(
            f"{config}: parquet round-trip changed row count "
            f"({loaded.num_rows} -> {reloaded.num_rows})"
        )
    if not _features_equivalent(loaded.features, reloaded.features):
        raise BuildError(
            f"{config}: parquet round-trip changed features: "
            f"{loaded.features} -> {reloaded.features}"
        )
    if loaded[0] != reloaded[0] or loaded[-1] != reloaded[-1]:
        raise BuildError(f"{config}: parquet round-trip changed row content")
    return ConversionResult(config=config, rows=reloaded.num_rows, path=path)


def build_v2_parquet(
    dataset_root: Path, output_root: Path | None = None
) -> dict[str, int]:
    """Convert every v2 source; returns config -> row count."""
    out_root = output_root or dataset_root
    results = {
        config: convert_source(dataset_root, config, out_root)
        for config in V2_SOURCES
    }
    return {config: result.rows for config, result in results.items()}

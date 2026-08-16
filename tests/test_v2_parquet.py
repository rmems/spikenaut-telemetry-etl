"""Tests for the v2 JSONL -> Parquet config conversions.

The conversion exists because a script-less Hub dataset gets exactly one
packaged builder, inferred from the first config's files — mixed JSONL and
Parquet configs made the viewer read Parquet bytes with the JSON loader.
These tests pin the fidelity contract of the replacement Parquet serving.
"""

from __future__ import annotations

import json

import datasets
import pytest

from spikenaut_etl.v2_parquet import V2_SOURCES, build_v2_parquet, convert_source
from spikenaut_etl.v3_build import BuildError


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture()
def v2_repo(tmp_path):
    """A miniature dataset repo with all four published v2 shapes."""
    write_jsonl(
        tmp_path / "full_data" / "neuromorphic_data.jsonl",
        [
            {"vddcr_gfx_v": 0.7 + i * 0.001, "gpu_temp_c": 30.0 + i, "row_index": i}
            for i in range(20)
        ],
    )
    # Mining: mixed null timestamps and chain attribution, as published.
    write_jsonl(
        tmp_path / "full_data" / "node_sync_harvest.jsonl",
        [
            {
                "hashrate_mh": 1.0 + i,
                "timestamp": f"2026-03-19T12:00:{i:02d}" if i % 3 else None,
                "blockchain": "dynex" if i % 3 == 0 else None,
                "block_height": 919876 + i if i % 3 == 0 else None,
            }
            for i in range(20)
        ],
    )
    write_jsonl(
        tmp_path / "full_data" / "ghost_market_log.jsonl",
        [
            {
                "timestamp": f"2026-03-11T18:22:{i:02d}.433458521+00:00",
                "step": i,
                "action": "observe",
                "price_usd": 70000.0 + i,
            }
            for i in range(20)
        ],
    )
    # Qubic: tz-suffixed second-resolution timestamps, the documented
    # timestamp[s] -> timestamp[ms] representation shift.
    write_jsonl(
        tmp_path / "full_data" / "qubic_ticks_snn.jsonl",
        [
            {
                "timestamp": f"2026-03-20T08:55:{i:02d}+00:00",
                "tick": 46538099 + i,
                "tick_rate": 0.4333,
            }
            for i in range(20)
        ],
    )
    return tmp_path


def test_all_sources_convert_and_report_counts(v2_repo):
    counts = build_v2_parquet(v2_repo)
    assert counts == {config: 20 for config in V2_SOURCES}
    for config in V2_SOURCES:
        assert (v2_repo / "v2_parquet" / config / "train-00000.parquet").exists()


def test_parquet_serves_identical_content(v2_repo):
    build_v2_parquet(v2_repo)
    for config, rel in V2_SOURCES.items():
        js = datasets.load_dataset(
            "json", data_files=str(v2_repo / rel), split="train"
        )
        pq_ds = datasets.load_dataset(
            "parquet",
            data_files=str(v2_repo / "v2_parquet" / config / "train-00000.parquet"),
            split="train",
        )
        assert pq_ds.num_rows == js.num_rows
        assert pq_ds[0] == js[0]
        assert pq_ds[-1] == js[-1]
        assert set(pq_ds.features) == set(js.features)


def test_nulls_survive_conversion(v2_repo):
    """Missing means null on the way in and null on the way out."""
    build_v2_parquet(v2_repo)
    pq_ds = datasets.load_dataset(
        "parquet",
        data_files=str(v2_repo / "v2_parquet" / "mining" / "train-00000.parquet"),
        split="train",
    )
    rows = list(pq_ds)
    assert any(r["timestamp"] is None for r in rows)
    assert any(r["blockchain"] is None for r in rows)
    assert not any(r["blockchain"] == "" for r in rows)


def test_timestamp_resolution_shift_is_the_only_feature_change(v2_repo):
    build_v2_parquet(v2_repo)
    for config, rel in V2_SOURCES.items():
        js = datasets.load_dataset(
            "json", data_files=str(v2_repo / rel), split="train"
        )
        pq_ds = datasets.load_dataset(
            "parquet",
            data_files=str(v2_repo / "v2_parquet" / config / "train-00000.parquet"),
            split="train",
        )
        for name, json_feature in js.features.items():
            parquet_feature = pq_ds.features[name]
            if json_feature != parquet_feature:
                assert (json_feature.dtype, parquet_feature.dtype) == (
                    "timestamp[s]",
                    "timestamp[ms]",
                ), f"{config}.{name}: {json_feature} -> {parquet_feature}"


def test_missing_source_refuses(tmp_path):
    with pytest.raises(BuildError, match="missing v2 source"):
        convert_source(tmp_path, "mining", tmp_path)


def test_empty_source_refuses(tmp_path):
    write_jsonl(tmp_path / "full_data" / "node_sync_harvest.jsonl", [])
    with pytest.raises(BuildError):
        convert_source(tmp_path, "mining", tmp_path)


def test_unknown_config_refuses(tmp_path):
    with pytest.raises(BuildError, match="unknown v2 config"):
        convert_source(tmp_path, "nonsense", tmp_path)


def test_rebuild_replaces_stale_shards(v2_repo):
    build_v2_parquet(v2_repo)
    stale = v2_repo / "v2_parquet" / "hft" / "train-00001.parquet"
    stale.write_bytes(b"stale shard")
    build_v2_parquet(v2_repo)
    assert not stale.exists()

#!/usr/bin/env python3
"""Regenerate test fixtures from the recovered originals.

Fixtures are **seeded random samples**, never prefixes.

The published dataset's ``*_SAMPLE_*.jsonl`` files were byte-exact prefixes of
their parent files. That made them unrepresentative -- ``ch2_mode`` is constant
``"NVDA"`` for the first 200 rows of ``ghost_market_log.jsonl`` but varies across
the file -- and it meant loading a sample config alongside the full file counted
the same rows twice. Sampling here follows the rule the dataset should have.

Rows that exercise schema drift are force-included, since a uniform sample of
813,973 rows will usually miss a rare variant.

Usage:
    python tools/make_fixtures.py --source ~/spikenaut-data-backup/2026-08-03
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 20260803
N_ROWS = 200

DRIFT_MARKERS = ("kaspa_", "monero_")


def reservoir_sample(path: Path, n: int, rng: random.Random) -> list[str]:
    """Uniform sample of ``n`` lines in one pass, without loading the file."""
    picked: list[str] = []
    for i, line in enumerate(path.open("r", encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        if len(picked) < n:
            picked.append(line)
        else:
            j = rng.randrange(i + 1)
            if j < n:
                picked[j] = line
    return picked


def collect_drift(path: Path, limit: int) -> list[str]:
    """Rows carrying drift fields, so the strict schema is actually exercised."""
    found: list[str] = []
    for line in path.open("r", encoding="utf-8"):
        if len(found) >= limit:
            break
        if any(marker in line for marker in DRIFT_MARKERS):
            found.append(line.strip())
    return found


def stratify_by_timestamp_form(path: Path, per_form: int) -> list[str]:
    """Sample each timestamp form separately.

    ``node_sync_harvest.jsonl`` holds three forms in very uneven proportion
    (114,250 datetime / 5,001 ``coin:height`` / 1,083 ``coin:epoch:tick``). A
    uniform sample would be ~95% datetime and would not exercise coin parsing.
    """
    buckets: dict[str, list[str]] = {}
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ts = json.loads(line).get("timestamp", "")
        if ts[:4].isdigit():
            key = "datetime"
        elif ts.count(":") == 2 and not ts[:1].isdigit():
            key = "coin_epoch_tick"
        elif ":" in ts:
            key = "coin_height"
        else:
            key = "other"
        bucket = buckets.setdefault(key, [])
        if len(bucket) < per_form:
            bucket.append(line)
        if all(len(b) >= per_form for b in buckets.values()) and len(buckets) >= 3:
            if sum(len(b) for b in buckets.values()) >= per_form * 3:
                break
    return [line for _, bucket in sorted(buckets.items()) for line in bucket]


def write(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {path.name:<32} {len(lines):>4} rows")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="directory holding the recovered original JSONL files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "fixtures",
    )
    args = parser.parse_args()
    rng = random.Random(SEED)

    print(f"Regenerating fixtures from {args.source} (seed={SEED}):")

    # Uniform random samples.
    for name in ("ghost_market_log.jsonl", "qubic_ticks.jsonl"):
        src = args.source / name
        if not src.exists():
            print(f"  SKIP {name}: not found")
            continue
        write(reservoir_sample(src, N_ROWS, rng), args.out / name)

    # GPU telemetry: random sample plus forced drift rows.
    src = args.source / "neuromorphic_data.jsonl"
    if src.exists():
        sample = reservoir_sample(src, N_ROWS - 50, rng)
        drift = collect_drift(src, 50)
        write(sample + drift, args.out / "neuromorphic_data.jsonl")

    # Mining telemetry: stratified so all three timestamp forms appear.
    src = args.source / "node_sync_harvest.jsonl"
    if src.exists():
        write(stratify_by_timestamp_form(src, 70), args.out / "node_sync_harvest.jsonl")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

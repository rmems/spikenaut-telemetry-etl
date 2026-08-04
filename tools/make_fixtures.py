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
import re
from collections import deque
from pathlib import Path

SEED = 20260803
N_ROWS = 200

DRIFT_MARKERS = ("kaspa_", "monero_")


def reservoir_sample(path: Path, n: int, rng: random.Random) -> list[str]:
    """Uniform sample of ``n`` lines in one pass, in original file order.

    Reservoir order is shuffled; these are time series, so the sample is sorted
    back into source order before being written. A sample whose records are out
    of chronological order is not a sample of a capture -- and it would trip the
    trailing-disorder rule for a reason that has nothing to do with the data.
    """
    picked: list[tuple[int, str]] = []
    for i, line in enumerate(path.open("r", encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        if len(picked) < n:
            picked.append((i, line))
        else:
            j = rng.randrange(i + 1)
            if j < n:
                picked[j] = (i, line)
    picked.sort(key=lambda pair: pair[0])
    return [line for _, line in picked]


def collect_drift(path: Path, limit: int) -> list[str]:
    """Rows carrying drift fields, so the strict schema is actually exercised."""
    found: list[str] = []
    for line in path.open("r", encoding="utf-8"):
        if len(found) >= limit:
            break
        if any(marker in line for marker in DRIFT_MARKERS):
            found.append(line.strip())
    return found


_UTC_OFFSET = re.compile(r"[+-]\d\d:\d\d$")

# Rows appended after the capture ended, out of time order. In the real file
# these are the last 12 rows and the only ones carrying a UTC offset.
PLACEHOLDER_TAIL = "placeholder_tail"


def _timestamp_form(ts: str) -> str:
    if ts[:4].isdigit():
        return PLACEHOLDER_TAIL if _UTC_OFFSET.search(ts) else "datetime"
    if ts.count(":") == 2:
        return "coin_epoch_tick"
    if ":" in ts:
        return "coin_height"
    return "other"


def stratify_by_timestamp_form(path: Path, per_form: int) -> list[str]:
    """Sample each timestamp form separately, preserving the real file layout.

    ``node_sync_harvest.jsonl`` holds its forms in very uneven proportion
    (114,238 datetime / 5,001 ``coin:height`` / 1,083 ``coin:epoch:tick`` / 12
    placeholder). A uniform sample would be ~95% datetime and would exercise
    neither coin parsing nor the trailing-disorder rule.

    Ordering matters here in a way it does not for the other fixtures: the
    placeholder rows are only detectable *because* they sit at the end, after a
    backward jump in time. They are emitted last, exactly as in the source.
    """
    buckets: dict[str, list[str]] = {}
    # The datetime rows must be the ones immediately *preceding* the placeholders,
    # not the first ones in the file. The placeholders sit at 2026-03-19 17:00,
    # which is later than the start of the capture -- so sampling from the front
    # would leave the fixture in ascending order and reproduce no backward jump.
    # Keeping a trailing window preserves the real adjacency.
    trailing_window: deque[str] = deque(maxlen=per_form)

    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        key = _timestamp_form(json.loads(line).get("timestamp", ""))
        if key == "datetime":
            trailing_window.append(line)
            continue
        bucket = buckets.setdefault(key, [])
        # Keep every placeholder row; they are few and the point of the fixture.
        if key == PLACEHOLDER_TAIL or len(bucket) < per_form:
            bucket.append(line)

    buckets["datetime"] = list(trailing_window)
    ordered = ["coin_height", "coin_epoch_tick", "datetime", "other", PLACEHOLDER_TAIL]
    return [line for key in ordered for line in buckets.get(key, [])]


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

    # Mining telemetry: stratified so every timestamp form appears.
    #
    # 100 per form, not 70, so the 12 placeholder rows stay under the 5% row-count
    # delta gate (12/312 = 3.8%). Shrinking the fixture would mean loosening that
    # threshold on the real 120,334-row file, where the same 12 rows are 0.01%.
    src = args.source / "node_sync_harvest.jsonl"
    if src.exists():
        write(stratify_by_timestamp_form(src, 100), args.out / "node_sync_harvest.jsonl")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

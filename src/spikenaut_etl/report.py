"""Per-file data-quality reports.

The prior pipeline printed ``✓ Cleanup complete!`` while emitting 813,973
identical empty records. A report that states distinct-row ratio and per-column
variance makes that failure visible at a glance, whether or not a gate catches it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .clean import CleanResult
from .validate import ValidationResult, column_stats


@dataclass
class ColumnProfile:
    name: str
    distinct: int
    nulls: int
    fill_rate: float
    sample: list[Any] = field(default_factory=list)


@dataclass
class FileReport:
    source: str
    rows_in: int
    rows_out: int
    distinct_rows: int
    distinct_ratio: float
    quarantined: int
    quarantined_by_reason: dict[str, int]
    dead_columns: list[str]
    dead_column_drift: list[str]
    coin_counts: dict[str, int]
    columns: list[ColumnProfile]
    gates_passed: bool
    gate_failures: list[str]
    generated_at: str


def profile(rows: Sequence[dict[str, Any]]) -> list[ColumnProfile]:
    if not rows:
        return []
    out: list[ColumnProfile] = []
    for name, counter in column_stats(rows).items():
        nulls = counter.get(None, 0)
        non_null = [v for v in counter if v is not None]
        out.append(
            ColumnProfile(
                name=name,
                distinct=len(non_null),
                nulls=nulls,
                fill_rate=round(1 - nulls / len(rows), 6),
                sample=sorted(non_null, key=repr)[:5],
            )
        )
    return out


def build(result: CleanResult, validation: ValidationResult) -> FileReport:
    rows = result.rows
    distinct = len({repr(sorted(r.items(), key=lambda kv: kv[0])) for r in rows})
    return FileReport(
        source=result.name,
        rows_in=result.n_in,
        rows_out=len(rows),
        distinct_rows=distinct,
        distinct_ratio=round(distinct / len(rows), 8) if rows else 0.0,
        quarantined=len(result.quarantine),
        quarantined_by_reason=result.quarantine.counts(),
        dead_columns=sorted(result.dead_columns),
        dead_column_drift=list(result.dead_column_drift),
        coin_counts=dict(result.coin_counts),
        columns=profile(rows),
        gates_passed=validation.ok,
        gate_failures=[str(f) for f in validation.failures],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def ingest_failure(source: str, detail: str, *, n_in: int = 0) -> FileReport:
    """Diagnostic for an ingest abort: nothing cleaned, nothing written."""
    return FileReport(
        source=source,
        rows_in=n_in,
        rows_out=0,
        distinct_rows=0,
        distinct_ratio=0.0,
        quarantined=0,
        quarantined_by_reason={},
        dead_columns=[],
        dead_column_drift=[],
        coin_counts={},
        columns=[],
        gates_passed=False,
        gate_failures=[f"ingest: {detail}"],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def write(report: FileReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.source}.json"
    path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
    return path


def render(report: FileReport) -> str:
    status = "PASS" if report.gates_passed else "FAIL"
    lines = [
        f"{status}  {report.source}",
        f"      rows      {report.rows_in} in -> {report.rows_out} out",
        f"      distinct  {report.distinct_rows} ({report.distinct_ratio:.4%})",
    ]
    if report.quarantined:
        detail = ", ".join(
            f"{k}={v}" for k, v in sorted(report.quarantined_by_reason.items())
        )
        lines.append(f"      quarantined  {report.quarantined} rows ({detail})")
    if report.dead_columns:
        lines.append(f"      dropped   {', '.join(report.dead_columns)}")
    for message in report.dead_column_drift:
        lines.append(f"      DRIFT     {message}")
    if report.coin_counts:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(report.coin_counts.items()))
        lines.append(f"      coins     {counts}")
    for failure in report.gate_failures:
        lines.append(f"      {failure}")
    return "\n".join(lines)

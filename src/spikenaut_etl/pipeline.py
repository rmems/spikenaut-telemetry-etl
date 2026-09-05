"""Pipeline orchestration: ingest -> clean -> validate -> write.

The ordering is load-bearing. Output is written only after
:func:`~spikenaut_etl.validate.assert_publishable` returns, so a run that fails a
gate leaves the previous artifact untouched rather than overwriting it with
degenerate data.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import clean, report
from .ingest import IngestError
from .schemas import CLEAN_COLUMNS
from .validate import GateConfig, ValidationError, assert_publishable, check_all

Cleaner = Callable[[Path], clean.CleanResult]


@dataclass(frozen=True)
class SourceSpec:
    """One source file and the gate configuration it is held to."""

    key: str
    filename: str
    cleaner: Cleaner
    output: str
    gates: GateConfig
    # Published sample prefix, e.g. "mining" -> samples/mining_SAMPLE_1k.jsonl.
    # None means this source publishes no sample.
    sample_prefix: str | None = None

    def input_path(self, root: Path) -> Path:
        return root / self.filename


# ``timestamp`` on node_sync is legitimately null for coin-tagged rows, and
# ``epoch`` on qubic is a single epoch (205) across the whole capture -- both are
# genuine invariants of the source, not collapse, so they are exempted from the
# constant-column gate rather than silently dropped.
SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="neuromorphic_data",
        filename="neuromorphic_data.jsonl",
        cleaner=clean.clean_gpu_telemetry,
        output="full_data/neuromorphic_data.jsonl",
        # No timestamp column exists in this source, so the fabrication gate
        # has nothing to check.
        gates=GateConfig(require_timestamp_jitter=False),
        sample_prefix="gpu",
    ),
    SourceSpec(
        key="node_sync_harvest",
        filename="node_sync_harvest.jsonl",
        cleaner=clean.clean_node_sync,
        output="full_data/node_sync_harvest.jsonl",
        gates=GateConfig(
            allow_constant=frozenset(
                {"timestamp", "blockchain", "block_height", "chain_epoch"}
            ),
            identity_columns=GateConfig.identity_columns | {"chain_epoch"},
        ),
        sample_prefix="mining",
    ),
    SourceSpec(
        key="qubic_ticks_snn",
        filename="qubic_ticks.jsonl",
        cleaner=clean.clean_qubic_ticks,
        output="full_data/qubic_ticks_snn.jsonl",
        gates=GateConfig(allow_constant=frozenset({"epoch", "epoch_progress"})),
        sample_prefix="qubic",
    ),
    SourceSpec(
        key="ghost_market_log",
        filename="ghost_market_log.jsonl",
        cleaner=clean.clean_trading_log,
        output="full_data/ghost_market_log.jsonl",
        gates=GateConfig(),
        sample_prefix="hft",
    ),
)

# Published sample sizes, as (row count, filename suffix).
SAMPLE_SIZES: tuple[tuple[int, str], ...] = ((100, "100"), (1000, "1k"))


@dataclass
class RunOutcome:
    key: str
    ok: bool
    rendered: str
    report_path: Path | None = None
    output_path: Path | None = None
    sample_paths: list[Path] = field(default_factory=list)


def run_source(
    spec: SourceSpec,
    input_root: Path,
    output_root: Path,
    report_dir: Path,
    *,
    write_output: bool = True,
) -> RunOutcome:
    """Clean and validate one source. Writes output only if every gate passes."""
    path = spec.input_path(input_root)
    if not path.exists():
        return RunOutcome(spec.key, False, f"SKIP  {spec.key}: {path} not found")

    try:
        result = spec.cleaner(path)
    except IngestError as exc:
        file_report = report.ingest_failure(spec.key, str(exc))
        report_path = report.write(file_report, report_dir)
        return RunOutcome(
            spec.key,
            False,
            f"{report.render(file_report)}\n{exc}",
            report_path,
        )
    except (UnicodeDecodeError, OSError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
        file_report = report.ingest_failure(spec.key, detail)
        report_path = report.write(file_report, report_dir)
        return RunOutcome(
            spec.key,
            False,
            f"{report.render(file_report)}\n{detail}",
            report_path,
        )
    validation = check_all(
        spec.key,
        result.rows,
        n_in=result.n_in,
        config=spec.gates,
        expected_columns=_expected_columns(spec, result.rows),
        timestamps=result.epochs or None,
    )
    file_report = report.build(result, validation)
    report_path = report.write(file_report, report_dir)
    rendered = report.render(file_report)

    try:
        assert_publishable(validation)
    except ValidationError as exc:
        return RunOutcome(spec.key, False, f"{rendered}\n{exc}", report_path)

    output_path = None
    sample_paths: list[Path] = []
    if write_output:
        output_path = output_root / spec.output
        write_jsonl(result.rows, output_path)

        if spec.sample_prefix:
            for size, suffix in SAMPLE_SIZES:
                sample_paths.append(
                    write_sample(
                        result.rows,
                        output_root
                        / "samples"
                        / f"{spec.sample_prefix}_SAMPLE_{suffix}.jsonl",
                        size,
                    )
                )

    return RunOutcome(spec.key, True, rendered, report_path, output_path, sample_paths)


def run_all(
    input_root: Path,
    output_root: Path,
    report_dir: Path,
    *,
    only: Sequence[str] | None = None,
    write_output: bool = True,
) -> list[RunOutcome]:
    selected = [s for s in SOURCES if not only or s.key in only]
    return [
        run_source(s, input_root, output_root, report_dir, write_output=write_output)
        for s in selected
    ]


def write_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), default=str))
            handle.write("\n")
    return path


def write_sample(
    rows: Sequence[dict[str, Any]], path: Path, n: int, *, seed: int = 20260803
) -> Path:
    """Write a random sample.

    The published samples were byte-exact *prefixes* of their parent files, so
    loading a sample config alongside the full file counted the same rows twice
    and the "sample" only ever showed the opening minutes of a capture. A seeded
    random draw is reproducible and actually representative.

    The draw is sorted back into source order. These are time series: a sample
    whose records are shuffled is not a sample of a capture, and it would trip
    the trailing-disorder rule for a reason having nothing to do with the data.
    """
    if len(rows) <= n:
        return write_jsonl(rows, path)
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(rows)), n))
    return write_jsonl([rows[i] for i in picked], path)


def _expected_columns(
    spec: SourceSpec, rows: Sequence[dict[str, Any]]
) -> AbstractSet[str] | None:
    """Declared columns minus any dropped as dead in this run.

    Dead-column removal is data-dependent, so the schema gate checks that what
    survived is a subset of the contract -- not that every declared field is
    present regardless of whether the source still carries it.
    """
    declared = CLEAN_COLUMNS.get(spec.key)
    if declared is None or not rows:
        return None
    present: set[str] = set()
    for row in rows:
        present.update(row)
    undeclared = present - declared
    if undeclared:
        # Let the gate report it rather than silently narrowing the contract.
        return declared
    return present

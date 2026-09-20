"""Command line entry point.

    spikenaut-etl validate --input <dir>     gates only, writes nothing
    spikenaut-etl clean    --input <dir> --output <dir>
    spikenaut-etl report   --input <dir>
    spikenaut-etl build-v3 --input <dataset-repo> [--output <dataset-repo>]
    spikenaut-etl audit-v3 --input <dataset-repo> --output <audit-dir>
    spikenaut-etl prepare-anticipation --input <campaign.json> --output <prepared-dir>

Exit status is 1 if any source fails a gate, so CI fails on corrupt data.

``validate`` / ``clean`` accept collector JSONL *or* published Vault
``full_data`` Clean* JSONL (flattened GPU, null timestamps on coin-tagged
harvest rows). That is the ``--profile auto`` published-shape path.

Spikenaut LiveStimAdapter live-bank input is **not** ``full_data`` GPU JSONL.
It is the v3 parquet projection:

    v3/state_telemetry/{train,validation,test}-00000.parquet

which carries ``sm_clock_mhz`` after STATE_BACKFILL. Published v2 GPU JSONL
has ``gpu_clock_mhz``; this CLI will not invent ``sm_clock_mhz`` from it.
Use ``--profile live-columns`` to validate a stripped LIVE_COLUMNS JSONL
projection of that parquet (this CLI does not read the parquet shards).
``--profile published`` requires already-cleaned Vault ``full_data`` rows;
``--profile raw`` requires nested collector JSONL.

``build-v3`` reads the *published v2 JSONL* in a dataset-repo checkout (not the
raw backup) and writes the additive ``v3/`` Parquet tree plus the
``v2_parquet/`` config conversions; --output defaults to the same checkout. It
needs the ``v3`` optional dependency (pyarrow + datasets).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contracts import IngestProfile
from .pipeline import SOURCES, run_all

DEFAULT_REPORTS = Path("reports")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spikenaut-etl",
        description="Clean and validate Spikenaut telemetry before publication.",
    )
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "clean",
            "report",
            "build-v3",
            "audit-v3",
            "prepare-anticipation",
        ),
        help="action to perform",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="directory holding collector JSONL, published Vault full_data/, "
        "or a dataset-repo checkout (full_data/*.jsonl is discovered). "
        "--profile live-columns needs stripped sm_clock_mhz JSONL; "
        "v3/state_telemetry parquet is the LiveStimAdapter consumer path, "
        "not something validate reads. For build-v3: the dataset repo "
        "checkout holding full_data/. For audit-v3: the v3 dataset checkout; "
        "for prepare-anticipation: the preassigned campaign JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory (required for clean, audit-v3, prepare-anticipation)",
    )
    parser.add_argument(
        "--reports", type=Path, default=DEFAULT_REPORTS, help="report output directory"
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[s.key for s in SOURCES],
        help="restrict to specific sources",
    )
    parser.add_argument(
        "--profile",
        choices=("auto", "raw", "published", "live-columns"),
        default="auto",
        help="ingest shape: auto accepts collector JSONL or published Clean* "
        "full_data; raw requires nested telemetry; published requires cleaned "
        "Vault JSONL; live-columns requires stripped sm_clock_mhz LIVE_COLUMNS "
        "JSONL (project v3/state_telemetry parquet yourself). Does not invent "
        "sm_clock from gpu_clock",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "clean" and args.output is None:
        print("error: --output is required for 'clean'", file=sys.stderr)
        return 2

    if args.command == "prepare-anticipation":
        if args.output is None:
            print(
                "error: --output is required for 'prepare-anticipation'", file=sys.stderr
            )
            return 2
        try:
            from .anticipation import PreparationError, prepare_campaign

            prepared = prepare_campaign(args.input, args.output)
        except (ImportError, PreparationError) as exc:
            print(f"prepare-anticipation failed: {exc}", file=sys.stderr)
            return 1
        examples = sum(len(session["examples"]) for session in prepared["sessions"])
        print(
            f"prepare-anticipation complete: {len(prepared['sessions'])} sessions, "
            f"{examples} eligible examples; wrote {args.output / 'prepared.json'}"
        )
        return 0

    if args.command == "audit-v3":
        if args.output is None:
            print("error: --output is required for 'audit-v3'", file=sys.stderr)
            return 2
        try:
            from .audit_v3 import AuditError, audit_v3

            audit = audit_v3(args.input, args.output)
        except (ImportError, AuditError) as exc:
            print(f"audit-v3 failed: {exc}", file=sys.stderr)
            return 1
        counts = ", ".join(
            f"{split}={details['eligible_rows']}/{details['source_rows']}"
            for split, details in sorted(audit.splits.items())
        )
        print(f"audit-v3 complete: {audit.view_id}; {counts}")
        return 0

    if args.command == "build-v3":
        # Deferred import: the base install stays pyarrow-free for validate/clean.
        try:
            from .v3_build import BuildError, build_v3
        except ImportError as exc:
            print(
                f"error: build-v3 needs the 'v3' extra: {exc}\n"
                "install it with: pip install 'spikenaut-telemetry-etl[v3]'",
                file=sys.stderr,
            )
            return 2

        try:
            build_report = build_v3(args.input, args.output)
        except ImportError as exc:
            print(
                f"error: build-v3 needs the 'v3' extra: {exc}\n"
                "install it with: pip install 'spikenaut-telemetry-etl[v3]'",
                file=sys.stderr,
            )
            return 2
        except BuildError as exc:
            print(f"build-v3 failed: {exc}", file=sys.stderr)
            return 1
        for config, split_rows in sorted(build_report.configs.items()):
            counts = ", ".join(f"{s}={n}" for s, n in split_rows.items())
            print(f"  {config:<20} {counts}")
        print("build-v3 complete; report written to v3/build_report.json")
        return 0

    write_output = args.command == "clean"
    profile: IngestProfile = args.profile
    outcomes = run_all(
        input_root=args.input,
        output_root=args.output or Path("."),
        report_dir=args.reports,
        only=args.only,
        write_output=write_output,
        profile=profile,
    )

    for outcome in outcomes:
        print(outcome.rendered)
        if outcome.output_path:
            print(f"      wrote     {outcome.output_path}")
        for sample in outcome.sample_paths:
            print(f"      sample    {sample}")

    failed = [o for o in outcomes if not o.ok]
    print()
    if failed:
        print(
            f"{len(failed)}/{len(outcomes)} source(s) failed validation: "
            f"{', '.join(o.key for o in failed)}"
        )
        return 1
    print(f"All {len(outcomes)} source(s) passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
